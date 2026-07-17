"""Coupled stellar-mass, dust, filter, and Chimera I/O gold tests."""

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from jaxsedfit.benchmark import (
    CHIMERA_FILTER_NAMES,
    _build_chimera_filter_set,
    load_chimera_benchmark_dataset,
)
from jaxsedfit.filters import load_filter_curve, normalize_filter_curve
from jaxsedfit.model import (
    GRAHSP_BIATTENUATION_BREAK_A,
    _analytic_delayed_age_weights,
    _apply_biattenuation,
    _project_filters,
    _redshift_to_obs,
)
from jaxsedfit.preload import (
    _lnu_lsun_per_hz_to_llambda_w_per_a_np,
    _pack_loaded_filters,
    _pack_loaded_filters_jax,
    _prepare_loaded_filter,
    _surviving_fraction_for_imf,
)


REFERENCE = Path(__file__).parent / "fixtures" / "stellar_nebular_gold_v1.npz"
C_M_S = 2.99792458e8


@pytest.fixture(scope="module")
def stellar_gold():
    with np.load(REFERENCE) as data:
        yield {key: np.asarray(data[key]) for key in data.files}


def _exact_filter_mjy(obs_wave, obs_flambda, loaded_filter):
    in_band = (obs_wave >= loaded_filter.wave[0]) & (obs_wave <= loaded_filter.wave[-1])
    wave = np.unique(np.concatenate((obs_wave[in_band], loaded_filter.wave)))
    trans = np.interp(wave, loaded_filter.wave, loaded_filter.native_transmission)
    values = np.interp(wave, obs_wave, obs_flambda, left=0.0, right=0.0)
    numerator = np.trapezoid(values * trans, x=wave)
    denominator = np.trapezoid(trans / wave**2, x=wave)
    return 1.0e-10 / C_M_S * 1.0e29 * numerator / denominator


def _project_one_ssp(stellar_gold, redshift, curves, distance=2.3e25):
    rest_wave = stellar_gold["fsps_full_wave_a"]
    rest_lum = _lnu_lsun_per_hz_to_llambda_w_per_a_np(
        rest_wave, stellar_gold["fsps_full_lnu_lsun_per_hz"][1]
    )
    low = min(float(np.min(curve.wave)) for curve in curves)
    high = max(float(np.max(curve.wave)) for curve in curves)
    obs_wave = np.linspace(low, high, 40001)
    obs_flux = np.asarray(_redshift_to_obs(rest_wave, rest_lum, obs_wave, redshift, distance))
    loaded = [_prepare_loaded_filter(obs_wave, curve) for curve in curves]
    packed = _pack_loaded_filters_jax(_pack_loaded_filters(loaded))
    actual = np.asarray(_project_filters(obs_flux, packed))
    expected = np.asarray([_exact_filter_mjy(obs_wave, obs_flux, filt) for filt in loaded])
    return actual, expected


@pytest.mark.parametrize("redshift", [0.0, 0.49, 0.50, 0.51, 1.0, 2.0])
def test_one_formed_solar_mass_fsps_ssp_has_correct_flux_at_each_redshift(stellar_gold, redshift):
    """Test SSP units, 1+z dimming, distance scaling, and filter integration together."""
    from jaxsedfit.config import FilterCurve

    curves = []
    for center, width in ((3600.0, 700.0), (8000.0, 1600.0), (22000.0, 4000.0)):
        wave = np.linspace(center - width, center + width, 1001)
        transmission = np.maximum(1.0 - np.abs(wave - center) / width, 0.0)
        curves.append(normalize_filter_curve(FilterCurve(str(center), wave, transmission)))
    actual, expected = _project_one_ssp(stellar_gold, redshift, curves)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-30)


def test_current_stellar_mass_and_formed_mass_match_cigale_definition(stellar_gold):
    """CIGALE ``stellar.m_star`` is surviving mass; close the conversion explicitly."""
    ages = stellar_gold["delayed_fsps_lg_age_gyr"]
    age_weights = np.asarray(_analytic_delayed_age_weights(5.0, 1.7, ages))
    surviving_by_age = _surviving_fraction_for_imf(ages, "chabrier_2003")
    surviving_fraction = float(np.sum(age_weights * surviving_by_age))
    cigale_stellar_m_star = 10.0**9.4
    formed_mass = cigale_stellar_m_star / surviving_fraction

    assert 0.0 < surviving_fraction < 1.0
    assert formed_mass > cigale_stellar_m_star
    assert formed_mass * surviving_fraction == pytest.approx(cigale_stellar_m_star, rel=2e-12)


def test_full_delayed_sfh_fsps_spectrum_matches_independent_cigale_bin_convolution(stellar_gold):
    """Compare the analytic production weights with dense CIGALE delayed-SFH bin integrals."""
    ages_gyr = 10.0 ** stellar_gold["delayed_fsps_lg_age_gyr"]
    age, tau = 5.0, 1.7
    log_age = np.log10(ages_gyr)
    edges = 10.0 ** np.concatenate(
        (
            [log_age[0] - 0.5 * (log_age[1] - log_age[0])],
            0.5 * (log_age[:-1] + log_age[1:]),
            [log_age[-1] + 0.5 * (log_age[-1] - log_age[-2])],
        )
    )
    edges = np.clip(edges, 0.0, age)
    # CIGALE's delayed SFH is t exp(-t/tau)/tau^2 in formation time; an SSP
    # age bin [a0,a1] corresponds to formation times [age-a1, age-a0].
    def cumulative(t):
        return 1.0 - np.exp(-t / tau) * (1.0 + t / tau)

    dense_weights = cumulative(age - edges[:-1]) - cumulative(age - edges[1:])
    dense_weights = np.clip(dense_weights, 0.0, None)
    dense_weights /= dense_weights.sum()
    actual_weights = np.asarray(_analytic_delayed_age_weights(age, tau, np.log10(ages_gyr)))
    spectra = _lnu_lsun_per_hz_to_llambda_w_per_a_np(
        stellar_gold["fsps_full_wave_a"], stellar_gold["delayed_fsps_full_lnu_lsun_per_hz"]
    )
    actual_spectrum = np.tensordot(actual_weights, spectra, axes=1)
    expected_spectrum = np.tensordot(dense_weights, spectra, axes=1)

    np.testing.assert_allclose(actual_weights, dense_weights, rtol=2e-10, atol=2e-13)
    np.testing.assert_allclose(actual_spectrum, expected_spectrum, rtol=2e-10, atol=1e-25)


@pytest.mark.parametrize("ebv", [0.0, 0.05, 0.2, 0.8])
def test_stellar_dust_attenuation_and_absorbed_energy_match_grahsp_gold(stellar_gold, ebv):
    wave = stellar_gold["fsps_full_wave_a"]
    host = _lnu_lsun_per_hz_to_llambda_w_per_a_np(
        wave, stellar_gold["fsps_full_lnu_lsun_per_hz"][1]
    )
    curve = 1.2 * (wave / GRAHSP_BIATTENUATION_BREAK_A) ** np.where(
        wave < GRAHSP_BIATTENUATION_BREAK_A, -1.2, -3.0
    )
    expected = host * 10.0 ** (-0.4 * ebv * curve)
    actual, _, absorbed, absorbed_luminosity = _apply_biattenuation(
        wave, host, np.zeros_like(host), ebv, 0.0, -1.2, -3.0, 1.2,
        GRAHSP_BIATTENUATION_BREAK_A,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-30)
    np.testing.assert_allclose(absorbed, np.clip(host - expected, 0.0, None), rtol=2e-12, atol=1e-30)
    assert float(absorbed_luminosity) == pytest.approx(
        np.trapezoid(np.clip(host - expected, 0.0, None), x=wave), rel=2e-12
    )


def _chimera_curves():
    custom = {curve.name: normalize_filter_curve(curve) for curve in _build_chimera_filter_set().curves}
    return [custom.get(name, None) or load_filter_curve(name) for name in CHIMERA_FILTER_NAMES]


def test_all_chimera_filters_match_exact_cigale_fnu_integral_through_outlier_redshift(stellar_gold):
    """Sweep the real nine-band filter set densely through the suspicious z=0.4 region."""
    curves = _chimera_curves()
    for redshift in np.concatenate((np.linspace(0.0, 1.0, 11), np.linspace(0.37, 0.43, 13))):
        actual, expected = _project_one_ssp(stellar_gold, float(redshift), curves)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=1e-30)


def _write_chimera_table(path, row, *, truth):
    columns = [fits.Column(name="id", format="32A", array=[row["id"]])]
    if truth:
        values = {
            "MASS_MED_GAL": row["MASS_MED_GAL"], "resample_weight": 1,
            "chimera_QSO_weight": 0.0, "ID_COSMOS": 1, "redshift": row["redshift"],
        }
    else:
        values = {"ID_COSMOS": 1, "redshift": row["redshift"], "chimera_QSO_weight": 0.0, "resample_weight": 1}
        for name, value in row["fluxes"].items():
            column = {"spitzer.irac.I1": "IRAC1", "spitzer.irac.I2": "IRAC2"}.get(name, name)
            values[column] = value
            values[f"{column}_err"] = abs(value) * 0.01
    for name, value in values.items():
        fmt = "K" if isinstance(value, int) else "D"
        columns.append(fits.Column(name=name, format=fmt, array=[value]))
    fits.BinTableHDU.from_columns(columns).writeto(path)


def test_chimera_input_mjy_round_trip_recovers_injected_stellar_mass(tmp_path):
    """Prove that the FITS adapter applies no hidden magnitude or microJy conversion."""
    data_dir = tmp_path / "data" / "chimeras-2023-10-11"
    data_dir.mkdir(parents=True)
    log_mass = 9.35
    per_solar_mass = {name: (i + 1.0) * 2.0e-10 for i, name in enumerate(CHIMERA_FILTER_NAMES)}
    fluxes = {name: value * 10.0**log_mass for name, value in per_solar_mass.items()}
    row = {"id": "unit-roundtrip", "redshift": 0.5, "MASS_MED_GAL": log_mass, "fluxes": fluxes}
    _write_chimera_table(data_dir / "chimeras-grahsp.fits", row, truth=False)
    _write_chimera_table(data_dir / "chimeras-fullinfo.fits", row, truth=True)

    loaded = load_chimera_benchmark_dataset(tmp_path).rows[0]
    observed = np.asarray([loaded[name] for name in CHIMERA_FILTER_NAMES])
    template = np.asarray([per_solar_mass[name] for name in CHIMERA_FILTER_NAMES])
    recovered_log_mass = np.log10(np.dot(observed, template) / np.dot(template, template))

    np.testing.assert_allclose(observed, [fluxes[name] for name in CHIMERA_FILTER_NAMES], rtol=0.0, atol=0.0)
    assert loaded["log_stellar_mass_truth"] == pytest.approx(log_mass)
    assert recovered_log_mass == pytest.approx(log_mass, abs=2e-12)
