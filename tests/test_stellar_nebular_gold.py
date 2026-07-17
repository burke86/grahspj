"""External gold tests for the coupled FSPS + CIGALE nebular path."""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsedfit.model import (
    GRAHSP_BIATTENUATION_BREAK_A,
    _absorbed_line_luminosity,
    _attenuation_curve,
    _cigale_nebular_correction,
    _project_filters,
    _project_local_nebular_line_filters,
    _redshift_to_obs,
)
from jaxsedfit.preload import (
    PackedFilterCurvesJax,
    PackedFiltersJax,
    _lnu_lsun_per_hz_to_llambda_w_per_a_np,
    _load_nebular_templates_jax,
    _ssp_lyman_basis_np,
)


REFERENCE = Path(__file__).parent / "fixtures" / "stellar_nebular_gold_v1.npz"
C_M_S = 2.99792458e8


@pytest.fixture(scope="module")
def gold():
    with np.load(REFERENCE) as data:
        yield {key: np.asarray(data[key]) for key in data.files}


def _grid_point(templates, gold):
    iz = int(np.flatnonzero(np.isclose(np.asarray(templates.z_grid), gold["zgas"]))[0])
    iu = int(np.flatnonzero(np.isclose(np.asarray(templates.logu_grid), gold["logu"]))[0])
    ine = int(np.flatnonzero(np.isclose(np.asarray(templates.ne_grid), gold["ne"]))[0])
    return iz, iu, ine


def test_fsps_v3_2_ionizing_photon_rate_per_formed_mass(gold):
    """Recover Q(H) for three public FSPS SSPs, including the unit conversion."""
    assert str(gold["fsps_version"]) == "3.2"
    assert "DSPS_data" in str(gold["fsps_source_url"])
    llambda = _lnu_lsun_per_hz_to_llambda_w_per_a_np(
        gold["fsps_wave_a"], gold["fsps_lnu_lsun_per_hz"]
    )[None, :, :]

    n_ly, ly_lum = _ssp_lyman_basis_np(gold["fsps_wave_a"], llambda)

    np.testing.assert_allclose(n_ly[0], gold["fsps_n_ly_per_msun"], rtol=3e-13)
    np.testing.assert_allclose(ly_lum[0], gold["fsps_ly_lum_w_per_msun"], rtol=3e-13)
    assert n_ly[0, 0] > n_ly[0, 1] > n_ly[0, 2]


def test_cigale_line_luminosities_for_identical_nebular_parameters(gold):
    """Match all 129 CIGALE lines after applying the same photon budget."""
    templates = _load_nebular_templates_jax(True)
    iz, iu, ine = _grid_point(templates, gold)
    n_ly = 2.75e52
    f_esc, f_dust = 0.17, 0.23
    correction = float(_cigale_nebular_correction(f_esc, f_dust))
    actual = np.asarray(templates.line_lumin_per_photon)[iz, iu, ine] * n_ly * correction
    expected = gold["line_lumin_w_per_photon"] * n_ly * correction

    np.testing.assert_allclose(np.asarray(templates.line_wave_a), gold["line_wave_a"], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0.0)


def test_cigale_nebular_continuum_normalization(gold):
    """Match CIGALE's W/Angstrom/photon continuum and its integrated power."""
    templates = _load_nebular_templates_jax(True)
    iz, iu, ine = _grid_point(templates, gold)
    wave = np.asarray(templates.continuum_wave_a)
    actual = np.asarray(templates.continuum_lumin_per_a_per_photon)[iz, iu, ine]
    expected = np.interp(
        wave,
        gold["continuum_wave_a"],
        gold["continuum_lumin_w_per_a_per_photon"],
    )

    # One tabulated discontinuity is represented on opposite sides of the
    # resampling coordinate; its absolute power is negligible.
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-29)
    np.testing.assert_allclose(
        np.trapezoid(actual, x=wave),
        np.trapezoid(gold["continuum_lumin_w_per_a_per_photon"], x=gold["continuum_wave_a"]),
        # The vendored 1,600-point grid preserves the integral of CIGALE's
        # native 3,130-point table to 0.2%.
        rtol=2.5e-3,
    )


def test_line_attenuation_and_cigale_energy_balance_bookkeeping(gold):
    line_lumin = gold["line_lumin_w_per_photon"] * 4.0e52
    ebv, f_esc, f_dust = 0.21, 0.13, 0.27
    curve = 1.2 * (gold["line_wave_a"] / GRAHSP_BIATTENUATION_BREAK_A) ** np.where(
        gold["line_wave_a"] < GRAHSP_BIATTENUATION_BREAK_A, -1.2, -3.0
    )
    expected_absorbed_lines = np.sum(line_lumin * (1.0 - 10.0 ** (-0.4 * ebv * curve)))
    actual_absorbed_lines = _absorbed_line_luminosity(
        gold["line_wave_a"], line_lumin, ebv, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A
    )
    ly_lum = 8.5e35

    assert float(actual_absorbed_lines) == pytest.approx(expected_absorbed_lines, rel=2e-12)
    assert ly_lum * f_dust == pytest.approx(2.295e35)
    assert float(_cigale_nebular_correction(f_esc, f_dust)) == pytest.approx(
        (1.0 - f_esc - f_dust) / (1.0 + (1.54e-19 / 2.58e-19) * (f_esc + f_dust)), rel=2e-12
    )


def _local_context(filter_wave, filter_trans, rest_wave):
    denom = np.trapezoid(filter_trans, x=filter_wave, axis=1)
    effective = np.trapezoid(filter_wave * filter_trans, x=filter_wave, axis=1) / denom
    return SimpleNamespace(
        rest_wave_jax=jnp.asarray(rest_wave),
        packed_filter_curves_jax=PackedFilterCurvesJax(
            wave=jnp.asarray(filter_wave), transmission=jnp.asarray(filter_trans),
            denom=jnp.asarray(denom), valid_mask=jnp.ones_like(jnp.asarray(filter_trans), dtype=bool),
        ),
        filter_effective_wavelength_jax=jnp.asarray(effective),
    ), effective, denom


def test_filter_integrated_line_flux_tracks_redshift(gold):
    filter_wave = np.tile(np.linspace(5000.0, 9000.0, 4001), (1, 1))
    filter_trans = np.maximum(1.0 - np.abs(filter_wave - 7000.0) / 1800.0, 0.0)
    rest_wave = np.linspace(500.0, 10000.0, 2000)
    context, effective, denom = _local_context(filter_wave, filter_trans, rest_wave)
    line_wave = np.asarray([4862.68])
    line_lumin = np.asarray([3.2e35])
    distance = 2.0e25

    for redshift in (0.05, 0.25, 0.45, 0.70):
        actual = float(_project_local_nebular_line_filters(
            context, line_wave, line_lumin, 80.0, 0.0, redshift, distance, jnp.ones(rest_wave.shape)
        )[0])
        transmission = np.interp(line_wave[0] * (1.0 + redshift), filter_wave[0], filter_trans[0])
        f_lambda = line_lumin[0] * transmission / (4.0 * np.pi * distance**2 * denom[0])
        expected = 1e-10 / C_M_S * 1e29 * effective[0] ** 2 * f_lambda
        # The expected value is the delta-function limit; the production path
        # uses a finite, flux-conserving local Gaussian quadrature.
        assert actual == pytest.approx(expected, rel=5e-3, abs=1e-30)


def test_full_fsps_stellar_plus_cigale_nebular_broadband_photometry(gold):
    """Gold-check the coupled continuum + exact local-line broadband path."""
    rest_wave = gold["fsps_full_wave_a"]
    lnu_mix = np.tensordot(np.asarray([0.7, 0.2, 0.1]), gold["fsps_full_lnu_lsun_per_hz"], axes=1)
    stellar = _lnu_lsun_per_hz_to_llambda_w_per_a_np(rest_wave, lnu_mix)
    n_ly = float(np.dot([0.7, 0.2, 0.1], gold["fsps_n_ly_per_msun"]))
    correction = float(_cigale_nebular_correction(0.1, 0.2))
    continuum = np.interp(
        rest_wave, gold["continuum_wave_a"], gold["continuum_lumin_w_per_a_per_photon"], left=0.0, right=0.0
    ) * n_ly * correction
    direct = np.where(rest_wave < 912.0, stellar * 0.1, stellar) + continuum
    redshift, distance = 0.5, 2.4e25
    filter_wave = np.stack([np.linspace(3000, 6000, 3001), np.linspace(6000, 10000, 3001), np.linspace(10000, 18000, 3001)])
    centers = np.asarray([4500.0, 8000.0, 14000.0])[:, None]
    widths = np.asarray([1400.0, 1800.0, 3500.0])[:, None]
    filter_trans = np.maximum(1.0 - np.abs(filter_wave - centers) / widths, 0.0)
    context, effective, denom = _local_context(filter_wave, filter_trans, rest_wave)
    obs_wave = np.linspace(3000.0, 18000.0, 15001)
    obs = np.asarray(_redshift_to_obs(rest_wave, direct, obs_wave, redshift, distance))
    indices = np.searchsorted(obs_wave, filter_wave) - 1
    indices = np.clip(indices, 0, obs_wave.size - 2)
    weights = (filter_wave - obs_wave[indices]) / (obs_wave[indices + 1] - obs_wave[indices])
    packed = PackedFiltersJax(
        interp_indices=jnp.asarray(indices), interp_weight=jnp.asarray(weights), transmission=jnp.asarray(filter_trans),
        work_wave=jnp.asarray(filter_wave), effective_wavelength=jnp.asarray(effective), valid_mask=jnp.ones_like(jnp.asarray(filter_trans), dtype=bool),
    )
    actual = np.array(_project_filters(obs, packed), copy=True)
    actual = actual + np.asarray(_project_local_nebular_line_filters(
        context, gold["line_wave_a"], gold["line_lumin_w_per_photon"] * n_ly * correction,
        100.0, 0.0, redshift, distance, jnp.ones_like(jnp.asarray(rest_wave)),
    ))

    expected = []
    distance_scale = 4.0 * np.pi * distance**2 * (1.0 + redshift)
    wave_obs_native = rest_wave * (1.0 + redshift)
    flux_native = direct / distance_scale
    for wave, trans, den, eff in zip(filter_wave, filter_trans, denom, effective, strict=True):
        continuum_numer = np.trapezoid(np.interp(wave, wave_obs_native, flux_native, left=0.0, right=0.0) * trans, x=wave)
        line_trans = np.interp(gold["line_wave_a"] * (1.0 + redshift), wave, trans, left=0.0, right=0.0)
        line_numer = np.sum(gold["line_lumin_w_per_photon"] * n_ly * correction * line_trans) / (4.0 * np.pi * distance**2)
        f_lambda = (continuum_numer + line_numer) / den
        expected.append(1e-10 / C_M_S * 1e29 * eff**2 * f_lambda)
    # This includes both the compressed continuum grid and finite local-line
    # quadrature errors measured separately above.
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-30)
