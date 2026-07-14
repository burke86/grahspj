"""Physical and numerical invariants for the JAXSEDfit forward model."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsedfit.config import FilterCurve
from jaxsedfit.model import (
    _apply_biattenuation,
    _host_dust_emission,
    _powerlaw_jax,
    _project_filters,
    _redshift_to_obs,
    _torus_component,
)
from jaxsedfit.preload import (
    _build_filter_projection_matrices_for_redshift,
    _build_fixed_filter_projection_matrices,
    _load_vendored_dale2014_templates,
    _pack_loaded_filters,
    _pack_loaded_filters_jax,
    _prepare_loaded_filter,
)


def _smooth_filter(name: str, center: float, half_width: float, n: int = 257) -> FilterCurve:
    wave = np.linspace(center - half_width, center + half_width, n)
    transmission = np.maximum(1.0 - np.abs(wave - center) / half_width, 0.0)
    return FilterCurve(name=name, wave=wave, transmission=transmission)


@pytest.mark.parametrize("profile", ["top_hat", "triangle"])
@pytest.mark.parametrize("power", [0, 1, 2])
def test_synthetic_photometry_matches_analytic_polynomial_moments(profile, power):
    """Filter integration and f_lambda-to-mJy conversion match analytic moments."""
    center = 6000.0
    half_width = 1000.0
    obs_wave = np.linspace(3000.0, 9000.0, 12001)
    if profile == "top_hat":
        filter_wave = np.linspace(center - half_width, center + half_width, 1001)
        transmission = np.ones_like(filter_wave)
        second_moment = center**2 + half_width**2 / 3.0
    else:
        filter_wave = np.linspace(center - half_width, center + half_width, 1001)
        transmission = np.maximum(1.0 - np.abs(filter_wave - center) / half_width, 0.0)
        second_moment = center**2 + half_width**2 / 6.0
    curve = FilterCurve(name=profile, wave=filter_wave, transmission=transmission)
    packed = _pack_loaded_filters_jax(_pack_loaded_filters([_prepare_loaded_filter(obs_wave, curve)]))
    amplitude = 2.5e-20
    flux_lambda = amplitude * (obs_wave / center) ** power

    projected = float(np.asarray(_project_filters(flux_lambda, packed))[0])
    if power == 0:
        mean_flux_lambda = amplitude
    elif power == 1:
        mean_flux_lambda = amplitude
    else:
        mean_flux_lambda = amplitude * second_moment / center**2
    conversion = 1.0e-10 / 299792458.0 * 1.0e29 * center**2
    assert projected == pytest.approx(conversion * mean_flux_lambda, rel=2.0e-7)


def test_flat_fnu_matches_documented_effective_wavelength_convention():
    """A flat f_nu spectrum follows the code's explicit effective-lambda convention."""
    center = 7000.0
    curve = _smooth_filter("broad", center, 2500.0, n=2001)
    obs_wave = np.linspace(3000.0, 11000.0, 16001)
    loaded = _prepare_loaded_filter(obs_wave, curve)
    packed = _pack_loaded_filters_jax(_pack_loaded_filters([loaded]))
    target_mjy = 3.2
    conversion_at_wave = 1.0e-10 / 299792458.0 * 1.0e29 * obs_wave**2
    flat_fnu_as_flambda = target_mjy / conversion_at_wave

    projected = float(np.asarray(_project_filters(flat_fnu_as_flambda, packed))[0])
    work = loaded.work_wave
    trans = loaded.transmission
    mean_inverse_square = np.trapezoid(trans / work**2, work) / np.trapezoid(trans, work)
    expected = target_mjy * loaded.effective_wavelength**2 * mean_inverse_square
    assert projected == pytest.approx(expected, rel=2.0e-7)


def test_filter_curve_resampling_invariance():
    """Equivalent samplings of a triangular bandpass produce stable photometry."""
    obs_wave = np.linspace(3500.0, 8500.0, 20001)
    flux_lambda = 1.7e-20 * (obs_wave / 6000.0) ** 1.3
    results = []
    for n_filter in (17, 65, 257, 1025):
        curve = _smooth_filter("triangle", 6000.0, 1400.0, n=n_filter)
        packed = _pack_loaded_filters_jax(_pack_loaded_filters([_prepare_loaded_filter(obs_wave, curve)]))
        results.append(float(np.asarray(_project_filters(flux_lambda, packed))[0]))

    np.testing.assert_allclose(results, results[-1], rtol=2.0e-5, atol=0.0)


def test_narrow_feature_moves_continuously_across_filter_edge():
    """A narrow feature entering a linear filter edge has monotonic throughput."""
    obs_wave = np.linspace(4000.0, 8000.0, 40001)
    curve = _smooth_filter("triangle", 6000.0, 1500.0, n=2001)
    packed = _pack_loaded_filters_jax(_pack_loaded_filters([_prepare_loaded_filter(obs_wave, curve)]))
    centers = np.asarray([4400.0, 4550.0, 4800.0, 5200.0, 5600.0])
    projected = []
    for feature_center in centers:
        feature = np.exp(-0.5 * ((obs_wave - feature_center) / 8.0) ** 2)
        projected.append(float(np.asarray(_project_filters(feature, packed))[0]))

    assert projected[0] < projected[1] < projected[2] < projected[3] < projected[4]


@pytest.mark.parametrize("redshift", [0.01, 0.3, 1.5, 4.0])
@pytest.mark.parametrize("shape", ["flat", "powerlaw", "curved"])
def test_fast_and_direct_filter_projection_parity(redshift, shape):
    """Cached matrix projection must reproduce the direct observed-grid path."""
    rest_wave = np.geomspace(500.0, 3.0e5, 4096)
    obs_wave = rest_wave * (1.0 + redshift)
    x = rest_wave / 5100.0
    if shape == "flat":
        rest_lum = np.full_like(rest_wave, 2.0e31)
    elif shape == "powerlaw":
        rest_lum = 2.0e31 * x**-1.3
    else:
        rest_lum = 2.0e31 * x**-0.7 * np.exp(-0.5 * (np.log10(rest_wave / 9000.0) / 0.35) ** 2)
    igm = np.where(rest_wave < 1216.0, 0.63, 1.0)
    distance_m = 2.4e25
    curves = [
        _smooth_filter("optical", 5500.0 * (1.0 + redshift), 1100.0),
        _smooth_filter("nir", 18000.0 * (1.0 + redshift), 3500.0),
    ]
    loaded = [_prepare_loaded_filter(obs_wave, curve) for curve in curves]
    packed_np = _pack_loaded_filters(loaded)
    packed_jax = _pack_loaded_filters_jax(packed_np)

    obs_flux = _redshift_to_obs(rest_wave, rest_lum * igm, obs_wave, redshift, distance_m)
    direct = np.asarray(_project_filters(obs_flux, packed_jax))
    fixed, _ = _build_fixed_filter_projection_matrices(
        rest_wave, packed_np, igm, distance_m, redshift
    )
    arbitrary_z, _ = _build_filter_projection_matrices_for_redshift(
        rest_wave, packed_np, igm, distance_m, redshift
    )

    np.testing.assert_allclose(fixed @ rest_lum, direct, rtol=2.0e-12, atol=0.0)
    np.testing.assert_allclose(arbitrary_z @ rest_lum, direct, rtol=3.0e-4, atol=0.0)


@pytest.mark.parametrize("redshift", [0.0, 0.1, 1.0, 3.0, 7.0])
@pytest.mark.parametrize("shape", ["flat", "powerlaw", "gaussian"])
def test_redshift_projection_conserves_bolometric_flux(redshift, shape):
    """Integrating observed f_lambda must recover L/(4 pi D_L^2)."""
    rest_wave = np.geomspace(700.0, 2.0e5, 32768)
    if shape == "flat":
        rest_lum = np.full_like(rest_wave, 3.0e28)
    elif shape == "powerlaw":
        rest_lum = 3.0e28 * (rest_wave / 5000.0) ** -1.4
    else:
        rest_lum = 3.0e28 * np.exp(-0.5 * ((rest_wave - 8000.0) / 1200.0) ** 2)
    obs_wave = rest_wave * (1.0 + redshift)
    distance_m = 1.0e24 if redshift == 0.0 else 3.0e25
    obs_flux = np.asarray(_redshift_to_obs(rest_wave, rest_lum, obs_wave, redshift, distance_m))

    observed_integral = np.trapezoid(obs_flux, obs_wave)
    expected = np.trapezoid(rest_lum, rest_wave) / (4.0 * np.pi * distance_m**2)
    assert observed_integral == pytest.approx(expected, rel=2.0e-12)
    sample = rest_wave.size // 3
    expected_density = rest_lum[sample] / (4.0 * np.pi * distance_m**2 * (1.0 + redshift))
    assert obs_flux[sample] == pytest.approx(expected_density, rel=2.0e-12)


@pytest.mark.parametrize("ebv", [0.0, 0.05, 0.3, 1.0])
@pytest.mark.parametrize("dust_alpha", [1.0, 2.0, 3.0])
def test_host_dust_energy_balance(ebv, dust_alpha):
    """The normalized Dale template must reradiate absorbed host luminosity."""
    alpha_grid, dust_wave, dust_grid = _load_vendored_dale2014_templates()
    rest_wave = np.unique(np.concatenate([np.geomspace(100.0, 3.0e7, 32768), dust_wave]))
    host = 4.0e28 * (rest_wave / 5500.0) ** -0.8 * np.exp(-rest_wave / 8.0e5)
    zeros = np.zeros_like(host)
    _, _, _, absorbed = _apply_biattenuation(
        rest_wave, host, zeros, ebv, 0.0, -1.2, -3.0, 1.2, 11000.0
    )
    dust_on_rest = np.stack(
        [np.interp(rest_wave, dust_wave, row, left=0.0, right=0.0) for row in dust_grid]
    )
    context = SimpleNamespace(
        dust_alpha_grid_jax=jnp.asarray(alpha_grid),
        dust_lumin_rest_jax=jnp.asarray(dust_on_rest),
    )
    emitted = np.asarray(_host_dust_emission(context, absorbed, dust_alpha))

    emitted_integral = np.trapezoid(emitted, rest_wave)
    assert emitted_integral == pytest.approx(float(absorbed), rel=2.0e-3, abs=1.0e-20)
    assert np.all(emitted >= 0.0)


def _smooth_component_photometry(n_wave: int) -> tuple[np.ndarray, float]:
    rest_wave = np.geomspace(500.0, 3.0e6, n_wave)
    redshift = 0.4
    distance_m = 2.0e25
    disk = np.asarray(
        _powerlaw_jax(rest_wave, 1.0e34, 0.0, -1.8, 5100.0, 1000.0, 10.0, 100000.0)
    )
    torus = np.asarray(
        _torus_component(
            rest_wave,
            0.2,
            0.0,
            17.0,
            0.45,
            2.0,
            0.5,
            0.1,
            0.29,
            98410.0,
            142240.0,
            10253.0,
            11635.0,
            5.1e37,
        )
    )
    total = disk + torus
    obs_wave = rest_wave * (1.0 + redshift)
    filters = [
        _smooth_filter("uv", 2500.0 * (1.0 + redshift), 500.0),
        _smooth_filter("opt", 5100.0 * (1.0 + redshift), 1200.0),
        _smooth_filter("nir", 20000.0 * (1.0 + redshift), 5000.0),
        _smooth_filter("mir", 120000.0 * (1.0 + redshift), 30000.0),
    ]
    packed = _pack_loaded_filters_jax(
        _pack_loaded_filters([_prepare_loaded_filter(obs_wave, curve) for curve in filters])
    )
    obs_flux = _redshift_to_obs(rest_wave, total, obs_wave, redshift, distance_m)
    photometry = np.asarray(_project_filters(obs_flux, packed))
    bolometric = float(np.trapezoid(total, rest_wave))
    return photometry, bolometric


def test_smooth_components_converge_with_wavelength_grid_resolution():
    """Disk/torus photometry and luminosity should converge under grid refinement."""
    reference_photometry, reference_bolometric = _smooth_component_photometry(32768)
    errors = []
    for n_wave in (256, 2048, 8192):
        photometry, bolometric = _smooth_component_photometry(n_wave)
        error = max(
            float(np.max(np.abs(photometry / reference_photometry - 1.0))),
            abs(bolometric / reference_bolometric - 1.0),
        )
        errors.append(error)

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[-1] < 5.0e-4
