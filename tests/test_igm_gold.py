"""Gold tests for the CIGALE v2025.1 Meiksin IGM implementation."""

from pathlib import Path

import numpy as np

from jaxsedfit.preload import _build_fixed_igm_jax, _build_igm_cache_jax


REFERENCE = Path(__file__).parent / "fixtures" / "cigale_v2025_1_igm_reference.npz"


def _reference():
    with np.load(REFERENCE) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def test_igm_transmission_matches_cigale_v2025_1_at_all_wavelengths_and_redshifts():
    ref = _reference()
    assert str(ref["cigale_version"]) == "2025.1"
    assert str(ref["cigale_git_commit"]) == "29cb909fe2636800b4acdb1dfc7129d8c8494a24"
    assert str(ref["wavelength_convention"]) == "observed_nm"
    cache = _build_igm_cache_jax(ref["rest_wave_a"])

    actual = np.stack(
        [np.asarray(_build_fixed_igm_jax(cache, float(z))) for z in ref["redshift"]]
    )

    # This remains a valid gold test when JAX x64 is disabled: the maximum
    # float32 discrepancy is 1.5e-6, while x64 agrees at machine precision.
    np.testing.assert_allclose(actual, ref["transmission"], rtol=2e-5, atol=2e-7)
    assert np.all((actual >= 0.0) & (actual <= 1.0))
    # Redward of source-frame Ly-alpha there is no intervening Lyman-series
    # or Lyman-continuum absorption in CIGALE's prescription.
    np.testing.assert_allclose(actual[:, ref["rest_wave_a"] > 1216.0], 1.0, rtol=0.0, atol=0.0)


def test_filter_integrated_igm_attenuation_matches_cigale_across_lyman_features():
    """Compare broadband transmission, not just point samples of the curve."""
    ref = _reference()
    wave = ref["rest_wave_a"]
    cache = _build_igm_cache_jax(wave)
    # Smooth source continuum prevents a constant-spectrum test from hiding
    # wavelength-coordinate or weighting mistakes.
    intrinsic = (wave / 1000.0) ** -1.7
    bands = ((650.0, 900.0), (880.0, 1050.0), (1040.0, 1220.0), (1220.0, 1800.0))

    for z, cigale_transmission in zip(ref["redshift"], ref["transmission"], strict=True):
        jax_transmission = np.asarray(_build_fixed_igm_jax(cache, float(z)))
        for low, high in bands:
            response = np.maximum(
                1.0 - np.abs(wave - 0.5 * (low + high)) / (0.5 * (high - low)),
                0.0,
            )
            denominator = np.trapezoid(intrinsic * response, x=wave)
            expected = np.trapezoid(intrinsic * cigale_transmission * response, x=wave) / denominator
            actual = np.trapezoid(intrinsic * jax_transmission * response, x=wave) / denominator
            np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=3e-8)
