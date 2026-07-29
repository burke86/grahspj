"""Tests for smooth Fourier-domain velocity operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsedfit.velocity import C_KMS, shift_and_broaden_lnlam


jax.config.update("jax_enable_x64", True)


def _moments(lnwave, flux):
    flux = np.asarray(flux)
    lnwave = np.asarray(lnwave)
    norm = np.sum(flux)
    centroid = np.sum(lnwave * flux) / norm
    variance = np.sum((lnwave - centroid) ** 2 * flux) / norm
    return norm, centroid, np.sqrt(variance)


@pytest.mark.parametrize(
    ("velocity", "broadening"),
    [(0.0, 150.0), (375.0, 600.0), (-725.0, 1800.0)],
)
def test_fourier_velocity_operator_matches_gaussian_moments(velocity, broadening):
    lnwave = jnp.linspace(np.log(3000.0), np.log(8000.0), 4096)
    center = np.log(5000.0)
    intrinsic_sigma_kms = 350.0
    spectrum = jnp.exp(
        -0.5 * ((lnwave - center) / (intrinsic_sigma_kms / C_KMS)) ** 2
    )

    transformed = shift_and_broaden_lnlam(
        lnwave,
        spectrum,
        velocity,
        broadening,
    )
    norm_in, _, _ = _moments(lnwave, spectrum)
    norm_out, center_out, sigma_out = _moments(lnwave, transformed)

    assert norm_out == pytest.approx(norm_in, rel=2.0e-6)
    assert (center_out - center) * C_KMS == pytest.approx(velocity, abs=0.5)
    expected_sigma = np.hypot(intrinsic_sigma_kms, broadening)
    assert sigma_out * C_KMS == pytest.approx(expected_sigma, rel=2.0e-4)


def test_fourier_velocity_operator_does_not_wrap_flux_across_edges():
    lnwave = jnp.linspace(np.log(4000.0), np.log(5000.0), 2048)
    dln = float(lnwave[1] - lnwave[0])
    spectrum = jnp.exp(-0.5 * ((lnwave - lnwave[-40]) / (2.0 * dln)) ** 2)

    transformed = shift_and_broaden_lnlam(
        lnwave,
        spectrum,
        2500.0,
        300.0,
    )

    # The feature shifts out through the red edge; it must not reappear at blue.
    assert float(jnp.max(jnp.abs(transformed[:200]))) < 1.0e-10


def test_fourier_velocity_operator_handles_linear_wavelength_grid():
    wave = jnp.linspace(3000.0, 8000.0, 4096)
    lnwave = jnp.log(wave)
    center = np.log(5100.0)
    intrinsic_sigma_kms = 500.0
    velocity = -450.0
    broadening = 1300.0
    spectrum = jnp.exp(
        -0.5 * ((lnwave - center) / (intrinsic_sigma_kms / C_KMS)) ** 2
    )

    transformed = shift_and_broaden_lnlam(
        lnwave,
        spectrum,
        velocity,
        broadening,
    )
    sigma_out = np.hypot(intrinsic_sigma_kms, broadening) / C_KMS
    expected = (
        intrinsic_sigma_kms
        / np.hypot(intrinsic_sigma_kms, broadening)
        * jnp.exp(-0.5 * ((lnwave - center - velocity / C_KMS) / sigma_out) ** 2)
    )

    central = (wave > 4000.0) & (wave < 7000.0)
    relative_l2 = jnp.linalg.norm((transformed - expected)[central]) / jnp.linalg.norm(
        expected[central]
    )
    assert float(relative_l2) < 1.0e-3


def test_fourier_velocity_operator_has_finite_smooth_parameter_gradients():
    lnwave = jnp.linspace(np.log(3000.0), np.log(8000.0), 2048)
    spectrum = jnp.exp(-0.5 * ((lnwave - np.log(5000.0)) / 0.002) ** 2)
    weights = jnp.sin(jnp.linspace(0.0, 4.0 * np.pi, lnwave.size))

    def objective(parameters):
        velocity, broadening = parameters
        model = shift_and_broaden_lnlam(
            lnwave,
            spectrum,
            velocity,
            broadening,
        )
        return jnp.dot(model, weights)

    parameters = jnp.asarray([421.25, 777.75])
    gradient = jax.grad(objective)(parameters)
    jacobian = jax.jacfwd(jax.grad(objective))(parameters)

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.isfinite(np.asarray(jacobian)))


def test_generic_fft_velocity_convolution_uses_smooth_operator():
    from jaxsedfit.spectral_model import _convolve_velocity_space

    lnwave = jnp.linspace(np.log(3000.0), np.log(8000.0), 2048)
    signal = jnp.exp(-0.5 * ((lnwave - np.log(5200.0)) / 0.002) ** 2)
    sigma_ln = 1100.0 / C_KMS

    actual = _convolve_velocity_space(
        lnwave,
        signal,
        sigma_ln,
        method="fft",
    )
    expected = shift_and_broaden_lnlam(
        lnwave,
        signal,
        0.0,
        1100.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_balmer_continuum_fft_edge_and_width_gradients_are_smooth():
    from jaxsedfit.spectral_model import _balmer_continuum_jax

    wave = jnp.linspace(2500.0, 4500.0, 3072)
    weights = jnp.cos(jnp.linspace(0.0, 3.0 * np.pi, wave.size))

    def objective(velocity):
        continuum = _balmer_continuum_jax(
            wave,
            1.0,
            15000.0,
            1.0,
            velocity,
            convolution_method="fft",
        )
        return jnp.dot(continuum, weights)

    velocity = jnp.asarray(3200.0)
    value = objective(velocity)
    gradient = jax.grad(objective)(velocity)
    curvature = jax.grad(jax.grad(objective))(velocity)
    continuum = _balmer_continuum_jax(
        wave,
        1.0,
        15000.0,
        1.0,
        velocity,
        convolution_method="fft",
    )

    assert np.all(np.isfinite(np.asarray([value, gradient, curvature])))
    # Broadening must carry finite BC flux across the physical 3646-A edge.
    assert float(jnp.max(continuum[(wave > 3646.0) & (wave < 3700.0)])) > 0.0
    assert float(jnp.min(continuum)) > -1.0e-10


def test_model_wrappers_share_the_fourier_operator():
    from jaxsedfit.model import _shift_and_broaden_single_spectrum_lnlam as model_op
    from jaxsedfit.spectral_model import (
        _shift_and_broaden_single_spectrum_lnlam as spectral_op,
    )

    lnwave = jnp.linspace(np.log(3500.0), np.log(7500.0), 1024)
    spectrum = jnp.exp(-0.5 * ((lnwave - np.log(5100.0)) / 0.003) ** 2)
    expected = shift_and_broaden_lnlam(lnwave, spectrum, 325.0, 900.0)

    np.testing.assert_allclose(model_op(lnwave, spectrum, 325.0, 900.0), expected)
    np.testing.assert_allclose(
        spectral_op(
            lnwave,
            spectrum,
            325.0,
            900.0,
            convolution_method="fft",
        ),
        expected,
    )
