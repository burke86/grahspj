"""Smooth velocity shifts and broadening for spectra on log-wavelength grids."""

from __future__ import annotations

import jax.numpy as jnp


C_KMS = 299792.458


def shift_and_broaden_lnlam(
    lnwave,
    spectrum,
    v_kms,
    sigma_kms,
    *,
    max_pad: int = 512,
):
    """Doppler shift and Gaussian-broaden a spectrum in one Fourier operation.

    The calculation uses an internal uniform log-wavelength grid. Zero padding
    makes the FFT effectively linear over the returned interval, preventing
    flux from wrapping between the two ends of the spectrum.
    """
    lnwave = jnp.asarray(lnwave, dtype=jnp.float64)
    spectrum = jnp.asarray(spectrum, dtype=jnp.float64)
    n = lnwave.shape[0]
    if n < 2:
        return spectrum

    ln_uniform = jnp.linspace(lnwave[0], lnwave[-1], n)
    dln = jnp.maximum((lnwave[-1] - lnwave[0]) / (n - 1), 1.0e-12)
    spectrum_uniform = jnp.interp(
        ln_uniform,
        lnwave,
        spectrum,
        left=0.0,
        right=0.0,
    )

    # This static padding keeps compiled FFT shapes stable. Short arrays get a
    # full signal length on each side; long spectra cap the added work.
    pad = min(n, int(max_pad))
    padded = jnp.pad(spectrum_uniform, (pad, pad))
    npad = n + 2 * pad

    angular_frequency = 2.0 * jnp.pi * jnp.fft.rfftfreq(npad, d=dln)
    shift_ln = jnp.asarray(v_kms, dtype=jnp.float64) / C_KMS
    sigma_ln = jnp.asarray(sigma_kms, dtype=jnp.float64) / C_KMS
    # Smoothly regularize zero without introducing a gradient boundary.
    sigma_ln = jnp.sqrt(sigma_ln**2 + 1.0e-20)
    transfer = jnp.exp(
        -0.5 * (angular_frequency * sigma_ln) ** 2
        - 1j * angular_frequency * shift_ln
    )
    transformed = jnp.fft.irfft(jnp.fft.rfft(padded) * transfer, n=npad)
    transformed = transformed[pad : pad + n]
    return jnp.interp(
        lnwave,
        ln_uniform,
        transformed,
        left=0.0,
        right=0.0,
    )
