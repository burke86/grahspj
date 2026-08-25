"""Stable, unit-explicit public results for spectral fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .spectral_contract import (
    LineComponentResultBase,
    LineGroupResultBase,
    SpectralSites,
    fwhm_kms_from_sigma_ln,
    gaussian_fnu_flux_w_m2,
    line_component_metadata,
    velocity_offset_kms,
)


@dataclass(frozen=True)
class LineComponentResult(LineComponentResultBase):
    """Posterior draws and metadata for one Gaussian line component."""

    amplitude_mjy: np.ndarray
    center_rest_angstrom: np.ndarray
    sigma_ln_lambda: np.ndarray
    fwhm_kms: np.ndarray
    velocity_offset_kms: np.ndarray
    flux_w_m2: np.ndarray


@dataclass(frozen=True)
class LineGroupResult(LineGroupResultBase):
    """Aggregate posterior quantities for one physical emission line."""

    total_flux_w_m2: np.ndarray


@dataclass(frozen=True)
class SpectrumObservationResult:
    """Observed data and posterior model draws for one input spectrum."""

    index: int
    instrument: str
    wavelength_obs_angstrom: np.ndarray
    observed_flux_mjy: np.ndarray
    error_mjy: np.ndarray
    mask: np.ndarray
    model_flux_mjy: np.ndarray
    continuum_flux_mjy: np.ndarray
    host_flux_mjy: np.ndarray
    disk_flux_mjy: np.ndarray
    torus_flux_mjy: np.ndarray
    line_flux_density_mjy: np.ndarray
    feii_flux_density_mjy: np.ndarray
    balmer_flux_density_mjy: np.ndarray
    residual_mjy: np.ndarray


@dataclass(frozen=True)
class SpectralResult:
    """Stable public contract for posterior spectral results."""

    lines: Mapping[str, LineComponentResult]
    line_groups: Mapping[str, LineGroupResult]
    observations: tuple[SpectrumObservationResult, ...]
    wavelength_obs_angstrom: np.ndarray
    model_flux_mjy: np.ndarray
    continuum_flux_mjy: np.ndarray
    line_flux_density_mjy: np.ndarray
    feii_flux_density_mjy: np.ndarray
    balmer_flux_density_mjy: np.ndarray


def _line_results(
    predictive: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    redshift: float,
) -> tuple[dict[str, LineComponentResult], dict[str, LineGroupResult]]:
    metadata = metadata or {}
    names = [str(name) for name in metadata.get("names", ())]
    required = (
        SpectralSites.LINE_AMPLITUDE,
        SpectralSites.LINE_CENTER_LN,
        SpectralSites.LINE_SIGMA_LN,
    )
    if not names or any(name not in predictive for name in required):
        return {}, {}

    amplitudes = np.asarray(predictive[required[0]], dtype=float)
    centers_ln = np.asarray(predictive[required[1]], dtype=float)
    sigmas_ln = np.asarray(predictive[required[2]], dtype=float)
    if amplitudes.shape[-1] != len(names):
        raise ValueError(
            "Spectral line metadata does not match the predictive component axis."
        )
    component_metadata, group_metadata = line_component_metadata(metadata)
    redshift_draws = np.asarray(
        predictive.get(SpectralSites.REDSHIFT, redshift), dtype=float
    )
    while redshift_draws.ndim < amplitudes.ndim - 1:
        redshift_draws = redshift_draws[..., None]

    lines: dict[str, LineComponentResult] = {}
    group_members: dict[str, list[str]] = {}
    for index, identity in enumerate(component_metadata):
        center = np.exp(centers_ln[..., index])
        sigma_ln = sigmas_ln[..., index]
        amplitude = amplitudes[..., index]
        flux = gaussian_fnu_flux_w_m2(
            amplitude, centers_ln[..., index], sigma_ln, redshift_draws
        )
        lines[identity.public_name] = LineComponentResult(
            amplitude_mjy=amplitude,
            center_rest_angstrom=center,
            sigma_ln_lambda=sigma_ln,
            fwhm_kms=fwhm_kms_from_sigma_ln(sigma_ln),
            velocity_offset_kms=velocity_offset_kms(
                centers_ln[..., index], identity.rest_wavelength_angstrom
            ),
            flux_w_m2=flux,
            parent_line=identity.parent_line,
            component_index=identity.component_index,
            kind=identity.kind,
            rest_wavelength_angstrom=identity.rest_wavelength_angstrom,
        )
        group_members.setdefault(identity.parent_line, []).append(identity.public_name)

    groups: dict[str, LineGroupResult] = {}
    group_by_name = {item.name: item for item in group_metadata}
    for parent, members in group_members.items():
        identity = group_by_name[parent]
        groups[parent] = LineGroupResult(
            component_names=tuple(members),
            total_flux_w_m2=np.sum(
                np.stack([lines[name].flux_w_m2 for name in members], axis=0),
                axis=0,
            ),
            kind=identity.kind,
            rest_wavelength_angstrom=identity.rest_wavelength_angstrom,
        )
    return lines, groups


def _draw_pixel_array(
    predictive: Mapping[str, Any], name: str, n_pixels: int
) -> np.ndarray:
    value = predictive.get(name)
    if value is None:
        return np.zeros((0, n_pixels), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[-1] != n_pixels:
        return np.zeros((0, n_pixels), dtype=float)
    return array


def _calibration_per_pixel(
    predictive: Mapping[str, Any], spectrum_index: np.ndarray, n_draws: int
) -> np.ndarray:
    scale = np.asarray(predictive.get(SpectralSites.SCALE, 1.0), dtype=float)
    if scale.ndim == 0:
        return np.full((n_draws, spectrum_index.size), float(scale))
    if scale.ndim == 1:
        if scale.shape[0] == n_draws:
            return np.broadcast_to(scale[:, None], (n_draws, spectrum_index.size))
        return np.broadcast_to(scale[spectrum_index][None, :], (n_draws, spectrum_index.size))
    return scale[:, spectrum_index]


def build_spectral_result(
    predictive: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    *,
    redshift: float,
    context: Any,
) -> SpectralResult:
    """Adapt raw predictive sites into the stable public spectral contract."""
    lines, groups = _line_results(predictive, metadata, redshift)
    wavelength = np.asarray(
        predictive.get(
            SpectralSites.WAVELENGTH_OBS,
            getattr(context, "spec_wave_obs", np.zeros(0)),
        ),
        dtype=float,
    )
    if wavelength.ndim > 1:
        wavelength = wavelength[0]
    n_pixels = wavelength.size
    spectrum_index = np.asarray(
        predictive.get(
            SpectralSites.SPECTRUM_INDEX,
            getattr(context, "spec_spectrum_index", np.zeros(n_pixels, dtype=int)),
        ),
        dtype=int,
    )
    if spectrum_index.ndim > 1:
        spectrum_index = spectrum_index[0]

    model = _draw_pixel_array(predictive, SpectralSites.MODEL_FLUX, n_pixels)
    continuum_raw = _draw_pixel_array(
        predictive, SpectralSites.CONTINUUM_FLUX, n_pixels
    )
    n_draws = model.shape[0] or continuum_raw.shape[0]
    calibration = _calibration_per_pixel(predictive, spectrum_index, n_draws)

    def calibrated(name: str) -> np.ndarray:
        values = _draw_pixel_array(predictive, name, n_pixels)
        return values * calibration if values.shape[0] else values

    continuum = continuum_raw * calibration if continuum_raw.shape[0] else continuum_raw
    line_site = (
        SpectralSites.LINE_APERTURE_FLUX
        if SpectralSites.LINE_APERTURE_FLUX in predictive
        else SpectralSites.LINE_FLUX
    )
    line = calibrated(line_site)
    feii = calibrated(SpectralSites.FEII_FLUX)
    balmer = calibrated(SpectralSites.BALMER_FLUX)
    host = calibrated(SpectralSites.HOST_FLUX)
    disk = calibrated(SpectralSites.DISK_FLUX)
    torus = calibrated(SpectralSites.TORUS_FLUX)

    observed = np.asarray(getattr(context, "spec_fluxes", np.zeros(n_pixels)), dtype=float)
    errors = np.asarray(getattr(context, "spec_errors", np.zeros(n_pixels)), dtype=float)
    mask = np.asarray(getattr(context, "spec_mask", np.ones(n_pixels)), dtype=bool)
    instruments = tuple(getattr(context, "spec_instruments", ()))
    observations = []
    for index in sorted(set(spectrum_index.tolist())) if n_pixels else []:
        selected = spectrum_index == index
        model_selected = model[..., selected]
        observations.append(
            SpectrumObservationResult(
                index=int(index),
                instrument=(
                    str(instruments[index])
                    if index < len(instruments)
                    else f"spectrum_{index}"
                ),
                wavelength_obs_angstrom=wavelength[selected],
                observed_flux_mjy=observed[selected],
                error_mjy=errors[selected],
                mask=mask[selected],
                model_flux_mjy=model_selected,
                continuum_flux_mjy=continuum[..., selected],
                host_flux_mjy=host[..., selected],
                disk_flux_mjy=disk[..., selected],
                torus_flux_mjy=torus[..., selected],
                line_flux_density_mjy=line[..., selected],
                feii_flux_density_mjy=feii[..., selected],
                balmer_flux_density_mjy=balmer[..., selected],
                residual_mjy=observed[selected] - model_selected,
            )
        )
    return SpectralResult(
        lines=lines,
        line_groups=groups,
        observations=tuple(observations),
        wavelength_obs_angstrom=wavelength,
        model_flux_mjy=model,
        continuum_flux_mjy=continuum,
        line_flux_density_mjy=line,
        feii_flux_density_mjy=feii,
        balmer_flux_density_mjy=balmer,
    )


__all__ = [
    "LineComponentResult",
    "LineGroupResult",
    "SpectralResult",
    "SpectrumObservationResult",
    "SpectralSites",
    "build_spectral_result",
]
