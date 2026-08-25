"""Shared names, metadata, and units for every spectral fitting frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

C_KMS = 299792.458
C_ANGSTROM_PER_S = 2.99792458e18
GAUSSIAN_FWHM = 2.354820045


class SpectralSites:
    """Canonical posterior-site names for joint and standalone spectra.

    Model code, result adapters, plotting, and persistence should refer to
    these constants rather than repeat string literals.
    """

    # Joint JAXSEDfit predictive sites (observed-frame f_nu in mJy).
    MODEL_FLUX = "pred_spectrum_fluxes"
    CONTINUUM_FLUX = "spec_continuum_model_fluxes"
    HOST_FLUX = "spec_host_model_fluxes"
    DISK_FLUX = "spec_disk_model_fluxes"
    TORUS_FLUX = "spec_torus_model_fluxes"
    WAVELENGTH_OBS = "spec_wave_obs"
    SPECTRUM_INDEX = "spec_spectrum_index"
    SCALE = "spectrum_scale_fit"
    LINE_FLUX = "spectral_line_model"
    LINE_APERTURE_FLUX = "spectral_line_model_aperture"
    FEII_FLUX = "spectral_feii_model"
    BALMER_FLUX = "spectral_balmer_model"
    LINE_AMPLITUDE = "spectral_line_amp_per_component"
    LINE_CENTER_LN = "spectral_line_mu_per_component"
    LINE_SIGMA_LN = "spectral_line_sig_per_component"
    LINE_BROAD_MASK = "spectral_line_broad_mask_per_component"
    REDSHIFT = "redshift_fit"

    # Standalone JAXQSOFit predictive sites (rest-frame f_lambda in 1e-17 cgs).
    STANDALONE_MODEL_FLUX = "model"
    STANDALONE_CONTINUUM_FLUX = "continuum_model"
    STANDALONE_LINE_FLUX = "line_model"
    STANDALONE_LINE_PROFILES = "line_component_profiles"
    STANDALONE_LINE_AMPLITUDE = "line_amp_per_component"
    STANDALONE_LINE_CENTER_LN = "line_mu_per_component"
    STANDALONE_LINE_SIGMA_LN = "line_sig_per_component"
    STANDALONE_HOST_FLUX = "gal_model"
    STANDALONE_POWER_LAW_FLUX = "f_pl_model"
    STANDALONE_FEII_UV_FLUX = "f_fe_mgii_model"
    STANDALONE_FEII_OPTICAL_FLUX = "f_fe_balmer_model"
    STANDALONE_BALMER_FLUX = "f_bc_model"


@dataclass
class SpectrumConfig:
    """Shared high-level spectral feature configuration for both fitters.

    Every field is optional so a shared config can override only the features
    relevant to a particular fit while photometric-SED and standalone-specific
    settings remain in their native config sections.
    """

    power_law_enabled: bool | None = None
    host_enabled: bool | None = None
    lines_enabled: bool | None = None
    broad_lines_enabled: bool | None = None
    narrow_lines_enabled: bool | None = None
    tied_lines: bool | None = None
    feii_enabled: bool | None = None
    balmer_continuum_enabled: bool | None = None
    polynomial_tilt_enabled: bool | None = None
    broadening_convolution: str | None = None
    line_definitions: Sequence[Any] | None = None
    components: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.broadening_convolution is not None:
            method = str(self.broadening_convolution).strip().lower()
            if method not in {"fft", "direct"}:
                raise ValueError(
                    "SpectrumConfig.broadening_convolution must be 'fft' or 'direct'."
                )
            self.broadening_convolution = method
        self.components = tuple(self.components or ())


@dataclass(frozen=True)
class LineDefinition:
    """Typed public definition of one physical emission-line table row."""

    name: str
    rest_wavelength_angstrom: float
    component: str
    sigma_ln_initial: float
    sigma_ln_minimum: float
    sigma_ln_maximum: float
    max_center_offset_ln: float
    amplitude_initial: float = 0.0
    amplitude_minimum: float = 0.0
    amplitude_maximum: float = 1.0e10
    components: int = 1
    velocity_tie: int = 0
    width_tie: int = 0
    amplitude_tie: int = 0
    amplitude_ratio: float = 1.0
    vary: bool = True

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("LineDefinition.name must be non-empty.")
        if not str(self.component).strip():
            raise ValueError("LineDefinition.component must be non-empty.")
        if not np.isfinite(self.rest_wavelength_angstrom) or self.rest_wavelength_angstrom <= 0:
            raise ValueError("LineDefinition.rest_wavelength_angstrom must be positive.")
        if int(self.components) < 1:
            raise ValueError("LineDefinition.components must be at least one.")
        if not (0 < self.sigma_ln_minimum <= self.sigma_ln_initial <= self.sigma_ln_maximum):
            raise ValueError(
                "LineDefinition widths must satisfy 0 < minimum <= initial <= maximum."
            )
        if self.amplitude_minimum > self.amplitude_initial or self.amplitude_initial > self.amplitude_maximum:
            raise ValueError(
                "LineDefinition amplitudes must satisfy minimum <= initial <= maximum."
            )

    def to_mapping(self) -> dict[str, Any]:
        """Return the low-level line-table representation consumed by the model."""
        return {
            "lambda": float(self.rest_wavelength_angstrom),
            "compname": str(self.component),
            "linename": str(self.name),
            "ngauss": int(self.components),
            "inisca": float(self.amplitude_initial),
            "minsca": float(self.amplitude_minimum),
            "maxsca": float(self.amplitude_maximum),
            "inisig": float(self.sigma_ln_initial),
            "minsig": float(self.sigma_ln_minimum),
            "maxsig": float(self.sigma_ln_maximum),
            "voff": float(self.max_center_offset_ln),
            "vindex": int(self.velocity_tie),
            "windex": int(self.width_tie),
            "findex": int(self.amplitude_tie),
            "fvalue": float(self.amplitude_ratio),
            "vary": int(bool(self.vary)),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "LineDefinition":
        """Convert a legacy line-table row into the typed public definition."""
        return cls(
            name=str(row.get("linename", f"line_{float(row['lambda']):.1f}")),
            rest_wavelength_angstrom=float(row["lambda"]),
            component=str(row.get("compname", row.get("linename", "line"))),
            components=int(row.get("ngauss", 1)),
            amplitude_initial=float(row.get("inisca", 0.0)),
            amplitude_minimum=float(row.get("minsca", 0.0)),
            amplitude_maximum=float(row.get("maxsca", 1.0e10)),
            sigma_ln_initial=float(row["inisig"]),
            sigma_ln_minimum=float(row["minsig"]),
            sigma_ln_maximum=float(row["maxsig"]),
            max_center_offset_ln=float(row.get("voff", 0.0)),
            velocity_tie=int(row.get("vindex", 0)),
            width_tie=int(row.get("windex", 0)),
            amplitude_tie=int(row.get("findex", 0)),
            amplitude_ratio=float(row.get("fvalue", 1.0)),
            vary=bool(row.get("vary", 1)),
        )


def normalize_line_definitions(lines: Any) -> list[dict[str, Any]]:
    """Normalize typed, mapping, pandas, Astropy, or structured line tables."""
    if hasattr(lines, "to_dict"):
        records = lines.to_dict("records")
    elif hasattr(lines, "dtype") and getattr(lines.dtype, "names", None):
        records = [{name: row[name] for name in lines.dtype.names} for row in lines]
    elif hasattr(lines, "colnames"):
        records = [{name: row[name] for name in lines.colnames} for row in lines]
    else:
        records = list(lines)
    normalized = []
    for item in records:
        if isinstance(item, LineDefinition):
            normalized.append(item.to_mapping())
            continue
        row = dict(item)
        if "rest_wavelength_angstrom" in row:
            row = LineDefinition(**row).to_mapping()
        normalized.append(row)
    return normalized


@dataclass(frozen=True)
class LineComponentMetadata:
    """Backend-independent identity of one expanded Gaussian component."""

    internal_name: str
    public_name: str
    parent_line: str
    component_index: int
    kind: str
    rest_wavelength_angstrom: float


@dataclass(frozen=True)
class LineGroupMetadata:
    """Backend-independent identity of one physical emission line."""

    name: str
    component_names: tuple[str, ...]
    kind: str
    rest_wavelength_angstrom: float


@dataclass(frozen=True)
class LineComponentResultBase:
    """Shared scalar metadata for public Gaussian-component results."""

    parent_line: str
    component_index: int
    kind: str
    rest_wavelength_angstrom: float


@dataclass(frozen=True)
class LineGroupResultBase:
    """Shared scalar metadata for public physical-line groups."""

    component_names: tuple[str, ...]
    kind: str
    rest_wavelength_angstrom: float


def line_component_metadata(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[LineComponentMetadata, ...], tuple[LineGroupMetadata, ...]]:
    """Expand model line metadata into stable public component/group names."""
    metadata = metadata or {}
    names = [str(name) for name in metadata.get("names", ())]
    if not names:
        return (), ()
    rest = np.asarray(metadata["line_lambda"], dtype=float)
    broad = np.asarray(metadata["broad_mask"], dtype=bool)
    if rest.size != len(names) or broad.size != len(names):
        raise ValueError("Line metadata arrays must match the component names.")
    parsed: list[tuple[str, int]] = []
    for name in names:
        prefix, separator, suffix = name.rpartition("_")
        parsed.append((prefix, int(suffix)) if separator and suffix.isdigit() else (name, 1))
    counts = {parent: sum(item[0] == parent for item in parsed) for parent, _ in parsed}
    components = tuple(
        LineComponentMetadata(
            internal_name=name,
            public_name=name if counts[parent] > 1 else parent,
            parent_line=parent,
            component_index=component_index,
            kind="broad" if broad[index] else "narrow",
            rest_wavelength_angstrom=float(rest[index]),
        )
        for index, (name, (parent, component_index)) in enumerate(zip(names, parsed))
    )
    groups = tuple(
        LineGroupMetadata(
            name=parent,
            component_names=tuple(item.public_name for item in components if item.parent_line == parent),
            kind=next(item.kind for item in components if item.parent_line == parent),
            rest_wavelength_angstrom=next(
                item.rest_wavelength_angstrom for item in components if item.parent_line == parent
            ),
        )
        for parent in dict.fromkeys(item.parent_line for item in components)
    )
    return components, groups


def fwhm_kms_from_sigma_ln(sigma_ln: Any) -> np.ndarray:
    return GAUSSIAN_FWHM * C_KMS * np.asarray(sigma_ln, dtype=float)


def velocity_offset_kms(center_ln: Any, rest_wavelength_angstrom: float) -> np.ndarray:
    return C_KMS * (
        np.asarray(center_ln, dtype=float) - np.log(float(rest_wavelength_angstrom))
    )


def gaussian_flambda_flux_erg_s_cm2(amplitude_1e17: Any, center_ln: Any, sigma_ln: Any) -> np.ndarray:
    """Integrate an ln-wavelength Gaussian whose peak is rest-frame f_lambda."""
    amplitude = np.asarray(amplitude_1e17, dtype=float)
    center = np.asarray(center_ln, dtype=float)
    sigma = np.asarray(sigma_ln, dtype=float)
    return amplitude * np.sqrt(2.0 * np.pi) * sigma * np.exp(center + 0.5 * sigma**2) * 1.0e-17


def gaussian_fnu_flux_w_m2(amplitude_mjy: Any, center_ln: Any, sigma_ln: Any, redshift: Any) -> np.ndarray:
    """Integrate an ln-wavelength Gaussian whose peak is observed f_nu."""
    amplitude = np.asarray(amplitude_mjy, dtype=float)
    center = np.asarray(center_ln, dtype=float)
    sigma = np.asarray(sigma_ln, dtype=float)
    return (
        amplitude
        * 1.0e-29
        * C_ANGSTROM_PER_S
        * np.sqrt(2.0 * np.pi)
        * sigma
        * np.exp(-center + 0.5 * sigma**2)
        / (1.0 + np.asarray(redshift, dtype=float))
    )


class SpectralResultProtocol(Protocol):
    """Structural interface shared by joint and standalone spectral results."""

    lines: Mapping[str, Any]
    line_groups: Mapping[str, Any]


__all__ = [
    "C_ANGSTROM_PER_S",
    "C_KMS",
    "GAUSSIAN_FWHM",
    "LineComponentMetadata",
    "LineComponentResultBase",
    "LineDefinition",
    "LineGroupMetadata",
    "LineGroupResultBase",
    "SpectralResultProtocol",
    "SpectrumConfig",
    "SpectralSites",
    "fwhm_kms_from_sigma_ln",
    "gaussian_flambda_flux_erg_s_cm2",
    "gaussian_fnu_flux_w_m2",
    "line_component_metadata",
    "normalize_line_definitions",
    "velocity_offset_kms",
]
