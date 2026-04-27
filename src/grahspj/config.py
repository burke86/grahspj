from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class Observation:
    """Observation-level metadata for one fitted source."""
    object_id: str
    redshift: float
    fit_redshift: bool = False
    redshift_err: float = 0.0
    ra: float | None = None
    dec: float | None = None
    apply_mw_deredden: bool = False


@dataclass
class PhotometryData:
    """Observed photometric measurements and associated metadata."""
    filter_names: Sequence[str]
    fluxes: Sequence[float]
    errors: Sequence[float]
    is_upper_limit: Sequence[bool] | None = None
    psf_fwhm_arcsec: Sequence[float | None] | None = None
    aperture_diameter_arcsec: Sequence[float | None] | None = None
    photometry_method: Sequence[str | None] | None = None

    def validate(self) -> None:
        """Validate array lengths for one photometry payload."""
        n = len(self.filter_names)
        if len(self.fluxes) != n or len(self.errors) != n:
            raise ValueError("Photometry arrays must have the same length as filter_names.")
        if self.is_upper_limit is not None and len(self.is_upper_limit) != n:
            raise ValueError("is_upper_limit must match filter_names length.")
        if self.psf_fwhm_arcsec is not None and len(self.psf_fwhm_arcsec) != n:
            raise ValueError("psf_fwhm_arcsec must match filter_names length.")
        if self.aperture_diameter_arcsec is not None and len(self.aperture_diameter_arcsec) != n:
            raise ValueError("aperture_diameter_arcsec must match filter_names length.")
        if self.photometry_method is not None and len(self.photometry_method) != n:
            raise ValueError("photometry_method must match filter_names length.")


@dataclass
class SpectroscopyData:
    """Observed spectral measurements on an observed-frame wavelength grid."""
    wave_obs: Sequence[float]
    fluxes: Sequence[float]
    errors: Sequence[float]
    mask: Sequence[bool] | None = None
    instrument: str | None = None
    aperture_diameter_arcsec: float | None = None
    psf_fwhm_arcsec: float | None = None
    epoch_mjd: float | None = None

    def validate(self) -> None:
        """Validate array lengths for one spectrum payload."""
        n = len(self.wave_obs)
        if len(self.fluxes) != n or len(self.errors) != n:
            raise ValueError("Spectroscopy arrays must have the same length as wave_obs.")
        if self.mask is not None and len(self.mask) != n:
            raise ValueError("spectroscopy mask must match wave_obs length.")


@dataclass
class FilterCurve:
    """One explicit filter transmission curve."""
    name: str
    wave: Sequence[float]
    transmission: Sequence[float]
    effective_wavelength: float | None = None


@dataclass
class FilterSet:
    """Filter configuration used to construct synthetic photometry."""
    curves: Sequence[FilterCurve] = field(default_factory=list)
    speclite_names: Mapping[str, str] = field(default_factory=dict)
    use_grahsp_database: bool = True


@dataclass
class FeIITemplate:
    """Fe II template configuration or inline template data."""
    name: str = "BruhweilerVerner08"
    wave: Sequence[float] | None = None
    lumin: Sequence[float] | None = None


@dataclass
class EmissionLineTemplate:
    """Emission-line template tables for BLAGN, Sy2, and LINER branches."""
    wave: Sequence[float] | None = None
    lumin_blagn: Sequence[float] | None = None
    lumin_sy2: Sequence[float] | None = None
    lumin_liner: Sequence[float] | None = None


@dataclass
class GalaxyConfig:
    """Host-galaxy model, cosmology, and wavelength-grid settings."""
    fit_host: bool = True
    fit_host_kinematics: bool = False
    dsps_ssp_fn: str = "tempdata.h5"
    age_grid_gyr: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0)
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2)
    imf_type: int = 1
    zcontinuous: int = 1
    sfh: int = 0
    rest_wave_min: float = 100.0
    rest_wave_max: float = 3.0e6
    n_wave: int = 1024
    tau_host_prior_scale: float = 1.0
    sfh_n_steps: int = 64
    sfh_t_min_gyr: float = 0.01
    cosmology_h0: float = 70.0
    cosmology_om0: float = 0.3
    use_energy_balance: bool = True
    dust_alpha: float = 2.0


@dataclass
class NebularConfig:
    """CIGALE/GRAHSP-style host-galaxy nebular emission configuration."""
    enabled: bool = True
    emission: bool = True
    logU: float = -2.0
    zgas: float | None = None
    ne: float = 100.0
    f_esc: float = 0.0
    f_dust: float = 0.0
    lines_width: float = 300.0
    young_age_cut_myr: float = 10.0

    def validate(self) -> None:
        if self.zgas is not None and (not np.isfinite(float(self.zgas)) or float(self.zgas) <= 0.0):
            raise ValueError("nebular.zgas must be a positive finite metallicity when set.")
        if not np.isfinite(float(self.logU)):
            raise ValueError("nebular.logU must be finite.")
        if not np.isfinite(float(self.ne)) or float(self.ne) <= 0.0:
            raise ValueError("nebular.ne must be positive and finite.")
        if not np.isfinite(float(self.lines_width)) or float(self.lines_width) < 0.0:
            raise ValueError("nebular.lines_width must be finite and non-negative.")
        if not np.isfinite(float(self.young_age_cut_myr)) or float(self.young_age_cut_myr) < 0.0:
            raise ValueError("nebular.young_age_cut_myr must be finite and non-negative.")
        if not 0.0 <= float(self.f_esc) <= 1.0:
            raise ValueError("nebular.f_esc must be between 0 and 1.")
        if not 0.0 <= float(self.f_dust) <= 1.0:
            raise ValueError("nebular.f_dust must be between 0 and 1.")
        if float(self.f_esc) + float(self.f_dust) > 1.0:
            raise ValueError("nebular.f_esc + nebular.f_dust must be <= 1.")


@dataclass
class AGNConfig:
    """AGN component configuration, templates, and fixed branch settings."""
    fit_agn: bool = True
    use_powerlaw_disk: bool = True
    feii_template: FeIITemplate = field(default_factory=FeIITemplate)
    emission_line_template: EmissionLineTemplate = field(default_factory=EmissionLineTemplate)
    agn_type: int = 1
    line_width_kms_default: float = 3000.0
    lines_strength_default: float = 1.0
    feii_strength_default: float = 5.0
    balmer_continuum_default: float = 0.0


@dataclass
class LikelihoodConfig:
    """Likelihood and extra model-mismatch configuration."""
    systematics_width: float = 0.05
    student_t_df: float = 5.0
    fit_intrinsic_scatter: bool = True
    intrinsic_scatter_default: float = 1.0e-4
    variability_uncertainty: bool = True
    agn_nev: float = 0.1
    attenuation_model_uncertainty: bool = False
    lyman_break_uncertainty: bool = False
    use_absolute_flux_scale_prior: bool = True
    absolute_flux_scale_prior_sigma_dex: float = 0.5
    use_host_capture_model: bool = False
    use_fast_photometry_projection: bool = True


@dataclass
class JaxQSOFitConfig:
    """Spectroscopy-only jaxqsofit component configuration.

    These flags affect only the spectroscopic likelihood. Broadband
    photometry continues to use grahspj's native SED-scale AGN lines,
    Fe II, and Balmer continuum components.
    """
    use_spectral_lines: bool = True
    use_spectral_feii: bool = False
    use_spectral_balmer_continuum: bool = False
    use_tied_lines: bool = True
    use_spectral_smart_priors: bool = True
    use_multiplicative_tilt: bool = False
    line_flux_scale_mjy: float = 1.0
    include_elg_narrow_lines: bool = False
    include_high_ionization_lines: bool = False
    line_table: Sequence[Mapping[str, Any]] | None = None
    line_prior_config: Mapping[str, Any] | None = None


@dataclass
class SpectroscopyConfig:
    """Spectroscopic likelihood configuration."""
    enabled: bool = False
    backend: str = "grahspj"
    student_t_df: float = 5.0
    systematics_width: float = 0.05
    fit_scale: bool = True
    scale_prior_sigma_dex: float = 0.5
    jaxqsofit: JaxQSOFitConfig = field(default_factory=JaxQSOFitConfig)


@dataclass
class InferenceConfig:
    """Inference defaults for MAP optimization and NUTS sampling."""
    learning_rate: float = 5e-3
    map_steps: int = 1500
    num_warmup: int = 200
    num_samples: int = 200
    num_chains: int = 1
    target_accept_prob: float = 0.85
    seed: int = 0


@dataclass
class FitConfig:
    """Top-level configuration bundle for a single grahspj fit."""
    observation: Observation
    photometry: PhotometryData
    filters: FilterSet = field(default_factory=FilterSet)
    galaxy: GalaxyConfig = field(default_factory=GalaxyConfig)
    nebular: NebularConfig = field(default_factory=NebularConfig)
    agn: AGNConfig = field(default_factory=AGNConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    spectroscopy: SpectroscopyData | Sequence[SpectroscopyData] | None = None
    spectroscopy_config: SpectroscopyConfig = field(default_factory=SpectroscopyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    prior_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate nested config components that require runtime checks."""
        self.photometry.validate()
        self.nebular.validate()
        for spectrum in self.spectroscopy_list:
            spectrum.validate()
        if not self.galaxy.fit_host and not self.agn.fit_agn:
            raise ValueError("At least one of galaxy.fit_host or agn.fit_agn must be True.")
        redshift_pdf = self.prior_config.get("redshift_pdf")
        if redshift_pdf is not None:
            if not isinstance(redshift_pdf, Mapping):
                raise TypeError("prior_config['redshift_pdf'] must be a mapping with 'z_grid' and 'pdf'.")
            if "z_grid" not in redshift_pdf or "pdf" not in redshift_pdf:
                raise ValueError("prior_config['redshift_pdf'] must contain 'z_grid' and 'pdf'.")
            z_grid = np.asarray(redshift_pdf["z_grid"], dtype=float)
            pdf = np.asarray(redshift_pdf["pdf"], dtype=float)
            if z_grid.ndim != 1 or pdf.ndim != 1 or z_grid.size != pdf.size or z_grid.size < 2:
                raise ValueError("redshift_pdf z_grid and pdf must be one-dimensional arrays of the same length >= 2.")
            if not np.all(np.isfinite(z_grid)) or not np.all(np.isfinite(pdf)):
                raise ValueError("redshift_pdf z_grid and pdf must be finite.")
            if np.any(np.diff(z_grid) <= 0.0):
                raise ValueError("redshift_pdf z_grid must be strictly increasing.")
            if np.any(pdf < 0.0):
                raise ValueError("redshift_pdf pdf must be non-negative.")
            norm = float(np.trapezoid(pdf, z_grid))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError("redshift_pdf must integrate to a positive finite value.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass tree into a plain Python dictionary."""
        return asdict(self)

    @property
    def spectroscopy_list(self) -> list[SpectroscopyData]:
        """Return spectroscopy payloads as a list while preserving legacy single-spectrum input."""
        if self.spectroscopy is None:
            return []
        if isinstance(self.spectroscopy, SpectroscopyData):
            return [self.spectroscopy]
        return list(self.spectroscopy)


def _coerce_dataclass(cls, value: Any):
    """Convert a mapping or existing instance into the requested dataclass."""
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        kwargs = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name not in value:
                continue
            kwargs[field_name] = value[field_name]
        return cls(**kwargs)
    raise TypeError(f"Cannot coerce {type(value)!r} to {cls.__name__}")


def _coerce_jaxqsofit_config(value: Any) -> JaxQSOFitConfig:
    """Coerce jaxqsofit config and migrate older generic flag names."""
    if isinstance(value, JaxQSOFitConfig):
        return value
    if not isinstance(value, Mapping):
        return _coerce_dataclass(JaxQSOFitConfig, value)
    data = dict(value)
    aliases = {
        "use_lines": "use_spectral_lines",
        "use_feii": "use_spectral_feii",
        "use_balmer_continuum": "use_spectral_balmer_continuum",
    }
    for old_name, new_name in aliases.items():
        if old_name in data and new_name not in data:
            data[new_name] = data[old_name]
    return _coerce_dataclass(JaxQSOFitConfig, data)


def _coerce_spectroscopy_config(value: Any) -> SpectroscopyConfig:
    """Coerce spectroscopy config while supporting nested jaxqsofit config."""
    if isinstance(value, SpectroscopyConfig):
        return value
    if not isinstance(value, Mapping):
        return _coerce_dataclass(SpectroscopyConfig, value)
    kwargs = {}
    legacy_jaxqsofit = {}
    legacy_aliases = {
        "jaxqsofit_use_lines": "use_spectral_lines",
        "jaxqsofit_use_feii": "use_spectral_feii",
        "jaxqsofit_use_balmer_continuum": "use_spectral_balmer_continuum",
        "jaxqsofit_use_multiplicative_tilt": "use_multiplicative_tilt",
    }
    for old_name, new_name in legacy_aliases.items():
        if old_name in value:
            legacy_jaxqsofit[new_name] = value[old_name]
    for field_name in SpectroscopyConfig.__dataclass_fields__:
        if field_name not in value:
            continue
        if field_name == "jaxqsofit":
            merged_jaxqsofit = dict(legacy_jaxqsofit)
            if isinstance(value[field_name], Mapping):
                merged_jaxqsofit.update(value[field_name])
                kwargs[field_name] = _coerce_jaxqsofit_config(merged_jaxqsofit)
            else:
                kwargs[field_name] = _coerce_jaxqsofit_config(value[field_name])
        else:
            kwargs[field_name] = value[field_name]
    if "jaxqsofit" not in kwargs and legacy_jaxqsofit:
        kwargs["jaxqsofit"] = _coerce_jaxqsofit_config(legacy_jaxqsofit)
    return SpectroscopyConfig(**kwargs)


def fit_config_from_mapping(data: Mapping[str, Any]) -> FitConfig:
    """Build a validated FitConfig from a nested mapping."""
    filters_raw = data.get("filters", {})
    if isinstance(filters_raw, Mapping):
        curves_raw = filters_raw.get("curves", [])
        filters_obj = FilterSet(
            curves=[_coerce_dataclass(FilterCurve, curve) if isinstance(curve, Mapping) else curve for curve in curves_raw],
            speclite_names=dict(filters_raw.get("speclite_names", {})),
            use_grahsp_database=bool(filters_raw.get("use_grahsp_database", True)),
        )
    else:
        filters_obj = _coerce_dataclass(FilterSet, filters_raw)

    agn_raw = data.get("agn", {})
    if isinstance(agn_raw, Mapping):
        agn_obj = AGNConfig(
            fit_agn=bool(agn_raw.get("fit_agn", True)),
            use_powerlaw_disk=bool(agn_raw.get("use_powerlaw_disk", True)),
            feii_template=_coerce_dataclass(FeIITemplate, agn_raw.get("feii_template", {})),
            emission_line_template=_coerce_dataclass(EmissionLineTemplate, agn_raw.get("emission_line_template", {})),
            agn_type=int(agn_raw.get("agn_type", 1)),
            line_width_kms_default=float(agn_raw.get("line_width_kms_default", 3000.0)),
            lines_strength_default=float(agn_raw.get("lines_strength_default", 1.0)),
            feii_strength_default=float(agn_raw.get("feii_strength_default", 5.0)),
            balmer_continuum_default=float(agn_raw.get("balmer_continuum_default", 0.0)),
        )
    else:
        agn_obj = _coerce_dataclass(AGNConfig, agn_raw)

    spectroscopy_raw = data.get("spectroscopy")
    if spectroscopy_raw is None:
        spectroscopy_obj = None
    elif isinstance(spectroscopy_raw, SequenceABC) and not isinstance(spectroscopy_raw, (str, bytes, bytearray, Mapping, SpectroscopyData)):
        spectroscopy_obj = [
            _coerce_dataclass(SpectroscopyData, item)
            for item in spectroscopy_raw
        ]
    else:
        spectroscopy_obj = _coerce_dataclass(SpectroscopyData, spectroscopy_raw)

    cfg = FitConfig(
        observation=_coerce_dataclass(Observation, data["observation"]),
        photometry=_coerce_dataclass(PhotometryData, data["photometry"]),
        filters=filters_obj,
        galaxy=_coerce_dataclass(GalaxyConfig, data.get("galaxy", {})),
        nebular=_coerce_dataclass(NebularConfig, data.get("nebular", {})),
        agn=agn_obj,
        likelihood=_coerce_dataclass(LikelihoodConfig, data.get("likelihood", {})),
        spectroscopy=spectroscopy_obj,
        spectroscopy_config=_coerce_spectroscopy_config(data.get("spectroscopy_config", {})),
        inference=_coerce_dataclass(InferenceConfig, data.get("inference", {})),
        prior_config=dict(data.get("prior_config", {})),
    )
    cfg.validate()
    return cfg


def serialize_config(value: Any) -> Any:
    """Convert config-like objects into JSON-serializable Python values."""
    if is_dataclass(value):
        return {k: serialize_config(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: serialize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
