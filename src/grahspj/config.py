from __future__ import annotations

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

    def validate(self) -> None:
        """Validate array lengths for one photometry payload."""
        n = len(self.filter_names)
        if len(self.fluxes) != n or len(self.errors) != n:
            raise ValueError("Photometry arrays must have the same length as filter_names.")
        if self.is_upper_limit is not None and len(self.is_upper_limit) != n:
            raise ValueError("is_upper_limit must match filter_names length.")


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
    dsps_ssp_fn: str = "tempdata.h5"
    age_grid_gyr: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0)
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2)
    imf_type: int = 1
    zcontinuous: int = 1
    sfh: int = 0
    rest_wave_min: float = 900.0
    rest_wave_max: float = 3.0e5
    n_wave: int = 2048
    tau_host_prior_scale: float = 1.0
    sfh_n_steps: int = 64
    sfh_t_min_gyr: float = 0.01
    cosmology_h0: float = 70.0
    cosmology_om0: float = 0.3
    use_energy_balance: bool = True
    dust_alpha: float = 2.0


@dataclass
class AGNConfig:
    """AGN component configuration, templates, and fixed branch settings."""
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


@dataclass
class InferenceConfig:
    """Inference defaults for MAP optimization and NUTS sampling."""
    learning_rate: float = 5e-3
    map_steps: int = 1500
    num_warmup: int = 500
    num_samples: int = 1000
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
    agn: AGNConfig = field(default_factory=AGNConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    prior_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate nested config components that require runtime checks."""
        self.photometry.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass tree into a plain Python dictionary."""
        return asdict(self)


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

    cfg = FitConfig(
        observation=_coerce_dataclass(Observation, data["observation"]),
        photometry=_coerce_dataclass(PhotometryData, data["photometry"]),
        filters=filters_obj,
        galaxy=_coerce_dataclass(GalaxyConfig, data.get("galaxy", {})),
        agn=agn_obj,
        likelihood=_coerce_dataclass(LikelihoodConfig, data.get("likelihood", {})),
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
