from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class Observation:
    """Observation-level metadata for one fitted source."""
    redshift: float
    object_id: str = "result"
    redshift_mode: str = "fixed"
    redshift_err: float = 0.0
    ra: float | None = None
    dec: float | None = None
    apply_mw_deredden: bool = False

    @property
    def fits_redshift(self) -> bool:
        """Return True when redshift is inferred rather than fixed."""
        return str(self.redshift_mode).lower() == "fit"

    def validate(self) -> None:
        """Normalize and validate the redshift fitting mode."""
        mode = str(self.redshift_mode).lower()
        if mode not in {"fixed", "fit"}:
            raise ValueError("observation.redshift_mode must be either 'fixed' or 'fit'.")
        self.redshift_mode = mode


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
    host_sfh_model: str = "delayed"
    dsps_ssp_fn: str = "tempdata.h5"
    age_grid_gyr: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0)
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2)
    imf_type: int = 1
    zcontinuous: int = 1
    sfh: int = 0
    rest_wave_min: float = 100.0
    rest_wave_max: float = 3.0e6
    n_wave: int = 1024
    tau_host_prior_scale: float = 0.5
    sfh_n_steps: int = 64
    sfh_t_min_gyr: float = 0.01
    cosmology_h0: float = 70.0
    cosmology_om0: float = 0.3
    # Host-galaxy dust energy balance only. AGN torus emission is modeled by the
    # empirical AGN component, not by adding AGN-absorbed luminosity here.
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
        """Validate nebular-emission parameters and physical fractions."""
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
    fit_feii_broadening: bool = False
    fit_balmer_continuum: bool = False
    balmer_continuum_default: float = 0.0


@dataclass
class LikelihoodConfig:
    """Likelihood and extra model-mismatch configuration."""
    systematics_width: float = 0.05
    fit_systematics_width: bool = True
    systematics_width_prior_scale: float = 0.05
    likelihood_family: str = "gaussian"
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
    use_local_line_photometry: bool = True
    use_fixed_local_line_cache: bool = True
    fixed_local_line_cache_n_width: int = 128
    fixed_local_line_cache_min_width_kms: float = 200.0
    fixed_local_line_cache_max_width_kms: float = 30000.0
    use_redshift_projection_cache: bool = True
    redshift_projection_n_grid: int = 128
    redshift_projection_sigma: float = 6.0


@dataclass
class JaxQSOFitConfig:
    """Spectroscopy-only jaxqsofit component configuration.

    These flags affect only the spectroscopic likelihood. Broadband
    photometry continues to use jaxsedfit's native SED-scale AGN lines,
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
    backend: str = "jaxsedfit"
    student_t_df: float = 5.0
    systematics_width: float = 0.05
    fit_scale: bool = True
    scale_prior_sigma_dex: float = 0.5
    jaxqsofit: JaxQSOFitConfig = field(default_factory=JaxQSOFitConfig)


@dataclass
class InferenceConfig:
    """Inference defaults for MAP optimization, NUTS sampling, and nested sampling."""
    method: str = "optax+nuts"
    learning_rate: float = 5e-3
    map_steps: int = 1500
    staged_map: bool = True
    staged_steps: int | None = None
    num_warmup: int = 200
    num_samples: int = 200
    num_chains: int = 1
    target_accept_prob: float = 0.85
    dense_mass: bool = False
    max_tree_depth: int = 8
    use_map_init: bool = True
    ns_num_live_points: int | None = None
    ns_max_samples: int | None = None
    ns_dlogz: float | None = None
    ns_resamples: int | None = None
    ns_difficult_model: bool = False
    ns_parameter_estimation: bool = False
    ns_num_parallel_workers: int | None = None
    ns_init_efficiency_threshold: float | None = None
    ns_max_likelihood_evals: int | None = None
    ns_efficiency_threshold: float | None = None
    seed: int = 0


@dataclass
class OutputConfig:
    """Plotting and persistence defaults."""
    output_dir: str = "."
    fig_path: str | None = None
    result_path: str | None = None
    plot_fig: bool = False
    save_fig: bool = False
    save_result: bool = False
    show_plot: bool = False


def _scalar_or_list(value: Any) -> Any:
    """Convert scalar array-like distribution parameters into plain Python values."""
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return arr.tolist()


def _numpyro_distribution_to_mapping(value: Any) -> dict[str, Any] | None:
    """Convert supported NumPyro distributions into the model prior schema."""
    module = getattr(value.__class__, "__module__", "")
    if not module.startswith("numpyro.distributions"):
        return None

    name = value.__class__.__name__
    if name in {"Normal", "LogNormal"}:
        return {
            "dist": name,
            "loc": _scalar_or_list(value.loc),
            "scale": _scalar_or_list(value.scale),
        }
    if name == "TwoSidedTruncatedDistribution":
        base = value.base_dist
        if base.__class__.__name__ == "Normal":
            return {
                "dist": "TruncatedNormal",
                "loc": _scalar_or_list(base.loc),
                "scale": _scalar_or_list(base.scale),
                "low": _scalar_or_list(value.low),
                "high": _scalar_or_list(value.high),
            }
    if name == "TruncatedNormal":
        return {
            "dist": name,
            "loc": _scalar_or_list(value.loc),
            "scale": _scalar_or_list(value.scale),
            "low": _scalar_or_list(value.low),
            "high": _scalar_or_list(value.high),
        }
    if name == "HalfNormal":
        return {"dist": name, "scale": _scalar_or_list(value.scale)}
    if name == "StudentT":
        return {
            "dist": "student_t",
            "df": _scalar_or_list(value.df),
            "loc": _scalar_or_list(value.loc),
            "scale": _scalar_or_list(value.scale),
        }
    if name == "Uniform":
        return {
            "dist": "uniform",
            "low": _scalar_or_list(value.low),
            "high": _scalar_or_list(value.high),
        }
    if name == "Exponential":
        rate = _scalar_or_list(value.rate)
        return {"dist": "exponential", "scale": 1.0 / rate if np.isscalar(rate) else (1.0 / np.asarray(rate)).tolist()}
    raise TypeError(f"Unsupported NumPyro prior distribution: {name}")


def _prior_to_mapping(value: Any) -> Any:
    """Convert public prior specs to low-level mappings."""
    if isinstance(value, Mapping):
        return dict(value)
    prior = _numpyro_distribution_to_mapping(value)
    if prior is not None:
        return prior
    raise TypeError("Prior fields must be mappings or supported numpyro.distributions objects.")


@dataclass
class RedshiftPriorConfig:
    """Optional redshift-prior configuration."""
    z_grid: Sequence[float] | None = None
    pdf: Sequence[float] | None = None

    @property
    def enabled(self) -> bool:
        """Return True when a tabulated redshift prior is configured."""
        return self.z_grid is not None or self.pdf is not None

    def validate(self) -> None:
        """Validate the tabulated redshift PDF shape, ordering, and normalization."""
        if not self.enabled:
            return
        if self.z_grid is None or self.pdf is None:
            raise ValueError("redshift prior requires both z_grid and pdf.")
        z_grid = np.asarray(self.z_grid, dtype=float)
        pdf = np.asarray(self.pdf, dtype=float)
        if z_grid.ndim != 1 or pdf.ndim != 1 or z_grid.size != pdf.size or z_grid.size < 2:
            raise ValueError("redshift prior z_grid and pdf must be one-dimensional arrays of the same length >= 2.")
        if not np.all(np.isfinite(z_grid)) or not np.all(np.isfinite(pdf)):
            raise ValueError("redshift prior z_grid and pdf must be finite.")
        if np.any(np.diff(z_grid) <= 0.0):
            raise ValueError("redshift prior z_grid must be strictly increasing.")
        if np.any(pdf < 0.0):
            raise ValueError("redshift prior pdf must be non-negative.")
        norm = float(np.trapezoid(pdf, z_grid))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("redshift prior must integrate to a positive finite value.")

    def to_mapping(self) -> dict[str, Any]:
        """Convert the redshift prior into the low-level model mapping."""
        if not self.enabled:
            return {}
        return {"redshift_pdf": {"z_grid": self.z_grid, "pdf": self.pdf}}


@dataclass
class MassMetallicityPriorConfig:
    """Soft stellar mass-metallicity prior for host metallicity."""
    configured: bool = False
    enabled: bool = True
    pivot_mass: float = 10.0
    pivot_logzsol: float = -0.15
    pivot_lgmet: float | None = None
    slope: float = 0.35
    scale: float = 0.25
    redshift_ref: float = 0.0
    redshift_slope: float = -0.15
    min: float = -1.5
    max: float = 0.3
    min_lgmet: float | None = None
    max_lgmet: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert the mass-metallicity relation prior into model settings."""
        if not self.configured:
            return {}
        out: dict[str, Any] = {
            "enabled": bool(self.enabled),
            "pivot_mass": float(self.pivot_mass),
            "pivot_logzsol": float(self.pivot_logzsol),
            "slope": float(self.slope),
            "scale": float(self.scale),
            "redshift_ref": float(self.redshift_ref),
            "redshift_slope": float(self.redshift_slope),
            "min": float(self.min),
            "max": float(self.max),
        }
        if self.pivot_lgmet is not None:
            out["pivot_lgmet"] = float(self.pivot_lgmet)
        if self.min_lgmet is not None:
            out["min_lgmet"] = float(self.min_lgmet)
        if self.max_lgmet is not None:
            out["max_lgmet"] = float(self.max_lgmet)
        return {"mass_metallicity_relation": out}


@dataclass
class HostPriorConfig:
    """Host-galaxy prior options."""
    gal_lgmet: Any | None = None
    gal_lgmet_scatter: Any | None = None
    gal_v_kms: Any | None = None
    gal_sigma_kms: Any | None = None
    dust_alpha: Any | None = None
    ebv_gal: Any | None = None
    log_ebv_gal: Any | None = None
    log_sfh_tau_gyr: Any | None = None
    log_sfh_age_gyr: Any | None = None
    log_sfh_tau_over_age: Any | None = None
    u_lgmcrit: Any | None = None
    u_lgy_at_mcrit: Any | None = None
    u_indx_lo: Any | None = None
    u_indx_hi: Any | None = None
    u_tau_dep: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert host prior settings into model-site keys."""
        return _section_to_mapping(
            self,
            {
                "gal_lgmet": "gal_lgmet",
                "gal_lgmet_scatter": "gal_lgmet_scatter",
                "gal_v_kms": "gal_v_kms",
                "gal_sigma_kms": "gal_sigma_kms",
                "dust_alpha": "dust_alpha",
                "ebv_gal": "ebv_gal",
                "log_ebv_gal": "log_ebv_gal",
                "log_sfh_tau_gyr": "log_sfh_tau_gyr",
                "log_sfh_age_gyr": "log_sfh_age_gyr",
                "log_sfh_tau_over_age": "log_sfh_tau_over_age",
                "u_lgmcrit": "u_lgmcrit",
                "u_lgy_at_mcrit": "u_lgy_at_mcrit",
                "u_indx_lo": "u_indx_lo",
                "u_indx_hi": "u_indx_hi",
                "u_tau_dep": "u_tau_dep",
            },
        )


@dataclass
class AGNPriorConfig:
    """AGN prior options."""
    log_amp: Any | None = None
    pl_slope: Any | None = None
    uv_slope_delta: Any | None = None
    log_uv_slope_delta: Any | None = None
    pl_bend_loc: Any | None = None
    log_pl_bend_loc: Any | None = None
    pl_bend_width: Any | None = None
    log_pl_bend_width: Any | None = None
    pl_cutoff: Any | None = None
    log_pl_cutoff: Any | None = None
    fcov: Any | None = None
    log_fcov: Any | None = None
    si: Any | None = None
    cool_lam: Any | None = None
    log_cool_lam: Any | None = None
    cool_width: Any | None = None
    log_cool_width: Any | None = None
    hot_lam: Any | None = None
    log_hot_lam: Any | None = None
    hot_width: Any | None = None
    log_hot_width: Any | None = None
    hot_fcov: Any | None = None
    log_hot_fcov: Any | None = None
    ebv_agn: Any | None = None
    log_ebv_agn: Any | None = None
    lines_strength: Any | None = None
    log_lines_strength: Any | None = None
    line_width_kms: Any | None = None
    log_line_width_kms: Any | None = None
    balmer_norm: Any | None = None
    log_balmer_norm: Any | None = None
    balmer_tau: Any | None = None
    log_balmer_tau: Any | None = None
    balmer_vel: Any | None = None
    log_balmer_vel: Any | None = None
    feii_norm: Any | None = None
    log_feii_norm: Any | None = None
    feii_fwhm: Any | None = None
    log_feii_fwhm: Any | None = None
    feii_shift: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert AGN prior settings into model-site keys."""
        return _section_to_mapping(
            self,
            {
                "log_amp": "log_agn_amp",
                "pl_slope": "pl_slope",
                "uv_slope_delta": "uv_slope_delta",
                "log_uv_slope_delta": "log_uv_slope_delta",
                "pl_bend_loc": "pl_bend_loc",
                "log_pl_bend_loc": "log_pl_bend_loc",
                "pl_bend_width": "pl_bend_width",
                "log_pl_bend_width": "log_pl_bend_width",
                "pl_cutoff": "pl_cutoff",
                "log_pl_cutoff": "log_pl_cutoff",
                "fcov": "fcov",
                "log_fcov": "log_fcov",
                "si": "si",
                "cool_lam": "cool_lam",
                "log_cool_lam": "log_cool_lam",
                "cool_width": "cool_width",
                "log_cool_width": "log_cool_width",
                "hot_lam": "hot_lam",
                "log_hot_lam": "log_hot_lam",
                "hot_width": "hot_width",
                "log_hot_width": "log_hot_width",
                "hot_fcov": "hot_fcov",
                "log_hot_fcov": "log_hot_fcov",
                "ebv_agn": "ebv_agn",
                "log_ebv_agn": "log_ebv_agn",
                "lines_strength": "lines_strength",
                "log_lines_strength": "log_lines_strength",
                "line_width_kms": "line_width_kms",
                "log_line_width_kms": "log_line_width_kms",
                "balmer_norm": "balmer_norm",
                "log_balmer_norm": "log_balmer_norm",
                "balmer_tau": "balmer_tau",
                "log_balmer_tau": "log_balmer_tau",
                "balmer_vel": "balmer_vel",
                "log_balmer_vel": "log_balmer_vel",
                "feii_norm": "feii_norm",
                "log_feii_norm": "log_feii_norm",
                "feii_fwhm": "feii_fwhm",
                "log_feii_fwhm": "log_feii_fwhm",
                "feii_shift": "feii_shift",
            },
        )


@dataclass
class NebularPriorConfig:
    """Nebular-emission prior options."""
    logU: Any | None = None
    zgas: Any | None = None
    ne: Any | None = None
    f_esc: Any | None = None
    f_dust: Any | None = None
    lines_width: Any | None = None
    log_line_scale: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert nebular prior settings into model-site keys."""
        return _section_to_mapping(
            self,
            {
                "logU": "nebular_logU",
                "zgas": "nebular_zgas",
                "ne": "nebular_ne",
                "f_esc": "nebular_f_esc",
                "f_dust": "nebular_f_dust",
                "lines_width": "nebular_lines_width",
                "log_line_scale": "log_nebular_line_scale",
            },
        )


@dataclass
class LikelihoodPriorConfig:
    """Likelihood and calibration prior options."""
    systematics_width: Any | None = None
    log_systematics_width: Any | None = None
    intrinsic_scatter: Any | None = None
    log_intrinsic_scatter: Any | None = None
    host_capture_scale_arcsec: Any | None = None
    log_host_capture_scale_arcsec: Any | None = None
    host_capture_slope: Any | None = None
    log_host_capture_slope: Any | None = None
    spectrum_scale: Any | None = None
    log_spectrum_scale: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert likelihood prior settings into model-site keys."""
        return _section_to_mapping(
            self,
            {
                "systematics_width": "systematics_width",
                "log_systematics_width": "log_systematics_width",
                "intrinsic_scatter": "intrinsic_scatter",
                "log_intrinsic_scatter": "log_intrinsic_scatter",
                "host_capture_scale_arcsec": "host_capture_scale_arcsec",
                "log_host_capture_scale_arcsec": "log_host_capture_scale_arcsec",
                "host_capture_slope": "host_capture_slope",
                "log_host_capture_slope": "log_host_capture_slope",
                "spectrum_scale": "spectrum_scale",
                "log_spectrum_scale": "log_spectrum_scale",
            },
        )


def _section_to_mapping(section: Any, fields_to_keys: Mapping[str, str]) -> dict[str, Any]:
    """Convert non-None section fields into model prior mappings."""
    out: dict[str, Any] = {}
    for field_name, key in fields_to_keys.items():
        value = getattr(section, field_name)
        if value is not None:
            out[key] = _prior_to_mapping(value)
    return out


@dataclass
class PriorConfig:
    """Object-oriented prior configuration."""
    redshift: RedshiftPriorConfig = field(default_factory=RedshiftPriorConfig)
    stellar_mass: Any | None = None
    mass_metallicity: MassMetallicityPriorConfig = field(default_factory=MassMetallicityPriorConfig)
    host: HostPriorConfig = field(default_factory=HostPriorConfig)
    agn: AGNPriorConfig = field(default_factory=AGNPriorConfig)
    nebular: NebularPriorConfig = field(default_factory=NebularPriorConfig)
    likelihood: LikelihoodPriorConfig = field(default_factory=LikelihoodPriorConfig)

    def __post_init__(self) -> None:
        """Normalize nested prior sections passed as mappings."""
        self.redshift = _coerce_dataclass(RedshiftPriorConfig, self.redshift)
        self.mass_metallicity = _coerce_dataclass(MassMetallicityPriorConfig, self.mass_metallicity)
        self.host = _coerce_dataclass(HostPriorConfig, self.host)
        self.agn = _coerce_dataclass(AGNPriorConfig, self.agn)
        self.nebular = _coerce_dataclass(NebularPriorConfig, self.nebular)
        self.likelihood = _coerce_dataclass(LikelihoodPriorConfig, self.likelihood)

    def validate(self) -> None:
        """Validate nested semantic prior objects."""
        self.redshift.validate()

    def to_mapping(self) -> dict[str, Any]:
        """Return the flat prior mapping consumed by the NumPyro model."""
        out: dict[str, Any] = {}
        if self.stellar_mass is not None:
            out["log_stellar_mass"] = _prior_to_mapping(self.stellar_mass)
        out.update(self.redshift.to_mapping())
        out.update(self.mass_metallicity.to_mapping())
        out.update(self.host.to_mapping())
        out.update(self.agn.to_mapping())
        out.update(self.nebular.to_mapping())
        out.update(self.likelihood.to_mapping())
        return out


@dataclass
class FitConfig:
    """Top-level configuration bundle for a single jaxsedfit fit."""
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
    output: OutputConfig = field(default_factory=OutputConfig)
    prior_config: PriorConfig = field(default_factory=PriorConfig)

    def __post_init__(self) -> None:
        """Coerce mapping-style prior configs into :class:`PriorConfig`."""
        self.prior_config = _coerce_prior_config(self.prior_config)

    def validate(self) -> None:
        """Validate nested config components that require runtime checks."""
        self.observation.validate()
        self.photometry.validate()
        self.nebular.validate()
        for spectrum in self.spectroscopy_list:
            spectrum.validate()
        if not self.galaxy.fit_host and not self.agn.fit_agn:
            raise ValueError("At least one of galaxy.fit_host or agn.fit_agn must be True.")
        self.prior_config.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass tree into a plain Python dictionary."""
        return serialize_config(self)

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
        data = dict(value)
        if cls is Observation and "fit_redshift" in data and "redshift_mode" not in data:
            data["redshift_mode"] = "fit" if bool(data.pop("fit_redshift")) else "fixed"
        kwargs = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name not in data:
                continue
            kwargs[field_name] = data[field_name]
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


def _coerce_prior_config(value: Any) -> PriorConfig:
    """Coerce structured prior mappings into :class:`PriorConfig`."""
    if isinstance(value, PriorConfig):
        return value
    if value is None:
        return PriorConfig()
    if not isinstance(value, Mapping):
        return _coerce_dataclass(PriorConfig, value)

    data = dict(value)
    nested_keys = {"redshift", "stellar_mass", "mass_metallicity", "host", "agn", "nebular", "likelihood"}
    if data and not any(key in data for key in nested_keys):
        raise ValueError("prior_config mappings must use structured PriorConfig sections.")
    return PriorConfig(
        redshift=_coerce_dataclass(RedshiftPriorConfig, data.get("redshift", {})),
        stellar_mass=data.get("stellar_mass"),
        mass_metallicity=_coerce_dataclass(MassMetallicityPriorConfig, data.get("mass_metallicity", {})),
        host=_coerce_dataclass(HostPriorConfig, data.get("host", {})),
        agn=_coerce_dataclass(AGNPriorConfig, data.get("agn", {})),
        nebular=_coerce_dataclass(NebularPriorConfig, data.get("nebular", {})),
        likelihood=_coerce_dataclass(LikelihoodPriorConfig, data.get("likelihood", {})),
    )


def fit_config_from_mapping(data: Mapping[str, Any]) -> FitConfig:
    """Build a validated FitConfig from a nested mapping."""
    filters_raw = data.get("filters", {})
    if isinstance(filters_raw, Mapping):
        curves_raw = filters_raw.get("curves", [])
        filters_obj = FilterSet(
            curves=[_coerce_dataclass(FilterCurve, curve) if isinstance(curve, Mapping) else curve for curve in curves_raw],
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
            fit_feii_broadening=bool(agn_raw.get("fit_feii_broadening", False)),
            fit_balmer_continuum=bool(agn_raw.get("fit_balmer_continuum", False)),
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
        output=_coerce_dataclass(OutputConfig, data.get("output", {})),
        prior_config=_coerce_prior_config(data.get("prior_config", {})),
    )
    cfg.validate()
    return cfg


def serialize_config(value: Any) -> Any:
    """Convert config-like objects into JSON-serializable Python values."""
    prior = _numpyro_distribution_to_mapping(value)
    if prior is not None:
        return serialize_config(prior)
    if is_dataclass(value):
        return {k: serialize_config(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: serialize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
