from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpyro.distributions as dist


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
        if not np.isfinite(float(self.redshift)) or float(self.redshift) < 0.0:
            raise ValueError("observation.redshift must be finite and non-negative.")
        if not np.isfinite(float(self.redshift_err)) or float(self.redshift_err) < 0.0:
            raise ValueError("observation.redshift_err must be finite and non-negative.")
        self.redshift_mode = mode


@dataclass
class PhotometryData:
    """Observed photometric measurements and associated metadata.

    ``photometry_method`` records the measurement semantics for provenance and
    downstream diagnostics. Aperture/PSF corrections are controlled by
    ``psf_fwhm_arcsec``, ``aperture_diameter_arcsec``, and likelihood settings,
    not by this label alone. Use ``psf`` for point-source/PSF-like
    measurements, ``profile`` for profile-fit photometry, ``aperture`` for
    explicit fixed apertures, ``auto`` for Kron/AUTO-like photometry,
    ``model``/``cmodel``/``petrosian`` for extended-source model measurements,
    and ``catalog`` only when the catalog flux semantics are not known.
    """
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
        if self.photometry_method is not None:
            allowed_methods = {
                "aperture",
                "auto",
                "catalog",
                "cmodel",
                "fiber",
                "model",
                "petrosian",
                "profile",
                "psf",
                "unknown",
            }
            normalized_methods: list[str | None] = []
            for method in self.photometry_method:
                if method is None:
                    normalized_methods.append(None)
                    continue
                normalized = str(method).strip().lower()
                if normalized not in allowed_methods:
                    allowed = ", ".join(sorted(allowed_methods))
                    raise ValueError(f"Unknown photometry_method '{method}'. Allowed metadata labels: {allowed}.")
                normalized_methods.append(normalized)
            self.photometry_method = normalized_methods


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
    """Fe II template configuration or inline template data.

    ``wavelength_unit`` is required for inline ``wave`` values. Supported
    values are ``angstrom``, ``nm``, and ``micron`` (plus common aliases).
    """
    name: str = "BruhweilerVerner08"
    wave: Sequence[float] | None = None
    lumin: Sequence[float] | None = None
    wavelength_unit: str | None = None


@dataclass
class EmissionLineTemplate:
    """Emission-line template tables with explicit inline wavelength units."""
    wave: Sequence[float] | None = None
    lumin_blagn: Sequence[float] | None = None
    lumin_sy2: Sequence[float] | None = None
    lumin_liner: Sequence[float] | None = None
    wavelength_unit: str | None = None


@dataclass
class GalaxyConfig:
    """Host-galaxy model, SSP provenance, cosmology, and wavelength-grid settings.

    ``ssp_imf`` and ``ssp_metallicity_coordinate`` describe the already-built
    library at ``dsps_ssp_fn``; they do not regenerate or transform that file.
    The IMF declaration selects the matching stellar surviving-mass
    calibration used to convert between formed and surviving mass.
    """
    fit_host: bool = True
    fit_host_kinematics: bool = False
    host_sfh_model: str = "delayed"
    dsps_ssp_fn: str = "tempdata.h5"
    ssp_imf: str = "chabrier_2003"
    ssp_metallicity_coordinate: str = "absolute_log10_z"
    ssp_solar_metallicity: float = 0.019
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

    def validate(self) -> None:
        """Validate the internal Angstrom wavelength grid and SFH grid."""
        supported_imfs = {"chabrier_2003", "salpeter_1955", "kroupa_2001", "van_dokkum_2008"}
        self.ssp_imf = str(self.ssp_imf).strip().lower()
        if self.ssp_imf not in supported_imfs:
            supported = ", ".join(sorted(supported_imfs))
            raise ValueError(f"galaxy.ssp_imf must be one of: {supported}.")
        supported_coordinates = {"absolute_log10_z", "log10_z_over_zsun"}
        self.ssp_metallicity_coordinate = str(self.ssp_metallicity_coordinate).strip().lower()
        if self.ssp_metallicity_coordinate not in supported_coordinates:
            supported = ", ".join(sorted(supported_coordinates))
            raise ValueError(f"galaxy.ssp_metallicity_coordinate must be one of: {supported}.")
        if not np.isfinite(float(self.ssp_solar_metallicity)) or float(self.ssp_solar_metallicity) <= 0.0:
            raise ValueError("galaxy.ssp_solar_metallicity must be positive and finite.")
        if not np.isfinite(float(self.rest_wave_min)) or float(self.rest_wave_min) <= 0.0:
            raise ValueError("galaxy.rest_wave_min must be positive and finite (Angstrom).")
        if not np.isfinite(float(self.rest_wave_max)) or float(self.rest_wave_max) <= float(self.rest_wave_min):
            raise ValueError("galaxy.rest_wave_max must be finite and greater than rest_wave_min (Angstrom).")
        if int(self.n_wave) < 2:
            raise ValueError("galaxy.n_wave must be at least 2.")
        if int(self.sfh_n_steps) < 2:
            raise ValueError("galaxy.sfh_n_steps must be at least 2.")
        if not np.isfinite(float(self.sfh_t_min_gyr)) or float(self.sfh_t_min_gyr) <= 0.0:
            raise ValueError("galaxy.sfh_t_min_gyr must be positive and finite.")


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
    fit_feii_broadening: bool = False
    fit_balmer_continuum: bool = False

    def __post_init__(self) -> None:
        """Normalize nested template sections."""
        self.feii_template = _coerce_dataclass(FeIITemplate, self.feii_template)
        self.emission_line_template = _coerce_dataclass(EmissionLineTemplate, self.emission_line_template)

    def validate(self) -> None:
        """Validate inline template units and array shapes."""
        allowed_units = {"angstrom", "angstroms", "aa", "a", "nm", "nanometer", "nanometers", "um", "micron", "microns"}
        for label, template in (("feii_template", self.feii_template), ("emission_line_template", self.emission_line_template)):
            unit = None if template.wavelength_unit is None else str(template.wavelength_unit).strip().lower()
            if template.wave is not None and unit is None:
                raise ValueError(f"agn.{label}.wavelength_unit is required when inline wave values are provided.")
            if unit is not None and unit not in allowed_units:
                allowed = "angstrom, nm, micron"
                raise ValueError(f"agn.{label}.wavelength_unit must be one of: {allowed}.")
            if template.wave is not None:
                wave = np.asarray(template.wave, dtype=float)
                min_size = 2 if label == "feii_template" else 1
                if wave.ndim != 1 or wave.size < min_size or not np.all(np.isfinite(wave)) or np.any(wave <= 0.0):
                    raise ValueError(f"agn.{label}.wave must contain at least {min_size} positive finite wavelength value(s).")
            if label == "feii_template" and template.wave is not None:
                if template.lumin is None or len(template.lumin) != len(template.wave):
                    raise ValueError("agn.feii_template.lumin must match feii_template.wave.")
            if label == "emission_line_template" and template.wave is not None:
                for field_name in ("lumin_blagn", "lumin_sy2", "lumin_liner"):
                    values = getattr(template, field_name)
                    if values is None or len(values) != len(template.wave):
                        raise ValueError(f"agn.emission_line_template.{field_name} must match emission_line_template.wave.")


@dataclass
class LikelihoodConfig:
    """Likelihood and extra model-mismatch configuration."""
    systematics_width: float = 0.10
    fit_systematics_width: bool = True
    systematics_width_prior_scale: float = 0.10
    agn_systematics_width: float = 0.0
    fit_agn_systematics_width: bool = True
    agn_systematics_width_prior_scale: float = 0.20
    likelihood_family: str = "gaussian"
    student_t_df: float = 5.0
    variability_uncertainty: bool = True
    agn_nev: float = 0.1
    attenuation_model_uncertainty: bool = False
    lyman_break_uncertainty: bool = False
    use_host_capture_model: bool = False
    use_fast_photometry_projection: bool = True
    use_local_line_photometry: bool = True
    local_nebular_line_uncertainty_dex: float = 0.3
    use_fixed_local_line_cache: bool = True
    fixed_local_line_cache_n_width: int = 256
    fixed_local_line_cache_min_width_kms: float = 1.0
    fixed_local_line_cache_max_width_kms: float = 100000.0
    use_redshift_projection_cache: bool = True
    redshift_projection_n_grid: int = 128
    redshift_projection_sigma: float = 6.0


@dataclass
class JaxQSOFitConfig:
    """Joint jaxqsofit spectral-feature configuration.

    The spectral flags control Fe II, Balmer-continuum, and line components
    evaluated by jaxqsofit on the spectroscopic grid. Broadband photometry
    keeps the jaxsedfit continuum/torus/dust engine; when enabled, the native
    SED-scale AGN line template supplies only a simple global-strength
    correction for lines outside the spectroscopic coverage.
    """
    use_spectral_lines: bool = True
    use_spectral_feii: bool = False
    use_spectral_balmer_continuum: bool = False
    use_photometric_lines: bool = True
    use_tied_lines: bool = True
    use_spectral_smart_priors: bool = True
    use_multiplicative_tilt: bool = False
    line_flux_scale_mjy: float = 1.0
    line_coverage_margin_kms: float = 3000.0
    use_line_strength_priors: bool = True
    line_strength_prior_sigma_dex: float = 0.7
    use_nebular_line_prior: bool = True
    nebular_line_prior_sigma_dex: float = 1.0
    include_elg_narrow_lines: bool = False
    include_high_ionization_lines: bool = False
    line_table: Sequence[Mapping[str, Any]] | None = None
    broadening_convolution: str = "fft"

    def __post_init__(self) -> None:
        method = str(self.broadening_convolution).lower()
        if method not in {"fft", "direct"}:
            raise ValueError("JaxQSOFitConfig.broadening_convolution must be 'fft' or 'direct'.")
        self.broadening_convolution = method


@dataclass
class SpectroscopyConfig:
    """Spectroscopic likelihood configuration."""
    enabled: bool = False
    backend: str = "jaxsedfit"
    student_t_df: float = 5.0
    systematics_width: float = 0.05
    likelihood_weight_mode: str = "pixels"
    resolving_power: float | None = None
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
    """Convert scalar array-like distribution parameters into plain Python values.


    Parameters
    ----------
    value : object
        NumPyro distribution parameter, scalar, or array-like value to
        serialize.
    """
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return arr.tolist()


def _numpyro_distribution_to_mapping(value: Any) -> dict[str, Any] | None:
    """Convert supported NumPyro distributions into the model prior schema.

    Parameters
    ----------
    value : object
        Candidate NumPyro distribution instance. Unsupported non-distribution
        objects return ``None``.
    """
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


def _mapping_to_numpyro_distribution(value: Mapping[str, Any]) -> dist.Distribution:
    """Restore a distribution from a legacy serialized prior mapping."""
    family = str(value.get("dist", value.get("family", ""))).lower()
    if family in {"normal", "gaussian"}:
        return dist.Normal(value.get("loc", 0.0), value.get("scale", 1.0))
    if family in {"truncatednormal", "truncated_normal", "truncnormal", "truncnorm"}:
        return dist.TruncatedNormal(
            value.get("loc", 0.0), value.get("scale", 1.0),
            low=value.get("low", -np.inf), high=value.get("high", np.inf),
        )
    if family in {"lognormal", "log-normal", "log_normal"}:
        return dist.LogNormal(value.get("loc", 0.0), value.get("scale", 1.0))
    if family in {"halfnormal", "half_normal"}:
        return dist.HalfNormal(value.get("scale", 1.0))
    if family in {"student_t", "studentt", "t"}:
        return dist.StudentT(value.get("df", 5.0), value.get("loc", 0.0), value.get("scale", 1.0))
    if family in {"uniform", "flat"}:
        return dist.Uniform(value.get("low", 0.0), value.get("high", 1.0))
    if family in {"exponential", "exp"}:
        return dist.Exponential(1.0 / value.get("scale", 1.0))
    raise TypeError(f"Unsupported serialized prior distribution: {family!r}")


def _prior_to_mapping(value: Any) -> dist.Distribution:
    """Return the canonical in-memory NumPyro distribution.

    Parameters
    ----------
    value : object
        Public prior field value. Must be a supported
        ``numpyro.distributions`` object.
    """
    if isinstance(value, dist.Distribution):
        return value
    if isinstance(value, Mapping):
        return _mapping_to_numpyro_distribution(value)
    raise TypeError("Prior fields must be supported numpyro.distributions objects.")


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
    broad_lines_strength: Any | None = None
    log_broad_lines_strength: Any | None = None
    narrow_lines_strength: Any | None = None
    log_narrow_lines_strength: Any | None = None
    broad_line_width_kms: Any | None = None
    log_broad_line_width_kms: Any | None = None
    narrow_line_width_kms: Any | None = None
    log_narrow_line_width_kms: Any | None = None
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
                "broad_lines_strength": "broad_lines_strength",
                "log_broad_lines_strength": "log_broad_lines_strength",
                "narrow_lines_strength": "narrow_lines_strength",
                "log_narrow_lines_strength": "log_narrow_lines_strength",
                "broad_line_width_kms": "broad_line_width_kms",
                "log_broad_line_width_kms": "log_broad_line_width_kms",
                "narrow_line_width_kms": "narrow_line_width_kms",
                "log_narrow_line_width_kms": "log_narrow_line_width_kms",
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
    """Nebular-emission prior options.

    ``f_dust`` controls the smooth fraction of non-escaping ionizing photons
    absorbed by dust, so the physical dust fraction is
    ``(1 - f_esc) * f_dust`` and always satisfies ``f_esc + f_dust_physical <= 1``.
    """
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
                "f_dust": "nebular_f_dust_fraction",
                "lines_width": "nebular_lines_width",
                "log_line_scale": "log_nebular_line_scale",
            },
        )


@dataclass
class LikelihoodPriorConfig:
    """Likelihood and calibration prior options."""
    systematics_width: Any | None = None
    log_systematics_width: Any | None = None
    agn_systematics_width: Any | None = None
    log_agn_systematics_width: Any | None = None
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
                "agn_systematics_width": "agn_systematics_width",
                "log_agn_systematics_width": "log_agn_systematics_width",
                "host_capture_scale_arcsec": "host_capture_scale_arcsec",
                "log_host_capture_scale_arcsec": "log_host_capture_scale_arcsec",
                "host_capture_slope": "host_capture_slope",
                "log_host_capture_slope": "log_host_capture_slope",
                "spectrum_scale": "spectrum_scale",
                "log_spectrum_scale": "log_spectrum_scale",
            },
        )


def _section_to_mapping(section: Any, fields_to_keys: Mapping[str, str]) -> dict[str, Any]:
    """Convert non-None section fields into model prior mappings.

    Parameters
    ----------
    section : object
        section value.
    fields_to_keys : object
        fields_to_keys value.
    """
    out: dict[str, Any] = {}
    for field_name, key in fields_to_keys.items():
        value = getattr(section, field_name)
        if value is not None:
            out[key] = _prior_to_mapping(value)
    return out


@dataclass
class PriorConfig:
    """Object-oriented prior configuration for a jaxsedfit model.

    Parameters
    ----------
    redshift : RedshiftPriorConfig or mapping, optional
        Optional tabulated redshift prior used when
        ``Observation.redshift_mode='fit'``.
    stellar_mass : numpyro.distributions.Distribution, optional
        Prior for ``log_stellar_mass``.
    mass_metallicity : MassMetallicityPriorConfig or mapping, optional
        Soft stellar mass-metallicity relation prior.
    host : HostPriorConfig or mapping, optional
        Host-galaxy priors for dust, metallicity, SFH, and stellar kinematics.
    agn : AGNPriorConfig or mapping, optional
        AGN continuum, torus, line-strength, Balmer continuum, and Fe II priors.
    nebular : NebularPriorConfig or mapping, optional
        Nebular gas, escape/dust fraction, line-width, and line-scale priors.
    likelihood : LikelihoodPriorConfig or mapping, optional
        Priors for nuisance likelihood terms such as photometric systematics,
        host capture, and spectrum scale.
    """
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
        if self.host.log_sfh_tau_gyr is not None and self.host.log_sfh_tau_over_age is not None:
            raise ValueError("Configure only one of host.log_sfh_tau_gyr and host.log_sfh_tau_over_age.")

    def to_mapping(self) -> dict[str, Any]:
        """Return flat model-site keys with distributions kept as objects."""
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
    """Top-level configuration bundle for a single jaxsedfit fit.

    Parameters
    ----------
    observation : Observation or mapping
        Source metadata including redshift, object identifier, sky coordinates,
        and redshift-fitting mode.
    photometry : PhotometryData or mapping
        Broadband fluxes, uncertainties, upper-limit flags, and optional
        aperture/PSF metadata.
    filters : FilterSet or mapping, optional
        Explicit filter curves used for synthetic photometry. If omitted,
        filters are loaded from known filter names.
    galaxy : GalaxyConfig or mapping, optional
        Host-galaxy model, SSP grid, dust, cosmology, and wavelength-grid
        settings.
    nebular : NebularConfig or mapping, optional
        Host nebular-emission switches and fixed gas defaults.
    agn : AGNConfig or mapping, optional
        AGN continuum, torus, Fe II, Balmer continuum, and native line settings.
    likelihood : LikelihoodConfig or mapping, optional
        Photometric likelihood family, systematics, variability, local line
        projection, and aperture-capture options.
    spectroscopy : SpectroscopyData, sequence of SpectroscopyData, mapping, or None, optional
        Optional observed spectra for joint SED+spectrum fitting.
    spectroscopy_config : SpectroscopyConfig or mapping, optional
        Spectroscopic likelihood, scale, resolution weighting, and jaxqsofit
        backend options.
    inference : InferenceConfig or mapping, optional
        MAP, NUTS, and nested-sampling controls.
    output : OutputConfig or mapping, optional
        Plotting and persistence behavior.
    prior_config : PriorConfig or mapping, optional
        Priors consumed by the NumPyro model.
    """
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
        self.galaxy.validate()
        self.nebular.validate()
        self.agn.validate()
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
    """Convert a mapping or existing instance into the requested dataclass.

    Parameters
    ----------
    cls : type
        Dataclass type to construct.
    value : object
        Existing instance or mapping of dataclass field names to values.
    """
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        if cls is Observation and "fit_redshift" in data and "redshift_mode" not in data:
            data["redshift_mode"] = "fit" if bool(data.pop("fit_redshift")) else "fixed"
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown {cls.__name__} field(s): {unknown_list}")
        kwargs = {}
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name not in data:
                continue
            kwargs[field_name] = data[field_name]
        return cls(**kwargs)
    raise TypeError(f"Cannot coerce {type(value)!r} to {cls.__name__}")


def _coerce_jaxqsofit_config(value: Any) -> JaxQSOFitConfig:
    """Coerce jaxqsofit config into the structured config object.

    Parameters
    ----------
    value : object
        Existing :class:`JaxQSOFitConfig` instance or mapping.
    """
    return _coerce_dataclass(JaxQSOFitConfig, value)


def _coerce_spectroscopy_config(value: Any) -> SpectroscopyConfig:
    """Coerce spectroscopy config while preserving the nested jaxqsofit config.

    Parameters
    ----------
    value : object
        Existing :class:`SpectroscopyConfig` instance or mapping.
    """
    if isinstance(value, SpectroscopyConfig):
        return value
    if not isinstance(value, Mapping):
        return _coerce_dataclass(SpectroscopyConfig, value)
    unknown = set(value) - set(SpectroscopyConfig.__dataclass_fields__)
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown SpectroscopyConfig field(s): {unknown_list}")
    kwargs = {}
    for field_name in SpectroscopyConfig.__dataclass_fields__:
        if field_name not in value:
            continue
        if field_name == "jaxqsofit":
            kwargs[field_name] = _coerce_jaxqsofit_config(value[field_name])
        else:
            kwargs[field_name] = value[field_name]
    return SpectroscopyConfig(**kwargs)


def _coerce_prior_config(value: Any) -> PriorConfig:
    """Coerce structured prior mappings into :class:`PriorConfig`.

    Parameters
    ----------
    value : object
        Existing :class:`PriorConfig`, mapping, or ``None``.
    """
    if isinstance(value, PriorConfig):
        return value
    if value is None:
        return PriorConfig()
    if not isinstance(value, Mapping):
        return _coerce_dataclass(PriorConfig, value)

    data = dict(value)
    nested_keys = {"redshift", "stellar_mass", "mass_metallicity", "host", "agn", "nebular", "likelihood"}
    unknown = set(data) - nested_keys
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown PriorConfig section(s): {unknown_list}")
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
    """Build a validated FitConfig from a nested mapping.

    Parameters
    ----------
    data : object
        data value.
    """
    valid_top_level = set(FitConfig.__dataclass_fields__)
    unknown = set(data) - valid_top_level
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown FitConfig field(s): {unknown_list}")

    filters_raw = data.get("filters", {})
    if isinstance(filters_raw, Mapping):
        curves_raw = filters_raw.get("curves", [])
        filters_obj = FilterSet(
            curves=[_coerce_dataclass(FilterCurve, curve) if isinstance(curve, Mapping) else curve for curve in curves_raw],
        )
    else:
        filters_obj = _coerce_dataclass(FilterSet, filters_raw)

    agn_obj = _coerce_dataclass(AGNConfig, data.get("agn", {}))

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
    """Convert config-like objects into JSON-serializable Python values.


    Parameters
    ----------
    value : object
        Dataclass, mapping, sequence, NumPy array, NumPyro distribution, or
        scalar value to convert into JSON-compatible containers.
    """
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
