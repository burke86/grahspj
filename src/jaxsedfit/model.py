from __future__ import annotations

# Portions of this file are derived from or closely based on GRAHSP/pcigale
# model logic, translated into JAX/NumPyro for jaxsedfit.
# Relevant upstream sources include:
# - pcigale/creation_modules/activate.py
# - pcigale/creation_modules/activategtorus.py
# - pcigale/creation_modules/activatelines.py
# - pcigale/creation_modules/biattenuation.py
# - pcigale/creation_modules/redshifting.py
# - pcigale/creation_modules/galdale2014.py
# Upstream license: CeCILL v2. See LICENSES/CeCILL-v2.txt and
# LICENSES/THIRD_PARTY_NOTICES.md.

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_U_PARAMS, DiffstarUParams, calc_sfh_singlegal, get_bounded_diffstar_params
from diffstar.defaults import FB as DIFFSTAR_FB
from diffstar.defaults import LGT0 as DIFFSTAR_LGT0
from dsps.sed.ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf
import numpyro
import numpyro.distributions as dist

from .preload import ModelContext, _build_fixed_igm_jax as _igm_transmission

C_KMS = 299792.458
C_MS = 2.99792458e8
L_SUN_W = 3.828e26
ERG_PER_WATT = 1.0e7
AGN_BOLOMETRIC_CORRECTION_5100 = 9.26
DSPS_SOLAR_METALLICITY = 0.019
MPC_TO_M = 3.085677581491367e22
DEFAULT_BROAD_LINE_WIDTH_KMS_MIN = 1000.0
DEFAULT_BROAD_LINE_WIDTH_KMS_MAX = 15000.0
DEFAULT_NARROW_LINE_WIDTH_KMS_MIN = 100.0
DEFAULT_NARROW_LINE_WIDTH_KMS_MAX = 1500.0
DEFAULT_BROAD_LINE_WIDTH_KMS = 3000.0
DEFAULT_NARROW_LINE_WIDTH_KMS = 500.0
DEFAULT_BROAD_LINES_STRENGTH = 1.0
DEFAULT_NARROW_LINES_STRENGTH = 1.0
DEFAULT_FEII_STRENGTH = 5.0
DEFAULT_BALMER_CONTINUUM_STRENGTH = 1.0e-3
GRAHSP_BIATTENUATION_BREAK_A = 11000.0
GRAHSP_PL_BEND_LOC_A = 1000.0
GRAHSP_PL_BEND_WIDTH = 10.0
GRAHSP_PL_CUTOFF_A = 100000.0
GRAHSP_TORUS_NORM_A = 120000.0
GRAHSP_SI_EM_LAM_A = 98410.0
GRAHSP_SI_ABS_LAM_A = 142240.0
GRAHSP_SI_EM_WIDTH_A = 10253.0
GRAHSP_SI_ABS_WIDTH_A = 11635.0


def _np_to_jnp(x):
    """Convert an array-like object to a float64 JAX array."""
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def _bool_to_jnp(x):
    """Convert an array-like object to a boolean JAX array."""
    return jnp.asarray(np.asarray(x, dtype=bool))


@lru_cache(maxsize=16)
def _get_jax_cosmo_backend(h0: float, om0: float):
    """Return cached jax_cosmo helpers for a flat LCDM luminosity distance."""
    import jax_cosmo.background as bg
    from jax_cosmo.core import Cosmology

    omega_b = min(0.05, max(float(om0) - 1.0e-6, 1.0e-6))
    omega_c = max(float(om0) - omega_b, 1.0e-6)
    cosmo = Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=float(h0) / 100.0,
        n_s=0.96,
        sigma8=0.8,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )
    return bg, cosmo


def _flat_lcdm_luminosity_distance_m_jax(redshift, h0: float, om0: float):
    """Fallback flat-LCDM luminosity distance when jax_cosmo is unavailable."""
    z = jnp.maximum(jnp.asarray(redshift, dtype=jnp.float64), 0.0)
    grid = jnp.linspace(0.0, 1.0, 256, dtype=jnp.float64)
    z_grid = z[..., None] * grid
    e_z = jnp.sqrt(float(om0) * (1.0 + z_grid) ** 3 + (1.0 - float(om0)))
    integrand = 1.0 / jnp.maximum(e_z, 1.0e-30)
    dt = 1.0 / (grid.size - 1)
    integral_unit = dt * (
        0.5 * integrand[..., 0] + jnp.sum(integrand[..., 1:-1], axis=-1) + 0.5 * integrand[..., -1]
    )
    comoving_mpc = (C_KMS / float(h0)) * z * integral_unit
    return (1.0 + z) * comoving_mpc * MPC_TO_M


def _luminosity_distance_m_jax(redshift, h0: float, om0: float):
    """Return luminosity distance in meters using a JAX-native flat LCDM path."""
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    scalar_input = redshift.ndim == 0
    try:
        bg, cosmo = _get_jax_cosmo_backend(float(h0), float(om0))
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise
        d_l_m = _flat_lcdm_luminosity_distance_m_jax(redshift, h0, om0)
    else:
        a = 1.0 / (1.0 + jnp.maximum(redshift, 0.0))
        d_a_mpc_over_h = bg.angular_diameter_distance(cosmo, a)
        d_l_mpc_over_h = d_a_mpc_over_h / jnp.maximum(a * a, 1.0e-30)
        d_l_m = d_l_mpc_over_h / cosmo.h * MPC_TO_M
    return jnp.reshape(d_l_m, ()) if scalar_input else d_l_m


def _prior_distribution(prior_config: dict[str, Any], key: str, default_distribution):
    """Read a NumPyro distribution-like prior from the flat prior mapping."""
    cfg = prior_config.get(key, None)
    if cfg is None:
        return default_distribution
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return dist.Normal(jnp.asarray(cfg[0], dtype=jnp.float64), jnp.maximum(jnp.asarray(cfg[1], dtype=jnp.float64), 1.0e-6))
    if not isinstance(cfg, dict):
        return default_distribution

    default_name = default_distribution.__class__.__name__
    family = str(cfg.get("dist", cfg.get("family", default_name))).lower()
    if family in {"normal", "gaussian"}:
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.Normal(loc, scale)
    if family in {"truncatednormal", "truncated_normal", "truncnormal", "truncnorm"}:
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        low = jnp.asarray(cfg.get("low", -jnp.inf), dtype=jnp.float64)
        high = jnp.asarray(cfg.get("high", jnp.inf), dtype=jnp.float64)
        return dist.TruncatedNormal(loc, scale, low=low, high=high)
    if family in {"lognormal", "log-normal", "log_normal"}:
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.LogNormal(loc, scale)
    if family in {"halfnormal", "half_normal"}:
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.HalfNormal(scale)
    if family in {"student_t", "studentt", "t"}:
        df = jnp.maximum(jnp.asarray(cfg.get("df", 5.0), dtype=jnp.float64), 1.0e-6)
        loc = jnp.asarray(cfg.get("loc", 0.0), dtype=jnp.float64)
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-6)
        return dist.StudentT(df=df, loc=loc, scale=scale)
    if family in {"uniform", "flat"}:
        low = jnp.asarray(cfg.get("low", 0.0), dtype=jnp.float64)
        high = jnp.asarray(cfg.get("high", 1.0), dtype=jnp.float64)
        lo = jnp.minimum(low, high)
        hi = jnp.maximum(jnp.maximum(low, high), lo + 1.0e-6)
        return dist.Uniform(lo, hi)
    if family in {"exponential", "exp"}:
        scale = jnp.maximum(jnp.asarray(cfg.get("scale", 1.0), dtype=jnp.float64), 1.0e-30)
        return dist.Exponential(1.0 / scale)
    return default_distribution


def _sample_prior(prior_config: dict[str, Any], key: str, default_distribution):
    """Sample a scalar site from a configured distribution or a default."""
    return numpyro.sample(key, _prior_distribution(prior_config, key, default_distribution))


def _sample_log_positive_from_distribution(
    prior_config: dict[str, Any],
    *,
    value_key: str,
    log_key: str,
    default_distribution,
):
    """Sample a log-parameter from a distribution and expose its physical value."""
    log_value = numpyro.sample(log_key, _prior_distribution(prior_config, log_key, default_distribution))
    value = jnp.exp(log_value)
    numpyro.deterministic(value_key, value)
    return value


def _sample_positive_distribution(
    prior_config: dict[str, Any],
    *,
    value_key: str,
    log_key: str,
    default_value_distribution,
    default_log_distribution,
    default_to_log: bool = False,
):
    """Sample a positive parameter, honoring either physical or log prior keys."""
    if log_key in prior_config:
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=default_log_distribution,
        )
    if value_key not in prior_config and default_to_log:
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=default_log_distribution,
        )
    return _sample_prior(prior_config, value_key, default_value_distribution)


def _sample_positive(
    prior_config: dict[str, Any],
    *,
    value_key: str,
    log_key: str,
    default_value: float,
    default_log_scale: float,
    default_family: str = "lognormal",
):
    """Sample a positive parameter with an explicit prior family.

    ``prior_config[value_key]["family"]`` or ``["dist"]`` may be one of
    ``"exponential"`` or ``"lognormal"``. A direct ``prior_config[log_key]``
    override always selects the log-normal parameterization.
    """
    cfg = prior_config.get(value_key, None)
    family = default_family
    if isinstance(cfg, dict):
        family = str(cfg.get("family", cfg.get("dist", family))).lower()
    if log_key in prior_config:
        family = "lognormal"

    if family in {"exponential", "exp"}:
        return _sample_prior(prior_config, value_key, dist.Exponential(1.0 / max(default_value, 1.0e-30)))
    if family in {"lognormal", "log-normal", "log_normal", "normal_log"}:
        if isinstance(cfg, dict):
            return _sample_prior(
                prior_config,
                value_key,
                dist.LogNormal(np.log(max(default_value, 1.0e-30)), default_log_scale),
            )
        return _sample_log_positive_from_distribution(
            prior_config,
            value_key=value_key,
            log_key=log_key,
            default_distribution=dist.Normal(np.log(max(default_value, 1.0e-30)), default_log_scale),
        )
    raise ValueError(
        f"prior_config[{value_key!r}] family must be one of: "
        "'exponential', 'lognormal'."
    )

def _sample_optional_normal(prior_config: dict[str, Any], key: str, default: float, scale: float):
    """Return a fixed default unless a Normal-like prior is explicitly configured."""
    if key not in prior_config:
        return jnp.asarray(default, dtype=jnp.float64)
    return _sample_prior(prior_config, key, dist.Normal(default, scale))


def _sample_optional_truncnorm(prior_config: dict[str, Any], key: str, default: float, scale: float, low: float, high: float):
    """Return a fixed default unless a truncated Normal-like prior is explicitly configured."""
    if key not in prior_config:
        return jnp.asarray(default, dtype=jnp.float64)
    return _sample_prior(prior_config, key, dist.TruncatedNormal(default, scale, low=low, high=high))


def _safe_log10(x):
    """Take log10 after clipping to a tiny positive floor."""
    return jnp.log10(jnp.clip(jnp.asarray(x, dtype=jnp.float64), 1.0e-30, 1.0e300))


def _sample_log_stellar_mass(prior_config: dict[str, Any]):
    """Sample stellar mass with a less top-heavy default prior.

    By default this uses a heavy-tailed Student-t prior centered lower than the
    original Normal(10.5, 2.5) benchmark default. Existing Normal-like
    overrides with only ``loc`` and ``scale`` are still supported.
    """
    return _sample_prior(prior_config, "log_stellar_mass", dist.StudentT(df=5.0, loc=10.0, scale=2.0))


def _ssp_lgmet_solar_offset(ssp_lgmet):
    """Return the solar-metallicity offset for the SSP metallicity convention."""
    ssp_lgmet = jnp.asarray(ssp_lgmet, dtype=jnp.float64)
    # DSPS SSP grids use absolute log10(Z). Older/simple test grids may use
    # log10(Z/Zsun). A max below -1 is a robust signature of absolute log10(Z).
    return jnp.where(jnp.nanmax(ssp_lgmet) < -1.0, jnp.log10(DSPS_SOLAR_METALLICITY), 0.0)


def _gal_lgmet_to_absolute_z(gal_lgmet, ssp_lgmet):
    """Convert galaxy metallicity from the SSP-grid convention to absolute Z."""
    gal_lgmet = jnp.asarray(gal_lgmet, dtype=jnp.float64)
    ssp_lgmet = jnp.asarray(ssp_lgmet, dtype=jnp.float64)
    absolute_logz = jnp.where(
        jnp.nanmax(ssp_lgmet) < -1.0,
        gal_lgmet,
        gal_lgmet + jnp.log10(DSPS_SOLAR_METALLICITY),
    )
    return jnp.power(10.0, absolute_logz)


def _default_gal_lgmet_loc(ssp_lgmet):
    """Default galaxy metallicity center in the SSP grid's metallicity convention."""
    ssp_lgmet = jnp.asarray(ssp_lgmet, dtype=jnp.float64)
    loc = _ssp_lgmet_solar_offset(ssp_lgmet) - 0.3
    return jnp.clip(loc, jnp.nanmin(ssp_lgmet), jnp.nanmax(ssp_lgmet))


def _cfg_lgmet_value(
    cfg: dict[str, Any],
    solar_relative_key: str,
    default_logzsol: float,
    solar_offset,
    *,
    absolute_key: str | None = None,
):
    """Read metallicity config values and convert log(Z/Zsun) defaults to log10(Z)."""
    if absolute_key is not None and absolute_key in cfg:
        return jnp.asarray(cfg[absolute_key], dtype=jnp.float64)
    return jnp.asarray(cfg.get(solar_relative_key, default_logzsol), dtype=jnp.float64) + solar_offset


def _cumulative_trapezoid(y, x):
    """Return cumulative trapezoidal integral with an initial zero element."""
    dx = jnp.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.zeros((1,), dtype=jnp.result_type(y, x)), jnp.cumsum(area)])


def _mass_metallicity_relation_logprior(
    log_stellar_mass,
    gal_lgmet,
    prior_config: dict[str, Any],
    *,
    ssp_lgmet=None,
    redshift: float = 0.0,
):
    """Return an optional soft stellar mass-metallicity log-prior.

    The ``prior_config["mass_metallicity_relation"]`` mapping defines a broad,
    heuristic Gaussian prior on the host metallicity sampled by the stellar
    population model. By default the metallicity keys are solar-relative
    ``log10(Z/Zsun)`` values and are converted into the active SSP grid
    convention before evaluating the prior. For example, ``pivot_logzsol=-0.15``
    means 0.15 dex below solar regardless of whether the SSP grid stores
    absolute ``log10(Z)`` or relative ``log10(Z/Zsun)`` metallicities.

    Supported keys are:

    - ``enabled``: set ``False`` to disable the prior.
    - ``pivot_mass``: stellar-mass pivot in ``log10(M*/Msun)``.
    - ``pivot_logzsol``: solar-relative metallicity at ``pivot_mass``.
    - ``pivot_lgmet``: absolute value in the SSP grid convention, overriding
      ``pivot_logzsol``.
    - ``slope``: metallicity slope per dex in stellar mass.
    - ``scale``: Gaussian prior width in dex.
    - ``redshift_ref`` and ``redshift_slope``: optional linear redshift trend.
    - ``min`` and ``max``: solar-relative lower and upper bounds for the prior
      location, clipped to the SSP grid range.
    - ``min_lgmet`` and ``max_lgmet``: bounds in the SSP grid convention,
      overriding ``min`` and ``max``.
    """
    cfg = prior_config.get("mass_metallicity_relation", None)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict) or cfg.get("enabled", True) is False:
        return jnp.asarray(0.0, dtype=jnp.float64)

    solar_offset = _ssp_lgmet_solar_offset(ssp_lgmet) if ssp_lgmet is not None else jnp.asarray(0.0, dtype=jnp.float64)
    pivot_mass = jnp.asarray(cfg.get("pivot_mass", 10.0), dtype=jnp.float64)
    pivot_lgmet = _cfg_lgmet_value(cfg, "pivot_logzsol", -0.15, solar_offset, absolute_key="pivot_lgmet")
    slope = jnp.asarray(cfg.get("slope", 0.35), dtype=jnp.float64)
    scale = jnp.maximum(jnp.asarray(cfg.get("scale", 0.25), dtype=jnp.float64), 1.0e-6)
    redshift_ref = jnp.asarray(cfg.get("redshift_ref", 0.0), dtype=jnp.float64)
    redshift_slope = jnp.asarray(cfg.get("redshift_slope", -0.15), dtype=jnp.float64)
    min_loc = _cfg_lgmet_value(cfg, "min", -1.5, solar_offset, absolute_key="min_lgmet")
    max_loc = _cfg_lgmet_value(cfg, "max", 0.3, solar_offset, absolute_key="max_lgmet")
    if ssp_lgmet is not None:
        ssp_lgmet = jnp.asarray(ssp_lgmet, dtype=jnp.float64)
        min_loc = jnp.maximum(min_loc, jnp.nanmin(ssp_lgmet))
        max_loc = jnp.minimum(max_loc, jnp.nanmax(ssp_lgmet))

    loc = pivot_lgmet + slope * (jnp.asarray(log_stellar_mass, dtype=jnp.float64) - pivot_mass)
    loc = loc + redshift_slope * (jnp.asarray(redshift, dtype=jnp.float64) - redshift_ref)
    loc = jnp.clip(loc, jnp.minimum(min_loc, max_loc), jnp.maximum(min_loc, max_loc))
    return dist.Normal(loc, scale).log_prob(jnp.asarray(gal_lgmet, dtype=jnp.float64))


def _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=256):
    """Build a normalized 1D Gaussian convolution kernel."""
    sigma_pix = jnp.maximum(sigma_pix, 1e-3)
    x = jnp.arange(-max_half, max_half + 1, dtype=jnp.float64)
    half_dyn = jnp.maximum(3.0, jnp.ceil(radius_mult * sigma_pix))
    mask = jnp.abs(x) <= half_dyn
    k = jnp.exp(-0.5 * (x / sigma_pix) ** 2)
    k = jnp.where(mask, k, 0.0)
    return k / jnp.maximum(jnp.sum(k), 1e-30)


def _convolve_same_length(signal, kernel):
    """Convolve a signal and return an output with the original length."""
    full = jnp.convolve(signal, kernel, mode="same")
    n = signal.shape[0]
    m = full.shape[0]
    start = jnp.maximum((m - n) // 2, 0)
    return jax.lax.dynamic_slice(full, (start,), (n,))


def _shift_and_broaden_single_spectrum_lnlam(lnwave, spectrum, v_kms, sigma_kms):
    """Apply a velocity shift and Gaussian broadening in log-wavelength space."""
    dln = jnp.mean(jnp.diff(lnwave))
    sigma_ln = jnp.maximum(sigma_kms / C_KMS, 1e-5)
    sigma_pix = sigma_ln / jnp.maximum(dln, 1e-8)
    kern = _gaussian_kernel1d(sigma_pix, radius_mult=5.0, max_half=128)
    wave = jnp.exp(lnwave)
    shift_ln = v_kms / C_KMS
    shifted_wave = jnp.exp(lnwave - shift_ln)
    shifted = jnp.interp(shifted_wave, wave, spectrum, left=0.0, right=0.0)
    return _convolve_same_length(shifted, kern)


def _powerlaw_jax(wave, norm, lam1, lam2, x0, xbrk, bend_width, cutoff):
    """Evaluate the bent AGN disk power-law continuum."""
    expo = 1.0 / jnp.maximum(bend_width, 1e-6)
    lamaddexpo = (lam1 + lam2 + 2.0) / 2.0
    lamsubexpo = (lam2 - lam1) / 2.0 * jnp.maximum(bend_width, 1e-6)
    xpivratio = x0 / xbrk
    divisor = 1.0 / (xpivratio**expo + xpivratio**-expo)
    xratio = wave / xbrk
    bbb = norm * (wave / x0) ** lamaddexpo * ((xratio**expo + xratio**-expo) * divisor) ** lamsubexpo * (x0 / wave)
    cutoff_factor = -jnp.expm1(-(jnp.maximum(cutoff, 0.0) / wave))
    return jnp.where(cutoff > 0.0, bbb * cutoff_factor, bbb)


def _torus_component(wave, fcov, si, cool_lam, cool_width, hot_lam, hot_width, hot_fcov, si_ratio, si_em_lam, si_abs_lam, si_em_width, si_abs_width, l_agn):
    """Evaluate the empirical torus component on the rest-frame wavelength grid.

    The torus luminosity follows the GRAHSP-style empirical normalization from
    the AGN luminosity and covering factor proxy. It is not computed from the
    luminosity absorbed by the AGN attenuation curve.
    """
    log_wave_um = jnp.log10(wave / 10000.0)
    log_cool = jnp.log10(cool_lam)
    log_hot = jnp.log10(hot_lam)
    cool = jnp.exp(-((log_wave_um - log_cool) / cool_width) ** 2)
    hot = hot_fcov * 10 ** (log_cool - log_hot) * jnp.exp(-((log_wave_um - log_hot) / hot_width) ** 2)
    total = cool + hot
    norm_index = jnp.argmin(jnp.abs(wave - GRAHSP_TORUS_NORM_A))
    l_torus = 2.5 * l_agn * fcov
    torus = l_torus / GRAHSP_TORUS_NORM_A * total / jnp.maximum(total[norm_index], 1e-30)
    si_em_ampl = 0.4
    si_abs_ampl = si_em_ampl * si_ratio
    si_spec = l_torus / GRAHSP_TORUS_NORM_A * si * (
        si_em_ampl * jnp.exp(-0.5 * ((wave - si_em_lam) / si_em_width) ** 2)
        - si_abs_ampl * jnp.exp(-0.5 * ((wave - si_abs_lam) / si_abs_width) ** 2)
    )
    si_spec = jnp.maximum(si_spec, -torus)
    return torus + si_spec


def _feii_component(wave, template_flux_on_wave, norm, fwhm_kms, shift_frac):
    """Broaden, shift, and normalize the Fe II template contribution."""
    sigma_kms = jnp.maximum(fwhm_kms / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0))), 10.0)
    return norm * _shift_and_broaden_single_spectrum_lnlam(jnp.log(wave), jnp.maximum(template_flux_on_wave, 0.0), C_KMS * shift_frac, sigma_kms)


def _line_gaussians(wave, line_wave, line_lumin, width_kms):
    """Evaluate a summed Gaussian emission-line template with one shared width."""
    fwhm_to_sigma_conversion = 1 / (2 * jnp.sqrt(2 * jnp.log(2)))
    width_wave = line_wave * (width_kms * 1000.0) / 299792458.0
    sigma = width_wave * fwhm_to_sigma_conversion
    z = (wave[:, None] - line_wave[None, :]) / jnp.maximum(sigma[None, :], 1e-12)
    norm = 5100.0 / jnp.sqrt(jnp.pi * sigma**2)
    return jnp.sum(line_lumin[None, :] * jnp.exp(-0.5 * z * z) * norm[None, :], axis=1)


def _flux_conserving_line_gaussians(wave, line_wave, line_lumin, width_kms):
    """Evaluate CIGALE-style nebular lines preserving integrated luminosity."""
    fwhm_wave = jnp.maximum(line_wave * width_kms / C_KMS, 1.0e-8)
    sigma = fwhm_wave / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    z = (wave[:, None] - line_wave[None, :]) / jnp.maximum(sigma[None, :], 1.0e-12)
    profile = jnp.exp(-0.5 * z * z) / jnp.maximum(sigma[None, :] * jnp.sqrt(2.0 * jnp.pi), 1.0e-30)
    return jnp.sum(line_lumin[None, :] * profile, axis=1)


def _nearest_index(grid, value):
    """Return the index of the nearest tabulated grid value."""
    return jnp.argmin(jnp.abs(grid - jnp.asarray(value, dtype=jnp.float64)))


def _cigale_nebular_correction(f_esc, f_dust):
    """CIGALE nebular escape/dust correction factor."""
    alpha_b = jnp.asarray(2.58e-19, dtype=jnp.float64)
    alpha_1 = jnp.asarray(1.54e-19, dtype=jnp.float64)
    escaped_or_dust = f_esc + f_dust
    return jnp.clip((1.0 - escaped_or_dust) / (1.0 + alpha_1 / alpha_b * escaped_or_dust), 0.0, 1.0)


def _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel):
    """Evaluate the broadened Balmer continuum template."""
    lam_be = 3646.0
    h_c_per_k_B = 1.4388e8
    bb = (wave**-5) / jnp.expm1(jnp.clip(h_c_per_k_B / (balmer_te * wave), 1e-9, 700.0))
    bb0 = (lam_be**-5) / jnp.expm1(jnp.clip(h_c_per_k_B / (balmer_te * lam_be), 1e-9, 700.0))
    tau = balmer_tau * (wave / lam_be) ** 3
    bc = balmer_norm * (1.0 - jnp.exp(-tau)) * bb / jnp.maximum(bb0, 1e-30)
    bc = jnp.where(wave <= lam_be, bc, 0.0)
    return _shift_and_broaden_single_spectrum_lnlam(jnp.log(wave), bc, 0.0, balmer_vel)


def _attenuation_curve(wave_rest, opt_index, nir_index, norm, lam_break):
    """Return the broken power-law attenuation curve in magnitudes."""
    return norm * (wave_rest / lam_break) ** jnp.where(wave_rest < lam_break, opt_index, nir_index)


def _apply_biattenuation(wave_rest, gal_spec, agn_spec, ebv_gal, ebv_agn, opt_index, nir_index, norm, lam_break):
    """Apply differential attenuation to host and AGN components.

    The returned ``dust_luminosity`` is the host-galaxy luminosity absorbed by
    the galaxy attenuation curve. AGN light is attenuated separately, but its
    absorbed luminosity is not added to the host dust energy-balance budget.
    """
    curve = _attenuation_curve(wave_rest, opt_index, nir_index, norm, lam_break)
    gal_att = gal_spec * 10 ** (ebv_gal * curve / -2.5)
    agn_att = agn_spec * 10 ** ((ebv_gal + ebv_agn) * curve / -2.5)
    host_absorbed = jnp.clip(gal_spec - gal_att, 0.0, None)
    dust_luminosity = jnp.clip(jnp.trapezoid(host_absorbed, wave_rest), 0.0, None)
    return gal_att, agn_att, host_absorbed, dust_luminosity


def _attenuation_transmitted_fraction(direct_attenuated, direct_intrinsic):
    """Return the attenuation-only transmitted fraction for direct components.

    This excludes re-emitted host dust and empirical torus emission so the
    attenuation model uncertainty is controlled only by components that pass
    through the attenuation curve.
    """
    return jnp.clip(
        direct_attenuated / jnp.maximum(direct_intrinsic, 1.0e-30),
        1.0e-4,
        1.0,
    )


def _redshift_to_obs(rest_wave, rest_lum, obs_wave, redshift, luminosity_distance_m):
    """Project a rest-frame luminosity density to the observed frame."""
    wave_obs = rest_wave * (1.0 + redshift)
    flux_obs = rest_lum / (4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1e-12) ** 2 * jnp.maximum(1.0 + redshift, 1e-8))
    return jnp.interp(obs_wave, wave_obs, flux_obs, left=0.0, right=0.0)


def _redshift_scalar_to_obs(rest_wave, rest_value, obs_wave, redshift):
    """Interpolate a scalar rest-frame quantity onto the observed wavelength grid."""
    wave_obs = rest_wave * (1.0 + redshift)
    return jnp.interp(obs_wave, wave_obs, rest_value, left=0.0, right=0.0)


def _project_filters(obs_flux, packed_filters):
    """Project an observed-frame spectrum through prepared filters.

    The final f_lambda-to-mJy conversion intentionally follows the
    GRAHSP/pcigale convention of using each filter's effective wavelength
    squared. This is not an exact pivot-wavelength conversion, so very broad
    filters can retain small convention-level systematics.
    """
    interp_indices = packed_filters.interp_indices
    interp_weight = packed_filters.interp_weight
    transmission = packed_filters.transmission
    work_wave = packed_filters.work_wave
    effective_wavelength = packed_filters.effective_wavelength
    valid_mask = packed_filters.valid_mask

    left = obs_flux[interp_indices]
    right = obs_flux[interp_indices + 1]
    values = left * (1.0 - interp_weight) + right * interp_weight
    values = jnp.where(valid_mask, values, 0.0)
    weighted_trans = jnp.where(valid_mask, transmission, 0.0)
    weighted_wave = work_wave
    numer = jnp.trapezoid(values * weighted_trans, weighted_wave, axis=1)
    denom = jnp.maximum(jnp.trapezoid(weighted_trans, weighted_wave, axis=1), 1e-30)
    f_lambda = numer / denom
    return 1e-10 / 299792458.0 * 1e29 * effective_wavelength * effective_wavelength * f_lambda


def _can_use_fixed_filter_projection(context: ModelContext, cfg) -> bool:
    """Return whether cached fixed-z photometric projection matrices are valid."""
    return bool(
        cfg.likelihood.use_fast_photometry_projection
        and not cfg.observation.fits_redshift
        and context.fixed_filter_projection_jax is not None
        and context.fixed_scalar_filter_projection_jax is not None
    )


def _project_rest_luminosity_filters(context: ModelContext, rest_lum):
    """Project fixed-redshift rest luminosity density directly into filter mJy."""
    return context.fixed_filter_projection_jax @ rest_lum


def _project_rest_scalar_filters(context: ModelContext, rest_value):
    """Project fixed-redshift scalar rest quantity through filters like legacy code."""
    return context.fixed_scalar_filter_projection_jax @ rest_value


def _interp_redshift_projection_matrix(redshift, grid, matrices):
    """Linearly interpolate a redshift-tabulated projection matrix."""
    z = jnp.asarray(redshift, dtype=jnp.float64)
    idx_hi = jnp.searchsorted(grid, z, side="right")
    idx_hi = jnp.clip(idx_hi, 1, grid.shape[0] - 1)
    idx_lo = idx_hi - 1
    z_lo = grid[idx_lo]
    z_hi = grid[idx_hi]
    weight = jnp.clip((z - z_lo) / jnp.maximum(z_hi - z_lo, 1.0e-30), 0.0, 1.0)
    return matrices[idx_lo] * (1.0 - weight) + matrices[idx_hi] * weight


def _can_use_redshift_filter_projection(context: ModelContext, cfg) -> bool:
    """Return whether cached variable-redshift photometric matrices are valid."""
    return bool(
        cfg.likelihood.use_redshift_projection_cache
        and cfg.observation.fits_redshift
        and context.redshift_projection_cache_jax is not None
    )


def _project_redshift_luminosity_filters(context: ModelContext, rest_lum, redshift):
    """Project rest luminosity density through redshift-interpolated matrices."""
    cache = context.redshift_projection_cache_jax
    matrix = _interp_redshift_projection_matrix(redshift, cache.redshift_grid, cache.filter_projection)
    return matrix @ rest_lum


def _project_redshift_scalar_filters(context: ModelContext, rest_value, redshift):
    """Project rest-frame scalar values through redshift-interpolated matrices."""
    cache = context.redshift_projection_cache_jax
    matrix = _interp_redshift_projection_matrix(redshift, cache.redshift_grid, cache.scalar_projection)
    return matrix @ rest_value


def _local_line_gaussian_grid(line_wave, line_lumin, width_kms, *, n_local: int = 9):
    """Evaluate CIGALE-style line profiles on per-line local grids."""
    line_wave = jnp.asarray(line_wave, dtype=jnp.float64)
    line_lumin = jnp.asarray(line_lumin, dtype=jnp.float64)
    offsets = jnp.linspace(-3.0, 3.0, int(n_local), dtype=jnp.float64)
    fwhm_wave = jnp.maximum(line_wave * (width_kms * 1000.0) / 299792458.0, 1.0e-8)
    sigma = fwhm_wave / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    wave = line_wave[:, None] + offsets[None, :] * fwhm_wave[:, None]
    z = (wave - line_wave[:, None]) / jnp.maximum(sigma[:, None], 1.0e-12)
    norm = 5100.0 / jnp.sqrt(jnp.pi * sigma * sigma)
    lumin = line_lumin[:, None] * jnp.exp(-0.5 * z * z) * norm[:, None]
    return wave, lumin


def _local_flux_conserving_line_grid(line_wave, line_lumin, width_kms, *, n_local: int = 9):
    """Evaluate integrated-luminosity line profiles on per-line local grids."""
    line_wave = jnp.asarray(line_wave, dtype=jnp.float64)
    line_lumin = jnp.asarray(line_lumin, dtype=jnp.float64)
    offsets = jnp.linspace(-3.0, 3.0, int(n_local), dtype=jnp.float64)
    fwhm_wave = jnp.maximum(line_wave * jnp.asarray(width_kms, dtype=jnp.float64) / C_KMS, 1.0e-8)
    sigma = fwhm_wave / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    wave = jnp.maximum(line_wave[:, None] + offsets[None, :] * fwhm_wave[:, None], 1.0e-6)
    z = (wave - line_wave[:, None]) / jnp.maximum(sigma[:, None], 1.0e-12)
    norm = 1.0 / jnp.maximum(sigma * jnp.sqrt(2.0 * jnp.pi), 1.0e-30)
    lumin = line_lumin[:, None] * jnp.exp(-0.5 * z * z) * norm[:, None]
    return wave, lumin


def _project_local_line_filters(
    context: ModelContext,
    line_wave,
    line_lumin,
    width_kms,
    ebv_total,
    redshift,
    luminosity_distance_m,
    igm,
):
    """Project Gaussian line emission through filters using local line grids."""
    line_lumin = jnp.asarray(line_lumin, dtype=jnp.float64)
    rest_line_wave, rest_lumin = _local_line_gaussian_grid(line_wave, line_lumin, width_kms)
    rest_line_wave = jnp.maximum(rest_line_wave, 1.0e-6)
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    luminosity_distance_m = jnp.asarray(luminosity_distance_m, dtype=jnp.float64)
    distance_scale = 4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1.0e-12) ** 2 * jnp.maximum(1.0 + redshift, 1.0e-8)
    obs_line_wave = rest_line_wave * (1.0 + redshift)
    igm = jnp.interp(rest_line_wave, context.rest_wave_jax, igm, left=0.0, right=0.0)
    attenuation_curve = _attenuation_curve(rest_line_wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    attenuation_factor = 10 ** (jnp.asarray(ebv_total, dtype=jnp.float64) * attenuation_curve / -2.5)
    flux_lambda = rest_lumin * attenuation_factor * igm / distance_scale
    curves = context.packed_filter_curves_jax

    def _one_filter(filt_wave, filt_trans, denom, eff_wave):
        """Project the local AGN line grid through one packed filter curve."""
        trans = jnp.interp(obs_line_wave, filt_wave, filt_trans, left=0.0, right=0.0)
        numer = jnp.sum(jnp.trapezoid(trans * flux_lambda, obs_line_wave, axis=1))
        f_lambda = numer / jnp.maximum(denom, 1.0e-30)
        return 1.0e-10 / 299792458.0 * 1.0e29 * eff_wave * eff_wave * f_lambda

    return jax.vmap(_one_filter)(
        curves.wave,
        curves.transmission,
        curves.denom,
        context.filter_effective_wavelength_jax,
    )


def _project_local_nebular_line_filters(
    context: ModelContext,
    line_wave,
    line_lumin,
    width_kms,
    ebv_total,
    redshift,
    luminosity_distance_m,
    igm,
):
    """Project flux-conserving nebular line emission through filters using local line grids."""
    rest_line_wave, rest_lumin = _local_flux_conserving_line_grid(line_wave, line_lumin, width_kms)
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    luminosity_distance_m = jnp.asarray(luminosity_distance_m, dtype=jnp.float64)
    distance_scale = 4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1.0e-12) ** 2 * jnp.maximum(1.0 + redshift, 1.0e-8)
    obs_line_wave = rest_line_wave * (1.0 + redshift)
    igm_local = jnp.interp(rest_line_wave, context.rest_wave_jax, igm, left=0.0, right=0.0)
    attenuation_curve = _attenuation_curve(rest_line_wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    attenuation_factor = 10 ** (jnp.asarray(ebv_total, dtype=jnp.float64) * attenuation_curve / -2.5)
    flux_lambda = rest_lumin * attenuation_factor * igm_local / distance_scale
    curves = context.packed_filter_curves_jax

    def _one_filter(filt_wave, filt_trans, denom, eff_wave):
        """Project the local nebular line grid through one packed filter curve."""
        trans = jnp.interp(obs_line_wave, filt_wave, filt_trans, left=0.0, right=0.0)
        numer = jnp.sum(jnp.trapezoid(trans * flux_lambda, obs_line_wave, axis=1))
        f_lambda = numer / jnp.maximum(denom, 1.0e-30)
        return 1.0e-10 / 299792458.0 * 1.0e29 * eff_wave * eff_wave * f_lambda

    return jax.vmap(_one_filter)(
        curves.wave,
        curves.transmission,
        curves.denom,
        context.filter_effective_wavelength_jax,
    )


def _local_nebular_line_obs_sed(context: ModelContext, line_wave, line_lumin, width_kms, ebv_total, redshift, luminosity_distance_m, igm):
    """Return observed-frame local-grid wavelengths and F_lambda for nebular lines."""
    rest_line_wave, rest_lumin = _local_flux_conserving_line_grid(line_wave, line_lumin, width_kms)
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    luminosity_distance_m = jnp.asarray(luminosity_distance_m, dtype=jnp.float64)
    distance_scale = 4.0 * jnp.pi * jnp.maximum(luminosity_distance_m, 1.0e-12) ** 2 * jnp.maximum(1.0 + redshift, 1.0e-8)
    obs_line_wave = rest_line_wave * (1.0 + redshift)
    igm_local = jnp.interp(rest_line_wave, context.rest_wave_jax, igm, left=0.0, right=0.0)
    attenuation_curve = _attenuation_curve(rest_line_wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    attenuation_factor = 10 ** (jnp.asarray(ebv_total, dtype=jnp.float64) * attenuation_curve / -2.5)
    flux_lambda = rest_lumin * attenuation_factor * igm_local / distance_scale
    separator = jnp.full((obs_line_wave.shape[0], 1), jnp.nan, dtype=jnp.float64)
    obs_plot = jnp.concatenate([obs_line_wave, separator], axis=1)
    flux_plot = jnp.concatenate([flux_lambda, separator], axis=1)
    return jnp.ravel(obs_plot), jnp.ravel(flux_plot)


def _interp_fixed_local_line_terms(width_kms, cache):
    """Interpolate cached fixed-z local line projection terms in log width."""
    log_width = jnp.log(jnp.maximum(jnp.asarray(width_kms, dtype=jnp.float64), 1.0e-12))
    grid = cache.log_width_grid
    idx_hi = jnp.searchsorted(grid, log_width, side="right")
    idx_hi = jnp.clip(idx_hi, 1, grid.shape[0] - 1)
    idx_lo = idx_hi - 1
    w = jnp.clip((log_width - grid[idx_lo]) / jnp.maximum(grid[idx_hi] - grid[idx_lo], 1.0e-30), 0.0, 1.0)
    profile_norm = cache.profile_norm[idx_lo] * (1.0 - w) + cache.profile_norm[idx_hi] * w
    attenuation_curve = cache.attenuation_curve[idx_lo] * (1.0 - w) + cache.attenuation_curve[idx_hi] * w
    projection_weight = cache.projection_weight[idx_lo] * (1.0 - w) + cache.projection_weight[idx_hi] * w
    return profile_norm, attenuation_curve, projection_weight


def _project_fixed_cached_local_line_filters(context: ModelContext, line_lumin, width_kms, ebv_total):
    """Project local AGN lines with fixed-z cached filter overlap terms."""
    cache = context.fixed_local_line_projection_cache_jax
    profile_norm, attenuation_curve, projection_weight = _interp_fixed_local_line_terms(width_kms, cache)
    attenuation_factor = 10 ** (jnp.asarray(ebv_total, dtype=jnp.float64) * attenuation_curve / -2.5)
    line_flux_density = jnp.asarray(line_lumin, dtype=jnp.float64)[:, None] * profile_norm * attenuation_factor
    return jnp.sum(projection_weight * line_flux_density[None, :, :], axis=(1, 2))


def _interp_rest_sed(target_wave, source_wave, source_sed):
    """Interpolate one rest-frame SED onto a new wavelength grid."""
    return jnp.interp(target_wave, source_wave, source_sed, left=0.0, right=0.0)


def _interp_dale_template(alpha, alpha_grid, dust_lumin_grid):
    """Interpolate the Dale 2014 host-dust grid in alpha."""
    alpha = jnp.clip(alpha, jnp.min(alpha_grid), jnp.max(alpha_grid))
    return jax.vmap(lambda row: jnp.interp(alpha, alpha_grid, row))(dust_lumin_grid.T)


def _host_dust_emission(context: ModelContext, dust_luminosity, dust_alpha):
    """Convert absorbed host luminosity into a Dale-template dust SED."""
    dust_template = _interp_dale_template(dust_alpha, context.dust_alpha_grid_jax, context.dust_lumin_rest_jax)
    dust_rest_native = jnp.clip(dust_luminosity, 0.0, None) * jnp.clip(dust_template, 0.0, None)
    return dust_rest_native


def _lnu_lsun_per_hz_to_llambda_w_per_a(wave_a, lnu_lsun_per_hz):
    """Convert DSPS `Lnu` in Lsun/Hz to `Llambda` in W/Angstrom."""
    wave_m = jnp.maximum(wave_a, 1e-12) * 1.0e-10
    lnu_w_per_hz = lnu_lsun_per_hz * L_SUN_W
    return lnu_w_per_hz * C_MS / (wave_m * wave_m) * 1.0e-10


def _build_diffstar_host(context: ModelContext, prior_config: dict[str, Any], *, full_output: bool = True):
    """Build the host-galaxy SED from Diffstar SFH and a precomputed SSP basis."""
    ssp_lgmet = context.host_basis_jax.ssp_lgmet
    ssp_lg_age_gyr = context.host_basis_jax.ssp_lg_age_gyr
    host_basis_rest = context.host_basis_jax.rest_llambda
    surviving_frac_by_age = context.host_basis_jax.surviving_frac_by_age
    gal_t_table = context.host_basis_jax.gal_t_table
    t_obs_gyr = jnp.asarray(context.t_obs_gyr, dtype=jnp.float64)

    log_stellar_mass = _sample_log_stellar_mass(prior_config)

    u_params = {}
    for key in DEFAULT_DIFFSTAR_U_PARAMS._fields:
        default_loc = float(np.asarray(getattr(DEFAULT_DIFFSTAR_U_PARAMS, key)))
        u_params[key] = _sample_prior(prior_config, key, dist.Normal(default_loc, 1.0))
    bounded = get_bounded_diffstar_params(DiffstarUParams(**u_params))
    base_history = calc_sfh_singlegal(
        bounded,
        DEFAULT_MAH_PARAMS,
        gal_t_table,
        lgt0=DIFFSTAR_LGT0,
        fb=DIFFSTAR_FB,
        return_smh=True,
    )

    gal_lgmet = _sample_prior(prior_config, "gal_lgmet", dist.Normal(_default_gal_lgmet_loc(ssp_lgmet), 0.5))
    gal_lgmet_scatter = _sample_positive_distribution(
        prior_config,
        value_key="gal_lgmet_scatter",
        log_key="log_gal_lgmet_scatter",
        default_value_distribution=dist.LogNormal(np.log(0.15), 0.8),
        default_log_distribution=dist.Normal(np.log(0.15), 0.8),
        default_to_log=True,
    )
    mmr_logprior = _mass_metallicity_relation_logprior(
        log_stellar_mass,
        gal_lgmet,
        prior_config,
        ssp_lgmet=ssp_lgmet,
        redshift=float(context.fit_config.observation.redshift),
    )
    numpyro.factor("mass_metallicity_relation_prior", mmr_logprior)
    host_weights_info = calc_ssp_weights_sfh_table_lognormal_mdf(
        gal_t_table,
        base_history.sfh,
        gal_lgmet,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs_gyr,
    )
    host_weights = host_weights_info.weights
    surviving_mass_fraction = jnp.clip(jnp.sum(host_weights_info.age_weights * surviving_frac_by_age), 1e-12, 1.0)
    target_surviving_mass = 10.0**log_stellar_mass
    target_formed_mass = target_surviving_mass / surviving_mass_fraction
    base_formed_mass = jnp.clip(base_history.smh[-1], 1e-30, 1.0e40)
    sfh_scale = target_formed_mass / base_formed_mass
    host_rest = target_formed_mass * jnp.tensordot(
        host_weights,
        host_basis_rest,
        axes=((0, 1), (0, 1)),
    )
    state = {
        "host_rest": host_rest,
        "log_stellar_mass": log_stellar_mass,
        "surviving_mass_fraction": surviving_mass_fraction,
        "formed_mass": target_formed_mass,
        "sfh_scale": sfh_scale,
        "gal_lgmet": gal_lgmet,
        "gal_lgmet_scatter": gal_lgmet_scatter,
        "mass_metallicity_relation_logprior": mmr_logprior,
        "host_age_weights": host_weights_info.age_weights,
        "host_lgmet_weights": host_weights_info.lgmet_weights,
        "host_ssp_weights": host_weights,
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
        "sfh_age_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_tau_gyr": jnp.asarray(0.0, dtype=jnp.float64),
    }
    if not full_output:
        return state
    scaled_sfh = base_history.sfh * sfh_scale
    scaled_smh = base_history.smh * sfh_scale
    state.update(
        {
            "host_age_weights": host_weights_info.age_weights,
            "host_lgmet_weights": host_weights_info.lgmet_weights,
            "host_ssp_weights": host_weights,
            "gal_sfr_table": scaled_sfh,
            "gal_smh_table": scaled_smh,
        }
    )
    return state


def _build_delayed_host(context: ModelContext, prior_config: dict[str, Any], *, full_output: bool = True):
    """Build the host-galaxy SED from a CIGALE-like delayed-tau SFH."""
    ssp_lgmet = context.host_basis_jax.ssp_lgmet
    ssp_lg_age_gyr = context.host_basis_jax.ssp_lg_age_gyr
    host_basis_rest = context.host_basis_jax.rest_llambda
    surviving_frac_by_age = context.host_basis_jax.surviving_frac_by_age
    gal_t_table = context.host_basis_jax.gal_t_table
    t_obs_gyr = jnp.asarray(context.t_obs_gyr, dtype=jnp.float64)
    cfg = context.fit_config.galaxy

    log_stellar_mass = _sample_log_stellar_mass(prior_config)
    min_age = jnp.asarray(max(float(cfg.sfh_t_min_gyr), 1.0e-3), dtype=jnp.float64)
    max_age = jnp.maximum(t_obs_gyr, min_age * 1.01)
    log_age_gyr = _sample_prior(
        prior_config,
        "log_sfh_age_gyr",
        dist.TruncatedNormal(
            np.log(min(3.0, max(float(context.t_obs_gyr), 1.0e-3))),
            1.0,
            low=jnp.log(min_age),
            high=jnp.log(max_age),
        ),
    )
    log_tau_over_age = _sample_prior(
        prior_config,
        "log_sfh_tau_over_age",
        dist.Normal(0.0, float(cfg.tau_host_prior_scale)),
    )
    age_gyr = jnp.exp(log_age_gyr)
    tau_gyr = jnp.clip(age_gyr * jnp.exp(log_tau_over_age), 0.03, 30.0)
    log_tau_gyr = numpyro.deterministic("log_sfh_tau_gyr", jnp.log(tau_gyr))
    stellar_age_gyr = jnp.maximum(t_obs_gyr - gal_t_table, 0.0)
    sfh_age_gyr = age_gyr - stellar_age_gyr
    base_sfh = jnp.where((sfh_age_gyr > 0.0) & (sfh_age_gyr <= age_gyr), sfh_age_gyr * jnp.exp(-sfh_age_gyr / tau_gyr), 0.0)
    base_smh = _cumulative_trapezoid(base_sfh, gal_t_table) * 1.0e9

    gal_lgmet = _sample_prior(prior_config, "gal_lgmet", dist.Normal(_default_gal_lgmet_loc(ssp_lgmet), 0.5))
    gal_lgmet_scatter = _sample_positive_distribution(
        prior_config,
        value_key="gal_lgmet_scatter",
        log_key="log_gal_lgmet_scatter",
        default_value_distribution=dist.LogNormal(np.log(0.15), 0.8),
        default_log_distribution=dist.Normal(np.log(0.15), 0.8),
        default_to_log=True,
    )
    mmr_logprior = _mass_metallicity_relation_logprior(
        log_stellar_mass,
        gal_lgmet,
        prior_config,
        ssp_lgmet=ssp_lgmet,
        redshift=float(context.fit_config.observation.redshift),
    )
    numpyro.factor("mass_metallicity_relation_prior", mmr_logprior)
    host_weights_info = calc_ssp_weights_sfh_table_lognormal_mdf(
        gal_t_table,
        base_sfh,
        gal_lgmet,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs_gyr,
    )
    host_weights = host_weights_info.weights
    surviving_mass_fraction = jnp.clip(jnp.sum(host_weights_info.age_weights * surviving_frac_by_age), 1e-12, 1.0)
    target_surviving_mass = 10.0**log_stellar_mass
    target_formed_mass = target_surviving_mass / surviving_mass_fraction
    base_formed_mass = jnp.clip(base_smh[-1], 1e-30, 1.0e40)
    sfh_scale = target_formed_mass / base_formed_mass
    host_rest = target_formed_mass * jnp.tensordot(
        host_weights,
        host_basis_rest,
        axes=((0, 1), (0, 1)),
    )
    state = {
        "host_rest": host_rest,
        "log_stellar_mass": log_stellar_mass,
        "surviving_mass_fraction": surviving_mass_fraction,
        "formed_mass": target_formed_mass,
        "sfh_scale": sfh_scale,
        "gal_lgmet": gal_lgmet,
        "gal_lgmet_scatter": gal_lgmet_scatter,
        "mass_metallicity_relation_logprior": mmr_logprior,
        "host_age_weights": host_weights_info.age_weights,
        "host_lgmet_weights": host_weights_info.lgmet_weights,
        "host_ssp_weights": host_weights,
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
        "sfh_age_gyr": age_gyr,
        "sfh_tau_gyr": tau_gyr,
    }
    if not full_output:
        return state
    state.update(
        {
            "host_age_weights": host_weights_info.age_weights,
            "host_lgmet_weights": host_weights_info.lgmet_weights,
            "host_ssp_weights": host_weights,
            "gal_sfr_table": base_sfh * sfh_scale,
            "gal_smh_table": base_smh * sfh_scale,
        }
    )
    return state


def _build_host_state(context: ModelContext, prior_config: dict[str, Any], *, full_output: bool = True):
    """Dispatch to the configured host SFH model."""
    model_name = str(context.fit_config.galaxy.host_sfh_model).lower()
    if model_name in {"delayed", "sfhdelayed", "delayed_tau", "delayed-tau"}:
        return _build_delayed_host(context, prior_config, full_output=full_output)
    if model_name in {"diffstar", "dsps_diffstar"}:
        return _build_diffstar_host(context, prior_config, full_output=full_output)
    raise ValueError("galaxy.host_sfh_model must be one of: 'delayed', 'diffstar'.")


def _host_rest_on_basis(host_state: dict[str, Any], host_basis_jax):
    """Evaluate the sampled SSP mixture on an alternate wavelength basis."""
    return host_state["formed_mass"] * jnp.tensordot(
        host_state["host_ssp_weights"],
        host_basis_jax.rest_llambda,
        axes=((0, 1), (0, 1)),
    )


def _empty_host_state(context: ModelContext):
    """Return zero-valued host placeholders for AGN-only fits."""
    rest_wave = _np_to_jnp(context.rest_wave)
    ssp_lgmet = _np_to_jnp(context.ssp_data.ssp_lgmet)
    ssp_lg_age_gyr = _np_to_jnp(context.ssp_data.ssp_lg_age_gyr)
    gal_t_table = _np_to_jnp(context.gal_t_table)
    zero_host_weights = jnp.zeros((ssp_lgmet.shape[0], ssp_lg_age_gyr.shape[0]), dtype=jnp.float64)
    return {
        "host_rest": jnp.zeros_like(rest_wave),
        "host_age_weights": jnp.zeros_like(ssp_lg_age_gyr),
        "host_lgmet_weights": jnp.zeros_like(ssp_lgmet),
        "host_ssp_weights": zero_host_weights,
        "surviving_mass_fraction": jnp.asarray(0.0, dtype=jnp.float64),
        "formed_mass": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_scale": jnp.asarray(0.0, dtype=jnp.float64),
        "gal_lgmet": jnp.asarray(0.0, dtype=jnp.float64),
        "gal_lgmet_scatter": jnp.asarray(0.0, dtype=jnp.float64),
        "mass_metallicity_relation_logprior": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_age_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "sfh_tau_gyr": jnp.asarray(0.0, dtype=jnp.float64),
        "gal_sfr_table": jnp.zeros_like(gal_t_table),
        "gal_smh_table": jnp.zeros_like(gal_t_table),
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
    }


def _build_nebular_components(context: ModelContext, host_state: dict[str, Any], host_rest, prior_config: dict[str, Any]):
    """Build CIGALE/GRAHSP-style host nebular absorption, lines, and continuum."""
    rest_wave = context.rest_wave_jax
    cfg = context.fit_config.nebular
    zeros = jnp.zeros_like(rest_wave)
    if not (context.fit_config.galaxy.fit_host and cfg.enabled):
        return {
            "absorption_rest": zeros,
            "lines_rest": zeros,
            "continuum_rest": zeros,
            "emission_rest": zeros,
            "dust_luminosity": jnp.asarray(0.0, dtype=jnp.float64),
            "n_ly_young": jnp.asarray(0.0, dtype=jnp.float64),
            "n_ly_old": jnp.asarray(0.0, dtype=jnp.float64),
            "ly_lum_young": jnp.asarray(0.0, dtype=jnp.float64),
            "ly_lum_old": jnp.asarray(0.0, dtype=jnp.float64),
            "logU": jnp.asarray(float(cfg.logU), dtype=jnp.float64),
            "zgas": jnp.asarray(float(cfg.zgas) if cfg.zgas is not None else 0.02, dtype=jnp.float64),
            "ne": jnp.asarray(float(cfg.ne), dtype=jnp.float64),
            "f_esc": jnp.asarray(float(cfg.f_esc), dtype=jnp.float64),
            "f_dust": jnp.asarray(float(cfg.f_dust), dtype=jnp.float64),
            "lines_width": jnp.asarray(float(cfg.lines_width), dtype=jnp.float64),
            "line_scale": jnp.asarray(1.0, dtype=jnp.float64),
            "corr": jnp.asarray(0.0, dtype=jnp.float64),
            "line_lumin": jnp.zeros((1,), dtype=jnp.float64),
        }

    logu = _sample_optional_normal(prior_config, "nebular_logU", float(cfg.logU), 0.3)
    default_zgas = float(cfg.zgas) if cfg.zgas is not None else None
    host_zgas = _gal_lgmet_to_absolute_z(host_state["gal_lgmet"], host_state["ssp_lgmet"])
    zgas_default = host_zgas if default_zgas is None else jnp.asarray(default_zgas, dtype=jnp.float64)
    zgas = _sample_optional_normal(prior_config, "nebular_zgas", float(default_zgas) if default_zgas is not None else 0.02, 0.01)
    zgas = jnp.where(default_zgas is None and "nebular_zgas" not in prior_config, zgas_default, jnp.clip(zgas, 1.0e-6, 1.0))
    ne = jnp.clip(_sample_optional_normal(prior_config, "nebular_ne", float(cfg.ne), 100.0), 1.0e-6, 1.0e8)
    f_esc = _sample_optional_truncnorm(prior_config, "nebular_f_esc", float(cfg.f_esc), 0.1, 0.0, 1.0)
    f_dust_raw = _sample_optional_truncnorm(prior_config, "nebular_f_dust", float(cfg.f_dust), 0.1, 0.0, 1.0)
    f_dust = jnp.minimum(f_dust_raw, jnp.maximum(1.0 - f_esc, 0.0))
    tie_width_to_jaxqsofit = bool(
        context.fit_config.spectroscopy_config.enabled
        and str(context.fit_config.spectroscopy_config.backend).lower() == "jaxqsofit"
        and context.fit_config.spectroscopy_config.jaxqsofit.use_spectral_lines
    )
    if tie_width_to_jaxqsofit and "nebular_lines_width" not in prior_config:
        lines_width = numpyro.sample(
            "nebular_lines_width",
            dist.TruncatedNormal(
                loc=jnp.asarray(float(cfg.lines_width), dtype=jnp.float64),
                scale=jnp.asarray(100.0, dtype=jnp.float64),
                low=jnp.asarray(1.0, dtype=jnp.float64),
                high=jnp.asarray(1.0e5, dtype=jnp.float64),
            ),
        )
    else:
        lines_width = jnp.clip(_sample_optional_normal(prior_config, "nebular_lines_width", float(cfg.lines_width), 100.0), 0.0, 1.0e5)
    if tie_width_to_jaxqsofit or "log_nebular_line_scale" in prior_config:
        line_scale = _sample_log_positive_from_distribution(
            prior_config,
            value_key="nebular_line_scale",
            log_key="log_nebular_line_scale",
            default_distribution=dist.Normal(0.0, 0.5),
        )
    else:
        line_scale = jnp.asarray(1.0, dtype=jnp.float64)

    weights = host_state["formed_mass"] * host_state["host_ssp_weights"]
    young_mask = (jnp.power(10.0, context.host_basis_jax.ssp_lg_age_gyr) * 1.0e3) <= float(cfg.young_age_cut_myr)
    young_mask_2d = young_mask[None, :]
    old_mask_2d = ~young_mask_2d
    n_basis = context.host_basis_jax.n_ly_per_msun
    l_basis = context.host_basis_jax.ly_lum_per_msun
    n_ly_young = jnp.sum(weights * n_basis * young_mask_2d)
    n_ly_old = jnp.sum(weights * n_basis * old_mask_2d)
    ly_lum_young = jnp.sum(weights * l_basis * young_mask_2d)
    ly_lum_old = jnp.sum(weights * l_basis * old_mask_2d)
    n_ly_total = jnp.clip(n_ly_young + n_ly_old, 0.0, 1.0e300)
    ly_lum_total = jnp.clip(ly_lum_young + ly_lum_old, 0.0, 1.0e300)

    templates = context.nebular_templates_jax
    z_idx = _nearest_index(templates.z_grid, zgas)
    u_idx = _nearest_index(templates.logu_grid, logu)
    ne_idx = _nearest_index(templates.ne_grid, ne)
    corr = _cigale_nebular_correction(f_esc, f_dust)
    rest_templates = context.nebular_rest_templates_jax
    continuum_rest = rest_templates.continuum_lumin_per_a_per_photon[z_idx, u_idx, ne_idx] * n_ly_total * corr
    line_lumin = templates.line_lumin_per_photon[z_idx, u_idx, ne_idx] * n_ly_total * corr
    if context.fixed_nebular_line_profile_jax is not None:
        lines_rest = context.fixed_nebular_line_profile_jax * n_ly_total * corr
    elif rest_templates.line_profile_per_photon is not None:
        lines_rest = rest_templates.line_profile_per_photon[z_idx, u_idx, ne_idx] * n_ly_total * corr
    else:
        lines_rest = _flux_conserving_line_gaussians(rest_wave, templates.line_wave_a, line_lumin, lines_width)
    emission_scale = jnp.asarray(1.0 if cfg.emission else 0.0, dtype=jnp.float64)
    line_lumin = line_lumin * emission_scale * line_scale
    lines_rest = lines_rest * emission_scale * line_scale
    continuum_rest = continuum_rest * emission_scale
    absorption_rest = jnp.where(rest_wave < 912.0, -host_rest * (1.0 - f_esc), 0.0)
    return {
        "absorption_rest": absorption_rest,
        "lines_rest": lines_rest,
        "continuum_rest": continuum_rest,
        "emission_rest": lines_rest + continuum_rest,
        "dust_luminosity": ly_lum_total * f_dust,
        "n_ly_young": n_ly_young,
        "n_ly_old": n_ly_old,
        "ly_lum_young": ly_lum_young,
        "ly_lum_old": ly_lum_old,
        "logU": logu,
        "zgas": zgas,
        "ne": ne,
        "f_esc": f_esc,
        "f_dust": f_dust,
        "lines_width": lines_width,
        "line_scale": line_scale,
        "corr": corr,
        "line_lumin": line_lumin,
    }


def _estimate_log_agn_amp_prior_loc(context: ModelContext, redshift: float) -> float:
    """Estimate a rough log(lambda L_lambda, 5100 A) prior center from the photometry."""
    obs_fluxes_mjy = np.asarray(context.fluxes, dtype=float)
    mask = np.asarray(context.positive_detected_mask, dtype=bool) & np.isfinite(obs_fluxes_mjy) & (obs_fluxes_mjy > 0.0)
    if not np.any(mask):
        return float(np.log(1.0e36))
    filter_wavelength = np.array([f.effective_wavelength for f in context.filters], dtype=float)
    target_obs_wave = 5100.0 * (1.0 + max(float(redshift), 0.0))
    valid_indices = np.flatnonzero(mask)
    best_index = valid_indices[int(np.argmin(np.abs(filter_wavelength[valid_indices] - target_obs_wave)))]
    flux_w_m2_hz = obs_fluxes_mjy[best_index] * 1.0e-29
    nu_obs_hz = C_MS / max(filter_wavelength[best_index] * 1.0e-10, 1.0e-30)
    agn_amp_w = 4.0 * np.pi * float(context.luminosity_distance_m) ** 2 * nu_obs_hz * flux_w_m2_hz
    return float(np.log(np.clip(agn_amp_w, 1.0e30, 1.0e50)))


def _sample_redshift(context: ModelContext, prior_config: dict[str, Any], cfg) -> jnp.ndarray:
    """Sample redshift from either the legacy Gaussian prior or a tabulated p(z)."""
    redshift_pdf = prior_config.get("redshift_pdf")
    if redshift_pdf is None:
        return numpyro.sample(
            "redshift",
            dist.TruncatedNormal(
                cfg.observation.redshift,
                max(cfg.observation.redshift_err, 1e-3),
                low=0.0,
            ),
        )

    z_grid = np.asarray(redshift_pdf["z_grid"], dtype=float)
    pdf = np.asarray(redshift_pdf["pdf"], dtype=float)
    pdf_norm = pdf / max(float(np.trapezoid(pdf, z_grid)), 1.0e-300)
    z_grid_jnp = _np_to_jnp(z_grid)
    pdf_jnp = _np_to_jnp(pdf_norm)
    redshift = numpyro.sample(
        "redshift",
        dist.Uniform(
            low=float(z_grid[0]),
            high=float(z_grid[-1]),
        ),
    )
    pz_val = jnp.interp(redshift, z_grid_jnp, pdf_jnp, left=0.0, right=0.0)
    numpyro.factor("redshift_pdf_prior", jnp.log(jnp.clip(pz_val, 1.0e-300, None)))
    return redshift


def _chi2_upper_limit(obs_fluxes, model_fluxes, total_variance):
    """Return the one-sided chi-square contribution for upper limits."""
    z = (obs_fluxes - model_fluxes) / jnp.sqrt(2.0 * jnp.maximum(total_variance, 1e-60))
    return -2.0 * jnp.log(0.5 * (1.0 + jax.scipy.special.erf(z)) + 1e-300)


def _robust_flux_scale(fluxes, valid_mask):
    """Estimate a robust characteristic flux scale from valid photometric points."""
    fluxes = jnp.asarray(fluxes, dtype=jnp.float64)
    valid_mask = jnp.asarray(valid_mask, dtype=bool)
    safe_flux = jnp.where(valid_mask, jnp.abs(fluxes), jnp.nan)
    scale = jnp.nanmedian(safe_flux)
    fallback = jnp.nanmax(safe_flux)
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, fallback)
    return jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0e-6)


def _absolute_flux_scale_logprior(pred_fluxes, obs_fluxes, valid_mask, sigma_dex):
    """Penalize solutions whose overall flux scale is far from the data."""
    n_valid = jnp.sum(valid_mask.astype(jnp.int32))

    def _compute():
        """Evaluate the Gaussian prior on log10 model/data flux scale."""
        obs_scale = _robust_flux_scale(obs_fluxes, valid_mask)
        pred_scale = _robust_flux_scale(pred_fluxes, valid_mask)
        log_ratio = jnp.log10(jnp.maximum(pred_scale, 1.0e-30) / jnp.maximum(obs_scale, 1.0e-30))
        return dist.Normal(0.0, jnp.maximum(jnp.asarray(sigma_dex, dtype=jnp.float64), 1.0e-6)).log_prob(log_ratio)

    return jax.lax.cond(n_valid > 0, _compute, lambda: jnp.asarray(0.0, dtype=jnp.float64))


def _agn_variability_nev(agn_bol_lum_w, max_nev):
    """Return the Simm+2016-inspired fractional variability variance cap."""
    agn_bol_lum_w = jnp.maximum(jnp.asarray(agn_bol_lum_w, dtype=jnp.float64), 1.0e-30)
    max_nev = jnp.maximum(jnp.asarray(max_nev, dtype=jnp.float64), 1.0e-6)
    log_lbol_erg_s = jnp.log10(agn_bol_lum_w * ERG_PER_WATT)
    l45 = log_lbol_erg_s - 45.0
    simm_nev = 10.0 ** (-1.43 - 0.74 * l45)
    return jnp.minimum(max_nev, simm_nev)


def _host_capture_fraction(spatial_scale_arcsec, turnover_arcsec, slope):
    """Map a band's effective spatial scale to the captured host-light fraction."""
    spatial_scale_arcsec = jnp.asarray(spatial_scale_arcsec, dtype=jnp.float64)
    valid = jnp.isfinite(spatial_scale_arcsec) & (spatial_scale_arcsec > 0.0)
    turnover_arcsec = jnp.maximum(jnp.asarray(turnover_arcsec, dtype=jnp.float64), 1.0e-3)
    slope = jnp.maximum(jnp.asarray(slope, dtype=jnp.float64), 1.0e-3)
    safe_scale = jnp.where(valid, jnp.clip(spatial_scale_arcsec, 1.0e-3, 1.0e6), turnover_arcsec)
    frac = jax.nn.sigmoid(slope * (jnp.log(safe_scale) - jnp.log(turnover_arcsec)))
    return jnp.where(valid, frac, 1.0)


def photometric_loglike(pred_fluxes, obs_fluxes, obs_errors, upper_limits, data_mask, systematics_width, intrinsic_scatter, likelihood_family, student_t_df, agn_component, agn_bol_lum_w, agn_nev, variability_uncertainty, attenuation_model_uncertainty, transmitted_fraction, lyman_break_uncertainty, filter_wavelength, redshift):
    """Evaluate the broadband photometric log-likelihood for one model state."""
    pred_fluxes = jnp.nan_to_num(pred_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    agn_component = jnp.nan_to_num(agn_component, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    transmitted_fraction = jnp.nan_to_num(transmitted_fraction, nan=1.0e-4, posinf=1.0, neginf=1.0e-4)
    obs_variance = obs_errors**2 + jnp.maximum(intrinsic_scatter, 0.0) ** 2
    variability_nev = _agn_variability_nev(agn_bol_lum_w, agn_nev)
    var_variance = jnp.where(variability_uncertainty, variability_nev * agn_component**2, 0.0)
    sys_variance = (systematics_width * pred_fluxes) ** 2
    if attenuation_model_uncertainty:
        tf = jnp.clip(transmitted_fraction, 1e-4, 1.0)
        neg_log = -jnp.log10(tf + 1e-4)
        log_unc_frac = jnp.minimum(-4.0 + 2.0 * neg_log, -1.0)
        att_unc = 10 ** log_unc_frac / tf
        sys_variance = sys_variance + (att_unc * pred_fluxes) ** 2
    if lyman_break_uncertainty:
        ly_unc = jnp.where(filter_wavelength / (1.0 + redshift) < 1500.0, 1.0e8, 0.0)
        sys_variance = sys_variance + (ly_unc * pred_fluxes) ** 2
    total_variance = jnp.nan_to_num(obs_variance + sys_variance + var_variance, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    scale = jnp.sqrt(jnp.clip(total_variance, 1e-30, 1.0e60))
    family = str(likelihood_family).lower()
    if family in {"gaussian", "normal"}:
        data_dist = dist.Normal(loc=pred_fluxes, scale=scale)
    elif family in {"student_t", "student-t", "studentt", "t"}:
        data_dist = dist.StudentT(df=student_t_df, loc=pred_fluxes, scale=scale)
    else:
        raise ValueError("likelihood.likelihood_family must be one of: 'gaussian', 'student_t'.")
    logl_data = jnp.sum(jnp.where(data_mask, data_dist.log_prob(obs_fluxes), 0.0))
    logl_lim = jnp.sum(jnp.where(upper_limits, -0.5 * _chi2_upper_limit(obs_fluxes, pred_fluxes, total_variance), 0.0))
    invalid = ~(jnp.isfinite(pred_fluxes) & jnp.isfinite(scale) & jnp.isfinite(obs_fluxes) & jnp.isfinite(obs_errors))
    penalty = -1.0e6 * jnp.sum(invalid.astype(jnp.float64))
    return logl_data + logl_lim + penalty


def spectroscopic_loglike(pred_fluxes, obs_fluxes, obs_errors, mask, systematics_width, student_t_df):
    """Evaluate the observed-frame spectral log-likelihood."""
    pred_fluxes = jnp.nan_to_num(pred_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    obs_fluxes = jnp.nan_to_num(obs_fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    obs_errors = jnp.nan_to_num(obs_errors, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    variance = obs_errors**2 + (jnp.maximum(systematics_width, 0.0) * pred_fluxes) ** 2
    scale = jnp.sqrt(jnp.clip(variance, 1e-30, 1.0e60))
    student = dist.StudentT(df=student_t_df, loc=pred_fluxes, scale=scale)
    valid = mask & jnp.isfinite(pred_fluxes) & jnp.isfinite(obs_fluxes) & jnp.isfinite(scale)
    penalty = -1.0e6 * jnp.sum((mask & ~valid).astype(jnp.float64))
    return jnp.sum(jnp.where(valid, student.log_prob(obs_fluxes), 0.0)) + penalty


def spectroscopic_likelihood_weight(wave_obs, mask, spectrum_index, likelihood_weight_mode, resolving_power):
    """Return a scalar spectral likelihood weight."""
    if str(likelihood_weight_mode).lower() not in {"resolution_elements", "resolution"}:
        return jnp.asarray(1.0, dtype=jnp.float64)
    if resolving_power is None or not np.isfinite(float(resolving_power)) or float(resolving_power) <= 0.0:
        return jnp.asarray(1.0, dtype=jnp.float64)
    wave_obs = jnp.asarray(wave_obs, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    spectrum_index = jnp.asarray(spectrum_index, dtype=jnp.int32)
    if wave_obs.size < 2:
        return jnp.asarray(1.0, dtype=jnp.float64)

    prev_delta = jnp.zeros_like(wave_obs)
    next_delta = jnp.zeros_like(wave_obs)
    same_adjacent = spectrum_index[1:] == spectrum_index[:-1]
    delta = jnp.abs(wave_obs[1:] - wave_obs[:-1])
    prev_delta = prev_delta.at[1:].set(jnp.where(same_adjacent, delta, 0.0))
    next_delta = next_delta.at[:-1].set(jnp.where(same_adjacent, delta, 0.0))
    pixel_width = 0.5 * (prev_delta + next_delta)
    valid = mask & jnp.isfinite(wave_obs) & (wave_obs > 0.0) & (pixel_width > 0.0)
    resolution_width = wave_obs / float(resolving_power)
    n_eff = jnp.sum(jnp.where(valid, pixel_width / jnp.maximum(resolution_width, 1.0e-30), 0.0))
    n_pix = jnp.sum(valid.astype(jnp.float64))
    return jnp.where(n_pix > 0.0, jnp.minimum(n_eff / n_pix, 1.0), jnp.asarray(1.0, dtype=jnp.float64))


def _flambda_to_mjy(wave_obs, flux_lambda):
    """Convert internal f_lambda values on an observed wavelength grid to mJy."""
    return 1.0e-10 / 299792458.0 * 1.0e29 * wave_obs * wave_obs * flux_lambda


def _fixed_spectral_line_coverage_rest(context: ModelContext, cfg: FitConfig) -> tuple[float, float] | None:
    """Return fixed-redshift rest coverage for the jaxqsofit tied-line table."""
    if cfg.observation.fits_redshift:
        return None
    if str(cfg.spectroscopy_config.backend).lower() != "jaxqsofit":
        return None
    if not bool(cfg.spectroscopy_config.jaxqsofit.use_spectral_lines):
        return None
    spec_wave = np.asarray(context.spec_wave_obs, dtype=float)
    spec_mask = np.asarray(context.spec_mask, dtype=bool)
    valid = spec_mask & np.isfinite(spec_wave) & (spec_wave > 0.0)
    if not np.any(valid):
        return None
    redshift = max(float(cfg.observation.redshift), 0.0)
    margin = max(float(cfg.spectroscopy_config.jaxqsofit.line_coverage_margin_kms), 0.0) / C_KMS
    rest_min = float(np.min(spec_wave[valid]) / (1.0 + redshift))
    rest_max = float(np.max(spec_wave[valid]) / (1.0 + redshift))
    return (rest_min * (1.0 - margin), rest_max * (1.0 + margin))


def _photometric_agn_line_mask(context: ModelContext, cfg: FitConfig, line_wave, redshift):
    """Mask native AGN SED lines to those not covered by jaxqsofit spectroscopy."""
    if str(cfg.spectroscopy_config.backend).lower() != "jaxqsofit":
        return jnp.ones_like(line_wave)
    jqf_cfg = cfg.spectroscopy_config.jaxqsofit
    if not bool(jqf_cfg.use_photometric_lines):
        return jnp.zeros_like(line_wave)
    if (
        not cfg.spectroscopy_config.enabled
        or not bool(jqf_cfg.use_spectral_lines)
        or len(context.spec_wave_obs) == 0
        or not np.any(context.spec_mask)
    ):
        return jnp.ones_like(line_wave)

    spec_wave = jnp.asarray(context.spec_wave_obs, dtype=jnp.float64)
    spec_mask = jnp.asarray(context.spec_mask, dtype=bool)
    valid = spec_mask & jnp.isfinite(spec_wave) & (spec_wave > 0.0)
    spec_min = jnp.min(jnp.where(valid, spec_wave, jnp.inf))
    spec_max = jnp.max(jnp.where(valid, spec_wave, -jnp.inf))
    margin = jnp.asarray(max(float(jqf_cfg.line_coverage_margin_kms), 0.0) / C_KMS, dtype=jnp.float64)
    line_obs = jnp.asarray(line_wave, dtype=jnp.float64) * jnp.maximum(1.0 + redshift, 1.0e-8)
    covered = (line_obs >= spec_min * (1.0 - margin)) & (line_obs <= spec_max * (1.0 + margin))
    return jnp.where(covered, 0.0, 1.0)


def _integrated_spectral_flux_proxy(wave_obs, flux_mjy, mask):
    """Integrate positive line flux density on the observed spectral grid."""
    wave_obs = jnp.asarray(wave_obs, dtype=jnp.float64)
    flux_mjy = jnp.asarray(flux_mjy, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)
    positive_flux = jnp.where(mask, jnp.clip(flux_mjy, 0.0, None), 0.0)
    return jnp.trapezoid(positive_flux, wave_obs)


def _line_strength_bridge_logprob(pred_flux, ref_flux, sigma_dex):
    """Broad log-normal bridge between same-grid integrated line-flux proxies."""
    floor = jnp.asarray(1.0e-30, dtype=jnp.float64)
    sigma_ln = jnp.maximum(jnp.asarray(float(sigma_dex), dtype=jnp.float64) * jnp.log(10.0), 1.0e-6)
    pred = jnp.maximum(jnp.asarray(pred_flux, dtype=jnp.float64), floor)
    ref = jnp.maximum(jnp.asarray(ref_flux, dtype=jnp.float64), floor)
    resid = (jnp.log(pred) - jnp.log(ref)) / sigma_ln
    active = ref_flux > floor * 10.0
    return jnp.where(active, -0.5 * resid * resid, jnp.asarray(0.0, dtype=jnp.float64))


def _evaluate_jaxqsofit_backend(
    wave_obs,
    redshift,
    continuum_mjy,
    cfg,
    line_prior_config,
    rest_wave,
    feii_template_flux,
    line_coverage_rest=None,
    fixed_narrow_fwhm_kms=None,
    fixed_narrow_amp_scale=None,
):
    """Evaluate optional jaxqsofit spectral components inside jaxsedfit."""
    try:
        from jaxqsofit.components import SpectralComponentConfig, evaluate_joint_spectral_components
    except Exception as exc:  # pragma: no cover - exercised only without optional dependency
        raise ImportError(
            "SpectroscopyConfig.backend='jaxqsofit' requires jaxqsofit on PYTHONPATH."
        ) from exc

    jqf_cfg = cfg.spectroscopy_config.jaxqsofit
    component_cfg = SpectralComponentConfig(
        use_lines=bool(jqf_cfg.use_spectral_lines),
        use_tied_lines=bool(jqf_cfg.use_tied_lines),
        use_feii=bool(jqf_cfg.use_spectral_feii),
        use_balmer_continuum=bool(jqf_cfg.use_spectral_balmer_continuum),
        use_multiplicative_tilt=bool(jqf_cfg.use_multiplicative_tilt),
        line_table=jqf_cfg.line_table,
        line_prior_config=line_prior_config,
        line_flux_scale_mjy=float(jqf_cfg.line_flux_scale_mjy),
        line_coverage_rest=line_coverage_rest,
        include_elg_narrow_lines=bool(jqf_cfg.include_elg_narrow_lines),
        include_high_ionization_lines=bool(jqf_cfg.include_high_ionization_lines),
        broad_fwhm_kms_default=DEFAULT_BROAD_LINE_WIDTH_KMS,
        feii_fwhm_kms_default=DEFAULT_BROAD_LINE_WIDTH_KMS,
        balmer_velocity_kms_default=DEFAULT_BROAD_LINE_WIDTH_KMS,
        fixed_narrow_fwhm_kms=fixed_narrow_fwhm_kms,
        fixed_narrow_amp_scale=fixed_narrow_amp_scale,
    )
    return evaluate_joint_spectral_components(
        wave_obs,
        redshift,
        continuum_mjy,
        config=component_cfg,
        feii_template_wave_rest=rest_wave,
        feii_template_flux=feii_template_flux,
        site_prefix="jqf",
    )


def evaluate_photometric_state(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
    add_likelihood: bool = True,
    return_state: bool = True,
    force_component_fluxes: bool = False,
):
    """Evaluate one jaxsedfit photometric model state inside a NumPyro trace."""
    cfg = context.fit_config
    prior_config = cfg.prior_config.to_mapping()
    rest_wave = context.rest_wave_jax
    obs_wave = context.obs_wave_jax
    feii_template_on_rest = context.feii_template_on_rest_jax
    line_wave = _np_to_jnp(context.templates.line_wave)
    line_blagn = _np_to_jnp(context.templates.line_blagn)
    line_sy2 = _np_to_jnp(context.templates.line_sy2)
    line_liner = _np_to_jnp(context.templates.line_liner)
    filter_wavelength = context.filter_effective_wavelength_jax
    obs_fluxes = _np_to_jnp(context.fluxes)
    obs_errors = _np_to_jnp(context.errors)
    upper_limits = _bool_to_jnp(context.upper_limits)
    data_mask = _bool_to_jnp(context.data_mask)
    positive_detected_mask = _bool_to_jnp(context.positive_detected_mask)
    spec_wave_obs = _np_to_jnp(context.spec_wave_obs)
    spec_fluxes = _np_to_jnp(context.spec_fluxes)
    spec_errors = _np_to_jnp(context.spec_errors)
    spec_mask = _bool_to_jnp(context.spec_mask)
    spec_spectrum_index = jnp.asarray(context.spec_spectrum_index, dtype=jnp.int32)
    spec_spatial_scale_arcsec = _np_to_jnp(context.spec_effective_spatial_scale_arcsec)
    dust_alpha_grid = np.asarray(context.templates.dust_alpha_grid, dtype=float)
    fit_host = bool(cfg.galaxy.fit_host)
    fit_agn = bool(cfg.agn.fit_agn)
    spatial_scale_arcsec = _np_to_jnp(context.effective_spatial_scale_arcsec)
    spectroscopy_enabled = bool(
        cfg.spectroscopy is not None
        and cfg.spectroscopy_config.enabled
        and len(context.spec_wave_obs) > 0
        and np.any(context.spec_mask)
    )
    jqf_cfg = cfg.spectroscopy_config.jaxqsofit
    jaxqsofit_backend_enabled = bool(
        spectroscopy_enabled
        and str(cfg.spectroscopy_config.backend).lower() == "jaxqsofit"
    )
    use_native_feii = bool(
        include_sed_agn_features
        and not (jaxqsofit_backend_enabled and bool(jqf_cfg.use_spectral_feii))
    )
    use_native_balmer = bool(
        include_sed_agn_features
        and not (jaxqsofit_backend_enabled and bool(jqf_cfg.use_spectral_balmer_continuum))
    )
    jaxqsofit_line_coverage_rest = _fixed_spectral_line_coverage_rest(context, cfg) if spectroscopy_enabled else None
    has_phot_spatial_scale = bool(
        np.any(np.isfinite(context.effective_spatial_scale_arcsec) & (np.asarray(context.effective_spatial_scale_arcsec, dtype=float) > 0.0))
    )
    has_spec_spatial_scale = bool(
        spectroscopy_enabled
        and np.any(np.isfinite(context.spec_effective_spatial_scale_arcsec) & (np.asarray(context.spec_effective_spatial_scale_arcsec, dtype=float) > 0.0))
    )
    host_capture_enabled = bool(
        fit_host
        and cfg.likelihood.use_host_capture_model
        and (has_phot_spatial_scale or has_spec_spatial_scale)
    )
    skip_coarse_agn_line_grid = bool(
        fit_agn
        and include_sed_agn_features
        and cfg.likelihood.use_local_line_photometry
        and not include_components
        and not spectroscopy_enabled
        and not cfg.likelihood.attenuation_model_uncertainty
    )
    needs_spec_host_basis = bool(
        spectroscopy_enabled
        and str(cfg.spectroscopy_config.backend).lower() == "jaxqsofit"
        and context.spec_host_basis_jax is not None
        and context.spec_rest_wave_jax.size == context.spec_wave_obs.size
        and not cfg.observation.fits_redshift
    )
    host_state = (
        _build_host_state(context, prior_config, full_output=include_components or needs_spec_host_basis)
        if fit_host
        else _empty_host_state(context)
    )
    host_rest = host_state["host_rest"]
    host_kinematics_enabled = bool(fit_host and cfg.galaxy.fit_host_kinematics and spectroscopy_enabled)
    if host_kinematics_enabled:
        gal_v_kms = _sample_prior(prior_config, "gal_v_kms", dist.Normal(0.0, 150.0))
        gal_sigma_kms = _sample_prior(prior_config, "gal_sigma_kms", dist.HalfNormal(150.0))
        host_rest = _shift_and_broaden_single_spectrum_lnlam(jnp.log(rest_wave), host_rest, gal_v_kms, gal_sigma_kms)
    else:
        gal_v_kms = jnp.asarray(0.0, dtype=jnp.float64)
        gal_sigma_kms = jnp.asarray(0.0, dtype=jnp.float64)

    host_llambda_5100 = jnp.interp(5100.0, rest_wave, host_rest, left=0.0, right=0.0) if fit_host else jnp.asarray(0.0, dtype=jnp.float64)
    if fit_agn:
        # Infer the AGN normalization directly from the photometry rather than
        # forcing the host 5100 A continuum to set the AGN amplitude.
        fracagn_5100 = jnp.asarray(0.999, dtype=jnp.float64)
        log_agn_amp = _sample_prior(
            prior_config,
            "log_agn_amp",
            dist.Normal(_estimate_log_agn_amp_prior_loc(context, cfg.observation.redshift), 2.0),
        )
        agn_amp = jnp.exp(log_agn_amp)
        if fit_host:
            agn_llambda_5100 = agn_amp / 5100.0
            fracagn_5100 = jnp.clip(
                agn_llambda_5100 / jnp.maximum(agn_llambda_5100 + jnp.clip(host_llambda_5100, 0.0, 1.0e60), 1.0e-30),
                1.0e-4,
                0.999,
            )
    else:
        fracagn_5100 = jnp.asarray(1.0e-4, dtype=jnp.float64)
        agn_amp = jnp.asarray(0.0, dtype=jnp.float64)
        log_agn_amp = jnp.log(jnp.clip(agn_amp, 1.0e-30, 1.0e80))

    agn_type = int(cfg.agn.agn_type)
    if fit_agn:
        pl_slope = _sample_prior(prior_config, "pl_slope", dist.TruncatedNormal(-1.8, 0.4, low=-2.5, high=-1.0))
        uv_slope_delta = _sample_positive_distribution(
            prior_config,
            value_key="uv_slope_delta",
            log_key="log_uv_slope_delta",
            default_value_distribution=dist.LogNormal(np.log(1.8), 0.4),
            default_log_distribution=dist.Normal(np.log(1.8), 0.4),
        )
        uv_slope = pl_slope + uv_slope_delta
        numpyro.deterministic("uv_slope", uv_slope)
        pl_bend_loc = _sample_positive_distribution(
            prior_config,
            value_key="pl_bend_loc",
            log_key="log_pl_bend_loc",
            default_value_distribution=dist.LogNormal(np.log(GRAHSP_PL_BEND_LOC_A), 0.2),
            default_log_distribution=dist.Normal(np.log(GRAHSP_PL_BEND_LOC_A), 0.2),
        )
        pl_bend_width = _sample_positive_distribution(
            prior_config,
            value_key="pl_bend_width",
            log_key="log_pl_bend_width",
            default_value_distribution=dist.LogNormal(np.log(GRAHSP_PL_BEND_WIDTH), 0.4),
            default_log_distribution=dist.Normal(np.log(GRAHSP_PL_BEND_WIDTH), 0.4),
        )
        pl_cutoff = _sample_positive_distribution(
            prior_config,
            value_key="pl_cutoff",
            log_key="log_pl_cutoff",
            default_value_distribution=dist.LogNormal(np.log(GRAHSP_PL_CUTOFF_A), 0.6),
            default_log_distribution=dist.Normal(np.log(GRAHSP_PL_CUTOFF_A), 0.6),
        )
        disk_rest = _powerlaw_jax(rest_wave, agn_amp / 5100.0, uv_slope, pl_slope, 5100.0, pl_bend_loc, pl_bend_width, pl_cutoff)

        fcov = _sample_positive_distribution(
            prior_config,
            value_key="fcov",
            log_key="log_fcov",
            default_value_distribution=dist.LogNormal(np.log(0.2), 0.6),
            default_log_distribution=dist.Normal(np.log(0.2), 0.6),
            default_to_log=True,
        )
        si = _sample_prior(prior_config, "si", dist.Normal(0.0, 1.0))
        cool_lam = _sample_positive_distribution(
            prior_config,
            value_key="cool_lam",
            log_key="log_cool_lam",
            default_value_distribution=dist.LogNormal(np.log(17.0), 0.2),
            default_log_distribution=dist.Normal(np.log(17.0), 0.2),
        )
        cool_width = _sample_positive_distribution(
            prior_config,
            value_key="cool_width",
            log_key="log_cool_width",
            default_value_distribution=dist.LogNormal(np.log(0.45), 0.2),
            default_log_distribution=dist.Normal(np.log(0.45), 0.2),
        )
        hot_lam = _sample_positive_distribution(
            prior_config,
            value_key="hot_lam",
            log_key="log_hot_lam",
            default_value_distribution=dist.LogNormal(np.log(2.0), 0.3),
            default_log_distribution=dist.Normal(np.log(2.0), 0.3),
        )
        hot_width = _sample_positive_distribution(
            prior_config,
            value_key="hot_width",
            log_key="log_hot_width",
            default_value_distribution=dist.LogNormal(np.log(0.5), 0.2),
            default_log_distribution=dist.Normal(np.log(0.5), 0.2),
        )
        hot_fcov = _sample_positive_distribution(
            prior_config,
            value_key="hot_fcov",
            log_key="log_hot_fcov",
            default_value_distribution=dist.LogNormal(np.log(0.1), 0.8),
            default_log_distribution=dist.Normal(np.log(0.1), 0.8),
            default_to_log=True,
        )
        torus_rest = _torus_component(
            rest_wave,
            fcov,
            si,
            cool_lam,
            cool_width,
            hot_lam,
            hot_width,
            hot_fcov,
            0.29,
            GRAHSP_SI_EM_LAM_A,
            GRAHSP_SI_ABS_LAM_A,
            GRAHSP_SI_EM_WIDTH_A,
            GRAHSP_SI_ABS_WIDTH_A,
            agn_amp,
        )

        if include_sed_agn_features:
            broad_lines_strength = _sample_positive_distribution(
                prior_config,
                value_key="broad_lines_strength",
                log_key="log_broad_lines_strength",
                default_value_distribution=dist.LogNormal(np.log(DEFAULT_BROAD_LINES_STRENGTH), 0.5),
                default_log_distribution=dist.Normal(np.log(DEFAULT_BROAD_LINES_STRENGTH), 0.5),
            )
            narrow_lines_strength = _sample_positive_distribution(
                prior_config,
                value_key="narrow_lines_strength",
                log_key="log_narrow_lines_strength",
                default_value_distribution=dist.LogNormal(np.log(DEFAULT_NARROW_LINES_STRENGTH), 0.5),
                default_log_distribution=dist.Normal(np.log(DEFAULT_NARROW_LINES_STRENGTH), 0.5),
            )
            broad_line_width = _sample_log_positive_from_distribution(
                prior_config,
                value_key="broad_line_width_kms",
                log_key="log_broad_line_width_kms",
                default_distribution=dist.TruncatedNormal(
                    np.log(DEFAULT_BROAD_LINE_WIDTH_KMS),
                    0.4,
                    low=np.log(DEFAULT_BROAD_LINE_WIDTH_KMS_MIN),
                    high=np.log(DEFAULT_BROAD_LINE_WIDTH_KMS_MAX),
                ),
            )
            narrow_line_width = _sample_log_positive_from_distribution(
                prior_config,
                value_key="narrow_line_width_kms",
                log_key="log_narrow_line_width_kms",
                default_distribution=dist.TruncatedNormal(
                    np.log(DEFAULT_NARROW_LINE_WIDTH_KMS),
                    0.3,
                    low=np.log(DEFAULT_NARROW_LINE_WIDTH_KMS_MIN),
                    high=np.log(DEFAULT_NARROW_LINE_WIDTH_KMS_MAX),
                ),
            )
            balmer_enabled = bool(use_native_balmer and cfg.agn.fit_balmer_continuum and agn_type == 1)
            if balmer_enabled:
                balmer_norm = _sample_positive_distribution(
                    prior_config,
                    value_key="balmer_norm",
                    log_key="log_balmer_norm",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_BALMER_CONTINUUM_STRENGTH), 1.0),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_BALMER_CONTINUUM_STRENGTH), 1.0),
                )
                balmer_tau = _sample_positive_distribution(
                    prior_config,
                    value_key="balmer_tau",
                    log_key="log_balmer_tau",
                    default_value_distribution=dist.LogNormal(np.log(1.0), 0.5),
                    default_log_distribution=dist.Normal(np.log(1.0), 0.5),
                )
                balmer_vel = _sample_positive_distribution(
                    prior_config,
                    value_key="balmer_vel",
                    log_key="log_balmer_vel",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.4),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.4),
                )
            else:
                balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
                balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
                balmer_vel = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        else:
            broad_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
            narrow_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
            broad_line_width = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
            narrow_line_width = jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS, dtype=jnp.float64)
            balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
            balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
            balmer_vel = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
            balmer_enabled = False
        l_agn_lambda_5100 = agn_amp / 5100.0
        agn_bol_luminosity = agn_amp * AGN_BOLOMETRIC_CORRECTION_5100
        l_broadlines = 0.02 * l_agn_lambda_5100 * broad_lines_strength
        l_narrowlines = 0.002 * l_agn_lambda_5100 * narrow_lines_strength

        if skip_coarse_agn_line_grid:
            line_bl_rest = jnp.zeros_like(rest_wave)
            line_nl_rest = jnp.zeros_like(rest_wave)
            line_liner_rest = jnp.zeros_like(rest_wave)
            line_rest = jnp.zeros_like(rest_wave)
        else:
            line_bl_rest = jnp.where(
                agn_type == 1,
                _line_gaussians(rest_wave, line_wave, l_broadlines * line_blagn, broad_line_width),
                jnp.zeros_like(rest_wave),
            )
            line_nl_rest = jnp.where(
                agn_type in (1, 2),
                _line_gaussians(rest_wave, line_wave, l_narrowlines * line_sy2, narrow_line_width),
                jnp.zeros_like(rest_wave),
            )
            line_liner_rest = jnp.where(
                agn_type == 3,
                _line_gaussians(rest_wave, line_wave, l_narrowlines * line_liner, narrow_line_width),
                jnp.zeros_like(rest_wave),
            )
            line_rest = line_bl_rest + line_nl_rest + line_liner_rest

        if use_native_feii and agn_type == 1:
            feii_norm = _sample_positive_distribution(
                prior_config,
                value_key="feii_norm",
                log_key="log_feii_norm",
                default_value_distribution=dist.LogNormal(np.log(DEFAULT_FEII_STRENGTH), 0.8),
                default_log_distribution=dist.Normal(np.log(DEFAULT_FEII_STRENGTH), 0.8),
            )
            l_feii = feii_norm * l_broadlines
            if cfg.agn.fit_feii_broadening:
                feii_fwhm = _sample_positive_distribution(
                    prior_config,
                    value_key="feii_fwhm",
                    log_key="log_feii_fwhm",
                    default_value_distribution=dist.LogNormal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.3),
                    default_log_distribution=dist.Normal(np.log(DEFAULT_BROAD_LINE_WIDTH_KMS), 0.3),
                )
                feii_shift = _sample_prior(prior_config, "feii_shift", dist.Normal(0.0, 0.01))
                feii_rest = _feii_component(rest_wave, feii_template_on_rest, l_feii, feii_fwhm, feii_shift)
            else:
                feii_fwhm = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
                feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
                feii_rest = l_feii * jnp.maximum(feii_template_on_rest, 0.0)
        else:
            feii_norm = jnp.asarray(0.0, dtype=jnp.float64)
            feii_fwhm = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
            feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
            feii_rest = jnp.zeros_like(rest_wave)
        balmer_rest = (
            _balmer_continuum_jax(rest_wave, balmer_norm, 15000.0, balmer_tau, balmer_vel)
            if balmer_enabled
            else jnp.zeros_like(rest_wave)
        )
    else:
        uv_slope = jnp.asarray(0.0, dtype=jnp.float64)
        pl_slope = jnp.asarray(-1.0, dtype=jnp.float64)
        pl_bend_loc = jnp.asarray(GRAHSP_PL_BEND_LOC_A, dtype=jnp.float64)
        pl_bend_width = jnp.asarray(GRAHSP_PL_BEND_WIDTH, dtype=jnp.float64)
        pl_cutoff = jnp.asarray(GRAHSP_PL_CUTOFF_A, dtype=jnp.float64)
        disk_rest = jnp.zeros_like(rest_wave)
        fcov = jnp.asarray(0.0, dtype=jnp.float64)
        si = jnp.asarray(0.0, dtype=jnp.float64)
        cool_lam = jnp.asarray(17.0, dtype=jnp.float64)
        cool_width = jnp.asarray(0.45, dtype=jnp.float64)
        hot_lam = jnp.asarray(2.0, dtype=jnp.float64)
        hot_width = jnp.asarray(0.5, dtype=jnp.float64)
        hot_fcov = jnp.asarray(0.0, dtype=jnp.float64)
        torus_rest = jnp.zeros_like(rest_wave)
        broad_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
        narrow_lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
        broad_line_width = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        narrow_line_width = jnp.asarray(DEFAULT_NARROW_LINE_WIDTH_KMS, dtype=jnp.float64)
        balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
        balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
        balmer_vel = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        l_agn_lambda_5100 = jnp.asarray(0.0, dtype=jnp.float64)
        agn_bol_luminosity = jnp.asarray(0.0, dtype=jnp.float64)
        line_bl_rest = jnp.zeros_like(rest_wave)
        line_nl_rest = jnp.zeros_like(rest_wave)
        line_liner_rest = jnp.zeros_like(rest_wave)
        line_rest = jnp.zeros_like(rest_wave)
        feii_norm = jnp.asarray(0.0, dtype=jnp.float64)
        feii_fwhm = jnp.asarray(DEFAULT_BROAD_LINE_WIDTH_KMS, dtype=jnp.float64)
        feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
        feii_rest = jnp.zeros_like(rest_wave)
        balmer_rest = jnp.zeros_like(rest_wave)

    ebv_gal = (
        _sample_positive_distribution(
            prior_config,
            value_key="ebv_gal",
            log_key="log_ebv_gal",
            default_value_distribution=dist.LogNormal(np.log(0.05), 1.0),
            default_log_distribution=dist.Normal(np.log(0.05), 1.0),
            default_to_log=True,
        )
        if fit_host
        else jnp.asarray(0.0, dtype=jnp.float64)
    )
    ebv_agn = (
        _sample_positive_distribution(
            prior_config,
            value_key="ebv_agn",
            log_key="log_ebv_agn",
            default_value_distribution=dist.LogNormal(np.log(0.05), 1.0),
            default_log_distribution=dist.Normal(np.log(0.05), 1.0),
            default_to_log=True,
        )
        if fit_agn
        else jnp.asarray(0.0, dtype=jnp.float64)
    )
    if cfg.galaxy.use_energy_balance and fit_host:
        dust_alpha = _sample_prior(
            prior_config,
            "dust_alpha",
            dist.TruncatedNormal(cfg.galaxy.dust_alpha, 0.4, low=float(np.min(dust_alpha_grid)), high=float(np.max(dust_alpha_grid))),
        )
    else:
        dust_alpha = jnp.asarray(float(cfg.galaxy.dust_alpha), dtype=jnp.float64)
    if cfg.likelihood.fit_intrinsic_scatter:
        log_intrinsic_scatter = _sample_prior(
            prior_config,
            "log_intrinsic_scatter",
            dist.Normal(np.log(max(cfg.likelihood.intrinsic_scatter_default, 1.0e-8)), 1.0),
        )
        intrinsic_scatter = jnp.exp(log_intrinsic_scatter)
    else:
        intrinsic_scatter = jnp.asarray(float(cfg.likelihood.intrinsic_scatter_default), dtype=jnp.float64)
    if cfg.likelihood.fit_systematics_width:
        systematics_width = _sample_positive(
            prior_config,
            value_key="systematics_width",
            log_key="log_systematics_width",
            default_value=float(cfg.likelihood.systematics_width_prior_scale),
            default_log_scale=1.0,
            default_family="exponential",
        )
    else:
        systematics_width = jnp.asarray(float(cfg.likelihood.systematics_width), dtype=jnp.float64)
    if cfg.observation.fits_redshift:
        redshift = _sample_redshift(context, prior_config, cfg)
        luminosity_distance_m = _luminosity_distance_m_jax(
            redshift,
            cfg.galaxy.cosmology_h0,
            cfg.galaxy.cosmology_om0,
        )
        igm = _igm_transmission(context.igm_cache_jax, redshift)
    else:
        redshift = context.fixed_redshift_jax
        luminosity_distance_m = context.fixed_luminosity_distance_m_jax
        igm = context.fixed_igm_jax
    nebular = _build_nebular_components(context, host_state, host_rest, prior_config)
    host_with_nebular_rest = host_rest + nebular["absorption_rest"] + nebular["emission_rest"]
    agn_attenuated_input_rest = disk_rest + feii_rest + line_rest + balmer_rest
    gal_att_rest, agn_attenuated_rest, host_absorbed_rest, dust_luminosity = _apply_biattenuation(
        rest_wave,
        host_with_nebular_rest,
        agn_attenuated_input_rest,
        ebv_gal,
        ebv_agn,
        -1.2,
        -3.0,
        1.2,
        GRAHSP_BIATTENUATION_BREAK_A,
    )
    dust_luminosity = dust_luminosity + nebular["dust_luminosity"]
    attenuation_curve = _attenuation_curve(rest_wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    gal_att_factor = 10 ** (ebv_gal * attenuation_curve / -2.5)
    agn_att_factor = 10 ** ((ebv_gal + ebv_agn) * attenuation_curve / -2.5)
    host_stellar_att_rest = (host_rest + nebular["absorption_rest"]) * gal_att_factor
    nebular_att_rest = nebular["emission_rest"] * gal_att_factor
    nebular_lines_att_rest = nebular["lines_rest"] * gal_att_factor
    nebular_continuum_att_rest = nebular["continuum_rest"] * gal_att_factor
    disk_att_rest = disk_rest * agn_att_factor
    torus_att_rest = torus_rest
    feii_att_rest = feii_rest * agn_att_factor
    line_bl_att_rest = line_bl_rest * agn_att_factor
    line_nl_att_rest = line_nl_rest * agn_att_factor
    line_liner_att_rest = line_liner_rest * agn_att_factor
    line_att_rest = line_rest * agn_att_factor
    balmer_att_rest = balmer_rest * agn_att_factor
    dust_rest = jnp.where(
        cfg.galaxy.use_energy_balance and fit_host,
        _host_dust_emission(context, dust_luminosity, dust_alpha),
        jnp.zeros_like(rest_wave),
    )
    agn_rest = agn_attenuated_rest + torus_att_rest
    total_rest = gal_att_rest + dust_rest + agn_rest
    transmitted_fraction = _attenuation_transmitted_fraction(
        gal_att_rest + agn_attenuated_rest,
        host_with_nebular_rest + agn_attenuated_input_rest,
    )
    fast_projection_enabled = _can_use_fixed_filter_projection(context, cfg) and not include_components
    redshift_projection_enabled = (
        _can_use_redshift_filter_projection(context, cfg)
        and not include_components
        and not spectroscopy_enabled
    )
    if fast_projection_enabled:
        total_obs = _redshift_to_obs(rest_wave, total_rest * igm, obs_wave, redshift, luminosity_distance_m) if spectroscopy_enabled else jnp.zeros_like(obs_wave)
        pred_fluxes_raw = _project_rest_luminosity_filters(context, total_rest)
    elif redshift_projection_enabled:
        total_obs = jnp.zeros_like(obs_wave)
        pred_fluxes_raw = _project_redshift_luminosity_filters(context, total_rest, redshift)
    else:
        total_obs = _redshift_to_obs(rest_wave, total_rest * igm, obs_wave, redshift, luminosity_distance_m)
        pred_fluxes_raw = _project_filters(total_obs, context.packed_filters_jax)
    local_agn_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    coarse_agn_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    correct_agn_line_photometry = (
        fit_agn
        and include_sed_agn_features
        and bool(cfg.likelihood.use_local_line_photometry)
    )
    if correct_agn_line_photometry:
        photometric_agn_line_mask = _photometric_agn_line_mask(context, cfg, line_wave, redshift)
        if agn_type == 1:
            local_broad_line_lumin = l_broadlines * line_blagn
            local_narrow_line_lumin = l_narrowlines * line_sy2
        elif agn_type == 2:
            local_broad_line_lumin = jnp.zeros_like(line_wave)
            local_narrow_line_lumin = l_narrowlines * line_sy2
        elif agn_type == 3:
            local_broad_line_lumin = jnp.zeros_like(line_wave)
            local_narrow_line_lumin = l_narrowlines * line_liner
        else:
            local_broad_line_lumin = jnp.zeros_like(line_wave)
            local_narrow_line_lumin = jnp.zeros_like(line_wave)
        local_broad_line_lumin = local_broad_line_lumin * photometric_agn_line_mask
        local_narrow_line_lumin = local_narrow_line_lumin * photometric_agn_line_mask
        if context.fixed_local_line_projection_cache_jax is not None and not cfg.observation.fits_redshift:
            local_broad_line_fluxes = _project_fixed_cached_local_line_filters(
                context,
                local_broad_line_lumin,
                broad_line_width,
                ebv_gal + ebv_agn,
            )
            local_narrow_line_fluxes = _project_fixed_cached_local_line_filters(
                context,
                local_narrow_line_lumin,
                narrow_line_width,
                ebv_gal + ebv_agn,
            )
        else:
            local_broad_line_fluxes = _project_local_line_filters(
                context,
                line_wave,
                local_broad_line_lumin,
                broad_line_width,
                ebv_gal + ebv_agn,
                redshift,
                luminosity_distance_m,
                igm,
            )
            local_narrow_line_fluxes = _project_local_line_filters(
                context,
                line_wave,
                local_narrow_line_lumin,
                narrow_line_width,
                ebv_gal + ebv_agn,
                redshift,
                luminosity_distance_m,
                igm,
            )
        local_agn_line_fluxes = local_broad_line_fluxes + local_narrow_line_fluxes
        if fast_projection_enabled:
            coarse_agn_line_fluxes = _project_rest_luminosity_filters(context, line_att_rest)
        elif redshift_projection_enabled:
            coarse_agn_line_fluxes = _project_redshift_luminosity_filters(context, line_att_rest, redshift)
        else:
            coarse_line_obs = _redshift_to_obs(rest_wave, line_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
            coarse_agn_line_fluxes = _project_filters(coarse_line_obs, context.packed_filters_jax)
        pred_fluxes_raw = pred_fluxes_raw - coarse_agn_line_fluxes + local_agn_line_fluxes
    local_nebular_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    coarse_nebular_line_fluxes = jnp.zeros_like(pred_fluxes_raw)
    correct_nebular_line_photometry = (
        fit_host
        and bool(cfg.nebular.enabled)
        and bool(cfg.nebular.emission)
        and bool(cfg.likelihood.use_local_line_photometry)
    )
    if correct_nebular_line_photometry:
        local_nebular_line_fluxes = _project_local_nebular_line_filters(
            context,
            context.nebular_templates_jax.line_wave_a,
            nebular["line_lumin"],
            nebular["lines_width"],
            ebv_gal,
            redshift,
            luminosity_distance_m,
            igm,
        )
        if fast_projection_enabled:
            coarse_nebular_line_fluxes = _project_rest_luminosity_filters(context, nebular_lines_att_rest)
        elif redshift_projection_enabled:
            coarse_nebular_line_fluxes = _project_redshift_luminosity_filters(context, nebular_lines_att_rest, redshift)
        else:
            coarse_nebular_line_obs = _redshift_to_obs(rest_wave, nebular_lines_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
            coarse_nebular_line_fluxes = _project_filters(coarse_nebular_line_obs, context.packed_filters_jax)
        pred_fluxes_raw = pred_fluxes_raw - coarse_nebular_line_fluxes + local_nebular_line_fluxes
    host_dust_fluxes_total = jnp.zeros_like(pred_fluxes_raw)
    if host_capture_enabled or include_components or spectroscopy_enabled:
        host_obs = (
            jnp.zeros_like(obs_wave)
            if redshift_projection_enabled and not (include_components or spectroscopy_enabled)
            else _redshift_to_obs(rest_wave, gal_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        )
        if fast_projection_enabled:
            host_fluxes_total = _project_rest_luminosity_filters(context, gal_att_rest)
        elif redshift_projection_enabled:
            host_fluxes_total = _project_redshift_luminosity_filters(context, gal_att_rest, redshift)
        else:
            host_fluxes_total = _project_filters(host_obs, context.packed_filters_jax)
        if correct_nebular_line_photometry:
            host_fluxes_total = host_fluxes_total - coarse_nebular_line_fluxes + local_nebular_line_fluxes
        if host_capture_enabled:
            if fast_projection_enabled:
                host_dust_fluxes_total = _project_rest_luminosity_filters(context, dust_rest)
            elif redshift_projection_enabled:
                host_dust_fluxes_total = _project_redshift_luminosity_filters(context, dust_rest, redshift)
            else:
                host_dust_obs = _redshift_to_obs(rest_wave, dust_rest * igm, obs_wave, redshift, luminosity_distance_m)
                host_dust_fluxes_total = _project_filters(host_dust_obs, context.packed_filters_jax)
    else:
        host_obs = jnp.zeros_like(total_obs)
        host_fluxes_total = jnp.zeros_like(pred_fluxes_raw)
    if host_capture_enabled:
        log_host_capture_scale_arcsec = _sample_prior(
            prior_config,
            "log_host_capture_scale_arcsec",
            dist.Normal(np.log(3.0), 1.0),
        )
        host_capture_slope = _sample_positive_distribution(
            prior_config,
            value_key="host_capture_slope",
            log_key="log_host_capture_slope",
            default_value_distribution=dist.LogNormal(np.log(2.0), 0.5),
            default_log_distribution=dist.Normal(np.log(2.0), 0.5),
        )
        phot_capture_raw = _host_capture_fraction(
            spatial_scale_arcsec,
            jnp.exp(log_host_capture_scale_arcsec),
            host_capture_slope,
        )
        phot_scale_valid = jnp.isfinite(spatial_scale_arcsec) & (spatial_scale_arcsec > 0.0)
        host_capture_fraction = jnp.where(phot_scale_valid, phot_capture_raw, 1.0)
        spec_capture_raw = _host_capture_fraction(
            spec_spatial_scale_arcsec,
            jnp.exp(log_host_capture_scale_arcsec),
            host_capture_slope,
        )
        spec_scale_valid = jnp.isfinite(spec_spatial_scale_arcsec) & (spec_spatial_scale_arcsec > 0.0)
        spec_host_capture_fraction_by_spectrum = jnp.where(spec_scale_valid, spec_capture_raw, 1.0)
    else:
        log_host_capture_scale_arcsec = jnp.asarray(np.log(3.0), dtype=jnp.float64)
        host_capture_slope = jnp.asarray(2.0, dtype=jnp.float64)
        host_capture_fraction = jnp.ones_like(pred_fluxes_raw)
        spec_host_capture_fraction_by_spectrum = jnp.ones_like(spec_spatial_scale_arcsec)
    host_capture_source_fluxes = host_fluxes_total + host_dust_fluxes_total
    host_fluxes = host_fluxes_total * host_capture_fraction
    captured_host_dust_fluxes = host_dust_fluxes_total * host_capture_fraction
    captured_host_source_fluxes = host_capture_source_fluxes * host_capture_fraction
    pred_fluxes = (
        pred_fluxes_raw
        if not host_capture_enabled
        else pred_fluxes_raw - host_capture_source_fluxes + captured_host_source_fluxes
    )

    spec_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_host_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_disk_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_torus_model_fluxes = jnp.zeros_like(spec_fluxes)
    spec_continuum_model_fluxes = jnp.zeros_like(spec_fluxes)
    spectrum_scale = jnp.asarray(1.0, dtype=jnp.float64)
    spec_likelihood_weight = jnp.asarray(1.0, dtype=jnp.float64)
    if spectroscopy_enabled:
        backend = str(cfg.spectroscopy_config.backend).lower()
        if backend == "jaxqsofit":
            use_spec_resolution_continuum = bool(
                context.spec_host_basis_jax is not None
                and context.spec_rest_wave_jax.size == context.spec_wave_obs.size
                and not cfg.observation.fits_redshift
            )
            if use_spec_resolution_continuum:
                spec_rest_wave = context.spec_rest_wave_jax
                spec_igm = jnp.interp(spec_rest_wave, rest_wave, igm, left=0.0, right=0.0)
                spec_att_curve = _attenuation_curve(
                    spec_rest_wave,
                    -1.2,
                    -3.0,
                    1.2,
                    GRAHSP_BIATTENUATION_BREAK_A,
                )
                spec_gal_att_factor = 10 ** (ebv_gal * spec_att_curve / -2.5)
                spec_agn_att_factor = 10 ** ((ebv_gal + ebv_agn) * spec_att_curve / -2.5)
                spec_nebular_absorption_rest = jnp.interp(
                    spec_rest_wave,
                    rest_wave,
                    nebular["absorption_rest"],
                    left=0.0,
                    right=0.0,
                )
                spec_host_intrinsic = _host_rest_on_basis(host_state, context.spec_host_basis_jax)
                if host_kinematics_enabled:
                    spec_host_intrinsic = _shift_and_broaden_single_spectrum_lnlam(
                        jnp.log(spec_rest_wave),
                        spec_host_intrinsic,
                        gal_v_kms,
                        gal_sigma_kms,
                    )
                spec_host_rest = (spec_host_intrinsic + spec_nebular_absorption_rest) * spec_gal_att_factor
                spec_disk_rest = (
                    _powerlaw_jax(
                        spec_rest_wave,
                        agn_amp / 5100.0,
                        uv_slope,
                        pl_slope,
                        5100.0,
                        pl_bend_loc,
                        pl_bend_width,
                        pl_cutoff,
                    )
                    * spec_agn_att_factor
                )
                spec_torus_rest = _torus_component(
                    spec_rest_wave,
                    fcov,
                    si,
                    cool_lam,
                    cool_width,
                    hot_lam,
                    hot_width,
                    hot_fcov,
                    0.29,
                    GRAHSP_SI_EM_LAM_A,
                    GRAHSP_SI_ABS_LAM_A,
                    GRAHSP_SI_EM_WIDTH_A,
                    GRAHSP_SI_ABS_WIDTH_A,
                    agn_amp,
                )
                spec_denom = (
                    4.0
                    * jnp.pi
                    * jnp.maximum(luminosity_distance_m, 1e-12) ** 2
                    * jnp.maximum(1.0 + redshift, 1e-8)
                )
                spec_host_lambda = spec_host_rest * spec_igm / spec_denom
                spec_disk_lambda = spec_disk_rest * spec_igm / spec_denom
                spec_torus_lambda = spec_torus_rest * spec_igm / spec_denom
                if host_capture_enabled:
                    spec_capture_at_pixel = spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                    spec_host_lambda = spec_capture_at_pixel * spec_host_lambda
                spec_host_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_host_lambda)
                spec_disk_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_disk_lambda)
                spec_torus_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_torus_lambda)
                spec_model_fluxes = spec_host_model_fluxes + spec_disk_model_fluxes + spec_torus_model_fluxes
            else:
                spec_source_obs = host_obs + _redshift_to_obs(
                    rest_wave,
                    (disk_rest + torus_rest) * igm,
                    obs_wave,
                    redshift,
                    luminosity_distance_m,
                )
                spec_model_lambda = jnp.interp(spec_wave_obs, obs_wave, spec_source_obs, left=0.0, right=0.0)
                spec_host_lambda = jnp.interp(spec_wave_obs, obs_wave, host_obs, left=0.0, right=0.0)
                if host_capture_enabled:
                    spec_capture_at_pixel = spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                    spec_model_lambda = spec_model_lambda - spec_host_lambda + spec_capture_at_pixel * spec_host_lambda
                    spec_host_lambda = spec_capture_at_pixel * spec_host_lambda
                spec_host_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_host_lambda)
                spec_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_model_lambda)
            spec_continuum_model_fluxes = spec_model_fluxes
        elif backend != "jaxsedfit":
            raise ValueError(f"Unsupported spectroscopy backend: {cfg.spectroscopy_config.backend!r}")
        else:
            spec_model_lambda = jnp.interp(spec_wave_obs, obs_wave, total_obs, left=0.0, right=0.0)
            spec_host_lambda = jnp.interp(spec_wave_obs, obs_wave, host_obs, left=0.0, right=0.0)
            if host_capture_enabled:
                spec_capture_at_pixel = spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                spec_model_lambda = spec_model_lambda - spec_host_lambda + spec_capture_at_pixel * spec_host_lambda
                spec_host_lambda = spec_capture_at_pixel * spec_host_lambda
            spec_host_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_host_lambda)
            spec_model_fluxes = _flambda_to_mjy(spec_wave_obs, spec_model_lambda)
            spec_continuum_model_fluxes = spec_model_fluxes
        if backend == "jaxqsofit" and include_spectral_features:
            jqf_components = _evaluate_jaxqsofit_backend(
                spec_wave_obs,
                redshift,
                spec_model_fluxes,
                cfg,
                context.jaxqsofit_prior_config,
                rest_wave,
                feii_template_on_rest,
                line_coverage_rest=jaxqsofit_line_coverage_rest,
                fixed_narrow_fwhm_kms=nebular["lines_width"],
            )
            jqf_cfg = cfg.spectroscopy_config.jaxqsofit
            if bool(jqf_cfg.use_line_strength_priors) and bool(jqf_cfg.use_spectral_lines):
                sed_broad_line_obs = _redshift_to_obs(
                    rest_wave,
                    line_bl_att_rest * igm,
                    obs_wave,
                    redshift,
                    luminosity_distance_m,
                )
                sed_narrow_line_obs = _redshift_to_obs(
                    rest_wave,
                    line_nl_att_rest * igm,
                    obs_wave,
                    redshift,
                    luminosity_distance_m,
                )
                sed_broad_line_mjy = _flambda_to_mjy(
                    spec_wave_obs,
                    jnp.interp(spec_wave_obs, obs_wave, sed_broad_line_obs, left=0.0, right=0.0),
                )
                sed_narrow_line_mjy = _flambda_to_mjy(
                    spec_wave_obs,
                    jnp.interp(spec_wave_obs, obs_wave, sed_narrow_line_obs, left=0.0, right=0.0),
                )
                sed_narrow_bridge_mjy = sed_narrow_line_mjy
                if bool(jqf_cfg.use_nebular_line_prior):
                    sed_nebular_line_obs = _redshift_to_obs(
                        rest_wave,
                        nebular_lines_att_rest * igm,
                        obs_wave,
                        redshift,
                        luminosity_distance_m,
                    )
                    sed_nebular_line_mjy = _flambda_to_mjy(
                        spec_wave_obs,
                        jnp.interp(spec_wave_obs, obs_wave, sed_nebular_line_obs, left=0.0, right=0.0),
                    )
                    if host_capture_enabled:
                        sed_nebular_line_mjy = sed_nebular_line_mjy * spec_host_capture_fraction_by_spectrum[spec_spectrum_index]
                    sed_narrow_bridge_mjy = sed_narrow_bridge_mjy + sed_nebular_line_mjy

                jqf_broad_flux = _integrated_spectral_flux_proxy(spec_wave_obs, jqf_components["line_broad"], spec_mask)
                sed_broad_flux = _integrated_spectral_flux_proxy(spec_wave_obs, sed_broad_line_mjy, spec_mask)
                jqf_narrow_flux = _integrated_spectral_flux_proxy(spec_wave_obs, jqf_components["line_narrow"], spec_mask)
                sed_narrow_flux = _integrated_spectral_flux_proxy(spec_wave_obs, sed_narrow_bridge_mjy, spec_mask)
                numpyro.factor(
                    "jqf_broad_to_sed_broad_line_prior",
                    _line_strength_bridge_logprob(
                        sed_broad_flux,
                        jqf_broad_flux,
                        jqf_cfg.line_strength_prior_sigma_dex,
                    ),
                )
                narrow_sigma_dex = (
                    jqf_cfg.nebular_line_prior_sigma_dex
                    if bool(jqf_cfg.use_nebular_line_prior)
                    else jqf_cfg.line_strength_prior_sigma_dex
                )
                numpyro.factor(
                    "jqf_narrow_to_sed_narrow_line_prior",
                    _line_strength_bridge_logprob(
                        sed_narrow_flux,
                        jqf_narrow_flux,
                        narrow_sigma_dex,
                    ),
                )
                numpyro.deterministic("jqf_broad_line_flux_proxy", jqf_broad_flux)
                numpyro.deterministic("sed_broad_line_flux_proxy", sed_broad_flux)
                numpyro.deterministic("jqf_narrow_line_flux_proxy", jqf_narrow_flux)
                numpyro.deterministic("sed_narrow_line_flux_proxy", sed_narrow_flux)
            spec_model_fluxes = jqf_components["total"]
        if cfg.spectroscopy_config.fit_scale:
            scale_prior = _prior_distribution(
                prior_config,
                "log_spectrum_scale",
                dist.Normal(0.0, np.log(10.0) * cfg.spectroscopy_config.scale_prior_sigma_dex),
            )
            n_spectra = len(context.spec_instruments)
            if n_spectra > 1:
                log_spectrum_scale = numpyro.sample(
                    "log_spectrum_scale",
                    scale_prior.expand([n_spectra]).to_event(1),
                )
                spectrum_scale = jnp.exp(log_spectrum_scale)
                spec_model_fluxes = spectrum_scale[spec_spectrum_index] * spec_model_fluxes
            else:
                log_spectrum_scale = numpyro.sample(
                    "log_spectrum_scale",
                    scale_prior,
                )
                spectrum_scale = jnp.exp(log_spectrum_scale)
                spec_model_fluxes = spectrum_scale * spec_model_fluxes
        else:
            log_spectrum_scale = jnp.asarray(0.0, dtype=jnp.float64)
        spec_logl = spectroscopic_loglike(
            spec_model_fluxes,
            spec_fluxes,
            spec_errors,
            spec_mask,
            cfg.spectroscopy_config.systematics_width,
            cfg.spectroscopy_config.student_t_df,
        )
        spec_likelihood_weight = spectroscopic_likelihood_weight(
            spec_wave_obs,
            spec_mask,
            spec_spectrum_index,
            cfg.spectroscopy_config.likelihood_weight_mode,
            cfg.spectroscopy_config.resolving_power,
        )
        spec_logl = spec_likelihood_weight * spec_logl
        if add_likelihood:
            numpyro.factor("spectroscopy_loglike_factor", spec_logl)
    else:
        log_spectrum_scale = jnp.asarray(0.0, dtype=jnp.float64)
        spec_logl = jnp.asarray(0.0, dtype=jnp.float64)
    need_agn_fluxes = fit_agn and (include_components or force_component_fluxes or cfg.likelihood.variability_uncertainty)
    need_trans_fluxes = include_components or cfg.likelihood.attenuation_model_uncertainty
    if include_components:
        agn_obs = _redshift_to_obs(rest_wave, agn_rest * igm, obs_wave, redshift, luminosity_distance_m)
        host_stellar_obs = _redshift_to_obs(rest_wave, host_stellar_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        dust_obs = _redshift_to_obs(rest_wave, dust_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_obs = _redshift_to_obs(rest_wave, nebular_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_lines_obs = _redshift_to_obs(rest_wave, nebular_lines_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_continuum_obs = _redshift_to_obs(rest_wave, nebular_continuum_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        nebular_lines_local_obs_wave, nebular_lines_local_obs = _local_nebular_line_obs_sed(
            context,
            context.nebular_templates_jax.line_wave_a,
            nebular["line_lumin"],
            nebular["lines_width"],
            ebv_gal,
            redshift,
            luminosity_distance_m,
            igm,
        )
        total_local_obs = (
            jnp.interp(nebular_lines_local_obs_wave, obs_wave, total_obs, left=0.0, right=0.0)
            - jnp.interp(nebular_lines_local_obs_wave, obs_wave, nebular_lines_obs, left=0.0, right=0.0)
            + nebular_lines_local_obs
        )
        disk_obs = _redshift_to_obs(rest_wave, disk_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        torus_obs = _redshift_to_obs(rest_wave, torus_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        feii_obs = _redshift_to_obs(rest_wave, feii_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_obs = _redshift_to_obs(rest_wave, line_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_bl_obs = _redshift_to_obs(rest_wave, line_bl_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_nl_obs = _redshift_to_obs(rest_wave, line_nl_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_liner_obs = _redshift_to_obs(rest_wave, line_liner_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        balmer_obs = _redshift_to_obs(rest_wave, balmer_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
        transmitted_fraction_obs = _redshift_scalar_to_obs(rest_wave, transmitted_fraction, obs_wave, redshift)
        agn_fluxes = _project_filters(agn_obs, context.packed_filters_jax)
        dust_fluxes = _project_filters(dust_obs, context.packed_filters_jax)
        nebular_fluxes = _project_filters(nebular_obs, context.packed_filters_jax)
        nebular_lines_fluxes = _project_filters(nebular_lines_obs, context.packed_filters_jax)
        nebular_continuum_fluxes = _project_filters(nebular_continuum_obs, context.packed_filters_jax)
        if correct_nebular_line_photometry:
            nebular_lines_fluxes = local_nebular_line_fluxes
            nebular_fluxes = nebular_continuum_fluxes + local_nebular_line_fluxes
        disk_fluxes = _project_filters(disk_obs, context.packed_filters_jax)
        torus_fluxes = _project_filters(torus_obs, context.packed_filters_jax)
        feii_fluxes = _project_filters(feii_obs, context.packed_filters_jax)
        line_fluxes = _project_filters(line_obs, context.packed_filters_jax)
        line_bl_fluxes = _project_filters(line_bl_obs, context.packed_filters_jax)
        line_nl_fluxes = _project_filters(line_nl_obs, context.packed_filters_jax)
        line_liner_fluxes = _project_filters(line_liner_obs, context.packed_filters_jax)
        balmer_fluxes = _project_filters(balmer_obs, context.packed_filters_jax)
        trans_fluxes = _project_filters(transmitted_fraction_obs, context.packed_filters_jax)
        if correct_agn_line_photometry:
            agn_fluxes = agn_fluxes - coarse_agn_line_fluxes + local_agn_line_fluxes
            line_fluxes = local_agn_line_fluxes
    else:
        nebular_lines_local_obs_wave = jnp.zeros((1,), dtype=jnp.float64)
        nebular_lines_local_obs = jnp.zeros((1,), dtype=jnp.float64)
        total_local_obs = jnp.zeros((1,), dtype=jnp.float64)
        if need_agn_fluxes:
            if fast_projection_enabled:
                agn_fluxes = _project_rest_luminosity_filters(context, agn_rest)
                if correct_agn_line_photometry:
                    agn_fluxes = agn_fluxes - coarse_agn_line_fluxes + local_agn_line_fluxes
            elif redshift_projection_enabled:
                agn_fluxes = _project_redshift_luminosity_filters(context, agn_rest, redshift)
                if correct_agn_line_photometry:
                    agn_fluxes = agn_fluxes - coarse_agn_line_fluxes + local_agn_line_fluxes
            else:
                agn_obs = _redshift_to_obs(rest_wave, agn_rest * igm, obs_wave, redshift, luminosity_distance_m)
                agn_fluxes = _project_filters(agn_obs, context.packed_filters_jax)
                if correct_agn_line_photometry:
                    agn_fluxes = agn_fluxes - coarse_agn_line_fluxes + local_agn_line_fluxes
        else:
            agn_fluxes = jnp.zeros_like(pred_fluxes)
        if need_trans_fluxes:
            if fast_projection_enabled:
                trans_fluxes = _project_rest_scalar_filters(context, transmitted_fraction)
            elif redshift_projection_enabled:
                trans_fluxes = _project_redshift_scalar_filters(context, transmitted_fraction, redshift)
            else:
                transmitted_fraction_obs = _redshift_scalar_to_obs(rest_wave, transmitted_fraction, obs_wave, redshift)
                trans_fluxes = _project_filters(transmitted_fraction_obs, context.packed_filters_jax)
        else:
            trans_fluxes = jnp.ones_like(pred_fluxes)

    logl = photometric_loglike(
        pred_fluxes=pred_fluxes,
        obs_fluxes=obs_fluxes,
        obs_errors=obs_errors,
        upper_limits=upper_limits,
        data_mask=data_mask,
        systematics_width=systematics_width,
        intrinsic_scatter=intrinsic_scatter,
        likelihood_family=cfg.likelihood.likelihood_family,
        student_t_df=cfg.likelihood.student_t_df,
        agn_component=agn_fluxes,
        agn_bol_lum_w=agn_bol_luminosity,
        agn_nev=cfg.likelihood.agn_nev,
        variability_uncertainty=cfg.likelihood.variability_uncertainty,
        attenuation_model_uncertainty=cfg.likelihood.attenuation_model_uncertainty,
        transmitted_fraction=trans_fluxes,
        lyman_break_uncertainty=cfg.likelihood.lyman_break_uncertainty,
        filter_wavelength=filter_wavelength,
        redshift=redshift,
    )
    if add_likelihood:
        numpyro.factor("photometry_loglike", logl)
    abs_flux_scale_logprior = jnp.asarray(0.0, dtype=jnp.float64)
    if cfg.likelihood.use_absolute_flux_scale_prior:
        abs_flux_scale_logprior = _absolute_flux_scale_logprior(
            pred_fluxes=pred_fluxes,
            obs_fluxes=obs_fluxes,
            valid_mask=positive_detected_mask,
            sigma_dex=cfg.likelihood.absolute_flux_scale_prior_sigma_dex,
        )
        if add_likelihood:
            numpyro.factor("absolute_flux_scale_prior", abs_flux_scale_logprior)

    numpyro.deterministic("pred_fluxes", pred_fluxes)
    numpyro.deterministic("pred_spectrum_fluxes", spec_model_fluxes)
    numpyro.deterministic("spec_continuum_model_fluxes", spec_continuum_model_fluxes)
    numpyro.deterministic("spec_host_model_fluxes", spec_host_model_fluxes)
    numpyro.deterministic("spec_disk_model_fluxes", spec_disk_model_fluxes)
    numpyro.deterministic("spec_torus_model_fluxes", spec_torus_model_fluxes)
    numpyro.deterministic("spectrum_scale_fit", spectrum_scale)
    numpyro.deterministic("log_spectrum_scale_fit", log_spectrum_scale)
    numpyro.deterministic("spectrum_host_capture_fraction", spec_host_capture_fraction_by_spectrum)
    numpyro.deterministic("spectroscopy_likelihood_weight", spec_likelihood_weight)
    numpyro.deterministic("spectroscopy_loglike", spec_logl)
    numpyro.deterministic("intrinsic_scatter_fit", intrinsic_scatter)
    numpyro.deterministic("fracAGN_5100_fit", fracagn_5100)
    numpyro.deterministic("log_agn_amp_fit", log_agn_amp)
    numpyro.deterministic("log_disk_luminosity_fit", _safe_log10(l_agn_lambda_5100))
    numpyro.deterministic("log_agn_bol_luminosity_fit", _safe_log10(agn_bol_luminosity))
    numpyro.deterministic("agn_variability_nev", _agn_variability_nev(agn_bol_luminosity, cfg.likelihood.agn_nev))
    numpyro.deterministic("transmitted_fraction_fluxes", trans_fluxes)
    numpyro.deterministic("host_total_fluxes", host_fluxes_total)
    numpyro.deterministic("host_capture_source_fluxes", host_capture_source_fluxes)
    numpyro.deterministic("captured_host_dust_fluxes", captured_host_dust_fluxes)
    numpyro.deterministic("host_capture_fraction_fluxes", host_capture_fraction)
    numpyro.deterministic("log_host_capture_scale_arcsec_fit", log_host_capture_scale_arcsec)
    numpyro.deterministic("host_capture_slope_fit", host_capture_slope)
    numpyro.deterministic("formed_stellar_mass", host_state["formed_mass"])
    numpyro.deterministic("surviving_mass_fraction", host_state["surviving_mass_fraction"])
    numpyro.deterministic("gal_lgmet_fit", host_state["gal_lgmet"])
    numpyro.deterministic("gal_lgmet_scatter_fit", host_state["gal_lgmet_scatter"])
    numpyro.deterministic("mass_metallicity_relation_logprior", host_state["mass_metallicity_relation_logprior"])
    numpyro.deterministic("sfh_age_gyr_fit", host_state["sfh_age_gyr"])
    numpyro.deterministic("sfh_tau_gyr_fit", host_state["sfh_tau_gyr"])
    numpyro.deterministic("log_dust_luminosity_fit", _safe_log10(dust_luminosity))
    numpyro.deterministic("dust_alpha_fit", dust_alpha)
    numpyro.deterministic("nebular_logU_fit", nebular["logU"])
    numpyro.deterministic("nebular_zgas_fit", nebular["zgas"])
    numpyro.deterministic("nebular_ne_fit", nebular["ne"])
    numpyro.deterministic("nebular_f_esc_fit", nebular["f_esc"])
    numpyro.deterministic("nebular_f_dust_fit", nebular["f_dust"])
    numpyro.deterministic("nebular_lines_width_fit", nebular["lines_width"])
    numpyro.deterministic("nebular_line_scale_fit", nebular["line_scale"])
    numpyro.deterministic("nebular_corr_fit", nebular["corr"])
    numpyro.deterministic("nebular_n_ly_young_fit", nebular["n_ly_young"])
    numpyro.deterministic("nebular_n_ly_old_fit", nebular["n_ly_old"])
    numpyro.deterministic("absolute_flux_scale_logprior", abs_flux_scale_logprior)
    numpyro.deterministic("rest_wave", rest_wave)
    numpyro.deterministic("obs_wave", obs_wave)
    numpyro.deterministic("spec_wave_obs", spec_wave_obs)
    numpyro.deterministic("spec_spectrum_index", spec_spectrum_index)
    numpyro.deterministic("redshift_fit", redshift)
    if include_components:
        numpyro.deterministic("host_age_weights", host_state["host_age_weights"])
        numpyro.deterministic("host_lgmet_weights", host_state["host_lgmet_weights"])
        numpyro.deterministic("host_ssp_weights", host_state["host_ssp_weights"])
        numpyro.deterministic("gal_sfr_table", host_state["gal_sfr_table"])
        numpyro.deterministic("gal_smh_table", host_state["gal_smh_table"])
        numpyro.deterministic("total_rest_sed", total_rest)
        numpyro.deterministic("agn_rest_sed", agn_rest)
        numpyro.deterministic("host_rest_sed", host_stellar_att_rest)
        numpyro.deterministic("host_total_rest_sed", gal_att_rest)
        numpyro.deterministic("host_absorbed_rest_sed", host_absorbed_rest)
        numpyro.deterministic("dust_rest_sed", dust_rest)
        numpyro.deterministic("nebular_rest_sed", nebular_att_rest)
        numpyro.deterministic("nebular_lines_rest_sed", nebular_lines_att_rest)
        numpyro.deterministic("nebular_continuum_rest_sed", nebular_continuum_att_rest)
        numpyro.deterministic("nebular_absorption_rest_sed", nebular["absorption_rest"])
        numpyro.deterministic("disk_rest_sed", disk_att_rest)
        numpyro.deterministic("torus_rest_sed", torus_att_rest)
        numpyro.deterministic("feii_rest_sed", feii_att_rest)
        numpyro.deterministic("line_rest_sed", line_att_rest)
        numpyro.deterministic("line_bl_rest_sed", line_bl_att_rest)
        numpyro.deterministic("line_nl_rest_sed", line_nl_att_rest)
        numpyro.deterministic("line_liner_rest_sed", line_liner_att_rest)
        numpyro.deterministic("balmer_rest_sed", balmer_att_rest)
        numpyro.deterministic("agn_fluxes", agn_fluxes)
        numpyro.deterministic("host_fluxes", host_fluxes)
        numpyro.deterministic("dust_fluxes", dust_fluxes)
        numpyro.deterministic("nebular_fluxes", nebular_fluxes)
        numpyro.deterministic("nebular_lines_fluxes", nebular_lines_fluxes)
        numpyro.deterministic("nebular_continuum_fluxes", nebular_continuum_fluxes)
        numpyro.deterministic("disk_fluxes", disk_fluxes)
        numpyro.deterministic("torus_fluxes", torus_fluxes)
        numpyro.deterministic("feii_fluxes", feii_fluxes)
        numpyro.deterministic("line_fluxes", line_fluxes)
        numpyro.deterministic("line_bl_fluxes", line_bl_fluxes)
        numpyro.deterministic("line_nl_fluxes", line_nl_fluxes)
        numpyro.deterministic("line_liner_fluxes", line_liner_fluxes)
        numpyro.deterministic("balmer_fluxes", balmer_fluxes)
        numpyro.deterministic("total_obs_sed", total_obs)
        numpyro.deterministic("total_local_lines_obs_wave", nebular_lines_local_obs_wave)
        numpyro.deterministic("total_local_lines_obs_sed", total_local_obs)
        numpyro.deterministic("agn_obs_sed", agn_obs)
        numpyro.deterministic("host_obs_sed", host_stellar_obs)
        numpyro.deterministic("host_total_obs_sed", host_obs)
        numpyro.deterministic("dust_obs_sed", dust_obs)
        numpyro.deterministic("nebular_obs_sed", nebular_obs)
        numpyro.deterministic("nebular_lines_obs_sed", nebular_lines_obs)
        numpyro.deterministic("nebular_lines_local_obs_wave", nebular_lines_local_obs_wave)
        numpyro.deterministic("nebular_lines_local_obs_sed", nebular_lines_local_obs)
        numpyro.deterministic("nebular_continuum_obs_sed", nebular_continuum_obs)
        numpyro.deterministic("disk_obs_sed", disk_obs)
        numpyro.deterministic("torus_obs_sed", torus_obs)
        numpyro.deterministic("feii_obs_sed", feii_obs)
        numpyro.deterministic("line_obs_sed", line_obs)
        numpyro.deterministic("line_bl_obs_sed", line_bl_obs)
        numpyro.deterministic("line_nl_obs_sed", line_nl_obs)
        numpyro.deterministic("line_liner_obs_sed", line_liner_obs)
        numpyro.deterministic("balmer_obs_sed", balmer_obs)

    if not return_state:
        return None

    state = {
        "pred_fluxes": pred_fluxes,
        "pred_spectrum_fluxes": spec_model_fluxes,
        "spec_continuum_model_fluxes": spec_continuum_model_fluxes,
        "spec_host_model_fluxes": spec_host_model_fluxes,
        "spec_disk_model_fluxes": spec_disk_model_fluxes,
        "spec_torus_model_fluxes": spec_torus_model_fluxes,
        "agn_fluxes": agn_fluxes,
        "host_fluxes": host_fluxes,
        "host_total_fluxes": host_fluxes_total,
        "host_capture_source_fluxes": host_capture_source_fluxes,
        "captured_host_dust_fluxes": captured_host_dust_fluxes,
        "dust_fluxes": dust_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "nebular_fluxes": nebular_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "nebular_lines_fluxes": nebular_lines_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "nebular_continuum_fluxes": nebular_continuum_fluxes if include_components else jnp.zeros_like(pred_fluxes),
        "rest_wave": rest_wave,
        "obs_wave": obs_wave,
        "redshift_fit": redshift,
        "photometry_loglike": logl,
        "spectroscopy_loglike": spec_logl,
        "spectroscopy_likelihood_weight": spec_likelihood_weight,
    }
    if include_components:
        state.update(
            {
                "total_rest_sed": total_rest,
                "agn_rest_sed": agn_rest,
                "host_rest_sed": host_stellar_att_rest,
                "host_total_rest_sed": gal_att_rest,
                "dust_rest_sed": dust_rest,
                "nebular_rest_sed": nebular_att_rest,
                "total_obs_sed": total_obs,
                "total_local_lines_obs_wave": nebular_lines_local_obs_wave,
                "total_local_lines_obs_sed": total_local_obs,
                "agn_obs_sed": agn_obs,
                "host_obs_sed": host_stellar_obs,
                "host_total_obs_sed": host_obs,
                "dust_obs_sed": dust_obs,
                "nebular_obs_sed": nebular_obs,
                "nebular_lines_obs_sed": nebular_lines_obs,
                "nebular_lines_local_obs_wave": nebular_lines_local_obs_wave,
                "nebular_lines_local_obs_sed": nebular_lines_local_obs,
                "nebular_continuum_obs_sed": nebular_continuum_obs,
                "disk_obs_sed": disk_obs,
                "torus_obs_sed": torus_obs,
                "feii_obs_sed": feii_obs,
                "line_obs_sed": line_obs,
                "line_bl_obs_sed": line_bl_obs,
                "line_nl_obs_sed": line_nl_obs,
                "line_liner_obs_sed": line_liner_obs,
                "balmer_obs_sed": balmer_obs,
            }
        )
    return state


def grahsp_photometric_model(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
):
    """NumPyro model for one jaxsedfit photometric fit or predictive expansion."""
    return evaluate_photometric_state(
        context,
        include_components=include_components,
        include_sed_agn_features=include_sed_agn_features,
        include_spectral_features=include_spectral_features,
        add_likelihood=True,
        return_state=False,
    )


def sed_numpyro_model(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
):
    """NumPyro SED model for one configured ``jaxsedfit`` target.

    This is the preferred public name for the low-level NumPyro model. The
    historical name :func:`grahsp_photometric_model` remains available as a
    compatibility alias.
    """
    return grahsp_photometric_model(
        context,
        include_components=include_components,
        include_sed_agn_features=include_sed_agn_features,
        include_spectral_features=include_spectral_features,
    )


def evaluate_sed_model(
    context: ModelContext,
    include_components: bool = False,
    include_sed_agn_features: bool = True,
    include_spectral_features: bool = True,
    add_likelihood: bool = True,
    return_state: bool = True,
    force_component_fluxes: bool = False,
):
    """Evaluate the SED model state for a configured target.

    This is the preferred public name for deterministic model evaluation. It
    delegates to :func:`evaluate_photometric_state`, which is kept for
    compatibility with earlier releases.
    """
    return evaluate_photometric_state(
        context,
        include_components=include_components,
        include_sed_agn_features=include_sed_agn_features,
        include_spectral_features=include_spectral_features,
        add_likelihood=add_likelihood,
        return_state=return_state,
        force_component_fluxes=force_component_fluxes,
    )


def photometric_log_likelihood(*args, **kwargs):
    """Return the photometric log likelihood for one model/data comparison."""
    return photometric_loglike(*args, **kwargs)


def spectroscopic_log_likelihood(*args, **kwargs):
    """Return the spectroscopic log likelihood for one model/data comparison."""
    return spectroscopic_loglike(*args, **kwargs)
