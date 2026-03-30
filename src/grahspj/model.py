from __future__ import annotations

# Portions of this file are derived from or closely based on GRAHSP/pcigale
# model logic, translated into JAX/NumPyro for grahspj.
# Relevant upstream sources include:
# - pcigale/creation_modules/activate.py
# - pcigale/creation_modules/activategtorus.py
# - pcigale/creation_modules/activatelines.py
# - pcigale/creation_modules/biattenuation.py
# - pcigale/creation_modules/redshifting.py
# - pcigale/creation_modules/galdale2014.py
# Upstream license: CeCILL v2. See LICENSES/CeCILL-v2.txt and
# THIRD_PARTY_NOTICES.md in the repository root.

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_U_PARAMS, DiffstarUParams, calc_sfh_singlegal, get_bounded_diffstar_params
from dsps.sed.ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf
import numpyro
import numpyro.distributions as dist

from .preload import ModelContext

C_KMS = 299792.458
C_MS = 2.99792458e8
L_SUN_W = 3.828e26
ERG_PER_WATT = 1.0e7
AGN_BOLOMETRIC_CORRECTION_5100 = 9.26
MPC_TO_M = 3.085677581491367e22


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


def _luminosity_distance_m_jax(redshift, h0: float, om0: float):
    """Return luminosity distance in meters using jax_cosmo."""
    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    scalar_input = redshift.ndim == 0
    bg, cosmo = _get_jax_cosmo_backend(float(h0), float(om0))
    a = 1.0 / (1.0 + jnp.maximum(redshift, 0.0))
    d_a_mpc_over_h = bg.angular_diameter_distance(cosmo, a)
    d_l_mpc_over_h = d_a_mpc_over_h / jnp.maximum(a * a, 1.0e-30)
    d_l_m = d_l_mpc_over_h / cosmo.h * MPC_TO_M
    return jnp.reshape(d_l_m, ()) if scalar_input else d_l_m


def _cfg_norm(prior_config: dict[str, Any], key: str, default_loc: float, default_scale: float):
    """Read a Normal-like prior `(loc, scale)` pair from prior_config."""
    cfg = prior_config.get(key, None)
    if isinstance(cfg, dict) and "loc" in cfg and "scale" in cfg:
        return jnp.asarray(cfg["loc"]), jnp.asarray(cfg["scale"])
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return jnp.asarray(cfg[0]), jnp.asarray(cfg[1])
    return jnp.asarray(default_loc), jnp.asarray(default_scale)


def _cfg_halfnorm(prior_config: dict[str, Any], key: str, default_scale: float):
    """Read a HalfNormal-like scale value from prior_config."""
    cfg = prior_config.get(key, None)
    if isinstance(cfg, dict) and "scale" in cfg:
        return jnp.asarray(cfg["scale"])
    if isinstance(cfg, (int, float)):
        return jnp.asarray(cfg)
    return jnp.asarray(default_scale)


def _cfg_mean_scale(prior_config: dict[str, Any], key: str, default_loc: float, default_scale: float):
    """Alias for reading mean/scale prior settings from prior_config."""
    return _cfg_norm(prior_config, key, default_loc, default_scale)


def _safe_log10(x):
    """Take log10 after clipping to a tiny positive floor."""
    return jnp.log10(jnp.clip(jnp.asarray(x, dtype=jnp.float64), 1.0e-30, 1.0e300))


def _sample_log_stellar_mass(prior_config: dict[str, Any]):
    """Sample stellar mass with a less top-heavy default prior.

    By default this uses a heavy-tailed Student-t prior centered lower than the
    original Normal(10.5, 2.5) benchmark default. Existing Normal-like
    overrides with only ``loc`` and ``scale`` are still supported.
    """
    cfg = prior_config.get("log_stellar_mass", None)
    if isinstance(cfg, dict):
        loc = jnp.asarray(cfg.get("loc", 10.0))
        scale = jnp.asarray(cfg.get("scale", 2.0))
        dist_name = str(cfg.get("dist", "student_t")).lower()
        if dist_name in {"student_t", "studentt", "t"}:
            df = jnp.asarray(cfg.get("df", 5.0))
            return numpyro.sample("log_stellar_mass", dist.StudentT(df=df, loc=loc, scale=scale))
        if dist_name in {"normal", "gaussian"}:
            return numpyro.sample("log_stellar_mass", dist.Normal(loc, scale))
    if isinstance(cfg, (tuple, list)) and len(cfg) >= 2:
        return numpyro.sample("log_stellar_mass", dist.Normal(jnp.asarray(cfg[0]), jnp.asarray(cfg[1])))
    return numpyro.sample("log_stellar_mass", dist.StudentT(df=5.0, loc=10.0, scale=2.0))


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
    """Evaluate the empirical torus component on the rest-frame wavelength grid."""
    log_wave_um = jnp.log10(wave / 1000.0)
    log_cool = jnp.log10(cool_lam)
    log_hot = jnp.log10(hot_lam)
    cool = jnp.exp(-((log_wave_um - log_cool) / cool_width) ** 2)
    hot = hot_fcov * 10 ** (log_cool - log_hot) * jnp.exp(-((log_wave_um - log_hot) / hot_width) ** 2)
    total = cool + hot
    norm_index = jnp.argmin(jnp.abs(wave - 12000.0))
    l_torus = 2.5 * l_agn * fcov
    torus = l_torus / 12000.0 * total / jnp.maximum(total[norm_index], 1e-30)
    si_em_ampl = 0.4
    si_abs_ampl = si_em_ampl * si_ratio
    si_spec = l_torus / 12000.0 * si * (
        si_em_ampl * jnp.exp(-0.5 * ((wave - si_em_lam) / si_em_width) ** 2)
        - si_abs_ampl * jnp.exp(-0.5 * ((wave - si_abs_lam) / si_abs_width) ** 2)
    )
    return torus + si_spec


def _feii_component(wave, template_wave, template_flux, norm, fwhm_kms, shift_frac):
    """Broaden, shift, and normalize the Fe II template contribution."""
    template_on_wave = jnp.interp(wave, template_wave, jnp.maximum(template_flux, 0.0), left=0.0, right=0.0)
    sigma_kms = jnp.maximum(fwhm_kms / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0))), 10.0)
    return norm * _shift_and_broaden_single_spectrum_lnlam(jnp.log(wave), template_on_wave, C_KMS * shift_frac, sigma_kms)


def _line_gaussians(wave, line_wave, line_lumin, width_kms):
    """Evaluate a summed Gaussian emission-line template with one shared width."""
    fwhm_to_sigma_conversion = 1 / (2 * jnp.sqrt(2 * jnp.log(2)))
    width_wave = line_wave * (width_kms * 1000.0) / 299792458.0
    sigma = width_wave * fwhm_to_sigma_conversion
    z = (wave[:, None] - line_wave[None, :]) / jnp.maximum(sigma[None, :], 1e-12)
    norm = 5100.0 / jnp.sqrt(jnp.pi * sigma**2)
    return jnp.sum(line_lumin[None, :] * jnp.exp(-0.5 * z * z) * norm[None, :], axis=1)


def _balmer_continuum_jax(wave, balmer_norm, balmer_te, balmer_tau, balmer_vel):
    """Evaluate the broadened Balmer continuum template."""
    lam_be = 3646.0
    h_c_per_k_B = 1.439e7
    bb = (wave**-5) / jnp.expm1(jnp.clip(h_c_per_k_B / (balmer_te * wave), 1e-9, 700.0))
    bb0 = (lam_be**-5) / jnp.expm1(jnp.clip(h_c_per_k_B / (balmer_te * lam_be), 1e-9, 700.0))
    tau = balmer_tau * (wave / lam_be) ** 3
    bc = balmer_norm * (1.0 - jnp.exp(-tau)) * bb / jnp.maximum(bb0, 1e-30)
    bc = jnp.where(wave <= lam_be, bc, 0.0)
    return _shift_and_broaden_single_spectrum_lnlam(jnp.log(wave), bc, 0.0, balmer_vel)


def _igm_transmission(wavelength, redshift):
    """Approximate IGM transmission following the GRAHSP-style implementation."""
    n_transitions_low = 10
    n_transitions_max = 31
    gamma = 0.2788
    n0 = 0.25
    lambda_limit = 91.2
    wavelength = jnp.asarray(wavelength)
    n_arr = jnp.arange(n_transitions_max, dtype=jnp.float64)
    lambda_n = jnp.where(n_arr >= 2, lambda_limit / (1.0 - 1.0 / jnp.maximum(n_arr * n_arr, 1.0)), 1.0)
    z_n = wavelength[None, :] / lambda_n[:, None] - 1.0
    fact = jnp.array([1., 1., 1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373, 0.0283], dtype=jnp.float64)
    tau_a = jnp.where(redshift <= 4, 0.00211 * (1.0 + redshift) ** 3.7, 0.00058 * (1.0 + redshift) ** 4.5)
    tau_n = jnp.zeros((n_transitions_max, wavelength.size), dtype=jnp.float64)
    tau2 = jnp.where(redshift <= 4, 0.00211 * (1.0 + z_n[2]) ** 3.7, 0.00058 * (1.0 + z_n[2]) ** 4.5)
    tau_n = tau_n.at[2, :].set(tau2)

    n_eval = jnp.arange(3, n_transitions_max, dtype=jnp.float64)
    z_eval = z_n[3:, :]
    fact_eval = jnp.where(n_eval <= 9.0, fact[n_eval.astype(jnp.int32)], 0.0)
    val_le5 = jnp.where(
        z_eval < 3.0,
        tau_a * fact_eval[:, None] * (0.25 * (1.0 + z_eval)) ** (1.0 / 3.0),
        tau_a * fact_eval[:, None] * (0.25 * (1.0 + z_eval)) ** (1.0 / 6.0),
    )
    val_6_9 = tau_a * fact_eval[:, None] * (0.25 * (1.0 + z_eval)) ** (1.0 / 3.0)
    tau9 = tau_a * fact[9] * (0.25 * (1.0 + z_n[9])) ** (1.0 / 3.0)
    val_gt9 = tau9[None, :] * 720.0 / (n_eval[:, None] * (n_eval[:, None] * n_eval[:, None] - 1.0))
    val_eval = jnp.where(
        n_eval[:, None] <= 5.0,
        val_le5,
        jnp.where(n_eval[:, None] <= 9.0, val_6_9, val_gt9),
    )
    tau_n = tau_n.at[3:, :].set(jnp.where(z_eval >= redshift, 0.0, val_eval))
    tau_n = tau_n.at[2, :].set(jnp.where(z_n[2] >= redshift, 0.0, tau_n[2]))
    z_l = wavelength / lambda_limit - 1.0
    w = z_l < redshift
    tau_l_igm = jnp.where(w, 0.805 * (1.0 + z_l) ** 3 * (1.0 / (1.0 + z_l) - 1.0 / (1.0 + redshift)), 0.0)
    term1 = gamma - jnp.exp(-1.0)
    n = jnp.arange(n_transitions_low - 1, dtype=jnp.float64)
    factorial_n = jnp.exp(jax.scipy.special.gammaln(n + 1.0))
    term2 = jnp.sum(jnp.power(-1.0, n) / (factorial_n * (2.0 * n - 1.0)))
    term3 = (1.0 + redshift) * (wavelength / lambda_limit) ** 1.5 - (wavelength / lambda_limit) ** 2.5
    ni = jnp.arange(1, n_transitions_low, dtype=jnp.float64)
    factorial_ni = jnp.exp(jax.scipy.special.gammaln(ni + 1.0))
    coeff = 2.0 * jnp.power(-1.0, ni) / (factorial_ni * ((6.0 * ni - 5.0) * (2.0 * ni - 1.0)))
    wl_ratio = wavelength / lambda_limit
    term4_terms = coeff[:, None] * (
        (1.0 + redshift) ** (2.5 - (3.0 * ni[:, None])) * wl_ratio[None, :] ** (3.0 * ni[:, None])
        - wl_ratio[None, :] ** 2.5
    )
    term4 = jnp.sum(term4_terms, axis=0)
    tau_l_lls = jnp.where(w, n0 * ((term1 - term2) * term3 - term4), 0.0)
    tau_taun = jnp.sum(tau_n[2:n_transitions_max, :], axis=0)
    lambda_min_igm = (1.0 + redshift) * 70.0
    weight = jnp.where(wavelength < lambda_min_igm, (wavelength / lambda_min_igm) ** 2, 1.0)
    return jnp.exp(-tau_taun - tau_l_igm - tau_l_lls) * weight


def _attenuation_curve(wave_rest, opt_index, nir_index, norm, lam_break):
    """Return the broken power-law attenuation curve in magnitudes."""
    return norm * (wave_rest / lam_break) ** jnp.where(wave_rest < lam_break, opt_index, nir_index)


def _apply_biattenuation(wave_rest, gal_spec, agn_spec, ebv_gal, ebv_agn, opt_index, nir_index, norm, lam_break):
    """Apply differential attenuation to host and AGN components."""
    curve = _attenuation_curve(wave_rest, opt_index, nir_index, norm, lam_break)
    gal_att = gal_spec * 10 ** (ebv_gal * curve / -2.5)
    agn_att = agn_spec * 10 ** ((ebv_gal + ebv_agn) * curve / -2.5)
    host_absorbed = jnp.clip(gal_spec - gal_att, 0.0, None)
    dust_luminosity = jnp.clip(jnp.trapezoid(host_absorbed, wave_rest), 0.0, None)
    return gal_att, agn_att, host_absorbed, dust_luminosity


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
    """Project an observed-frame spectrum through all prepared filters at once."""
    interp_indices = jnp.asarray(np.asarray(packed_filters.interp_indices, dtype=np.int32))
    interp_weight = _np_to_jnp(packed_filters.interp_weight)
    transmission = _np_to_jnp(packed_filters.transmission)
    work_wave = _np_to_jnp(packed_filters.work_wave)
    effective_wavelength = _np_to_jnp(packed_filters.effective_wavelength)
    valid_mask = _bool_to_jnp(packed_filters.valid_mask)

    left = obs_flux[interp_indices]
    right = obs_flux[interp_indices + 1]
    values = left * (1.0 - interp_weight) + right * interp_weight
    values = jnp.where(valid_mask, values, 0.0)
    weighted_trans = jnp.where(valid_mask, transmission, 0.0)
    weighted_wave = jnp.where(valid_mask, work_wave, 0.0)
    numer = jnp.trapezoid(values * weighted_trans, weighted_wave, axis=1)
    denom = jnp.maximum(jnp.trapezoid(weighted_trans, weighted_wave, axis=1), 1e-30)
    f_lambda = numer / denom
    return 1e-10 / 299792458.0 * 1e29 * effective_wavelength * effective_wavelength * f_lambda


def _interp_rest_sed(target_wave, source_wave, source_sed):
    """Interpolate one rest-frame SED onto a new wavelength grid."""
    return jnp.interp(target_wave, source_wave, source_sed, left=0.0, right=0.0)


def _interp_dale_template(alpha, alpha_grid, dust_lumin_grid):
    """Interpolate the Dale 2014 host-dust grid in alpha."""
    alpha = jnp.clip(alpha, jnp.min(alpha_grid), jnp.max(alpha_grid))
    return jax.vmap(lambda row: jnp.interp(alpha, alpha_grid, row))(dust_lumin_grid.T)


def _host_dust_emission(context: ModelContext, dust_luminosity, dust_alpha):
    """Convert absorbed host luminosity into a Dale-template dust SED."""
    dust_wave = _np_to_jnp(context.templates.dust_wave)
    dust_alpha_grid = _np_to_jnp(context.templates.dust_alpha_grid)
    dust_lumin_grid = _np_to_jnp(context.templates.dust_lumin)
    dust_template = _interp_dale_template(dust_alpha, dust_alpha_grid, dust_lumin_grid)
    dust_rest_native = jnp.clip(dust_luminosity, 0.0, None) * jnp.clip(dust_template, 0.0, None)
    return _interp_rest_sed(_np_to_jnp(context.rest_wave), dust_wave, dust_rest_native)


def _lnu_lsun_per_hz_to_llambda_w_per_a(wave_a, lnu_lsun_per_hz):
    """Convert DSPS `Lnu` in Lsun/Hz to `Llambda` in W/Angstrom."""
    wave_m = jnp.maximum(wave_a, 1e-12) * 1.0e-10
    lnu_w_per_hz = lnu_lsun_per_hz * L_SUN_W
    return lnu_w_per_hz * C_MS / (wave_m * wave_m) * 1.0e-10


def _build_diffstar_host(context: ModelContext, prior_config: dict[str, Any]):
    """Build the host-galaxy SED from Diffstar SFH and a precomputed SSP basis."""
    ssp_lgmet = _np_to_jnp(context.ssp_data.ssp_lgmet)
    ssp_lg_age_gyr = _np_to_jnp(context.ssp_data.ssp_lg_age_gyr)
    host_basis_rest = _np_to_jnp(context.host_basis.rest_llambda)
    surviving_frac_by_age = _np_to_jnp(context.host_basis.surviving_frac_by_age)
    gal_t_table = _np_to_jnp(context.gal_t_table)
    t_obs_gyr = jnp.asarray(context.t_obs_gyr, dtype=jnp.float64)

    log_stellar_mass = _sample_log_stellar_mass(prior_config)

    u_params = {}
    for key in DEFAULT_DIFFSTAR_U_PARAMS._fields:
        default_loc = float(np.asarray(getattr(DEFAULT_DIFFSTAR_U_PARAMS, key)))
        u_params[key] = numpyro.sample(key, dist.Normal(*_cfg_mean_scale(prior_config, key, default_loc, 1.0)))
    bounded = get_bounded_diffstar_params(DiffstarUParams(**u_params))
    base_history = calc_sfh_singlegal(bounded, DEFAULT_MAH_PARAMS, gal_t_table, return_smh=True)

    gal_lgmet = numpyro.sample("gal_lgmet", dist.Normal(*_cfg_mean_scale(prior_config, "gal_lgmet", -0.3, 0.5)))
    gal_lgmet_scatter = numpyro.sample("gal_lgmet_scatter", dist.HalfNormal(_cfg_halfnorm(prior_config, "gal_lgmet_scatter", 0.2)))
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
    scaled_sfh = base_history.sfh * sfh_scale
    scaled_smh = base_history.smh * sfh_scale
    host_rest = target_formed_mass * jnp.sum(
        host_basis_rest * host_weights.reshape((*host_weights.shape, 1)),
        axis=(0, 1),
    )
    return {
        "host_rest": host_rest,
        "log_stellar_mass": log_stellar_mass,
        "host_age_weights": host_weights_info.age_weights,
        "host_lgmet_weights": host_weights_info.lgmet_weights,
        "host_ssp_weights": host_weights,
        "surviving_mass_fraction": surviving_mass_fraction,
        "formed_mass": target_formed_mass,
        "sfh_scale": sfh_scale,
        "gal_lgmet": gal_lgmet,
        "gal_lgmet_scatter": gal_lgmet_scatter,
        "gal_sfr_table": scaled_sfh,
        "gal_smh_table": scaled_smh,
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
    }


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
        "gal_sfr_table": jnp.zeros_like(gal_t_table),
        "gal_smh_table": jnp.zeros_like(gal_t_table),
        "ssp_lg_age_gyr": ssp_lg_age_gyr,
        "ssp_lgmet": ssp_lgmet,
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
    z = (obs_fluxes - model_fluxes) / (jnp.sqrt(2.0) * jnp.maximum(total_variance, 1e-30))
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


def photometric_loglike(pred_fluxes, obs_fluxes, obs_errors, upper_limits, data_mask, systematics_width, intrinsic_scatter, student_t_df, agn_component, agn_bol_lum_w, agn_nev, variability_uncertainty, attenuation_model_uncertainty, transmitted_fraction, lyman_break_uncertainty, filter_wavelength, redshift):
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
        ly_unc = jnp.where(filter_wavelength / (1.0 + redshift) < 150.0, 1.0e8, 0.0)
        sys_variance = sys_variance + (ly_unc * pred_fluxes) ** 2
    total_variance = jnp.nan_to_num(obs_variance + sys_variance + var_variance, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    scale = jnp.sqrt(jnp.clip(total_variance, 1e-30, 1.0e60))
    student = dist.StudentT(df=student_t_df, loc=pred_fluxes, scale=scale)
    logl_data = jnp.sum(jnp.where(data_mask, student.log_prob(obs_fluxes), 0.0))
    logl_lim = jnp.sum(jnp.where(upper_limits, -0.5 * _chi2_upper_limit(obs_fluxes, pred_fluxes, total_variance), 0.0))
    invalid = ~(jnp.isfinite(pred_fluxes) & jnp.isfinite(scale) & jnp.isfinite(obs_fluxes) & jnp.isfinite(obs_errors))
    penalty = -1.0e6 * jnp.sum(invalid.astype(jnp.float64))
    return logl_data + logl_lim + penalty


def grahsp_photometric_model(context: ModelContext, include_components: bool = False):
    """NumPyro model for one grahspj photometric fit or predictive expansion."""
    cfg = context.fit_config
    prior_config = cfg.prior_config
    rest_wave = _np_to_jnp(context.rest_wave)
    obs_wave = _np_to_jnp(context.obs_wave)
    feii_wave = _np_to_jnp(context.templates.feii_wave)
    feii_lumin = _np_to_jnp(context.templates.feii_lumin)
    line_wave = _np_to_jnp(context.templates.line_wave)
    line_blagn = _np_to_jnp(context.templates.line_blagn)
    line_sy2 = _np_to_jnp(context.templates.line_sy2)
    line_liner = _np_to_jnp(context.templates.line_liner)
    filter_wavelength = _np_to_jnp(np.array([f.effective_wavelength for f in context.filters], dtype=float))
    obs_fluxes = _np_to_jnp(context.fluxes)
    obs_errors = _np_to_jnp(context.errors)
    upper_limits = _bool_to_jnp(context.upper_limits)
    data_mask = _bool_to_jnp(context.data_mask)
    positive_detected_mask = _bool_to_jnp(context.positive_detected_mask)
    dust_alpha_grid = np.asarray(context.templates.dust_alpha_grid, dtype=float)
    fit_host = bool(cfg.galaxy.fit_host)
    fit_agn = bool(cfg.agn.fit_agn)
    spatial_scale_arcsec = _np_to_jnp(context.effective_spatial_scale_arcsec)
    host_capture_enabled = bool(
        fit_host
        and cfg.likelihood.use_host_capture_model
        and np.any(np.isfinite(context.effective_spatial_scale_arcsec) & (np.asarray(context.effective_spatial_scale_arcsec, dtype=float) > 0.0))
    )
    host_state = _build_diffstar_host(context, prior_config) if fit_host else _empty_host_state(context)
    host_rest = host_state["host_rest"]
    if fit_host:
        gal_v_kms = numpyro.sample("gal_v_kms", dist.Normal(*_cfg_norm(prior_config, "gal_v_kms", 0.0, 150.0)))
        gal_sigma_kms = numpyro.sample("gal_sigma_kms", dist.HalfNormal(_cfg_halfnorm(prior_config, "gal_sigma_kms", 150.0)))
        host_rest = _shift_and_broaden_single_spectrum_lnlam(jnp.log(rest_wave), host_rest, gal_v_kms, gal_sigma_kms)
    else:
        gal_v_kms = jnp.asarray(0.0, dtype=jnp.float64)
        gal_sigma_kms = jnp.asarray(1.0, dtype=jnp.float64)

    if fit_agn and fit_host:
        fracagn_5100 = numpyro.sample(
            "fracAGN_5100",
            dist.TruncatedNormal(
                *_cfg_norm(prior_config, "fracAGN_5100", 0.3, 0.25),
                low=1.0e-4,
                high=0.999,
            ),
        )
        host_llambda_5100 = jnp.interp(5100.0, rest_wave, host_rest, left=0.0, right=0.0)
        agn_amp = jnp.clip(host_llambda_5100, 0.0, 1.0e60) * (1.0 / (1.0 - fracagn_5100) - 1.0) * 5100.0
        log_agn_amp = jnp.log(jnp.clip(agn_amp, 1.0e-30, 1.0e80))
    elif fit_agn:
        fracagn_5100 = jnp.asarray(0.999, dtype=jnp.float64)
        log_agn_amp = numpyro.sample(
            "log_agn_amp",
            dist.Normal(
                *_cfg_norm(
                    prior_config,
                    "log_agn_amp",
                    _estimate_log_agn_amp_prior_loc(context, cfg.observation.redshift),
                    2.0,
                ),
            ),
        )
        agn_amp = jnp.exp(log_agn_amp)
    else:
        fracagn_5100 = jnp.asarray(1.0e-4, dtype=jnp.float64)
        agn_amp = jnp.asarray(0.0, dtype=jnp.float64)
        log_agn_amp = jnp.log(jnp.clip(agn_amp, 1.0e-30, 1.0e80))

    agn_type = int(cfg.agn.agn_type)
    if fit_agn:
        uv_slope = numpyro.sample("uv_slope", dist.Normal(*_cfg_norm(prior_config, "uv_slope", 0.0, 0.5)))
        pl_slope = numpyro.sample("pl_slope", dist.Normal(*_cfg_norm(prior_config, "pl_slope", -1.0, 0.5)))
        pl_bend_loc = numpyro.sample("pl_bend_loc", dist.LogNormal(*_cfg_norm(prior_config, "log_pl_bend_loc", np.log(100.0), 0.3)))
        pl_bend_width = numpyro.sample("pl_bend_width", dist.LogNormal(*_cfg_norm(prior_config, "log_pl_bend_width", np.log(10.0), 0.4)))
        pl_cutoff = numpyro.sample("pl_cutoff", dist.LogNormal(*_cfg_norm(prior_config, "log_pl_cutoff", np.log(10000.0), 0.6)))
        disk_rest = _powerlaw_jax(rest_wave, agn_amp / 5100.0, uv_slope, pl_slope, 5100.0, pl_bend_loc, pl_bend_width, pl_cutoff)

        fcov = numpyro.sample("fcov", dist.Beta(2.0, 8.0))
        si = numpyro.sample("si", dist.Normal(*_cfg_norm(prior_config, "si", 0.0, 1.0)))
        cool_lam = numpyro.sample("cool_lam", dist.LogNormal(*_cfg_norm(prior_config, "log_cool_lam", np.log(17.0), 0.2)))
        cool_width = numpyro.sample("cool_width", dist.LogNormal(*_cfg_norm(prior_config, "log_cool_width", np.log(0.45), 0.2)))
        hot_lam = numpyro.sample("hot_lam", dist.LogNormal(*_cfg_norm(prior_config, "log_hot_lam", np.log(2.0), 0.3)))
        hot_width = numpyro.sample("hot_width", dist.LogNormal(*_cfg_norm(prior_config, "log_hot_width", np.log(0.5), 0.2)))
        hot_fcov = numpyro.sample("hot_fcov", dist.LogNormal(*_cfg_norm(prior_config, "log_hot_fcov", np.log(0.1), 0.8)))
        torus_rest = _torus_component(rest_wave, fcov, si, cool_lam, cool_width, hot_lam, hot_width, hot_fcov, 0.29, 9841.0, 14224.0, 1025.3, 1163.5, agn_amp)

        lines_strength = numpyro.sample("lines_strength", dist.LogNormal(*_cfg_norm(prior_config, "log_lines_strength", np.log(max(cfg.agn.lines_strength_default, 1e-3)), 0.5)))
        line_width = numpyro.sample("line_width_kms", dist.LogNormal(*_cfg_norm(prior_config, "log_line_width_kms", np.log(cfg.agn.line_width_kms_default), 0.4)))
        balmer_enabled = agn_type == 1 and (
            cfg.agn.balmer_continuum_default > 0.0
            or "log_balmer_norm" in prior_config
            or "balmer_norm" in prior_config
        )
        if balmer_enabled:
            balmer_norm = numpyro.sample(
                "balmer_norm",
                dist.LogNormal(*_cfg_norm(prior_config, "log_balmer_norm", np.log(max(cfg.agn.balmer_continuum_default, 1e-3)), 1.0)),
            )
            balmer_tau = numpyro.sample("balmer_tau", dist.LogNormal(*_cfg_norm(prior_config, "log_balmer_tau", np.log(1.0), 0.5)))
            balmer_vel = numpyro.sample("balmer_vel", dist.LogNormal(*_cfg_norm(prior_config, "log_balmer_vel", np.log(cfg.agn.line_width_kms_default), 0.4)))
        else:
            balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
            balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
            balmer_vel = jnp.asarray(float(cfg.agn.line_width_kms_default), dtype=jnp.float64)
        l_agn_lambda_5100 = agn_amp / 5100.0
        agn_bol_luminosity = agn_amp * AGN_BOLOMETRIC_CORRECTION_5100
        l_broadlines = 0.02 * l_agn_lambda_5100 * lines_strength
        l_narrowlines = 0.002 * l_agn_lambda_5100 * lines_strength

        line_bl_rest = jnp.where(
            agn_type == 1,
            _line_gaussians(rest_wave, line_wave, l_broadlines * line_blagn, line_width),
            jnp.zeros_like(rest_wave),
        )
        line_nl_rest = jnp.where(
            agn_type in (1, 2),
            _line_gaussians(rest_wave, line_wave, l_narrowlines * line_sy2, line_width),
            jnp.zeros_like(rest_wave),
        )
        line_liner_rest = jnp.where(
            agn_type == 3,
            _line_gaussians(rest_wave, line_wave, l_narrowlines * line_liner, line_width),
            jnp.zeros_like(rest_wave),
        )
        line_rest = line_bl_rest + line_nl_rest + line_liner_rest

        if agn_type == 1:
            feii_norm = numpyro.sample("feii_norm", dist.LogNormal(*_cfg_norm(prior_config, "log_feii_norm", np.log(max(cfg.agn.feii_strength_default, 1e-3)), 0.8)))
            feii_fwhm = numpyro.sample("feii_fwhm", dist.LogNormal(*_cfg_norm(prior_config, "log_feii_fwhm", np.log(cfg.agn.line_width_kms_default), 0.3)))
            feii_shift = numpyro.sample("feii_shift", dist.Normal(*_cfg_norm(prior_config, "feii_shift", 0.0, 0.01)))
            l_feii = feii_norm * l_broadlines
            feii_rest = _feii_component(rest_wave, feii_wave, feii_lumin, l_feii, feii_fwhm, feii_shift)
        else:
            feii_norm = jnp.asarray(0.0, dtype=jnp.float64)
            feii_fwhm = jnp.asarray(float(cfg.agn.line_width_kms_default), dtype=jnp.float64)
            feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
            feii_rest = jnp.zeros_like(rest_wave)
        balmer_rest = _balmer_continuum_jax(rest_wave, balmer_norm, 15000.0, balmer_tau, balmer_vel)
    else:
        uv_slope = jnp.asarray(0.0, dtype=jnp.float64)
        pl_slope = jnp.asarray(-1.0, dtype=jnp.float64)
        pl_bend_loc = jnp.asarray(100.0, dtype=jnp.float64)
        pl_bend_width = jnp.asarray(10.0, dtype=jnp.float64)
        pl_cutoff = jnp.asarray(10000.0, dtype=jnp.float64)
        disk_rest = jnp.zeros_like(rest_wave)
        fcov = jnp.asarray(0.0, dtype=jnp.float64)
        si = jnp.asarray(0.0, dtype=jnp.float64)
        cool_lam = jnp.asarray(17.0, dtype=jnp.float64)
        cool_width = jnp.asarray(0.45, dtype=jnp.float64)
        hot_lam = jnp.asarray(2.0, dtype=jnp.float64)
        hot_width = jnp.asarray(0.5, dtype=jnp.float64)
        hot_fcov = jnp.asarray(0.0, dtype=jnp.float64)
        torus_rest = jnp.zeros_like(rest_wave)
        lines_strength = jnp.asarray(0.0, dtype=jnp.float64)
        line_width = jnp.asarray(float(cfg.agn.line_width_kms_default), dtype=jnp.float64)
        balmer_norm = jnp.asarray(0.0, dtype=jnp.float64)
        balmer_tau = jnp.asarray(1.0, dtype=jnp.float64)
        balmer_vel = jnp.asarray(float(cfg.agn.line_width_kms_default), dtype=jnp.float64)
        l_agn_lambda_5100 = jnp.asarray(0.0, dtype=jnp.float64)
        agn_bol_luminosity = jnp.asarray(0.0, dtype=jnp.float64)
        line_bl_rest = jnp.zeros_like(rest_wave)
        line_nl_rest = jnp.zeros_like(rest_wave)
        line_liner_rest = jnp.zeros_like(rest_wave)
        line_rest = jnp.zeros_like(rest_wave)
        feii_norm = jnp.asarray(0.0, dtype=jnp.float64)
        feii_fwhm = jnp.asarray(float(cfg.agn.line_width_kms_default), dtype=jnp.float64)
        feii_shift = jnp.asarray(0.0, dtype=jnp.float64)
        feii_rest = jnp.zeros_like(rest_wave)
        balmer_rest = jnp.zeros_like(rest_wave)

    ebv_gal = numpyro.sample("ebv_gal", dist.HalfNormal(_cfg_halfnorm(prior_config, "ebv_gal", 0.4))) if fit_host else jnp.asarray(0.0, dtype=jnp.float64)
    ebv_agn = numpyro.sample("ebv_agn", dist.HalfNormal(_cfg_halfnorm(prior_config, "ebv_agn", 0.4))) if fit_agn else jnp.asarray(0.0, dtype=jnp.float64)
    if cfg.galaxy.use_energy_balance and fit_host:
        dust_alpha = numpyro.sample(
            "dust_alpha",
            dist.TruncatedNormal(
                *_cfg_norm(prior_config, "dust_alpha", cfg.galaxy.dust_alpha, 0.4),
                low=float(np.min(dust_alpha_grid)),
                high=float(np.max(dust_alpha_grid)),
            ),
        )
    else:
        dust_alpha = jnp.asarray(float(cfg.galaxy.dust_alpha), dtype=jnp.float64)
    if cfg.likelihood.fit_intrinsic_scatter:
        log_intrinsic_scatter = numpyro.sample(
            "log_intrinsic_scatter",
            dist.Normal(
                *_cfg_norm(
                    prior_config,
                    "log_intrinsic_scatter",
                    np.log(max(cfg.likelihood.intrinsic_scatter_default, 1.0e-8)),
                    1.0,
                ),
            ),
        )
        intrinsic_scatter = jnp.exp(log_intrinsic_scatter)
    else:
        intrinsic_scatter = jnp.asarray(float(cfg.likelihood.intrinsic_scatter_default), dtype=jnp.float64)
    if cfg.observation.fit_redshift:
        redshift = _sample_redshift(context, prior_config, cfg)
    else:
        redshift = jnp.asarray(float(cfg.observation.redshift), dtype=jnp.float64)
    luminosity_distance_m = _luminosity_distance_m_jax(
        redshift,
        cfg.galaxy.cosmology_h0,
        cfg.galaxy.cosmology_om0,
    )
    gal_att_rest, agn_att_rest, host_absorbed_rest, dust_luminosity = _apply_biattenuation(
        rest_wave,
        host_rest,
        disk_rest + torus_rest + feii_rest + line_rest + balmer_rest,
        ebv_gal,
        ebv_agn,
        -1.2,
        -3.0,
        1.2,
        1100.0,
    )
    dust_rest = jnp.where(
        cfg.galaxy.use_energy_balance and fit_host,
        _host_dust_emission(context, dust_luminosity, dust_alpha),
        jnp.zeros_like(rest_wave),
    )
    total_rest = gal_att_rest + dust_rest + agn_att_rest
    agn_rest = agn_att_rest
    igm = _igm_transmission(rest_wave, redshift)
    total_obs = _redshift_to_obs(rest_wave, total_rest * igm, obs_wave, redshift, luminosity_distance_m)
    host_obs = _redshift_to_obs(rest_wave, gal_att_rest * igm, obs_wave, redshift, luminosity_distance_m)
    transmitted_fraction = jnp.clip(total_rest / jnp.maximum(host_rest + disk_rest + torus_rest + feii_rest + line_rest + balmer_rest, 1e-30), 1e-4, 1.0)

    pred_fluxes_raw = _project_filters(total_obs, context.packed_filters)
    host_fluxes_total = _project_filters(host_obs, context.packed_filters)
    if host_capture_enabled:
        log_host_capture_scale_arcsec = numpyro.sample(
            "log_host_capture_scale_arcsec",
            dist.Normal(*_cfg_norm(prior_config, "log_host_capture_scale_arcsec", np.log(3.0), 1.0)),
        )
        host_capture_slope = numpyro.sample(
            "host_capture_slope",
            dist.LogNormal(*_cfg_norm(prior_config, "log_host_capture_slope", np.log(2.0), 0.5)),
        )
        host_capture_fraction = _host_capture_fraction(
            spatial_scale_arcsec,
            jnp.exp(log_host_capture_scale_arcsec),
            host_capture_slope,
        )
    else:
        log_host_capture_scale_arcsec = jnp.asarray(np.log(3.0), dtype=jnp.float64)
        host_capture_slope = jnp.asarray(2.0, dtype=jnp.float64)
        host_capture_fraction = jnp.ones_like(pred_fluxes_raw)
    host_fluxes = host_fluxes_total * host_capture_fraction
    pred_fluxes = pred_fluxes_raw - host_fluxes_total + host_fluxes
    need_agn_fluxes = include_components or cfg.likelihood.variability_uncertainty
    need_trans_fluxes = include_components or cfg.likelihood.attenuation_model_uncertainty
    if include_components:
        agn_obs = _redshift_to_obs(rest_wave, agn_rest * igm, obs_wave, redshift, luminosity_distance_m)
        dust_obs = _redshift_to_obs(rest_wave, dust_rest * igm, obs_wave, redshift, luminosity_distance_m)
        disk_obs = _redshift_to_obs(rest_wave, disk_rest * igm, obs_wave, redshift, luminosity_distance_m)
        torus_obs = _redshift_to_obs(rest_wave, torus_rest * igm, obs_wave, redshift, luminosity_distance_m)
        feii_obs = _redshift_to_obs(rest_wave, feii_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_obs = _redshift_to_obs(rest_wave, line_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_bl_obs = _redshift_to_obs(rest_wave, line_bl_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_nl_obs = _redshift_to_obs(rest_wave, line_nl_rest * igm, obs_wave, redshift, luminosity_distance_m)
        line_liner_obs = _redshift_to_obs(rest_wave, line_liner_rest * igm, obs_wave, redshift, luminosity_distance_m)
        balmer_obs = _redshift_to_obs(rest_wave, balmer_rest * igm, obs_wave, redshift, luminosity_distance_m)
        transmitted_fraction_obs = _redshift_scalar_to_obs(rest_wave, transmitted_fraction, obs_wave, redshift)
        agn_fluxes = _project_filters(agn_obs, context.packed_filters)
        dust_fluxes = _project_filters(dust_obs, context.packed_filters)
        disk_fluxes = _project_filters(disk_obs, context.packed_filters)
        torus_fluxes = _project_filters(torus_obs, context.packed_filters)
        feii_fluxes = _project_filters(feii_obs, context.packed_filters)
        line_fluxes = _project_filters(line_obs, context.packed_filters)
        line_bl_fluxes = _project_filters(line_bl_obs, context.packed_filters)
        line_nl_fluxes = _project_filters(line_nl_obs, context.packed_filters)
        line_liner_fluxes = _project_filters(line_liner_obs, context.packed_filters)
        balmer_fluxes = _project_filters(balmer_obs, context.packed_filters)
        trans_fluxes = _project_filters(transmitted_fraction_obs, context.packed_filters)
    else:
        if need_agn_fluxes:
            agn_obs = _redshift_to_obs(rest_wave, agn_rest * igm, obs_wave, redshift, luminosity_distance_m)
            agn_fluxes = _project_filters(agn_obs, context.packed_filters)
        else:
            agn_fluxes = jnp.zeros_like(pred_fluxes)
        if need_trans_fluxes:
            transmitted_fraction_obs = _redshift_scalar_to_obs(rest_wave, transmitted_fraction, obs_wave, redshift)
            trans_fluxes = _project_filters(transmitted_fraction_obs, context.packed_filters)
        else:
            trans_fluxes = jnp.ones_like(pred_fluxes)

    logl = photometric_loglike(
        pred_fluxes=pred_fluxes,
        obs_fluxes=obs_fluxes,
        obs_errors=obs_errors,
        upper_limits=upper_limits,
        data_mask=data_mask,
        systematics_width=cfg.likelihood.systematics_width,
        intrinsic_scatter=intrinsic_scatter,
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
    numpyro.factor("photometry_loglike", logl)
    abs_flux_scale_logprior = jnp.asarray(0.0, dtype=jnp.float64)
    if cfg.likelihood.use_absolute_flux_scale_prior:
        abs_flux_scale_logprior = _absolute_flux_scale_logprior(
            pred_fluxes=pred_fluxes,
            obs_fluxes=obs_fluxes,
            valid_mask=positive_detected_mask,
            sigma_dex=cfg.likelihood.absolute_flux_scale_prior_sigma_dex,
        )
        numpyro.factor("absolute_flux_scale_prior", abs_flux_scale_logprior)

    numpyro.deterministic("pred_fluxes", pred_fluxes)
    numpyro.deterministic("intrinsic_scatter_fit", intrinsic_scatter)
    numpyro.deterministic("log_agn_amp_fit", log_agn_amp)
    numpyro.deterministic("log_disk_luminosity_fit", _safe_log10(l_agn_lambda_5100))
    numpyro.deterministic("log_agn_bol_luminosity_fit", _safe_log10(agn_bol_luminosity))
    numpyro.deterministic("agn_variability_nev", _agn_variability_nev(agn_bol_luminosity, cfg.likelihood.agn_nev))
    numpyro.deterministic("transmitted_fraction_fluxes", trans_fluxes)
    numpyro.deterministic("host_total_fluxes", host_fluxes_total)
    numpyro.deterministic("host_capture_fraction_fluxes", host_capture_fraction)
    numpyro.deterministic("log_host_capture_scale_arcsec_fit", log_host_capture_scale_arcsec)
    numpyro.deterministic("host_capture_slope_fit", host_capture_slope)
    numpyro.deterministic("host_age_weights", host_state["host_age_weights"])
    numpyro.deterministic("host_lgmet_weights", host_state["host_lgmet_weights"])
    numpyro.deterministic("host_ssp_weights", host_state["host_ssp_weights"])
    numpyro.deterministic("gal_sfr_table", host_state["gal_sfr_table"])
    numpyro.deterministic("gal_smh_table", host_state["gal_smh_table"])
    numpyro.deterministic("formed_stellar_mass", host_state["formed_mass"])
    numpyro.deterministic("surviving_mass_fraction", host_state["surviving_mass_fraction"])
    numpyro.deterministic("gal_lgmet_fit", host_state["gal_lgmet"])
    numpyro.deterministic("gal_lgmet_scatter_fit", host_state["gal_lgmet_scatter"])
    numpyro.deterministic("log_dust_luminosity_fit", _safe_log10(dust_luminosity))
    numpyro.deterministic("dust_alpha_fit", dust_alpha)
    numpyro.deterministic("absolute_flux_scale_logprior", abs_flux_scale_logprior)
    numpyro.deterministic("rest_wave", rest_wave)
    numpyro.deterministic("obs_wave", obs_wave)
    numpyro.deterministic("redshift_fit", redshift)
    numpyro.deterministic("total_rest_sed", total_rest)
    numpyro.deterministic("agn_rest_sed", agn_rest)
    numpyro.deterministic("host_rest_sed", gal_att_rest)
    numpyro.deterministic("host_absorbed_rest_sed", host_absorbed_rest)
    numpyro.deterministic("dust_rest_sed", dust_rest)
    numpyro.deterministic("disk_rest_sed", disk_rest)
    numpyro.deterministic("torus_rest_sed", torus_rest)
    numpyro.deterministic("feii_rest_sed", feii_rest)
    numpyro.deterministic("line_rest_sed", line_rest)
    numpyro.deterministic("line_bl_rest_sed", line_bl_rest)
    numpyro.deterministic("line_nl_rest_sed", line_nl_rest)
    numpyro.deterministic("line_liner_rest_sed", line_liner_rest)
    numpyro.deterministic("balmer_rest_sed", balmer_rest)
    if include_components:
        numpyro.deterministic("agn_fluxes", agn_fluxes)
        numpyro.deterministic("host_fluxes", host_fluxes)
        numpyro.deterministic("dust_fluxes", dust_fluxes)
        numpyro.deterministic("disk_fluxes", disk_fluxes)
        numpyro.deterministic("torus_fluxes", torus_fluxes)
        numpyro.deterministic("feii_fluxes", feii_fluxes)
        numpyro.deterministic("line_fluxes", line_fluxes)
        numpyro.deterministic("line_bl_fluxes", line_bl_fluxes)
        numpyro.deterministic("line_nl_fluxes", line_nl_fluxes)
        numpyro.deterministic("line_liner_fluxes", line_liner_fluxes)
        numpyro.deterministic("balmer_fluxes", balmer_fluxes)
        numpyro.deterministic("total_obs_sed", total_obs)
        numpyro.deterministic("agn_obs_sed", agn_obs)
        numpyro.deterministic("host_obs_sed", host_obs)
        numpyro.deterministic("dust_obs_sed", dust_obs)
        numpyro.deterministic("disk_obs_sed", disk_obs)
        numpyro.deterministic("torus_obs_sed", torus_obs)
        numpyro.deterministic("feii_obs_sed", feii_obs)
        numpyro.deterministic("line_obs_sed", line_obs)
        numpyro.deterministic("line_bl_obs_sed", line_bl_obs)
        numpyro.deterministic("line_nl_obs_sed", line_nl_obs)
        numpyro.deterministic("line_liner_obs_sed", line_liner_obs)
        numpyro.deterministic("balmer_obs_sed", balmer_obs)
