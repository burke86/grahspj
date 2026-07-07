"""Run the jaxsedfit likelihood benchmarks used by PR benchmark workflows."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import platform
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_U_PARAMS, DiffstarUParams, calc_sfh_singlegal, get_bounded_diffstar_params
from diffstar.defaults import FB as DIFFSTAR_FB
from diffstar.defaults import LGT0 as DIFFSTAR_LGT0
from dsps.sed.ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf

from jaxsedfit.config import AGNConfig, FilterSet, FitConfig, GalaxyConfig, InferenceConfig, LikelihoodConfig, Observation, PhotometryData, PriorConfig
from jaxsedfit.core import JAXSEDFit
from jaxsedfit.filters import load_filter_curves
from jaxsedfit.model import (
    AGN_BOLOMETRIC_CORRECTION_5100,
    GRAHSP_BIATTENUATION_BREAK_A,
    GRAHSP_PL_BEND_LOC_A,
    GRAHSP_PL_BEND_WIDTH,
    GRAHSP_PL_CUTOFF_A,
    GRAHSP_SI_ABS_LAM_A,
    GRAHSP_SI_ABS_WIDTH_A,
    GRAHSP_SI_EM_LAM_A,
    GRAHSP_SI_EM_WIDTH_A,
    _apply_biattenuation,
    _balmer_continuum_jax,
    _build_nebular_components,
    _cigale_nebular_correction,
    _feii_component,
    _flux_conserving_line_gaussians,
    _host_dust_emission,
    _line_gaussians,
    _powerlaw_jax,
    _project_filters,
    _project_rest_luminosity_filters,
    _redshift_to_obs,
    _torus_component,
    grahsp_photometric_model,
    photometric_loglike,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DSP_SSP = REPO_ROOT.parent / "jaxqsofit" / "tempdata.h5"


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _percent_delta(candidate: float, baseline: float) -> float:
    return float(100.0 * (candidate - baseline) / baseline) if baseline else float("nan")


def _stderr(values: list[float]) -> float:
    return float(_stdev(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _metric_mean(row: dict[str, Any], key: str) -> float:
    return float(row.get(f"{key}_mean", row[key]))


def _metric_stderr(row: dict[str, Any], key: str) -> float:
    return float(row.get(f"{key}_stderr", 0.0))


def _percent_delta_stderr(candidate: float, baseline: float, candidate_stderr: float, baseline_stderr: float) -> float:
    if baseline == 0.0:
        return float("nan")
    term_candidate = candidate_stderr / baseline
    term_baseline = candidate * baseline_stderr / (baseline * baseline)
    return float(100.0 * np.sqrt(term_candidate * term_candidate + term_baseline * term_baseline))


def _block_until_ready_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        jax.block_until_ready(leaf)


def _preview_scalar(value: Any) -> float:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        return float("nan")
    return float(np.asarray(leaves[0]).ravel()[0])


_PHOTOMETRIC_LOGLIKE_PARAMETERS = set(inspect.signature(photometric_loglike).parameters)


def _photometric_loglike_compat(**kwargs):
    """Call photometric_loglike across benchmark base/head API revisions."""
    if "intrinsic_scatter" in _PHOTOMETRIC_LOGLIKE_PARAMETERS and "intrinsic_scatter" not in kwargs:
        kwargs["intrinsic_scatter"] = jnp.asarray(1.0e-4, dtype=jnp.float64)
    return photometric_loglike(
        **{key: value for key, value in kwargs.items() if key in _PHOTOMETRIC_LOGLIKE_PARAMETERS}
    )


def _workflow_url() -> str:
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    return f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else "local"


def _load_fairall9_payload() -> tuple[float, list[dict[str, object]]]:
    notebook_path = REPO_ROOT / "notebooks" / "04_fairall9_fake_photoz.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "phot_rows =" not in source or "true_redshift" not in source:
            continue
        module = ast.parse(source)
        true_redshift = None
        phot_rows = None
        for node in module.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name == "true_redshift":
                true_redshift = float(ast.literal_eval(node.value))
            elif name == "phot_rows":
                phot_rows = ast.literal_eval(node.value)
        if true_redshift is not None and phot_rows is not None:
            return true_redshift, phot_rows
    raise RuntimeError(f"Could not extract Fairall 9 photometry from {notebook_path}")


def _benchmark_prior_config():
    """Return priors compatible with both base and head benchmark installs."""
    flat_prior = {
        "log_stellar_mass": {"loc": 10.5, "scale": 1.0},
        "ebv_gal": {"scale": 0.15},
        "ebv_agn": {"scale": 0.15},
    }
    if "host" not in inspect.signature(PriorConfig).parameters:
        return flat_prior
    return PriorConfig(
        stellar_mass=dist.Normal(10.5, 1.0),
        host={"ebv_gal": dist.HalfNormal(0.15)},
        agn={"ebv_agn": dist.HalfNormal(0.15)},
    )


def build_fairall9_fixedz_config(dsps_ssp_fn: str | Path) -> FitConfig:
    """Build the representative fixed-z photometric benchmark config."""
    dsps_ssp_fn = Path(dsps_ssp_fn).expanduser()
    if not dsps_ssp_fn.is_file():
        raise FileNotFoundError(f"DSPS SSP file not found: {dsps_ssp_fn}")

    true_redshift, phot_rows = _load_fairall9_payload()
    return FitConfig(
        observation=Observation(
            object_id="Fairall 9 fixed-z PR benchmark",
            redshift=float(true_redshift),
            redshift_mode="fixed",
            redshift_err=0.0,
        ),
        photometry=PhotometryData(
            filter_names=[str(row["grahsp_filter"]) for row in phot_rows],
            fluxes=[float(row["flux_mjy"]) for row in phot_rows],
            errors=[float(row["err_mjy"]) for row in phot_rows],
            is_upper_limit=[False] * len(phot_rows),
            psf_fwhm_arcsec=[None if row["psf_fwhm_arcsec"] is None else float(row["psf_fwhm_arcsec"]) for row in phot_rows],
        ),
        filters=FilterSet(curves=load_filter_curves([str(row["grahsp_filter"]) for row in phot_rows])),
        galaxy=GalaxyConfig(dsps_ssp_fn=str(dsps_ssp_fn)),
        agn=AGNConfig(agn_type=1),
        likelihood=LikelihoodConfig(use_host_capture_model=False),
        inference=InferenceConfig(
            map_steps=40,
            learning_rate=5e-3,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=0,
        ),
        prior_config=_benchmark_prior_config(),
    )


def _bench_jitted(name: str, fn: Callable[[], Any], repeats: int, trials: int) -> dict[str, Any]:
    compile_start = time.perf_counter()
    compiled = jax.jit(fn)
    out = compiled()
    _block_until_ready_tree(out)
    compile_seconds = time.perf_counter() - compile_start
    trial_elapsed = []
    trial_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeats):
            out = compiled()
        _block_until_ready_tree(out)
        elapsed = time.perf_counter() - start
        trial_elapsed.append(float(elapsed))
        trial_ms.append(float(1.0e3 * elapsed / repeats))
    return {
        "name": name,
        "repeats": int(repeats),
        "trials": int(trials),
        "compile_seconds": float(compile_seconds),
        "elapsed_seconds": float(sum(trial_elapsed)),
        "trial_elapsed_seconds": trial_elapsed,
        "trial_ms_per_eval": trial_ms,
        "ms_per_eval": _mean(trial_ms),
        "ms_per_eval_stdev": _stdev(trial_ms),
        "ms_per_eval_stderr": _stderr(trial_ms),
        "value": _preview_scalar(out),
    }


def _build_component_functions(fitter: JAXSEDFit) -> dict[str, Callable[[], Any]]:
    ctx = fitter.context
    rest_wave = ctx.rest_wave_jax
    obs_wave = ctx.obs_wave_jax

    log_stellar_mass = jnp.asarray(10.0, dtype=jnp.float64)
    gal_lgmet = jnp.asarray(-0.3, dtype=jnp.float64)
    gal_lgmet_scatter = jnp.asarray(0.2, dtype=jnp.float64)
    u_defaults = {
        key: jnp.asarray(float(np.asarray(getattr(DEFAULT_DIFFSTAR_U_PARAMS, key))), dtype=jnp.float64)
        for key in DEFAULT_DIFFSTAR_U_PARAMS._fields
    }

    def host_outputs():
        bounded = get_bounded_diffstar_params(DiffstarUParams(**u_defaults))
        base_history = calc_sfh_singlegal(
            bounded,
            DEFAULT_MAH_PARAMS,
            ctx.host_basis_jax.gal_t_table,
            lgt0=DIFFSTAR_LGT0,
            fb=DIFFSTAR_FB,
            return_smh=True,
        )
        info = calc_ssp_weights_sfh_table_lognormal_mdf(
            ctx.host_basis_jax.gal_t_table,
            base_history.sfh,
            gal_lgmet,
            gal_lgmet_scatter,
            ctx.host_basis_jax.ssp_lgmet,
            ctx.host_basis_jax.ssp_lg_age_gyr,
            jnp.asarray(ctx.t_obs_gyr, dtype=jnp.float64),
        )
        surviving = jnp.clip(jnp.sum(info.age_weights * ctx.host_basis_jax.surviving_frac_by_age), 1.0e-12, 1.0)
        formed_mass = 10.0**log_stellar_mass / surviving
        host_rest = formed_mass * jnp.tensordot(info.weights, ctx.host_basis_jax.rest_llambda, axes=((0, 1), (0, 1)))
        return host_rest, formed_mass, info.weights

    host_rest, formed_mass, host_weights = jax.jit(host_outputs)()
    jax.block_until_ready(host_rest)

    agn_amp = jnp.asarray(1.0e37, dtype=jnp.float64)
    line_wave = jnp.asarray(ctx.templates.line_wave, dtype=jnp.float64)
    line_blagn = jnp.asarray(ctx.templates.line_blagn, dtype=jnp.float64)
    line_sy2 = jnp.asarray(ctx.templates.line_sy2, dtype=jnp.float64)
    feii_template = ctx.feii_template_on_rest_jax

    def host_diffstar_ssp_mix():
        return jnp.sum(host_outputs()[0])

    def host_sfh_weights_only():
        bounded = get_bounded_diffstar_params(DiffstarUParams(**u_defaults))
        base_history = calc_sfh_singlegal(
            bounded,
            DEFAULT_MAH_PARAMS,
            ctx.host_basis_jax.gal_t_table,
            lgt0=DIFFSTAR_LGT0,
            fb=DIFFSTAR_FB,
            return_smh=True,
        )
        info = calc_ssp_weights_sfh_table_lognormal_mdf(
            ctx.host_basis_jax.gal_t_table,
            base_history.sfh,
            gal_lgmet,
            gal_lgmet_scatter,
            ctx.host_basis_jax.ssp_lgmet,
            ctx.host_basis_jax.ssp_lg_age_gyr,
            jnp.asarray(ctx.t_obs_gyr, dtype=jnp.float64),
        )
        return jnp.sum(info.weights) + base_history.smh[-1]

    def agn_disk_plus_torus():
        disk = _powerlaw_jax(rest_wave, agn_amp / 5100.0, 0.0, -1.0, 5100.0, GRAHSP_PL_BEND_LOC_A, GRAHSP_PL_BEND_WIDTH, GRAHSP_PL_CUTOFF_A)
        torus = _torus_component(
            rest_wave,
            0.2,
            0.0,
            17.0,
            0.45,
            2.0,
            0.5,
            0.1,
            0.29,
            GRAHSP_SI_EM_LAM_A,
            GRAHSP_SI_ABS_LAM_A,
            GRAHSP_SI_EM_WIDTH_A,
            GRAHSP_SI_ABS_WIDTH_A,
            agn_amp,
        )
        return jnp.sum(disk + torus)

    def agn_line_gaussians_only():
        l5100 = agn_amp / 5100.0
        broad = _line_gaussians(rest_wave, line_wave, 0.02 * l5100 * line_blagn, 3000.0)
        narrow = _line_gaussians(rest_wave, line_wave, 0.002 * l5100 * line_sy2, 3000.0)
        return jnp.sum(broad + narrow)

    def agn_feii_only():
        return jnp.sum(_feii_component(rest_wave, feii_template, 5.0 * 0.02 * agn_amp / 5100.0, 3000.0, 0.0))

    def agn_balmer_only():
        return jnp.sum(_balmer_continuum_jax(rest_wave, 1.0e-6, 15000.0, 1.0, 3000.0))

    host_state = {
        "host_rest": host_rest,
        "formed_mass": formed_mass,
        "host_ssp_weights": host_weights,
        "gal_lgmet": gal_lgmet,
        "ssp_lgmet": ctx.host_basis_jax.ssp_lgmet,
    }
    neb = _build_nebular_components(ctx, host_state, host_rest, {})
    host_with_neb = host_rest + neb["absorption_rest"] + neb["emission_rest"]
    disk = _powerlaw_jax(rest_wave, agn_amp / 5100.0, 0.0, -1.0, 5100.0, GRAHSP_PL_BEND_LOC_A, GRAHSP_PL_BEND_WIDTH, GRAHSP_PL_CUTOFF_A)
    torus = _torus_component(rest_wave, 0.2, 0.0, 17.0, 0.45, 2.0, 0.5, 0.1, 0.29, GRAHSP_SI_EM_LAM_A, GRAHSP_SI_ABS_LAM_A, GRAHSP_SI_EM_WIDTH_A, GRAHSP_SI_ABS_WIDTH_A, agn_amp)
    agn_spec = disk + torus

    def nebular_lines_only():
        weights = formed_mass * host_weights
        n_ly_total = jnp.sum(weights * ctx.host_basis_jax.n_ly_per_msun)
        templates = ctx.nebular_templates_jax
        z_idx = jnp.argmin(jnp.abs(templates.z_grid - jnp.power(10.0, gal_lgmet)))
        u_idx = jnp.argmin(jnp.abs(templates.logu_grid - -2.0))
        ne_idx = jnp.argmin(jnp.abs(templates.ne_grid - 100.0))
        line_lumin = templates.line_lumin_per_photon[z_idx, u_idx, ne_idx] * n_ly_total
        return jnp.sum(_flux_conserving_line_gaussians(rest_wave, templates.line_wave_a, line_lumin, 300.0))

    def nebular_continuum_only():
        weights = formed_mass * host_weights
        n_ly_total = jnp.sum(weights * ctx.host_basis_jax.n_ly_per_msun)
        templates = ctx.nebular_templates_jax
        z_idx = jnp.argmin(jnp.abs(templates.z_grid - jnp.power(10.0, gal_lgmet)))
        u_idx = jnp.argmin(jnp.abs(templates.logu_grid - -2.0))
        ne_idx = jnp.argmin(jnp.abs(templates.ne_grid - 100.0))
        cont = jnp.interp(rest_wave, templates.continuum_wave_a, templates.continuum_lumin_per_a_per_photon[z_idx, u_idx, ne_idx], left=0.0, right=0.0)
        return jnp.sum(cont * n_ly_total * _cigale_nebular_correction(0.0, 0.0))

    def nebular_abs_lines_cont():
        built = _build_nebular_components(ctx, host_state, host_rest, {})
        return jnp.sum(built["absorption_rest"] + built["emission_rest"]) + built["dust_luminosity"]

    gal_att, agn_att, _, dust_lum = _apply_biattenuation(rest_wave, host_with_neb, agn_spec, 0.1, 0.1, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    dust_rest = _host_dust_emission(ctx, dust_lum + neb["dust_luminosity"], 2.0)
    total_rest = gal_att + agn_att + dust_rest

    def attenuation_plus_dale_dust():
        gal, agn, absorbed, dlum = _apply_biattenuation(rest_wave, host_with_neb, agn_spec, 0.1, 0.1, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
        dust = _host_dust_emission(ctx, dlum + neb["dust_luminosity"], 2.0)
        return jnp.sum(gal + agn + absorbed + dust)

    def fast_filter_projection():
        return jnp.sum(_project_rest_luminosity_filters(ctx, total_rest))

    def legacy_redshift_plus_projection():
        obs = _redshift_to_obs(rest_wave, total_rest * ctx.fixed_igm_jax, obs_wave, ctx.fixed_redshift_jax, ctx.fixed_luminosity_distance_m_jax)
        return jnp.sum(_project_filters(obs, ctx.packed_filters_jax))

    pred_fluxes = _project_rest_luminosity_filters(ctx, total_rest)

    def photometric_loglike_only():
        return _photometric_loglike_compat(
            pred_fluxes=pred_fluxes,
            obs_fluxes=jnp.asarray(ctx.fluxes, dtype=jnp.float64),
            obs_errors=jnp.asarray(ctx.errors, dtype=jnp.float64),
            upper_limits=jnp.asarray(ctx.upper_limits, dtype=bool),
            data_mask=jnp.asarray(ctx.data_mask, dtype=bool),
            systematics_width=ctx.fit_config.likelihood.systematics_width,
            likelihood_family=ctx.fit_config.likelihood.likelihood_family,
            student_t_df=ctx.fit_config.likelihood.student_t_df,
            agn_component=jnp.zeros_like(pred_fluxes),
            agn_bol_lum_w=agn_amp * AGN_BOLOMETRIC_CORRECTION_5100,
            agn_nev=ctx.fit_config.likelihood.agn_nev,
            variability_uncertainty=False,
            attenuation_model_uncertainty=False,
            transmitted_fraction=jnp.ones_like(pred_fluxes),
            lyman_break_uncertainty=False,
            filter_wavelength=ctx.filter_effective_wavelength_jax,
            redshift=ctx.fixed_redshift_jax,
        )

    return {
        "host_diffstar_ssp_mix": host_diffstar_ssp_mix,
        "host_sfh_weights_only": host_sfh_weights_only,
        "agn_disk_plus_torus": agn_disk_plus_torus,
        "agn_line_gaussians_only": agn_line_gaussians_only,
        "agn_feii_only": agn_feii_only,
        "agn_balmer_only": agn_balmer_only,
        "nebular_abs_lines_cont": nebular_abs_lines_cont,
        "nebular_lines_only": nebular_lines_only,
        "nebular_continuum_only": nebular_continuum_only,
        "attenuation_plus_dale_dust": attenuation_plus_dale_dust,
        "fast_filter_projection": fast_filter_projection,
        "legacy_redshift_plus_projection": legacy_redshift_plus_projection,
        "photometric_loglike_only": photometric_loglike_only,
    }


def _build_component_gradient_functions(fitter: JAXSEDFit) -> dict[str, Callable[[], Any]]:
    """Build scalar value-and-gradient probes for the major differentiable kernels."""
    ctx = fitter.context
    rest_wave = ctx.rest_wave_jax
    obs_wave = ctx.obs_wave_jax
    gal_lgmet = jnp.asarray(-0.3, dtype=jnp.float64)
    gal_lgmet_scatter = jnp.asarray(0.2, dtype=jnp.float64)
    u_defaults = {
        key: jnp.asarray(float(np.asarray(getattr(DEFAULT_DIFFSTAR_U_PARAMS, key))), dtype=jnp.float64)
        for key in DEFAULT_DIFFSTAR_U_PARAMS._fields
    }

    def host_outputs(log_stellar_mass):
        bounded = get_bounded_diffstar_params(DiffstarUParams(**u_defaults))
        base_history = calc_sfh_singlegal(
            bounded,
            DEFAULT_MAH_PARAMS,
            ctx.host_basis_jax.gal_t_table,
            lgt0=DIFFSTAR_LGT0,
            fb=DIFFSTAR_FB,
            return_smh=True,
        )
        info = calc_ssp_weights_sfh_table_lognormal_mdf(
            ctx.host_basis_jax.gal_t_table,
            base_history.sfh,
            gal_lgmet,
            gal_lgmet_scatter,
            ctx.host_basis_jax.ssp_lgmet,
            ctx.host_basis_jax.ssp_lg_age_gyr,
            jnp.asarray(ctx.t_obs_gyr, dtype=jnp.float64),
        )
        surviving = jnp.clip(jnp.sum(info.age_weights * ctx.host_basis_jax.surviving_frac_by_age), 1.0e-12, 1.0)
        formed_mass = 10.0**log_stellar_mass / surviving
        host_rest = formed_mass * jnp.tensordot(info.weights, ctx.host_basis_jax.rest_llambda, axes=((0, 1), (0, 1)))
        return host_rest, formed_mass, info.weights

    def host_diffstar_ssp_mix_grad(log_stellar_mass):
        return jnp.sum(host_outputs(log_stellar_mass)[0])

    def host_sfh_weights_grad(metallicity):
        bounded = get_bounded_diffstar_params(DiffstarUParams(**u_defaults))
        base_history = calc_sfh_singlegal(
            bounded,
            DEFAULT_MAH_PARAMS,
            ctx.host_basis_jax.gal_t_table,
            lgt0=DIFFSTAR_LGT0,
            fb=DIFFSTAR_FB,
            return_smh=True,
        )
        info = calc_ssp_weights_sfh_table_lognormal_mdf(
            ctx.host_basis_jax.gal_t_table,
            base_history.sfh,
            metallicity,
            gal_lgmet_scatter,
            ctx.host_basis_jax.ssp_lgmet,
            ctx.host_basis_jax.ssp_lg_age_gyr,
            jnp.asarray(ctx.t_obs_gyr, dtype=jnp.float64),
        )
        return jnp.sum(info.weights) + base_history.smh[-1]

    host_rest, formed_mass, host_weights = jax.jit(lambda: host_outputs(jnp.asarray(10.0, dtype=jnp.float64)))()
    jax.block_until_ready(host_rest)

    line_wave = jnp.asarray(ctx.templates.line_wave, dtype=jnp.float64)
    line_blagn = jnp.asarray(ctx.templates.line_blagn, dtype=jnp.float64)
    line_sy2 = jnp.asarray(ctx.templates.line_sy2, dtype=jnp.float64)
    feii_template = ctx.feii_template_on_rest_jax

    def agn_disk_plus_torus_grad(log_agn_amp):
        agn_amp = jnp.exp(log_agn_amp)
        disk = _powerlaw_jax(rest_wave, agn_amp / 5100.0, 0.0, -1.0, 5100.0, GRAHSP_PL_BEND_LOC_A, GRAHSP_PL_BEND_WIDTH, GRAHSP_PL_CUTOFF_A)
        torus = _torus_component(
            rest_wave,
            0.2,
            0.0,
            17.0,
            0.45,
            2.0,
            0.5,
            0.1,
            0.29,
            GRAHSP_SI_EM_LAM_A,
            GRAHSP_SI_ABS_LAM_A,
            GRAHSP_SI_EM_WIDTH_A,
            GRAHSP_SI_ABS_WIDTH_A,
            agn_amp,
        )
        return jnp.sum(disk + torus)

    def agn_line_gaussians_grad(log_width):
        agn_amp = jnp.asarray(1.0e37, dtype=jnp.float64)
        l5100 = agn_amp / 5100.0
        width = jnp.exp(log_width)
        broad = _line_gaussians(rest_wave, line_wave, 0.02 * l5100 * line_blagn, width)
        narrow = _line_gaussians(rest_wave, line_wave, 0.002 * l5100 * line_sy2, width)
        return jnp.sum(broad + narrow)

    def agn_feii_grad(log_fwhm):
        agn_amp = jnp.asarray(1.0e37, dtype=jnp.float64)
        return jnp.sum(_feii_component(rest_wave, feii_template, 5.0 * 0.02 * agn_amp / 5100.0, jnp.exp(log_fwhm), 0.0))

    def agn_balmer_grad(log_velocity):
        return jnp.sum(_balmer_continuum_jax(rest_wave, 1.0e-6, 15000.0, 1.0, jnp.exp(log_velocity)))

    host_state = {
        "host_rest": host_rest,
        "formed_mass": formed_mass,
        "host_ssp_weights": host_weights,
        "gal_lgmet": gal_lgmet,
        "ssp_lgmet": ctx.host_basis_jax.ssp_lgmet,
    }
    neb = _build_nebular_components(ctx, host_state, host_rest, {})
    host_with_neb = host_rest + neb["absorption_rest"] + neb["emission_rest"]
    agn_amp = jnp.asarray(1.0e37, dtype=jnp.float64)
    disk = _powerlaw_jax(rest_wave, agn_amp / 5100.0, 0.0, -1.0, 5100.0, GRAHSP_PL_BEND_LOC_A, GRAHSP_PL_BEND_WIDTH, GRAHSP_PL_CUTOFF_A)
    torus = _torus_component(rest_wave, 0.2, 0.0, 17.0, 0.45, 2.0, 0.5, 0.1, 0.29, GRAHSP_SI_EM_LAM_A, GRAHSP_SI_ABS_LAM_A, GRAHSP_SI_EM_WIDTH_A, GRAHSP_SI_ABS_WIDTH_A, agn_amp)
    agn_spec = disk + torus
    gal_att, agn_att, _, dust_lum = _apply_biattenuation(rest_wave, host_with_neb, agn_spec, 0.1, 0.1, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    dust_rest = _host_dust_emission(ctx, dust_lum + neb["dust_luminosity"], 2.0)
    total_rest = gal_att + agn_att + dust_rest
    pred_fluxes = _project_rest_luminosity_filters(ctx, total_rest)

    def nebular_lines_grad(log_width):
        weights = formed_mass * host_weights
        n_ly_total = jnp.sum(weights * ctx.host_basis_jax.n_ly_per_msun)
        templates = ctx.nebular_templates_jax
        z_idx = jnp.argmin(jnp.abs(templates.z_grid - jnp.power(10.0, gal_lgmet)))
        u_idx = jnp.argmin(jnp.abs(templates.logu_grid - -2.0))
        ne_idx = jnp.argmin(jnp.abs(templates.ne_grid - 100.0))
        line_lumin = templates.line_lumin_per_photon[z_idx, u_idx, ne_idx] * n_ly_total
        return jnp.sum(_flux_conserving_line_gaussians(rest_wave, templates.line_wave_a, line_lumin, jnp.exp(log_width)))

    def nebular_continuum_grad(scale):
        weights = formed_mass * host_weights
        n_ly_total = jnp.exp(scale) * jnp.sum(weights * ctx.host_basis_jax.n_ly_per_msun)
        templates = ctx.nebular_templates_jax
        z_idx = jnp.argmin(jnp.abs(templates.z_grid - jnp.power(10.0, gal_lgmet)))
        u_idx = jnp.argmin(jnp.abs(templates.logu_grid - -2.0))
        ne_idx = jnp.argmin(jnp.abs(templates.ne_grid - 100.0))
        cont = jnp.interp(rest_wave, templates.continuum_wave_a, templates.continuum_lumin_per_a_per_photon[z_idx, u_idx, ne_idx], left=0.0, right=0.0)
        return jnp.sum(cont * n_ly_total * _cigale_nebular_correction(0.0, 0.0))

    def attenuation_plus_dale_dust_grad(ebv_gal):
        gal, agn, absorbed, dlum = _apply_biattenuation(rest_wave, host_with_neb, agn_spec, ebv_gal, 0.1, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
        dust = _host_dust_emission(ctx, dlum + neb["dust_luminosity"], 2.0)
        return jnp.sum(gal + agn + absorbed + dust)

    def fast_filter_projection_grad(scale):
        return jnp.sum(_project_rest_luminosity_filters(ctx, total_rest * jnp.exp(scale)))

    def legacy_redshift_plus_projection_grad(scale):
        obs = _redshift_to_obs(rest_wave, total_rest * jnp.exp(scale) * ctx.fixed_igm_jax, obs_wave, ctx.fixed_redshift_jax, ctx.fixed_luminosity_distance_m_jax)
        return jnp.sum(_project_filters(obs, ctx.packed_filters_jax))

    def photometric_loglike_grad(scale):
        scaled_pred = pred_fluxes * jnp.exp(scale)
        return _photometric_loglike_compat(
            pred_fluxes=scaled_pred,
            obs_fluxes=jnp.asarray(ctx.fluxes, dtype=jnp.float64),
            obs_errors=jnp.asarray(ctx.errors, dtype=jnp.float64),
            upper_limits=jnp.asarray(ctx.upper_limits, dtype=bool),
            data_mask=jnp.asarray(ctx.data_mask, dtype=bool),
            systematics_width=ctx.fit_config.likelihood.systematics_width,
            likelihood_family=ctx.fit_config.likelihood.likelihood_family,
            student_t_df=ctx.fit_config.likelihood.student_t_df,
            agn_component=jnp.zeros_like(scaled_pred),
            agn_bol_lum_w=agn_amp * AGN_BOLOMETRIC_CORRECTION_5100,
            agn_nev=ctx.fit_config.likelihood.agn_nev,
            variability_uncertainty=False,
            attenuation_model_uncertainty=False,
            transmitted_fraction=jnp.ones_like(scaled_pred),
            lyman_break_uncertainty=False,
            filter_wavelength=ctx.filter_effective_wavelength_jax,
            redshift=ctx.fixed_redshift_jax,
        )

    def grad_probe(fn, value):
        x0 = jnp.asarray(value, dtype=jnp.float64)
        return lambda: jax.value_and_grad(fn)(x0)

    return {
        "host_diffstar_ssp_mix_grad_log_mass": grad_probe(host_diffstar_ssp_mix_grad, 10.0),
        "host_sfh_weights_grad_metallicity": grad_probe(host_sfh_weights_grad, -0.3),
        "agn_disk_plus_torus_grad_log_amp": grad_probe(agn_disk_plus_torus_grad, np.log(1.0e37)),
        "agn_line_gaussians_grad_log_width": grad_probe(agn_line_gaussians_grad, np.log(3000.0)),
        "agn_feii_grad_log_fwhm": grad_probe(agn_feii_grad, np.log(3000.0)),
        "agn_balmer_grad_log_velocity": grad_probe(agn_balmer_grad, np.log(3000.0)),
        "nebular_lines_grad_log_width": grad_probe(nebular_lines_grad, np.log(300.0)),
        "nebular_continuum_grad_scale": grad_probe(nebular_continuum_grad, 0.0),
        "attenuation_plus_dale_dust_grad_ebv": grad_probe(attenuation_plus_dale_dust_grad, 0.1),
        "fast_filter_projection_grad_scale": grad_probe(fast_filter_projection_grad, 0.0),
        "legacy_redshift_plus_projection_grad_scale": grad_probe(legacy_redshift_plus_projection_grad, 0.0),
        "photometric_loglike_grad_scale": grad_probe(photometric_loglike_grad, 0.0),
    }


def run_benchmark(
    *,
    label: str,
    sha: str,
    dsps_ssp_fn: str | Path,
    map_steps: int,
    repeats: int,
    component_repeats: int,
    trials: int,
) -> dict[str, Any]:
    if repeats < 1 or component_repeats < 1:
        raise ValueError("repeats and component_repeats must be at least 1")
    if trials < 1:
        raise ValueError("trials must be at least 1")

    benchmark_start = time.perf_counter()
    phase_timings: dict[str, float] = {}

    config_start = time.perf_counter()
    cfg = build_fairall9_fixedz_config(dsps_ssp_fn)
    cfg.inference.map_steps = int(map_steps)
    phase_timings["config_build_seconds"] = time.perf_counter() - config_start

    fitter_start = time.perf_counter()
    fitter = JAXSEDFit(cfg)
    phase_timings["fitter_init_seconds"] = time.perf_counter() - fitter_start
    phase_timings["setup_seconds"] = phase_timings["config_build_seconds"] + phase_timings["fitter_init_seconds"]

    fit_start = time.perf_counter()
    fitter.fit_map(steps=cfg.inference.map_steps, learning_rate=cfg.inference.learning_rate, progress_bar=False)
    phase_timings["map_seconds"] = time.perf_counter() - fit_start
    params = fitter.map_result["median"]

    model_setup_start = time.perf_counter()
    model = partial(grahsp_photometric_model, fitter.context, include_components=False)
    model_no_features = partial(
        grahsp_photometric_model,
        fitter.context,
        include_components=False,
        include_sed_agn_features=False,
        include_spectral_features=False,
    )
    phase_timings["model_callable_setup_seconds"] = time.perf_counter() - model_setup_start

    whole_start = time.perf_counter()
    whole = _bench_jitted("whole_log_density", lambda: log_density(model, (), {}, params)[0], repeats, trials)
    phase_timings["whole_log_density_benchmark_seconds"] = time.perf_counter() - whole_start
    whole_grad_start = time.perf_counter()
    whole_grad = _bench_jitted(
        "whole_value_and_grad",
        lambda: jax.value_and_grad(lambda p: log_density(model, (), {}, p)[0])(params),
        repeats,
        trials,
    )
    phase_timings["whole_value_and_grad_benchmark_seconds"] = time.perf_counter() - whole_grad_start
    no_features_start = time.perf_counter()
    whole_no_features = _bench_jitted(
        "whole_log_density_no_sed_agn_features",
        lambda: log_density(model_no_features, (), {}, params)[0],
        repeats,
        trials,
    )
    phase_timings["whole_no_features_benchmark_seconds"] = time.perf_counter() - no_features_start
    no_features_grad_start = time.perf_counter()
    whole_no_features_grad = _bench_jitted(
        "whole_value_and_grad_no_sed_agn_features",
        lambda: jax.value_and_grad(lambda p: log_density(model_no_features, (), {}, p)[0])(params),
        repeats,
        trials,
    )
    phase_timings["whole_no_features_value_and_grad_benchmark_seconds"] = time.perf_counter() - no_features_grad_start

    component_setup_start = time.perf_counter()
    component_functions = _build_component_functions(fitter)
    component_gradient_functions = _build_component_gradient_functions(fitter)
    phase_timings["component_callable_setup_seconds"] = time.perf_counter() - component_setup_start
    component_bench_start = time.perf_counter()
    components = [_bench_jitted(name, fn, component_repeats, trials) for name, fn in component_functions.items()]
    phase_timings["component_benchmark_seconds"] = time.perf_counter() - component_bench_start
    component_grad_bench_start = time.perf_counter()
    component_gradients = [_bench_jitted(name, fn, component_repeats, trials) for name, fn in component_gradient_functions.items()]
    phase_timings["component_gradient_benchmark_seconds"] = time.perf_counter() - component_grad_bench_start
    phase_timings["total_seconds"] = time.perf_counter() - benchmark_start

    return {
        "label": label,
        "sha": sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "n_wave": int(cfg.galaxy.n_wave),
        "n_filters": int(len(cfg.photometry.filter_names)),
        "map_steps": int(map_steps),
        "repeats": int(repeats),
        "component_repeats": int(component_repeats),
        "trials": int(trials),
        "phase_timings": {key: float(value) for key, value in phase_timings.items()},
        "setup_seconds": float(phase_timings["setup_seconds"]),
        "map_seconds": float(phase_timings["map_seconds"]),
        "whole_log_density": whole,
        "whole_value_and_grad": whole_grad,
        "whole_log_density_no_sed_agn_features": whole_no_features,
        "whole_value_and_grad_no_sed_agn_features": whole_no_features_grad,
        "components": components,
        "component_gradients": component_gradients,
    }


def _fmt_ms(value: float) -> str:
    return f"{value:.4f} ms"


def _fmt_ms_row(row: dict[str, Any]) -> str:
    mean = _metric_mean(row, "ms_per_eval")
    stderr = _metric_stderr(row, "ms_per_eval")
    if stderr <= 0.0:
        return _fmt_ms(mean)
    return f"{mean:.4f} +/- {stderr:.4f} ms"


def _fmt_percent_delta(candidate: float, baseline: float, candidate_stderr: float = 0.0, baseline_stderr: float = 0.0) -> str:
    delta = _percent_delta(candidate, baseline)
    stderr = _percent_delta_stderr(candidate, baseline, candidate_stderr, baseline_stderr)
    if not np.isfinite(stderr) or stderr <= 0.0:
        return f"{delta:+.2f}%"
    return f"{delta:+.2f}% +/- {stderr:.2f}%"


def _phase_timings(result: dict[str, Any]) -> dict[str, float]:
    phases = dict(result.get("phase_timings", {}))
    phases.setdefault("setup_seconds", float(result.get("setup_seconds", 0.0)))
    phases.setdefault("map_seconds", float(result.get("map_seconds", 0.0)))
    return phases


def _component_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in result["components"]}


def render_markdown(result: dict[str, Any], *, workflow_url: str) -> str:
    whole_ms = _metric_mean(result["whole_log_density"], "ms_per_eval")
    whole_grad_ms = _metric_mean(result["whole_value_and_grad"], "ms_per_eval")
    lines = [
        "<!-- jaxsedfit benchmark -->",
        "### jaxsedfit PR benchmark",
        "",
        "Benchmark input: fixed-z Fairall 9 photometry from `notebooks/04_fairall9_fake_photoz.ipynb`.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| commit | `{result['sha'][:12]}` |",
        f"| filters | {result['n_filters']} |",
        f"| wavelength grid | {result['n_wave']} |",
        f"| MAP steps | {result['map_steps']} |",
        f"| timing trials | {result.get('trials', 1)} x {result['repeats']} evals |",
        f"| setup time | {result['setup_seconds']:.3f} s |",
        f"| MAP time | {result['map_seconds']:.3f} s |",
        f"| whole log-density | {_fmt_ms_row(result['whole_log_density'])} |",
        f"| whole log-density compile | {result['whole_log_density'].get('compile_seconds', 0.0):.3f} s |",
        f"| whole value+grad | {_fmt_ms_row(result['whole_value_and_grad'])} |",
        f"| whole value+grad compile | {result['whole_value_and_grad'].get('compile_seconds', 0.0):.3f} s |",
        f"| whole log-density, no SED AGN features | {_fmt_ms_row(result['whole_log_density_no_sed_agn_features'])} |",
        f"| whole log-density, no SED AGN features compile | {result['whole_log_density_no_sed_agn_features'].get('compile_seconds', 0.0):.3f} s |",
        f"| whole value+grad, no SED AGN features | {_fmt_ms_row(result['whole_value_and_grad_no_sed_agn_features'])} |",
        f"| whole value+grad, no SED AGN features compile | {result['whole_value_and_grad_no_sed_agn_features'].get('compile_seconds', 0.0):.3f} s |",
        f"| total benchmark runtime | {_phase_timings(result).get('total_seconds', 0.0):.3f} s |",
        "",
        "| phase | seconds | share of total |",
        "| --- | ---: | ---: |",
    ]
    phases = _phase_timings(result)
    total_seconds = phases.get("total_seconds", 0.0)
    for name, seconds in sorted(phases.items(), key=lambda item: item[1], reverse=True):
        if name == "total_seconds":
            continue
        share = 100.0 * seconds / total_seconds if total_seconds else float("nan")
        lines.append(f"| `{name}` | {seconds:.3f} | {share:.1f}% |")
    lines.extend([
        "",
        "| component | ms/eval | share of whole |",
        "| --- | ---: | ---: |",
    ])
    for row in sorted(result["components"], key=lambda item: _metric_mean(item, "ms_per_eval"), reverse=True):
        row_ms = _metric_mean(row, "ms_per_eval")
        lines.append(f"| `{row['name']}` | {_fmt_ms_row(row)} | {100.0 * row_ms / whole_ms:.1f}% |")
    lines.extend([
        "",
        "| component gradient | ms/eval | share of whole value+grad |",
        "| --- | ---: | ---: |",
    ])
    for row in sorted(result.get("component_gradients", []), key=lambda item: _metric_mean(item, "ms_per_eval"), reverse=True):
        row_ms = _metric_mean(row, "ms_per_eval")
        lines.append(f"| `{row['name']}` | {_fmt_ms_row(row)} | {100.0 * row_ms / whole_grad_ms:.1f}% |")
    lines.extend(["", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def render_comparison_markdown(baseline: dict[str, Any], candidate: dict[str, Any], *, workflow_url: str) -> str:
    base_whole = _metric_mean(baseline["whole_log_density"], "ms_per_eval")
    cand_whole = _metric_mean(candidate["whole_log_density"], "ms_per_eval")
    base_whole_se = _metric_stderr(baseline["whole_log_density"], "ms_per_eval")
    cand_whole_se = _metric_stderr(candidate["whole_log_density"], "ms_per_eval")
    base_whole_grad = _metric_mean(baseline["whole_value_and_grad"], "ms_per_eval")
    cand_whole_grad = _metric_mean(candidate["whole_value_and_grad"], "ms_per_eval")
    base_whole_grad_se = _metric_stderr(baseline["whole_value_and_grad"], "ms_per_eval")
    cand_whole_grad_se = _metric_stderr(candidate["whole_value_and_grad"], "ms_per_eval")
    base_no_features = _metric_mean(baseline["whole_log_density_no_sed_agn_features"], "ms_per_eval")
    cand_no_features = _metric_mean(candidate["whole_log_density_no_sed_agn_features"], "ms_per_eval")
    base_no_features_se = _metric_stderr(baseline["whole_log_density_no_sed_agn_features"], "ms_per_eval")
    cand_no_features_se = _metric_stderr(candidate["whole_log_density_no_sed_agn_features"], "ms_per_eval")
    base_no_features_grad = _metric_mean(baseline["whole_value_and_grad_no_sed_agn_features"], "ms_per_eval")
    cand_no_features_grad = _metric_mean(candidate["whole_value_and_grad_no_sed_agn_features"], "ms_per_eval")
    base_no_features_grad_se = _metric_stderr(baseline["whole_value_and_grad_no_sed_agn_features"], "ms_per_eval")
    cand_no_features_grad_se = _metric_stderr(candidate["whole_value_and_grad_no_sed_agn_features"], "ms_per_eval")
    base_phases = _phase_timings(baseline)
    cand_phases = _phase_timings(candidate)
    lines = [
        "<!-- jaxsedfit benchmark -->",
        "### jaxsedfit PR benchmark",
        "",
        "Benchmark input: fixed-z Fairall 9 photometry from `notebooks/04_fairall9_fake_photoz.ipynb`.",
        "",
        "| metric | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| commit | `{baseline['sha'][:12]}` | `{candidate['sha'][:12]}` | |",
        f"| filters | {baseline['n_filters']} | {candidate['n_filters']} | |",
        f"| wavelength grid | {baseline['n_wave']} | {candidate['n_wave']} | |",
        f"| MAP steps | {baseline['map_steps']} | {candidate['map_steps']} | |",
        f"| timing trials | {baseline.get('trials', 1)} x {baseline['repeats']} evals | {candidate.get('trials', 1)} x {candidate['repeats']} evals | |",
        f"| MAP time | {baseline['map_seconds']:.3f} s | {candidate['map_seconds']:.3f} s | {_percent_delta(candidate['map_seconds'], baseline['map_seconds']):+.2f}% |",
        f"| whole log-density | {_fmt_ms_row(baseline['whole_log_density'])} | {_fmt_ms_row(candidate['whole_log_density'])} | {_fmt_percent_delta(cand_whole, base_whole, cand_whole_se, base_whole_se)} |",
        f"| whole log-density compile | {baseline['whole_log_density'].get('compile_seconds', 0.0):.3f} s | {candidate['whole_log_density'].get('compile_seconds', 0.0):.3f} s | {_percent_delta(candidate['whole_log_density'].get('compile_seconds', 0.0), baseline['whole_log_density'].get('compile_seconds', 0.0)):+.2f}% |",
        f"| whole value+grad | {_fmt_ms_row(baseline['whole_value_and_grad'])} | {_fmt_ms_row(candidate['whole_value_and_grad'])} | {_fmt_percent_delta(cand_whole_grad, base_whole_grad, cand_whole_grad_se, base_whole_grad_se)} |",
        f"| whole value+grad compile | {baseline['whole_value_and_grad'].get('compile_seconds', 0.0):.3f} s | {candidate['whole_value_and_grad'].get('compile_seconds', 0.0):.3f} s | {_percent_delta(candidate['whole_value_and_grad'].get('compile_seconds', 0.0), baseline['whole_value_and_grad'].get('compile_seconds', 0.0)):+.2f}% |",
        f"| whole log-density, no SED AGN features | {_fmt_ms_row(baseline['whole_log_density_no_sed_agn_features'])} | {_fmt_ms_row(candidate['whole_log_density_no_sed_agn_features'])} | {_fmt_percent_delta(cand_no_features, base_no_features, cand_no_features_se, base_no_features_se)} |",
        f"| whole log-density, no SED AGN features compile | {baseline['whole_log_density_no_sed_agn_features'].get('compile_seconds', 0.0):.3f} s | {candidate['whole_log_density_no_sed_agn_features'].get('compile_seconds', 0.0):.3f} s | {_percent_delta(candidate['whole_log_density_no_sed_agn_features'].get('compile_seconds', 0.0), baseline['whole_log_density_no_sed_agn_features'].get('compile_seconds', 0.0)):+.2f}% |",
        f"| whole value+grad, no SED AGN features | {_fmt_ms_row(baseline['whole_value_and_grad_no_sed_agn_features'])} | {_fmt_ms_row(candidate['whole_value_and_grad_no_sed_agn_features'])} | {_fmt_percent_delta(cand_no_features_grad, base_no_features_grad, cand_no_features_grad_se, base_no_features_grad_se)} |",
        f"| whole value+grad, no SED AGN features compile | {baseline['whole_value_and_grad_no_sed_agn_features'].get('compile_seconds', 0.0):.3f} s | {candidate['whole_value_and_grad_no_sed_agn_features'].get('compile_seconds', 0.0):.3f} s | {_percent_delta(candidate['whole_value_and_grad_no_sed_agn_features'].get('compile_seconds', 0.0), baseline['whole_value_and_grad_no_sed_agn_features'].get('compile_seconds', 0.0)):+.2f}% |",
        f"| total benchmark runtime | {base_phases.get('total_seconds', 0.0):.3f} s | {cand_phases.get('total_seconds', 0.0):.3f} s | {_percent_delta(cand_phases.get('total_seconds', 0.0), base_phases.get('total_seconds', 0.0)):+.2f}% |",
        "",
        "| phase | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(set(base_phases).intersection(cand_phases), key=lambda key: cand_phases[key], reverse=True):
        if name == "total_seconds":
            continue
        lines.append(f"| `{name}` | {base_phases[name]:.3f} s | {cand_phases[name]:.3f} s | {_percent_delta(cand_phases[name], base_phases[name]):+.2f}% |")
    lines.extend([
        "",
        "| component | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    base_components = _component_map(baseline)
    cand_components = _component_map(candidate)
    for name in sorted(set(base_components).intersection(cand_components), key=lambda key: _metric_mean(cand_components[key], "ms_per_eval"), reverse=True):
        base_ms = _metric_mean(base_components[name], "ms_per_eval")
        cand_ms = _metric_mean(cand_components[name], "ms_per_eval")
        base_se = _metric_stderr(base_components[name], "ms_per_eval")
        cand_se = _metric_stderr(cand_components[name], "ms_per_eval")
        lines.append(f"| `{name}` | {_fmt_ms_row(base_components[name])} | {_fmt_ms_row(cand_components[name])} | {_fmt_percent_delta(cand_ms, base_ms, cand_se, base_se)} |")
    lines.extend([
        "",
        "| component gradient | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    base_component_gradients = {row["name"]: row for row in baseline.get("component_gradients", [])}
    cand_component_gradients = {row["name"]: row for row in candidate.get("component_gradients", [])}
    for name in sorted(set(base_component_gradients).intersection(cand_component_gradients), key=lambda key: _metric_mean(cand_component_gradients[key], "ms_per_eval"), reverse=True):
        base_ms = _metric_mean(base_component_gradients[name], "ms_per_eval")
        cand_ms = _metric_mean(cand_component_gradients[name], "ms_per_eval")
        base_se = _metric_stderr(base_component_gradients[name], "ms_per_eval")
        cand_se = _metric_stderr(cand_component_gradients[name], "ms_per_eval")
        lines.append(f"| `{name}` | {_fmt_ms_row(base_component_gradients[name])} | {_fmt_ms_row(cand_component_gradients[name])} | {_fmt_percent_delta(cand_ms, base_ms, cand_se, base_se)} |")
    lines.extend(["", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def _run_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(
        label=args.label,
        sha=args.sha,
        dsps_ssp_fn=args.dsps_ssp_fn,
        map_steps=args.map_steps,
        repeats=args.repeats,
        component_repeats=args.component_repeats,
        trials=args.trials,
    )
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "output").write_text(render_markdown(result, workflow_url=_workflow_url()), encoding="utf-8")


def _compare_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "whole_log_density_delta_percent": _percent_delta(
            _metric_mean(candidate["whole_log_density"], "ms_per_eval"),
            _metric_mean(baseline["whole_log_density"], "ms_per_eval"),
        ),
        "map_seconds_delta_percent": _percent_delta(candidate["map_seconds"], baseline["map_seconds"]),
    }
    (args.output_dir / "benchmark-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "output").write_text(render_comparison_markdown(baseline, candidate, workflow_url=_workflow_url()), encoding="utf-8")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--sha", default=os.getenv("GITHUB_SHA", "local"))
    parser.add_argument("--dsps-ssp-fn", default=os.getenv("JAXSEDFIT_BENCH_DSPS_SSP_FN", str(DEFAULT_DSP_SSP)))
    parser.add_argument("--map-steps", type=int, default=int(os.getenv("JAXSEDFIT_BENCH_MAP_STEPS", "40")))
    parser.add_argument("--repeats", type=int, default=int(os.getenv("JAXSEDFIT_BENCH_REPEATS", "100")))
    parser.add_argument("--component-repeats", type=int, default=int(os.getenv("JAXSEDFIT_BENCH_COMPONENT_REPEATS", "100")))
    parser.add_argument("--trials", type=int, default=int(os.getenv("JAXSEDFIT_BENCH_TRIALS", "3")))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    _add_run_args(run_parser)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline-json", type=Path, required=True)
    compare_parser.add_argument("--candidate-json", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)

    argv = sys.argv[1:]
    if not argv or argv[0] not in {"run", "compare"}:
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command == "compare":
        _compare_command(args)
    else:
        _run_command(args)


if __name__ == "__main__":
    main()
