"""Run the grahspj likelihood benchmarks used by PR benchmark workflows."""

from __future__ import annotations

import argparse
import ast
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
from numpyro.infer.util import log_density

from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_U_PARAMS, DiffstarUParams, calc_sfh_singlegal, get_bounded_diffstar_params
from dsps.sed.ssp_weights import calc_ssp_weights_sfh_table_lognormal_mdf

from grahspj.config import AGNConfig, FilterSet, FitConfig, GalaxyConfig, InferenceConfig, LikelihoodConfig, Observation, PhotometryData
from grahspj.core import GRAHSPJ
from grahspj.model import (
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
            fit_redshift=False,
            redshift_err=0.0,
        ),
        photometry=PhotometryData(
            filter_names=[str(row["grahsp_filter"]) for row in phot_rows],
            fluxes=[float(row["flux_mjy"]) for row in phot_rows],
            errors=[float(row["err_mjy"]) for row in phot_rows],
            is_upper_limit=[False] * len(phot_rows),
            psf_fwhm_arcsec=[None if row["psf_fwhm_arcsec"] is None else float(row["psf_fwhm_arcsec"]) for row in phot_rows],
        ),
        filters=FilterSet(
            speclite_names={str(row["grahsp_filter"]): str(row["speclite_name"]) for row in phot_rows},
            use_grahsp_database=False,
        ),
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
        prior_config={
            "log_stellar_mass": {"loc": 10.5, "scale": 1.0},
            "ebv_gal": {"scale": 0.15},
            "ebv_agn": {"scale": 0.15},
        },
    )


def _bench_jitted(name: str, fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    compiled = jax.jit(fn)
    out = compiled()
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(repeats):
        out = compiled()
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "repeats": int(repeats),
        "elapsed_seconds": float(elapsed),
        "ms_per_eval": float(1.0e3 * elapsed / repeats),
        "value": float(np.asarray(out).ravel()[0]),
    }


def _build_component_functions(fitter: GRAHSPJ) -> dict[str, Callable[[], Any]]:
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
        base_history = calc_sfh_singlegal(bounded, DEFAULT_MAH_PARAMS, ctx.host_basis_jax.gal_t_table, return_smh=True)
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

    host_state = {"host_rest": host_rest, "formed_mass": formed_mass, "host_ssp_weights": host_weights, "gal_lgmet": gal_lgmet}
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
        return photometric_loglike(
            pred_fluxes,
            jnp.asarray(ctx.fluxes, dtype=jnp.float64),
            jnp.asarray(ctx.errors, dtype=jnp.float64),
            jnp.asarray(ctx.upper_limits, dtype=bool),
            jnp.asarray(ctx.data_mask, dtype=bool),
            ctx.fit_config.likelihood.systematics_width,
            1.0e-4,
            ctx.fit_config.likelihood.student_t_df,
            jnp.zeros_like(pred_fluxes),
            agn_amp * AGN_BOLOMETRIC_CORRECTION_5100,
            ctx.fit_config.likelihood.agn_nev,
            False,
            False,
            jnp.ones_like(pred_fluxes),
            False,
            ctx.filter_effective_wavelength_jax,
            ctx.fixed_redshift_jax,
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


def run_benchmark(
    *,
    label: str,
    sha: str,
    dsps_ssp_fn: str | Path,
    map_steps: int,
    repeats: int,
    component_repeats: int,
) -> dict[str, Any]:
    if repeats < 1 or component_repeats < 1:
        raise ValueError("repeats and component_repeats must be at least 1")

    setup_start = time.perf_counter()
    cfg = build_fairall9_fixedz_config(dsps_ssp_fn)
    cfg.inference.map_steps = int(map_steps)
    fitter = GRAHSPJ(cfg)
    setup_seconds = time.perf_counter() - setup_start

    fit_start = time.perf_counter()
    fitter.fit_map(steps=cfg.inference.map_steps, learning_rate=cfg.inference.learning_rate, progress_bar=False)
    map_seconds = time.perf_counter() - fit_start
    params = fitter.map_result["median"]

    model = partial(grahsp_photometric_model, fitter.context, include_components=False)
    model_no_features = partial(
        grahsp_photometric_model,
        fitter.context,
        include_components=False,
        include_sed_agn_features=False,
        include_spectral_features=False,
    )

    whole = _bench_jitted("whole_log_density", lambda: log_density(model, (), {}, params)[0], repeats)
    whole_no_features = _bench_jitted(
        "whole_log_density_no_sed_agn_features",
        lambda: log_density(model_no_features, (), {}, params)[0],
        repeats,
    )
    components = [_bench_jitted(name, fn, component_repeats) for name, fn in _build_component_functions(fitter).items()]

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
        "setup_seconds": float(setup_seconds),
        "map_seconds": float(map_seconds),
        "whole_log_density": whole,
        "whole_log_density_no_sed_agn_features": whole_no_features,
        "components": components,
    }


def _fmt_ms(value: float) -> str:
    return f"{value:.4f} ms"


def _component_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in result["components"]}


def render_markdown(result: dict[str, Any], *, workflow_url: str) -> str:
    whole_ms = result["whole_log_density"]["ms_per_eval"]
    lines = [
        "<!-- grahspj benchmark -->",
        "### grahspj PR benchmark",
        "",
        "Benchmark input: fixed-z Fairall 9 photometry from `notebooks/04_fairall9_fake_photoz.ipynb`.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| commit | `{result['sha'][:12]}` |",
        f"| filters | {result['n_filters']} |",
        f"| wavelength grid | {result['n_wave']} |",
        f"| MAP steps | {result['map_steps']} |",
        f"| setup time | {result['setup_seconds']:.3f} s |",
        f"| MAP time | {result['map_seconds']:.3f} s |",
        f"| whole log-density | {_fmt_ms(whole_ms)} |",
        f"| whole log-density, no SED AGN features | {_fmt_ms(result['whole_log_density_no_sed_agn_features']['ms_per_eval'])} |",
        "",
        "| component | ms/eval | share of whole |",
        "| --- | ---: | ---: |",
    ]
    for row in sorted(result["components"], key=lambda item: item["ms_per_eval"], reverse=True):
        lines.append(f"| `{row['name']}` | {_fmt_ms(row['ms_per_eval'])} | {100.0 * row['ms_per_eval'] / whole_ms:.1f}% |")
    lines.extend(["", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def render_comparison_markdown(baseline: dict[str, Any], candidate: dict[str, Any], *, workflow_url: str) -> str:
    base_whole = baseline["whole_log_density"]["ms_per_eval"]
    cand_whole = candidate["whole_log_density"]["ms_per_eval"]
    base_no_features = baseline["whole_log_density_no_sed_agn_features"]["ms_per_eval"]
    cand_no_features = candidate["whole_log_density_no_sed_agn_features"]["ms_per_eval"]
    lines = [
        "<!-- grahspj benchmark -->",
        "### grahspj PR benchmark",
        "",
        "Benchmark input: fixed-z Fairall 9 photometry from `notebooks/04_fairall9_fake_photoz.ipynb`.",
        "",
        "| metric | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| commit | `{baseline['sha'][:12]}` | `{candidate['sha'][:12]}` | |",
        f"| filters | {baseline['n_filters']} | {candidate['n_filters']} | |",
        f"| wavelength grid | {baseline['n_wave']} | {candidate['n_wave']} | |",
        f"| MAP steps | {baseline['map_steps']} | {candidate['map_steps']} | |",
        f"| MAP time | {baseline['map_seconds']:.3f} s | {candidate['map_seconds']:.3f} s | {_percent_delta(candidate['map_seconds'], baseline['map_seconds']):+.2f}% |",
        f"| whole log-density | {_fmt_ms(base_whole)} | {_fmt_ms(cand_whole)} | {_percent_delta(cand_whole, base_whole):+.2f}% |",
        f"| whole log-density, no SED AGN features | {_fmt_ms(base_no_features)} | {_fmt_ms(cand_no_features)} | {_percent_delta(cand_no_features, base_no_features):+.2f}% |",
        "",
        "| component | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    base_components = _component_map(baseline)
    cand_components = _component_map(candidate)
    for name in sorted(set(base_components).intersection(cand_components), key=lambda key: cand_components[key]["ms_per_eval"], reverse=True):
        base_ms = base_components[name]["ms_per_eval"]
        cand_ms = cand_components[name]["ms_per_eval"]
        lines.append(f"| `{name}` | {_fmt_ms(base_ms)} | {_fmt_ms(cand_ms)} | {_percent_delta(cand_ms, base_ms):+.2f}% |")
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
            candidate["whole_log_density"]["ms_per_eval"],
            baseline["whole_log_density"]["ms_per_eval"],
        ),
        "map_seconds_delta_percent": _percent_delta(candidate["map_seconds"], baseline["map_seconds"]),
    }
    (args.output_dir / "benchmark-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "output").write_text(render_comparison_markdown(baseline, candidate, workflow_url=_workflow_url()), encoding="utf-8")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--sha", default=os.getenv("GITHUB_SHA", "local"))
    parser.add_argument("--dsps-ssp-fn", default=os.getenv("GRAHSPJ_BENCH_DSPS_SSP_FN", str(DEFAULT_DSP_SSP)))
    parser.add_argument("--map-steps", type=int, default=int(os.getenv("GRAHSPJ_BENCH_MAP_STEPS", "40")))
    parser.add_argument("--repeats", type=int, default=int(os.getenv("GRAHSPJ_BENCH_REPEATS", "100")))
    parser.add_argument("--component-repeats", type=int, default=int(os.getenv("GRAHSPJ_BENCH_COMPONENT_REPEATS", "100")))


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
