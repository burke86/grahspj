from __future__ import annotations

import ast
from functools import partial
import json
import time
from pathlib import Path

import jax
import numpy as np
from numpyro.infer.util import log_density

from grahspj.config import AGNConfig, FilterSet, FitConfig, GalaxyConfig, InferenceConfig, LikelihoodConfig, Observation, PhotometryData
from grahspj.core import GRAHSPJ
from grahspj.model import grahsp_photometric_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_fairall9_payload() -> tuple[float, list[dict[str, object]]]:
    notebook_path = _repo_root() / "notebooks" / "04_fairall9_fake_photoz.ipynb"
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


def build_fairall9_fixedz_config() -> FitConfig:
    true_redshift, phot_rows = _load_fairall9_payload()
    dsps_ssp_fn = _repo_root().parent / "jaxqsofit" / "tempdata.h5"
    if not dsps_ssp_fn.is_file():
        raise FileNotFoundError(f"DSPS SSP file not found: {dsps_ssp_fn}")

    cfg = FitConfig(
        observation=Observation(
            object_id="Fairall 9 fixed-z benchmark",
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
        likelihood=LikelihoodConfig(use_host_capture_model=True),
        inference=InferenceConfig(
            map_steps=80,
            learning_rate=5e-3,
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            seed=0,
        ),
        prior_config={
            "log_stellar_mass": {"loc": 10.5, "scale": 1.0},
            "fracAGN_5100": {"loc": 0.8, "scale": 0.15},
            "ebv_gal": {"scale": 0.15},
            "ebv_agn": {"scale": 0.15},
        },
    )
    return cfg


def benchmark_log_density(num_calls: int = 200, map_steps: int = 80, progress_bar: bool = False) -> dict[str, float]:
    cfg = build_fairall9_fixedz_config()
    cfg.inference.map_steps = int(map_steps)
    fitter = GRAHSPJ(cfg)
    fitter.fit_map(steps=cfg.inference.map_steps, learning_rate=cfg.inference.learning_rate, progress_bar=progress_bar)
    params = fitter.map_result["median"]
    model = partial(grahsp_photometric_model, fitter.context, include_components=False)

    def compiled_log_density(p):
        return log_density(model, (), {}, p)[0]

    compiled = jax.jit(compiled_log_density)

    warm = compiled(params)
    jax.block_until_ready(warm)

    t0 = time.perf_counter()
    value = None
    for _ in range(int(num_calls)):
        value = compiled(params)
    jax.block_until_ready(value)
    elapsed = time.perf_counter() - t0

    evals_per_sec = float(num_calls) / elapsed
    ms_per_eval = 1e3 * elapsed / float(num_calls)
    return {
        "num_calls": int(num_calls),
        "map_steps": int(map_steps),
        "elapsed_s": float(elapsed),
        "evals_per_sec": evals_per_sec,
        "ms_per_eval": ms_per_eval,
        "log_density": float(np.asarray(value)),
    }


if __name__ == "__main__":
    result = benchmark_log_density()
    print(json.dumps(result, indent=2, sort_keys=True))
