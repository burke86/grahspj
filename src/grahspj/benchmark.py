from __future__ import annotations

# The Chimera benchmark path vendors selected filter and template files that
# originate from GRAHSP resources. Those assets are documented under
# src/grahspj/resources/ and are redistributed here with provenance notes.

import csv
import json
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import resources
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterable
import zlib

import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm

from .config import AGNConfig, EmissionLineTemplate, FeIITemplate, FilterCurve, FilterSet, FitConfig, InferenceConfig, Observation, PhotometryData
from .mplstyle import use_style

CHIMERA_FILTER_NAMES = (
    "u_sdss",
    "r_sdss",
    "i_sdss",
    "z_sdss",
    "J_2mass",
    "H_2mass",
    "Ks_2mass",
    "IRAC1",
    "IRAC2",
)
DEFAULT_RANDOM_SEED = 20231011
DEFAULT_MAX_WEIGHTED_MAE = 3.0
DEFAULT_MAX_ABS_WEIGHTED_BIAS = 2.0
DEFAULT_MIN_FINITE_FRACTION = 0.95
DEFAULT_BENCHMARK_OPTAX_STEPS = 600
DEFAULT_BENCHMARK_OPTAX_LR = 1.0e-2
DEFAULT_BENCHMARK_NUTS_WARMUP = 50
DEFAULT_BENCHMARK_NUTS_SAMPLES = 50
DEFAULT_BENCHMARK_NUTS_CHAINS = 1
DEFAULT_BENCHMARK_FIT_METHOD = "optax+nuts"
_C_LIGHT_ANG_PER_S = 2.99792458e18
_CHIMERA_FILTER_FILE_MAP = {
    "IRAC1": "resources/filters/IRAC1.dat",
    "IRAC2": "resources/filters/IRAC2.dat",
}
_CHIMERA_SPECLITE_NAME_MAP = {
    "u_sdss": "sdss2010-u",
    "r_sdss": "sdss2010-r",
    "i_sdss": "sdss2010-i",
    "z_sdss": "sdss2010-z",
    "J_2mass": "twomass-J",
    "H_2mass": "twomass-H",
    "Ks_2mass": "twomass-Ks",
}
_BENCHMARK_RESOURCE_CACHE: dict[str, Any] = {}
_CHIMERA_EFFECTIVE_WAVELENGTHS_A = {
    "u_sdss": 3543.0,
    "r_sdss": 6231.0,
    "i_sdss": 7625.0,
    "z_sdss": 9134.0,
    "J_2mass": 12350.0,
    "H_2mass": 16620.0,
    "Ks_2mass": 21590.0,
    "IRAC1": 35500.0,
    "IRAC2": 44930.0,
}
_C_MS = 2.99792458e8


@dataclass
class ChimeraBenchmarkDataset:
    """Joined Chimera benchmark rows used by the stellar-mass benchmark."""
    rows: list[dict[str, Any]]


@dataclass
class _BenchmarkWorkerTask:
    """One pickleable Chimera benchmark worker task."""
    index: int
    row: dict[str, Any]
    dsps_ssp_fn: str
    base_config: FitConfig | None
    z_edges: np.ndarray
    fit_method: str
    optax_steps: int
    optax_lr: float
    nuts_warmup: int
    nuts_samples: int
    nuts_chains: int
    target_accept_prob: float


def _package_root() -> Path:
    """Return the grahspj project root from the installed package layout."""
    return Path(__file__).resolve().parents[2]


def _package_resource_path(relpath: str) -> Path:
    """Return an absolute path to a packaged benchmark resource."""
    return Path(str(resources.files("grahspj").joinpath(relpath)))


def chimera_data_dir(root: str | Path | None = None) -> Path:
    """Return the Chimera benchmark data directory."""
    base = Path(root) if root is not None else _package_root()
    return base / "data" / "chimeras-2023-10-11"


def subset_ids_path(root: str | Path | None = None) -> Path:
    """Return the deterministic Chimera subset fixture path."""
    return chimera_data_dir(root) / "benchmark_subset_ids.txt"


def _load_filter_curve_from_text(path: Path, name: str) -> FilterCurve:
    """Load a two-column transmission curve from plain text."""
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Filter file {path} does not contain two-column transmission data.")
    wave = np.asarray(data[:, 0], dtype=float)
    trans = np.asarray(data[:, 1], dtype=float)
    if trans[0] != 0.0 or trans[-1] != 0.0:
        wave = wave.copy()
        trans = trans.copy()
        trans[0] = 0.0
        trans[-1] = 0.0
    return FilterCurve(name=name, wave=wave, transmission=trans)


def _build_chimera_filter_set() -> FilterSet:
    """Build the fixed filter set used by the Chimera benchmark."""
    curves = [
        _load_filter_curve_from_text(_package_resource_path(relpath), name)
        for name, relpath in _CHIMERA_FILTER_FILE_MAP.items()
    ]
    return FilterSet(
        curves=curves,
        speclite_names=dict(_CHIMERA_SPECLITE_NAME_MAP),
        use_grahsp_database=False,
    )


def _load_bruhweiler_feii_template() -> FeIITemplate:
    """Load and normalize the vendored Bruhweiler-Verner Fe II template."""
    cache_key = "feii::vendored"
    cached = _BENCHMARK_RESOURCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    path = _package_resource_path("resources/templates/Fe_d11-m20-20.5.txt")
    data = np.loadtxt(path)
    wave_observed = np.asarray(data[:, 0], dtype=float)
    lnu = np.asarray(data[:, 1], dtype=float)
    z_shift = 4593.4 / 4575.0 - 1.0
    wave_rest = wave_observed / (1.0 + z_shift)
    llam = lnu * _C_LIGHT_ANG_PER_S / np.clip(wave_observed * wave_observed, 1e-30, None)
    norm_idx = int(np.argmin(np.abs(wave_rest - 4575.0)))
    norm = float(llam[norm_idx])
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid FeII normalization derived from {path}.")
    tmpl = FeIITemplate(
        name="BruhweilerVerner08",
        wave=wave_rest.tolist(),
        lumin=(llam / norm).tolist(),
    )
    _BENCHMARK_RESOURCE_CACHE[cache_key] = tmpl
    return tmpl


def _load_mor_netzer_emission_lines() -> EmissionLineTemplate:
    """Load the vendored Mor and Netzer emission-line template table."""
    cache_key = "emlines::vendored"
    cached = _BENCHMARK_RESOURCE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    path = _package_resource_path("resources/templates/emission_line_table.formatted")
    data = np.loadtxt(
        path,
        comments="#",
        dtype=[
            ("name", "U32"),
            ("wave", "f8"),
            ("broad", "f8"),
            ("S2", "f8"),
            ("LINER", "f8"),
        ],
    )
    tmpl = EmissionLineTemplate(
        wave=np.asarray(data["wave"], dtype=float).tolist(),
        lumin_blagn=np.asarray(data["broad"], dtype=float).tolist(),
        lumin_sy2=np.asarray(data["S2"], dtype=float).tolist(),
        lumin_liner=np.asarray(data["LINER"], dtype=float).tolist(),
    )
    _BENCHMARK_RESOURCE_CACHE[cache_key] = tmpl
    return tmpl


def _build_chimera_agn_config() -> AGNConfig:
    """Build the AGN template configuration used by the Chimera benchmark."""
    return AGNConfig(
        feii_template=_load_bruhweiler_feii_template(),
        emission_line_template=_load_mor_netzer_emission_lines(),
    )


def _fits_rows_by_id(path: Path, columns: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Read selected FITS columns and index rows by object id."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        out: dict[str, dict[str, Any]] = {}
        for i in range(len(data)):
            row_id = str(data["id"][i])
            row = {}
            for col in columns:
                value = data[col][i]
                if np.ndim(value) == 0 and hasattr(value, "item"):
                    value = value.item()
                row[col] = value
            out[row_id] = row
        return out


def load_chimera_benchmark_dataset(root: str | Path | None = None) -> ChimeraBenchmarkDataset:
    """Load and join the Chimera photometry and truth tables."""
    data_dir = chimera_data_dir(root)
    phot_path = data_dir / "chimeras-grahsp.fits"
    truth_path = data_dir / "chimeras-fullinfo.fits"
    print(f"[benchmark] Loading Chimera photometry from {phot_path}")
    print(f"[benchmark] Loading Chimera truth table from {truth_path}")
    phot_cols = ["id", "ID_COSMOS", "redshift", "chimera_QSO_weight", "resample_weight", *CHIMERA_FILTER_NAMES, *[f"{name}_err" for name in CHIMERA_FILTER_NAMES]]
    truth_cols = ["id", "MASS_MED_GAL", "resample_weight", "chimera_QSO_weight", "ID_COSMOS", "redshift"]
    phot = _fits_rows_by_id(phot_path, phot_cols)
    truth = _fits_rows_by_id(truth_path, truth_cols)
    ids = sorted(set(phot).intersection(truth))
    rows = []
    for row_id in ids:
        prow = phot[row_id]
        trow = truth[row_id]
        row = {
            "id": row_id,
            "ID_COSMOS": str(prow["ID_COSMOS"]),
            "redshift": float(prow["redshift"]),
            "chimera_QSO_weight": float(prow["chimera_QSO_weight"]),
            "resample_weight": float(trow["resample_weight"]),
            "log_stellar_mass_truth": float(trow["MASS_MED_GAL"]),
            "MASS_MED_GAL": float(trow["MASS_MED_GAL"]),
        }
        for name in CHIMERA_FILTER_NAMES:
            row[name] = float(prow[name])
            row[f"{name}_err"] = float(prow[f"{name}_err"])
        rows.append(row)
    print(f"[benchmark] Joined {len(rows)} Chimera rows")
    return ChimeraBenchmarkDataset(rows=rows)


def select_chimera_subset(dataset: ChimeraBenchmarkDataset, root: str | Path | None = None) -> list[dict[str, Any]]:
    """Select the deterministic benchmark subset from the full Chimera table."""
    fixture = subset_ids_path(root)
    print(f"[benchmark] Loading deterministic subset from {fixture}")
    subset_ids = [line.strip() for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {row["id"]: row for row in dataset.rows}
    missing = [row_id for row_id in subset_ids if row_id not in lookup]
    if missing:
        raise RuntimeError(f"Subset fixture contains IDs missing from dataset: {missing[:5]}")
    rows = [lookup[row_id] for row_id in subset_ids]
    print(f"[benchmark] Selected {len(rows)} benchmark rows")
    return rows


def build_chimera_fit_config(row: dict[str, Any], dsps_ssp_fn: str = "tempdata.h5", base_config: FitConfig | None = None) -> FitConfig:
    """Build a grahspj FitConfig for one Chimera benchmark row."""
    if base_config is None:
        cfg = FitConfig(
            observation=Observation(object_id=str(row["id"]), redshift=float(row["redshift"])),
            photometry=PhotometryData(
                filter_names=list(CHIMERA_FILTER_NAMES),
                fluxes=[float(row[name]) for name in CHIMERA_FILTER_NAMES],
                errors=[float(row[f"{name}_err"]) for name in CHIMERA_FILTER_NAMES],
                is_upper_limit=[False] * len(CHIMERA_FILTER_NAMES),
            ),
            filters=_build_chimera_filter_set(),
            agn=_build_chimera_agn_config(),
        )
    else:
        cfg = FitConfig(
            observation=Observation(
                object_id=str(row["id"]),
                redshift=float(row["redshift"]),
                fit_redshift=base_config.observation.fit_redshift,
                redshift_err=base_config.observation.redshift_err,
                ra=base_config.observation.ra,
                dec=base_config.observation.dec,
                apply_mw_deredden=base_config.observation.apply_mw_deredden,
            ),
            photometry=PhotometryData(
                filter_names=list(CHIMERA_FILTER_NAMES),
                fluxes=[float(row[name]) for name in CHIMERA_FILTER_NAMES],
                errors=[float(row[f"{name}_err"]) for name in CHIMERA_FILTER_NAMES],
                is_upper_limit=[False] * len(CHIMERA_FILTER_NAMES),
            ),
            filters=base_config.filters,
            galaxy=base_config.galaxy,
            agn=base_config.agn,
            likelihood=base_config.likelihood,
            inference=base_config.inference,
            prior_config=dict(base_config.prior_config),
        )
        if cfg.agn.feii_template.wave is None or cfg.agn.emission_line_template.wave is None:
            cfg.agn = _build_chimera_agn_config()
    dsps_path = Path(dsps_ssp_fn).expanduser()
    if not dsps_path.is_file():
        raise FileNotFoundError(f"DSPS SSP file not found: {dsps_path}")
    cfg.galaxy.dsps_ssp_fn = str(dsps_path)
    cfg.inference = InferenceConfig(
        learning_rate=DEFAULT_BENCHMARK_OPTAX_LR,
        map_steps=DEFAULT_BENCHMARK_OPTAX_STEPS,
        num_warmup=DEFAULT_BENCHMARK_NUTS_WARMUP,
        num_samples=DEFAULT_BENCHMARK_NUTS_SAMPLES,
        num_chains=DEFAULT_BENCHMARK_NUTS_CHAINS,
        target_accept_prob=cfg.inference.target_accept_prob,
        seed=DEFAULT_RANDOM_SEED,
    )
    inferred_priors = _estimate_chimera_prior_config(row)
    for key, value in inferred_priors.items():
        cfg.prior_config.setdefault(key, value)
    return cfg


def _estimate_chimera_prior_config(row: dict[str, Any]) -> dict[str, Any]:
    """Seed a simple prior configuration from one Chimera photometric row."""
    redshift = float(row["redshift"])
    nir_fluxes = np.array(
        [float(row[name]) for name in ("J_2mass", "H_2mass", "Ks_2mass", "IRAC1", "IRAC2")],
        dtype=float,
    )
    nir_fluxes = nir_fluxes[np.isfinite(nir_fluxes) & (nir_fluxes > 0.0)]
    if nir_fluxes.size == 0:
        host_flux_mjy = 0.01
    else:
        host_flux_mjy = float(np.median(np.clip(nir_fluxes, 1.0e-6, None)))

    target_obs_wave = 5100.0 * (1.0 + redshift)
    candidate_bands = [
        name
        for name in CHIMERA_FILTER_NAMES
        if np.isfinite(float(row[name])) and float(row[name]) > 0.0
    ]
    if not candidate_bands:
        optical_flux_mjy = 0.01
    else:
        nearest_band = min(candidate_bands, key=lambda name: abs(_CHIMERA_EFFECTIVE_WAVELENGTHS_A[name] - target_obs_wave))
        optical_flux_mjy = float(np.clip(float(row[nearest_band]), 1.0e-6, None))
    fracagn_loc = float(np.clip(optical_flux_mjy / max(optical_flux_mjy + host_flux_mjy, 1.0e-12), 0.02, 0.95))

    return {
        "log_stellar_mass": {"dist": "student_t", "loc": 10.0, "scale": 2.0, "df": 5.0},
        "fracAGN_5100": {"loc": fracagn_loc, "scale": 0.2},
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute one weighted quantile for finite values and weights."""
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cdf, values))


def compute_weighted_metrics(log_mass_fit: np.ndarray, log_mass_truth: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Compute weighted stellar-mass recovery metrics for benchmark reporting."""
    resid = log_mass_fit - log_mass_truth
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    bias = float(np.sum(w * resid))
    mae = float(np.sum(w * np.abs(resid)))
    rmse = float(np.sqrt(np.sum(w * resid**2)))
    medae = _weighted_quantile(np.abs(resid), np.asarray(weights, dtype=float), 0.5)
    mx = float(np.sum(w * log_mass_fit))
    my = float(np.sum(w * log_mass_truth))
    cov = float(np.sum(w * (log_mass_fit - mx) * (log_mass_truth - my)))
    vx = float(np.sum(w * (log_mass_fit - mx) ** 2))
    vy = float(np.sum(w * (log_mass_truth - my) ** 2))
    pearson = cov / np.sqrt(max(vx * vy, 1e-30))
    return {
        "weighted_bias": bias,
        "weighted_mae": mae,
        "weighted_rmse": rmse,
        "weighted_medae": medae,
        "weighted_pearson": float(pearson),
    }


def _group_metrics(rows: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    """Compute weighted metrics separately for each value of a grouping key."""
    out: dict[str, dict[str, float]] = {}
    groups = sorted(set(row[group_key] for row in rows))
    for key in groups:
        chunk = [row for row in rows if row[group_key] == key and np.isfinite(row["log_stellar_mass_fit"])]
        if not chunk:
            continue
        out[str(key)] = compute_weighted_metrics(
            np.array([row["log_stellar_mass_fit"] for row in chunk], dtype=float),
            np.array([row["log_stellar_mass_truth"] for row in chunk], dtype=float),
            np.array([row["resample_weight"] for row in chunk], dtype=float),
        )
    return out


def _redshift_bin_label(z: float, edges: np.ndarray) -> str:
    """Return a readable redshift-bin label for one value and edge array."""
    idx = min(len(edges) - 2, max(0, int(np.searchsorted(edges, z, side="right") - 1)))
    return f"{edges[idx]:.3f}-{edges[idx+1]:.3f}"


def _stable_row_seed(row_id: str) -> int:
    """Derive a stable per-row seed from the benchmark base seed and row id."""
    return int((DEFAULT_RANDOM_SEED + zlib.crc32(str(row_id).encode("utf-8"))) % (2**31 - 1))


def _reduced_chi2_for_fit(fitter: Any) -> float:
    """Compute reduced chi-square using the model's effective median variance."""
    pred = fitter.predict()
    pred_fluxes = np.median(np.asarray(pred["pred_fluxes"], dtype=float), axis=0)
    agn_fluxes = np.median(np.asarray(pred["agn_fluxes"], dtype=float), axis=0) if "agn_fluxes" in pred else np.zeros_like(pred_fluxes)
    intrinsic_scatter = float(np.median(np.asarray(pred["intrinsic_scatter_fit"], dtype=float))) if "intrinsic_scatter_fit" in pred else 0.0
    agn_variability_nev = float(np.median(np.asarray(pred["agn_variability_nev"], dtype=float))) if "agn_variability_nev" in pred else 0.0
    transmitted_fraction = (
        np.median(np.asarray(pred["transmitted_fraction_fluxes"], dtype=float), axis=0)
        if "transmitted_fraction_fluxes" in pred
        else np.ones_like(pred_fluxes)
    )
    redshift = float(np.median(np.asarray(pred["redshift_fit"], dtype=float))) if "redshift_fit" in pred else 0.0
    obs_fluxes = np.asarray(fitter.context.fluxes, dtype=float)
    obs_errors = np.asarray(fitter.context.errors, dtype=float)
    filter_wavelength = (
        np.asarray([flt.effective_wavelength for flt in fitter.context.filters], dtype=float)
        if hasattr(fitter.context, "filters")
        else np.zeros_like(pred_fluxes, dtype=float)
    )
    upper_limits = np.asarray(fitter.context.upper_limits, dtype=bool)
    cfg = getattr(fitter.config, "likelihood", None)
    if cfg is None:
        class _FallbackLikelihood:
            systematics_width = 0.0
            variability_uncertainty = False
            attenuation_model_uncertainty = False
            lyman_break_uncertainty = False

        cfg = _FallbackLikelihood()
    obs_variance = obs_errors**2 + np.maximum(intrinsic_scatter, 0.0) ** 2
    sys_variance = (float(cfg.systematics_width) * pred_fluxes) ** 2
    var_variance = np.where(bool(cfg.variability_uncertainty), agn_variability_nev * agn_fluxes**2, 0.0)
    if cfg.attenuation_model_uncertainty:
        tf = np.clip(transmitted_fraction, 1e-4, 1.0)
        neg_log = -np.log10(tf + 1e-4)
        log_unc_frac = np.minimum(-4.0 + 2.0 * neg_log, -1.0)
        att_unc = 10 ** log_unc_frac / tf
        sys_variance = sys_variance + (att_unc * pred_fluxes) ** 2
    if cfg.lyman_break_uncertainty:
        ly_unc = np.where(filter_wavelength / (1.0 + redshift) < 150.0, 1.0e8, 0.0)
        sys_variance = sys_variance + (ly_unc * pred_fluxes) ** 2
    total_variance = np.nan_to_num(obs_variance + sys_variance + var_variance, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    sigma = np.sqrt(np.clip(total_variance, 1e-30, 1.0e60))
    valid = np.isfinite(pred_fluxes) & np.isfinite(obs_fluxes) & np.isfinite(sigma) & (sigma > 0.0) & (~upper_limits)
    if not np.any(valid):
        return float("nan")
    chi2 = np.sum(((obs_fluxes[valid] - pred_fluxes[valid]) / sigma[valid]) ** 2)
    # As in plotting.py, report chi2 per valid band rather than subtracting the
    # number of sampled variables, which is not a meaningful dof estimate here.
    return float(chi2 / max(1, int(np.sum(valid))))


def _failed_benchmark_row(task: _BenchmarkWorkerTask, exc: Exception) -> dict[str, Any]:
    """Build a NaN-filled benchmark row for one failed fit."""
    enriched = dict(task.row)
    enriched["log_stellar_mass_fit"] = float("nan")
    enriched["log_stellar_mass_fit_p16"] = float("nan")
    enriched["log_stellar_mass_fit_p84"] = float("nan")
    enriched["log_stellar_mass_fit_err_lo"] = float("nan")
    enriched["log_stellar_mass_fit_err_hi"] = float("nan")
    enriched["fracAGN_5100_fit"] = float("nan")
    enriched["reduced_chi2"] = float("nan")
    enriched["residual"] = float("nan")
    enriched["redshift_bin"] = _redshift_bin_label(float(task.row["redshift"]), task.z_edges)
    enriched["fit_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


def _run_single_chimera_fit(task: _BenchmarkWorkerTask, fitter_cls=None) -> tuple[int, dict[str, Any]]:
    """Run one Chimera benchmark fit and return an ordered row payload."""
    if fitter_cls is None:
        from .core import GRAHSPJ

        fitter_cls = GRAHSPJ
    cfg = build_chimera_fit_config(row=task.row, dsps_ssp_fn=task.dsps_ssp_fn, base_config=task.base_config)
    cfg.inference.seed = _stable_row_seed(str(task.row["id"]))
    fitter = fitter_cls(cfg)
    fitter.fit(
        fit_method=task.fit_method,
        progress_bar=False,
        optax_steps=task.optax_steps,
        optax_lr=task.optax_lr,
        nuts_warmup=task.nuts_warmup,
        nuts_samples=task.nuts_samples,
        nuts_chains=task.nuts_chains,
        target_accept_prob=task.target_accept_prob,
    )
    logm_samples = np.asarray(fitter.samples["log_stellar_mass"], dtype=float).reshape(-1)
    logm16, logm50, logm84 = np.percentile(logm_samples, [16.0, 50.0, 84.0])
    fracagn_raw = (fitter.samples or {}).get("fracAGN_5100", None)
    if fracagn_raw is None:
        fracagn50 = float("nan")
    else:
        fracagn_samples = np.asarray(fracagn_raw, dtype=float).reshape(-1)
        fracagn50 = float(np.percentile(fracagn_samples, 50.0))
    log_fit = float(logm50)
    reduced_chi2 = _reduced_chi2_for_fit(fitter)
    enriched = dict(task.row)
    enriched["log_stellar_mass_fit"] = log_fit
    enriched["log_stellar_mass_fit_p16"] = float(logm16)
    enriched["log_stellar_mass_fit_p84"] = float(logm84)
    enriched["log_stellar_mass_fit_err_lo"] = float(max(0.0, log_fit - logm16))
    enriched["log_stellar_mass_fit_err_hi"] = float(max(0.0, logm84 - log_fit))
    enriched["fracAGN_5100_fit"] = fracagn50
    enriched["reduced_chi2"] = reduced_chi2
    enriched["residual"] = log_fit - enriched["log_stellar_mass_truth"]
    enriched["redshift_bin"] = _redshift_bin_label(float(task.row["redshift"]), task.z_edges)
    enriched["fit_error"] = ""
    return task.index, enriched


def _write_artifact_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-object benchmark results to a CSV artifact table."""
    fieldnames = [
        "id",
        "ID_COSMOS",
        "redshift",
        "chimera_QSO_weight",
        "resample_weight",
        "log_stellar_mass_truth",
        "log_stellar_mass_fit",
        "log_stellar_mass_fit_p16",
        "log_stellar_mass_fit_p84",
        "log_stellar_mass_fit_err_lo",
        "log_stellar_mass_fit_err_hi",
        "fracAGN_5100_fit",
        "reduced_chi2",
        "residual",
        "fit_error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write the benchmark scatter and residual diagnostic plots."""
    import matplotlib.pyplot as plt

    truth = np.array([row["log_stellar_mass_truth"] for row in rows], dtype=float)
    fit = np.array([row["log_stellar_mass_fit"] for row in rows], dtype=float)
    fit_err_lo = np.array([row["log_stellar_mass_fit_err_lo"] for row in rows], dtype=float)
    fit_err_hi = np.array([row["log_stellar_mass_fit_err_hi"] for row in rows], dtype=float)
    fracagn = np.array([row.get("fracAGN_5100_fit", np.nan) for row in rows], dtype=float)
    qso_w = np.array([row["chimera_QSO_weight"] for row in rows], dtype=float)
    finite = np.isfinite(fit) & np.isfinite(fit_err_lo) & np.isfinite(fit_err_hi)
    finite_color = finite & np.isfinite(fracagn)

    with use_style():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(
            truth[finite],
            fit[finite],
            yerr=np.vstack([fit_err_lo[finite], fit_err_hi[finite]]),
            fmt="o",
            ms=4,
            alpha=0.75,
            lw=0.8,
            capsize=2,
            color="0.7",
            zorder=1,
        )
        sc = ax.scatter(
            truth[finite_color],
            fit[finite_color],
            c=fracagn[finite_color],
            s=18,
            alpha=0.9,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            zorder=2,
        )
        lo = min(np.nanmin(truth[finite]), np.nanmin(fit[finite]))
        hi = max(np.nanmax(truth[finite]), np.nanmax(fit[finite]))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--")
        ax.set_xlabel("Chimera log stellar mass")
        ax.set_ylabel("Recovered log stellar mass")
        if np.any(finite_color):
            fig.colorbar(sc, ax=ax, label="Recovered fracAGN_5100")
        fig.tight_layout()
        fig.savefig(output_dir / "chimera_mass_scatter.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(
            qso_w[finite_color],
            (fit - truth)[finite_color],
            c=fracagn[finite_color],
            s=16,
            alpha=0.85,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xscale("log")
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xlabel("chimera_QSO_weight")
        ax.set_ylabel("Residual logM_fit - logM_truth")
        if np.any(finite_color):
            fig.colorbar(sc, ax=ax, label="Recovered fracAGN_5100")
        fig.tight_layout()
        fig.savefig(output_dir / "chimera_mass_residual_vs_qso_weight.png")
        plt.close(fig)


def run_chimera_mass_benchmark(
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
    dsps_ssp_fn: str = "tempdata.h5",
    fitter_cls=None,
    base_config: FitConfig | None = None,
    max_weighted_mae: float = DEFAULT_MAX_WEIGHTED_MAE,
    max_abs_weighted_bias: float = DEFAULT_MAX_ABS_WEIGHTED_BIAS,
    min_finite_fraction: float = DEFAULT_MIN_FINITE_FRACTION,
    limit: int | None = None,
    num_workers: int | None = None,
    fit_method: str = DEFAULT_BENCHMARK_FIT_METHOD,
    optax_steps: int = DEFAULT_BENCHMARK_OPTAX_STEPS,
    optax_lr: float = DEFAULT_BENCHMARK_OPTAX_LR,
    nuts_warmup: int = DEFAULT_BENCHMARK_NUTS_WARMUP,
    nuts_samples: int = DEFAULT_BENCHMARK_NUTS_SAMPLES,
    nuts_chains: int = DEFAULT_BENCHMARK_NUTS_CHAINS,
    target_accept_prob: float = 0.85,
) -> dict[str, Any]:
    """Run the Chimera stellar-mass recovery benchmark end to end."""
    if fitter_cls is None:
        from .core import GRAHSPJ

        fitter_cls = GRAHSPJ
    print("[benchmark] Preparing Chimera stellar-mass recovery benchmark")
    print(f"[benchmark] Using DSPS SSP file: {Path(dsps_ssp_fn).expanduser()}")
    print(
        "[benchmark] Inference settings: "
        f"fit_method={fit_method}, "
        f"optax_steps={optax_steps}, "
        f"optax_lr={optax_lr}, "
        f"nuts_warmup={nuts_warmup}, "
        f"nuts_samples={nuts_samples}, "
        f"nuts_chains={nuts_chains}, "
        f"target_accept_prob={target_accept_prob}"
    )
    dataset = load_chimera_benchmark_dataset(root=root)
    rows = select_chimera_subset(dataset, root=root)
    if limit is not None:
        limit = max(0, int(limit))
        rows = rows[:limit]
        print(f"[benchmark] Applying row limit: {len(rows)} rows")
    if not rows:
        raise ValueError("Benchmark row selection is empty after applying limit.")
    z_edges = np.quantile(np.array([row["redshift"] for row in rows], dtype=float), np.linspace(0.0, 1.0, 6))
    z_edges[0] -= 1e-9
    z_edges[-1] += 1e-9
    if num_workers is None:
        num_workers = max(1, os.cpu_count() or 1)
    num_workers = max(1, min(int(num_workers), len(rows)))
    print(f"[benchmark] Using {num_workers} worker(s)")
    tasks = [
        _BenchmarkWorkerTask(
            index=i,
            row=row,
            dsps_ssp_fn=dsps_ssp_fn,
            base_config=base_config,
            z_edges=z_edges,
            fit_method=str(fit_method),
            optax_steps=int(optax_steps),
            optax_lr=float(optax_lr),
            nuts_warmup=int(nuts_warmup),
            nuts_samples=int(nuts_samples),
            nuts_chains=int(nuts_chains),
            target_accept_prob=float(target_accept_prob),
        )
        for i, row in enumerate(rows)
    ]
    benchmark_rows: list[dict[str, Any]] = [dict() for _ in tasks]
    progress = tqdm(total=len(tasks), desc=f"Chimera {fit_method} fits", unit="obj")
    if num_workers == 1:
        for task in tasks:
            progress.set_postfix_str(str(task.row["id"]))
            try:
                idx, enriched = _run_single_chimera_fit(task, fitter_cls=fitter_cls)
            except Exception as exc:
                idx, enriched = task.index, _failed_benchmark_row(task, exc)
            benchmark_rows[idx] = enriched
            print(f"[benchmark] {task.row['id']} reduced chi2 = {enriched['reduced_chi2']:.3f}")
            progress.update(1)
    else:
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
                future_map = {pool.submit(_run_single_chimera_fit, task, fitter_cls): task for task in tasks}
                for future in as_completed(future_map):
                    task = future_map[future]
                    progress.set_postfix_str(str(task.row["id"]))
                    try:
                        idx, enriched = future.result()
                    except Exception as exc:
                        idx, enriched = task.index, _failed_benchmark_row(task, exc)
                    benchmark_rows[idx] = enriched
                    print(f"[benchmark] {task.row['id']} reduced chi2 = {enriched['reduced_chi2']:.3f}")
                    progress.update(1)
        except (PermissionError, NotImplementedError, OSError) as exc:
            print(f"[benchmark] Parallel execution unavailable ({type(exc).__name__}: {exc}); falling back to serial execution")
            for task in tasks:
                progress.set_postfix_str(str(task.row["id"]))
                try:
                    idx, enriched = _run_single_chimera_fit(task, fitter_cls=fitter_cls)
                except Exception as exc:
                    idx, enriched = task.index, _failed_benchmark_row(task, exc)
                benchmark_rows[idx] = enriched
                print(f"[benchmark] {task.row['id']} reduced chi2 = {enriched['reduced_chi2']:.3f}")
                progress.update(1)
    progress.close()
    print(f"[benchmark] Finished {fit_method} fitting")

    fit = np.array([row["log_stellar_mass_fit"] for row in benchmark_rows], dtype=float)
    truth = np.array([row["log_stellar_mass_truth"] for row in benchmark_rows], dtype=float)
    weights = np.array([row["resample_weight"] for row in benchmark_rows], dtype=float)
    finite_fraction = float(np.isfinite(fit).mean())
    finite_rows = [row for row in benchmark_rows if np.isfinite(row["log_stellar_mass_fit"])]
    if finite_rows:
        metrics = compute_weighted_metrics(
            np.array([row["log_stellar_mass_fit"] for row in finite_rows], dtype=float),
            np.array([row["log_stellar_mass_truth"] for row in finite_rows], dtype=float),
            np.array([row["resample_weight"] for row in finite_rows], dtype=float),
        )
    else:
        metrics = {
            "weighted_bias": float("nan"),
            "weighted_mae": float("nan"),
            "weighted_rmse": float("nan"),
            "weighted_medae": float("nan"),
            "weighted_pearson": float("nan"),
        }
    metrics["finite_fit_fraction"] = finite_fraction
    metrics["n_rows"] = len(benchmark_rows)
    metrics["n_finite_rows"] = len(finite_rows)
    by_redshift = _group_metrics(finite_rows, "redshift_bin")
    by_qso_weight = _group_metrics(finite_rows, "chimera_QSO_weight")

    passed = (
        metrics["weighted_mae"] <= max_weighted_mae
        and abs(metrics["weighted_bias"]) <= max_abs_weighted_bias
        and metrics["finite_fit_fraction"] >= min_finite_fraction
    )

    out: dict[str, Any] = {
        "metrics": metrics,
        "by_redshift_bin": by_redshift,
        "by_chimera_qso_weight": by_qso_weight,
        "rows": benchmark_rows,
        "passed": passed,
        "num_workers": num_workers,
        "thresholds": {
            "max_weighted_mae": max_weighted_mae,
            "max_abs_weighted_bias": max_abs_weighted_bias,
            "min_finite_fraction": min_finite_fraction,
        },
        "inference": {
            "fit_method": str(fit_method),
            "optax_steps": int(optax_steps),
            "optax_lr": float(optax_lr),
            "nuts_warmup": int(nuts_warmup),
            "nuts_samples": int(nuts_samples),
            "nuts_chains": int(nuts_chains),
            "target_accept_prob": float(target_accept_prob),
        },
    }
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[benchmark] Writing benchmark artifacts to {outdir.resolve()}")
        _write_artifact_table(outdir / "chimera_mass_recovery_rows.csv", benchmark_rows)
        with open(outdir / "chimera_mass_recovery_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "metrics": metrics,
                    "by_redshift_bin": by_redshift,
                    "by_chimera_qso_weight": by_qso_weight,
                    "num_workers": num_workers,
                    "inference": out["inference"],
                    "thresholds": out["thresholds"],
                    "passed": passed,
                },
                fh,
                indent=2,
            )
        _write_plots(outdir, benchmark_rows)
        print("[benchmark] Wrote CSV, JSON, and plot artifacts")
    print(
        "[benchmark] Metrics: "
        f"weighted_mae={metrics['weighted_mae']:.4f}, "
        f"weighted_bias={metrics['weighted_bias']:.4f}, "
        f"finite_fit_fraction={metrics['finite_fit_fraction']:.4f}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the Chimera benchmark CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run the Chimera stellar-mass recovery benchmark.")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory for benchmark artifacts.")
    parser.add_argument("--dsps-ssp-fn", default="tempdata.h5", help="Path to the DSPS SSP HDF5 file.")
    parser.add_argument("--root", default=None, help="Optional grahspj project root override.")
    parser.add_argument("--max-weighted-mae", type=float, default=DEFAULT_MAX_WEIGHTED_MAE)
    parser.add_argument("--max-abs-weighted-bias", type=float, default=DEFAULT_MAX_ABS_WEIGHTED_BIAS)
    parser.add_argument("--min-finite-fraction", type=float, default=DEFAULT_MIN_FINITE_FRACTION)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of benchmark rows to run from the deterministic subset.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional number of worker processes for parallel MAP fitting.")
    parser.add_argument("--fit-method", default=DEFAULT_BENCHMARK_FIT_METHOD, choices=["optax", "nuts", "optax+nuts", "ns"], help="Inference path to use for each benchmark object.")
    parser.add_argument("--optax-steps", type=int, default=DEFAULT_BENCHMARK_OPTAX_STEPS, help="Optax/MAP optimization steps per object.")
    parser.add_argument("--optax-lr", type=float, default=DEFAULT_BENCHMARK_OPTAX_LR, help="Optax/MAP learning rate.")
    parser.add_argument("--nuts-warmup", type=int, default=DEFAULT_BENCHMARK_NUTS_WARMUP, help="NUTS warmup steps per object.")
    parser.add_argument("--nuts-samples", type=int, default=DEFAULT_BENCHMARK_NUTS_SAMPLES, help="NUTS posterior samples per object.")
    parser.add_argument("--nuts-chains", type=int, default=DEFAULT_BENCHMARK_NUTS_CHAINS, help="NUTS chains per object.")
    parser.add_argument("--target-accept-prob", type=float, default=0.85, help="NUTS target acceptance probability.")
    args = parser.parse_args(argv)

    result = run_chimera_mass_benchmark(
        root=args.root,
        output_dir=args.output_dir,
        dsps_ssp_fn=args.dsps_ssp_fn,
        max_weighted_mae=args.max_weighted_mae,
        max_abs_weighted_bias=args.max_abs_weighted_bias,
        min_finite_fraction=args.min_finite_fraction,
        limit=args.limit,
        num_workers=args.num_workers,
        fit_method=args.fit_method,
        optax_steps=args.optax_steps,
        optax_lr=args.optax_lr,
        nuts_warmup=args.nuts_warmup,
        nuts_samples=args.nuts_samples,
        nuts_chains=args.nuts_chains,
        target_accept_prob=args.target_accept_prob,
    )
    summary = {
        "passed": result["passed"],
        "metrics": result["metrics"],
        "num_workers": result["num_workers"],
        "inference": result["inference"],
        "thresholds": result["thresholds"],
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
