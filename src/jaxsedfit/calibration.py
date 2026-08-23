"""Small utilities for simulation-based calibration regression checks."""

from __future__ import annotations

import numpy as np


def summarize_mass_sbc(records: list[dict]) -> dict:
    """Summarize stellar-mass recovery and posterior ranks."""
    if len(records) < 2:
        raise ValueError("At least two SBC records are required.")
    truth = np.asarray([row["truth"] for row in records], dtype=float)
    median = np.asarray([row["posterior_median"] for row in records], dtype=float)
    residual = median - truth
    rank = np.asarray([row["rank_fraction"] for row in records], dtype=float)
    covered = np.asarray(
        [row["posterior_p16"] <= row["truth"] <= row["posterior_p84"] for row in records]
    )
    slope, intercept = np.polyfit(truth, median, 1)
    n_samples = sum(int(row["n_samples"]) for row in records)
    n_divergent = sum(int(row["n_divergent"]) for row in records)
    return {
        "n": len(records),
        "median_residual_dex": float(np.median(residual)),
        "mean_residual_dex": float(np.mean(residual)),
        "rms_residual_dex": float(np.sqrt(np.mean(residual**2))),
        "mass_recovery_slope": float(slope),
        "mass_recovery_intercept": float(intercept),
        "rank_mean": float(np.mean(rank)),
        "rank_std": float(np.std(rank)),
        "central_68_coverage": float(np.mean(covered)),
        "divergence_fraction": float(n_divergent / n_samples),
        "records": records,
    }


def mass_sbc_regression_checks(summary: dict) -> dict[str, bool]:
    """Return deliberately broad finite-SBC checks for the catastrophic bias."""
    return {
        "median_abs_bias_below_0p15_dex": abs(summary["median_residual_dex"]) < 0.15,
        "recovery_slope_between_0p8_and_1p2": 0.8 < summary["mass_recovery_slope"] < 1.2,
        "rank_mean_between_0p15_and_0p85": 0.15 < summary["rank_mean"] < 0.85,
        "central_68_coverage_between_0p4_and_0p9": 0.4 < summary["central_68_coverage"] < 0.9,
        "divergence_fraction_below_0p1": summary["divergence_fraction"] < 0.1,
    }
