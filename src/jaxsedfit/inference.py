"""Inference diagnostics shared by SED and standalone spectral fitters."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def summarize_nuts_transition_fields(
    extra_fields: Mapping[str, Any],
    max_tree_depth: int,
    *,
    include_raw_fields: bool = False,
) -> dict[str, Any]:
    """Summarize grouped NumPyro transition fields."""
    fields = {name: np.asarray(value) for name, value in extra_fields.items()}
    num_steps = np.asarray(fields.get("num_steps", []), dtype=float)
    accept_prob = np.asarray(fields.get("accept_prob", []), dtype=float)
    diverging = np.asarray(fields.get("diverging", []), dtype=bool)
    energy = np.asarray(fields.get("energy", []), dtype=float)
    max_num_steps = 2 ** int(max_tree_depth) - 1
    final_level_min_steps = 2 ** max(int(max_tree_depth) - 1, 0)
    finite_steps = num_steps[np.isfinite(num_steps) & (num_steps >= 0.0)]
    depth_lower_bound = (
        np.ceil(np.log2(finite_steps + 1.0))
        if finite_steps.size
        else np.asarray([], dtype=float)
    )
    if energy.ndim == 1:
        energy = energy[None, :]
    bfmi = []
    for chain_energy in energy:
        finite = chain_energy[np.isfinite(chain_energy)]
        variance = np.var(finite) if finite.size > 1 else np.nan
        bfmi.append(
            float(np.mean(np.diff(finite) ** 2) / variance)
            if finite.size > 1 and variance > 0.0
            else np.nan
        )
    summary = {
        "n_transitions": int(diverging.size),
        "n_divergent": int(np.count_nonzero(diverging)),
        "divergence_fraction": (
            float(np.mean(diverging)) if diverging.size else np.nan
        ),
        "mean_accept_prob": (
            float(np.nanmean(accept_prob)) if accept_prob.size else np.nan
        ),
        "mean_num_steps": (
            float(np.mean(finite_steps)) if finite_steps.size else np.nan
        ),
        "median_num_steps": (
            float(np.nanmedian(num_steps)) if num_steps.size else np.nan
        ),
        "p90_num_steps": (
            float(np.nanpercentile(num_steps, 90.0)) if num_steps.size else np.nan
        ),
        "p99_num_steps": (
            float(np.nanpercentile(num_steps, 99.0)) if num_steps.size else np.nan
        ),
        "total_num_steps": int(np.nansum(num_steps)) if num_steps.size else 0,
        "max_num_steps": int(np.nanmax(num_steps)) if num_steps.size else 0,
        "max_tree_depth": int(max_tree_depth),
        "median_tree_depth_lower_bound": (
            float(np.median(depth_lower_bound))
            if depth_lower_bound.size
            else np.nan
        ),
        "p90_tree_depth_lower_bound": (
            float(np.percentile(depth_lower_bound, 90.0))
            if depth_lower_bound.size
            else np.nan
        ),
        "final_tree_level_fraction": (
            float(np.mean(num_steps >= final_level_min_steps))
            if num_steps.size
            else np.nan
        ),
        "n_max_num_steps": (
            int(np.count_nonzero(num_steps >= max_num_steps))
            if num_steps.size
            else 0
        ),
        "max_num_steps_fraction": (
            float(np.mean(num_steps >= max_num_steps))
            if num_steps.size
            else np.nan
        ),
        "full_trajectory_fraction": (
            float(np.mean(num_steps >= max_num_steps))
            if num_steps.size
            else np.nan
        ),
        "bfmi": np.asarray(bfmi, dtype=float),
    }
    if include_raw_fields:
        summary["extra_fields"] = fields
    return summary


def nuts_transition_diagnostics(mcmc, max_tree_depth: int) -> dict[str, Any]:
    """Summarize transitions from a fitted NumPyro MCMC object."""
    return summarize_nuts_transition_fields(
        mcmc.get_extra_fields(group_by_chain=True),
        max_tree_depth,
        include_raw_fields=True,
    )


def nuts_metric_diagnostics(mcmc) -> dict[str, Any]:
    """Summarize the adapted step size and mass-matrix conditioning."""
    last_state = getattr(mcmc, "last_state", None)
    adapt_state = getattr(last_state, "adapt_state", None)
    inverse_mass = getattr(adapt_state, "inverse_mass_matrix", None)
    step_size = getattr(adapt_state, "step_size", None)
    if inverse_mass is None:
        return {}
    matrices = (
        inverse_mass
        if isinstance(inverse_mass, dict)
        else {("all",): inverse_mass}
    )
    num_chains = int(getattr(mcmc, "num_chains", 1))
    blocks = []
    for site_names, value in matrices.items():
        array = np.asarray(value, dtype=float)
        chain_values = [array]
        if num_chains > 1 and array.ndim >= 2 and array.shape[0] == num_chains:
            chain_values = list(array)
        for chain_index, chain_value in enumerate(chain_values):
            dimension = (
                chain_value.size
                if chain_value.ndim == 1
                else chain_value.shape[-1]
            )
            if chain_value.ndim == 1:
                eigenvalues = chain_value
            elif np.all(np.isfinite(chain_value)):
                try:
                    eigenvalues = np.linalg.eigvalsh(chain_value)
                except np.linalg.LinAlgError:
                    eigenvalues = np.full(dimension, np.nan, dtype=float)
            else:
                eigenvalues = np.full(dimension, np.nan, dtype=float)
            n_nonpositive = int(
                np.count_nonzero(
                    np.isfinite(eigenvalues) & (eigenvalues <= 0.0)
                )
            )
            n_nonfinite = int(np.count_nonzero(~np.isfinite(eigenvalues)))
            finite_positive = eigenvalues[
                np.isfinite(eigenvalues) & (eigenvalues > 0.0)
            ]
            min_eigenvalue = (
                float(np.min(finite_positive))
                if finite_positive.size
                else np.nan
            )
            max_eigenvalue = (
                float(np.max(finite_positive))
                if finite_positive.size
                else np.nan
            )
            blocks.append(
                {
                    "sites": tuple(site_names),
                    "chain": chain_index,
                    "dimension": int(dimension),
                    "min_eigenvalue": min_eigenvalue,
                    "max_eigenvalue": max_eigenvalue,
                    "n_nonpositive_eigenvalues": n_nonpositive,
                    "n_nonfinite_eigenvalues": n_nonfinite,
                    "condition_number": (
                        np.inf
                        if n_nonpositive or n_nonfinite
                        else (
                            max_eigenvalue / min_eigenvalue
                            if min_eigenvalue > 0.0
                            else np.nan
                        )
                    ),
                }
            )
    return {
        "adapted_step_size": (
            np.asarray(step_size, dtype=float)
            if step_size is not None
            else np.asarray([])
        ),
        "blocks": blocks,
    }


__all__ = [
    "nuts_metric_diagnostics",
    "nuts_transition_diagnostics",
    "summarize_nuts_transition_fields",
]
