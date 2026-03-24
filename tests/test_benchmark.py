from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grahspj.benchmark import (
    build_chimera_fit_config,
    chimera_data_dir,
    load_chimera_benchmark_dataset,
    run_chimera_mass_benchmark,
    select_chimera_subset,
)


def test_chimera_dataset_adapter_and_subset():
    dataset = load_chimera_benchmark_dataset()
    assert len(dataset.rows) > 1000
    subset = select_chimera_subset(dataset)
    assert len(subset) > 200
    assert subset[0]["id"] == "143816.78+191955.5_493880_0.03"
    assert all(np.isfinite(row["log_stellar_mass_truth"]) for row in subset)
    assert all(np.isfinite(row["resample_weight"]) for row in subset)


def test_build_chimera_fit_config(tmp_path):
    row = select_chimera_subset(load_chimera_benchmark_dataset())[0]
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path))
    assert cfg.observation.object_id == row["id"]
    assert cfg.photometry.filter_names[0] == "u_sdss"
    assert cfg.galaxy.dsps_ssp_fn == str(ssp_path)
    assert "log_stellar_mass" in cfg.prior_config
    assert "log_agn_amp" in cfg.prior_config


def test_build_chimera_fit_config_preserves_user_prior_overrides(tmp_path):
    row = select_chimera_subset(load_chimera_benchmark_dataset())[0]
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    base = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path))
    base.prior_config["log_stellar_mass"] = {"loc": 9.9, "scale": 0.1}
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path), base_config=base)
    assert cfg.prior_config["log_stellar_mass"] == {"loc": 9.9, "scale": 0.1}
    assert "log_agn_amp" in cfg.prior_config


def test_chimera_mass_benchmark_with_surrogate_fitter(tmp_path):
    class _FakeFitter:
        def __init__(self, config):
            self.config = config
            self._mass = np.nan

        def fit_map(self):
            row_id = str(self.config.observation.object_id)
            token = sum(ord(ch) for ch in row_id) % 11
            self._mass = 9.5 + 0.05 * token
            return {"median": {"log_stellar_mass": self._mass}}

        def recovered_log_stellar_mass(self):
            return float(self._mass)

    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    benchmark = run_chimera_mass_benchmark(
        output_dir=tmp_path,
        dsps_ssp_fn=str(ssp_path),
        fitter_cls=_FakeFitter,
        max_weighted_mae=10.0,
        max_abs_weighted_bias=10.0,
        min_finite_fraction=0.99,
    )
    assert benchmark["passed"] is True
    assert benchmark["metrics"]["n_rows"] > 200
    assert benchmark["metrics"]["finite_fit_fraction"] == 1.0
    assert (tmp_path / "chimera_mass_recovery_rows.csv").exists()
    assert (tmp_path / "chimera_mass_recovery_metrics.json").exists()
    assert (tmp_path / "chimera_mass_scatter.png").exists()
    assert (tmp_path / "chimera_mass_residual_vs_qso_weight.png").exists()
    metrics = json.loads((tmp_path / "chimera_mass_recovery_metrics.json").read_text(encoding="utf-8"))
    assert "weighted_mae" in metrics["metrics"]
