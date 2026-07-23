from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpyro.distributions as dist
import pytest

from jaxsedfit.benchmark import (
    build_chimera_fit_config,
    chimera_data_dir,
    fit_quality_metadata,
    load_chimera_benchmark_dataset,
    run_chimera_mass_benchmark,
    select_chimera_subset,
)


def _require_chimera_data():
    data_dir = chimera_data_dir()
    needed = [
        data_dir / "chimeras-grahsp.fits",
        data_dir / "chimeras-fullinfo.fits",
        data_dir / "benchmark_subset_ids.txt",
    ]
    missing = [path for path in needed if not path.exists()]
    if missing:
        pytest.skip(f"Chimera benchmark fixtures not available: {missing[0]}")


class _FakeFitter:
    def __init__(self, config):
        self.config = config
        self._mass = np.nan
        self.context = type(
            "_Ctx",
            (),
            {
                "fluxes": np.asarray(config.photometry.fluxes, dtype=float),
                "errors": np.asarray(config.photometry.errors, dtype=float),
                "upper_limits": np.asarray(config.photometry.is_upper_limit, dtype=bool),
            },
        )()
        self._predictive = None
        self.samples = None
        self.nuts_result = None

    def fit_map(self):
        row_id = str(self.config.observation.object_id)
        token = sum(ord(ch) for ch in row_id) % 11
        self._mass = 9.5 + 0.05 * token
        draws = np.array([self._mass - 0.1, self._mass, self._mass + 0.1], dtype=float)
        self.samples = {"log_stellar_mass": draws}
        pred = np.asarray(self.config.photometry.fluxes, dtype=float) * (1.0 + 0.01 * token)
        self._predictive = {
            "pred_fluxes": pred[None, :],
            "fracAGN_5100_fit": np.array([min(0.99, 0.2 + 0.05 * token)], dtype=float),
        }
        return {"median": {"log_stellar_mass": self._mass}}

    def fit(self, **kwargs):
        assert self.config.inference.method == "optax+nuts"
        return {"fit": self.fit_map()}

    def recovered_log_stellar_mass(self):
        return float(self._mass)

    def predict(self):
        return self._predictive


class _FakeMCMC:
    def get_extra_fields(self, group_by_chain=False):
        assert group_by_chain is False
        return {"diverging": np.array([False, True, False, True])}


def test_fit_quality_metadata_includes_reduced_chi2_and_divergence_fraction():
    fitter = type(
        "_DiagnosticFitter",
        (),
        {
            "nuts_result": {"mcmc": _FakeMCMC()},
            "predict": lambda self: {
                "pred_fluxes": np.array([[1.2, 1.8]]),
                "agn_fluxes": np.zeros((1, 2)),
            },
            "context": type(
                "_Ctx",
                (),
                {
                    "fluxes": np.array([1.0, 2.0]),
                    "errors": np.array([0.1, 0.2]),
                    "upper_limits": np.array([False, False]),
                    "filters": [],
                },
            )(),
            "config": type(
                "_Cfg",
                (),
                {
                    "likelihood": type(
                        "_Likelihood",
                        (),
                        {
                            "systematics_width": 0.0,
                            "agn_systematics_width": 0.0,
                            "variability_uncertainty": False,
                            "attenuation_model_uncertainty": False,
                            "lyman_break_uncertainty": False,
                        },
                    )()
                },
            )(),
        },
    )()

    metadata = fit_quality_metadata(fitter)

    assert metadata["reduced_chi2"] == pytest.approx(2.5)
    assert metadata["n_divergent"] == 2
    assert metadata["n_transition_samples"] == 4
    assert metadata["divergence_fraction"] == pytest.approx(0.5)


def test_fit_quality_metadata_uses_saved_transition_diagnostics_without_mcmc():
    fitter = type(
        "_LoadedDiagnosticFitter",
        (),
        {
            "nuts_result": {
                "transition_diagnostics": {
                    "n_divergent": 1,
                    "n_transitions": 5,
                    "divergence_fraction": 0.2,
                    "mean_accept_prob": 0.88,
                    "median_num_steps": 31.0,
                    "p90_num_steps": 120.0,
                    "p99_num_steps": 127.0,
                    "final_tree_level_fraction": 0.4,
                    "max_num_steps_fraction": 0.2,
                    "bfmi": np.array([0.8, 0.55]),
                }
            },
            "predict": lambda self: {
                "pred_fluxes": np.array([[1.0]]),
                "agn_fluxes": np.zeros((1, 1)),
            },
            "context": type(
                "_Ctx",
                (),
                {
                    "fluxes": np.array([1.0]),
                    "errors": np.array([0.1]),
                    "upper_limits": np.array([False]),
                    "filters": [],
                },
            )(),
            "config": type(
                "_Cfg",
                (),
                {
                    "likelihood": type(
                        "_Likelihood",
                        (),
                        {
                            "systematics_width": 0.0,
                            "agn_systematics_width": 0.0,
                            "variability_uncertainty": False,
                            "attenuation_model_uncertainty": False,
                            "lyman_break_uncertainty": False,
                        },
                    )()
                },
            )(),
        },
    )()

    metadata = fit_quality_metadata(fitter)

    assert metadata["n_divergent"] == 1
    assert metadata["n_transition_samples"] == 5
    assert metadata["divergence_fraction"] == pytest.approx(0.2)
    assert metadata["mean_accept_prob"] == pytest.approx(0.88)
    assert metadata["final_tree_level_fraction"] == pytest.approx(0.4)
    assert metadata["max_num_steps_fraction"] == pytest.approx(0.2)
    assert metadata["bfmi_min"] == pytest.approx(0.55)


class _FailingFakeFitter(_FakeFitter):
    def fit_map(self):
        if str(self.config.observation.object_id).endswith("_0.0001"):
            raise RuntimeError("intentional benchmark failure")
        return super().fit_map()

    def fit(self, **kwargs):
        assert self.config.inference.method == "optax+nuts"
        return {"fit": self.fit_map()}


def test_chimera_dataset_adapter_and_subset():
    _require_chimera_data()
    dataset = load_chimera_benchmark_dataset()
    assert len(dataset.rows) > 1000
    subset = select_chimera_subset(dataset)
    assert len(subset) > 200
    assert subset[0]["id"] == "142746.39+293038.2_736084_0.0001"
    assert all(np.isfinite(row["log_stellar_mass_truth"]) for row in subset)
    assert all(np.isfinite(row["resample_weight"]) for row in subset)


def test_build_chimera_fit_config(tmp_path):
    _require_chimera_data()
    row = select_chimera_subset(load_chimera_benchmark_dataset())[0]
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path))
    assert cfg.observation.object_id == row["id"]
    assert cfg.photometry.filter_names[0] == "u_sdss"
    assert cfg.galaxy.dsps_ssp_fn == str(ssp_path)
    prior = cfg.prior_config.to_mapping()
    assert "log_stellar_mass" in prior
    assert isinstance(prior["log_stellar_mass"], dist.StudentT)
    assert float(prior["log_stellar_mass"].loc) == 10.0


def test_build_chimera_fit_config_preserves_user_prior_overrides(tmp_path):
    _require_chimera_data()
    row = select_chimera_subset(load_chimera_benchmark_dataset())[0]
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    base = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path))
    base.prior_config.stellar_mass = dist.Normal(9.9, 0.1)
    cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path), base_config=base)
    prior = cfg.prior_config.to_mapping()["log_stellar_mass"]
    assert isinstance(prior, dist.Normal)
    assert float(prior.loc) == pytest.approx(9.9)
    assert float(prior.scale) == pytest.approx(0.1)


def test_chimera_mass_benchmark_with_surrogate_fitter(tmp_path):
    _require_chimera_data()
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    benchmark = run_chimera_mass_benchmark(
        output_dir=tmp_path,
        dsps_ssp_fn=str(ssp_path),
        fitter_cls=_FakeFitter,
        max_weighted_mae=10.0,
        max_abs_weighted_bias=10.0,
        min_finite_fraction=0.99,
        num_workers=1,
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
    assert metrics["num_workers"] == 1


def test_chimera_mass_benchmark_parallel_matches_serial(tmp_path):
    _require_chimera_data()
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    serial = run_chimera_mass_benchmark(
        dsps_ssp_fn=str(ssp_path),
        fitter_cls=_FakeFitter,
        max_weighted_mae=10.0,
        max_abs_weighted_bias=10.0,
        min_finite_fraction=0.99,
        limit=5,
        num_workers=1,
    )
    parallel = run_chimera_mass_benchmark(
        dsps_ssp_fn=str(ssp_path),
        fitter_cls=_FakeFitter,
        max_weighted_mae=10.0,
        max_abs_weighted_bias=10.0,
        min_finite_fraction=0.99,
        limit=5,
        num_workers=2,
    )
    assert [row["id"] for row in serial["rows"]] == [row["id"] for row in parallel["rows"]]
    assert [row["log_stellar_mass_fit"] for row in serial["rows"]] == [row["log_stellar_mass_fit"] for row in parallel["rows"]]


def test_chimera_mass_benchmark_worker_failure_returns_nan(tmp_path):
    _require_chimera_data()
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    benchmark = run_chimera_mass_benchmark(
        dsps_ssp_fn=str(ssp_path),
        fitter_cls=_FailingFakeFitter,
        max_weighted_mae=10.0,
        max_abs_weighted_bias=10.0,
        min_finite_fraction=0.0,
        limit=5,
        num_workers=2,
    )
    failed = [row for row in benchmark["rows"] if not np.isfinite(row["log_stellar_mass_fit"])]
    assert failed
    assert failed[0]["fit_error"]
