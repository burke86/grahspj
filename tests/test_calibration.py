"""Regression tests for simulation-based stellar-mass calibration metrics."""

import pytest

from jaxsedfit.calibration import mass_sbc_regression_checks, summarize_mass_sbc


def _record(truth, median, rank, *, covered=True, divergences=0):
    width = 0.3 if covered else 0.01
    return {
        "truth": truth,
        "posterior_median": median,
        "posterior_p16": median - width,
        "posterior_p84": median + width,
        "rank_fraction": rank,
        "n_samples": 200,
        "n_divergent": divergences,
    }


def test_mass_sbc_regression_accepts_calibrated_recovery():
    records = [
        _record(6.5, 6.48, 0.15),
        _record(7.5, 7.55, 0.35),
        _record(8.5, 8.47, 0.55),
        _record(9.5, 9.53, 0.75),
        _record(10.5, 10.48, 0.65, covered=False),
    ]
    summary = summarize_mass_sbc(records)

    assert summary["median_residual_dex"] == pytest.approx(-0.02)
    assert all(mass_sbc_regression_checks(summary).values())


def test_mass_sbc_regression_rejects_low_mass_pull_toward_intermediate_mass():
    records = [
        _record(6.5, 8.0, 0.0, covered=False),
        _record(7.0, 8.1, 0.0, covered=False),
        _record(7.5, 8.2, 0.0, covered=False),
        _record(8.0, 8.3, 0.0, covered=False),
    ]
    checks = mass_sbc_regression_checks(summarize_mass_sbc(records))

    assert not checks["median_abs_bias_below_0p15_dex"]
    assert not checks["recovery_slope_between_0p8_and_1p2"]
    assert not checks["rank_mean_between_0p15_and_0p85"]
    assert not checks["central_68_coverage_between_0p4_and_0p9"]
