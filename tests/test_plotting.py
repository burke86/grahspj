from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxsedfit.plotting import plot_corner, plot_fit_sed, plot_trace


def test_plot_fit_sed_writes_output(tmp_path):
    class _Filter:
        def __init__(self, lam):
            self.effective_wavelength = lam

    class _Obs:
        object_id = "demo-object"

    class _Phot:
        fluxes = [1.0, 2.0, 1.5]
        errors = [0.1, 0.2, 0.15]
        filter_names = ["f1", "f2", "f3"]

    class _Cfg:
        observation = _Obs()
        photometry = _Phot()

    wave = np.array([1000.0, 2000.0, 4000.0, 8000.0])
    flux = np.array([0.8, 1.5, 1.8, 1.0])
    phot = np.array([0.9, 1.9, 1.4])

    class _Fitter:
        config = _Cfg()
        context = type("_Context", (), {"filters": [_Filter(1200.0), _Filter(2500.0), _Filter(6000.0)]})()

        def predict(self, posterior="latest"):
            return {
                "obs_wave": wave[None, :],
                "pred_fluxes": phot[None, :],
                "host_obs_sed": (0.5 * flux)[None, :],
                "dust_obs_sed": (0.12 * flux)[None, :],
                "disk_obs_sed": (0.2 * flux)[None, :],
                "torus_obs_sed": (0.1 * flux)[None, :],
                "feii_obs_sed": (0.05 * flux)[None, :],
                "line_obs_sed": (0.05 * flux)[None, :],
                "line_bl_obs_sed": (0.03 * flux)[None, :],
                "line_nl_obs_sed": (0.02 * flux)[None, :],
                "line_liner_obs_sed": np.zeros((1, flux.size)),
                "balmer_obs_sed": (0.03 * flux)[None, :],
                "agn_obs_sed": (0.4 * flux)[None, :],
                "total_obs_sed": flux[None, :],
            }

    output = tmp_path / "sed_plot.png"
    fig = plot_fit_sed(_Fitter(), output_path=output)
    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_fit_sed_can_disable_band_annotations(tmp_path):
    class _Filter:
        def __init__(self, lam):
            self.effective_wavelength = lam

    class _Obs:
        object_id = "demo-object"

    class _Phot:
        fluxes = [1.0, 2.0, 1.5]
        errors = [0.1, 0.2, 0.15]
        filter_names = ["f1", "f2", "f3"]

    class _Cfg:
        observation = _Obs()
        photometry = _Phot()

    wave = np.array([1000.0, 2000.0, 4000.0, 8000.0])
    flux = np.array([0.8, 1.5, 1.8, 1.0])
    phot = np.array([0.9, 1.9, 1.4])

    class _Fitter:
        config = _Cfg()
        context = type("_Context", (), {"filters": [_Filter(1200.0), _Filter(2500.0), _Filter(6000.0)]})()

        def predict(self, posterior="latest"):
            return {
                "obs_wave": wave[None, :],
                "pred_fluxes": phot[None, :],
                "host_obs_sed": (0.5 * flux)[None, :],
                "agn_obs_sed": (0.4 * flux)[None, :],
                "total_obs_sed": flux[None, :],
            }

    output = tmp_path / "sed_plot_no_labels.png"
    fig = plot_fit_sed(_Fitter(), output_path=output, annotate_band_names=False)
    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_corner_writes_output_and_calls_corner(tmp_path, monkeypatch):
    captured = {}
    fake_corner_module = types.ModuleType("corner")

    def _corner(data, labels=None, truths=None, **kwargs):
        captured["data"] = data
        captured["labels"] = labels
        captured["truths"] = truths
        captured["kwargs"] = kwargs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return fig

    fake_corner_module.corner = _corner
    monkeypatch.setitem(sys.modules, "corner", fake_corner_module)

    output = tmp_path / "corner.pdf"
    fig = plot_corner(
        {
            "alpha": np.array([1.0, 2.0, 3.0]),
            "vector_site": np.ones((3, 2)),
            "beta": np.array([3.0, 2.0, 1.0]),
        },
        output_path=output,
        params=["beta", "alpha"],
        labels={"beta": "Beta", "alpha": "Alpha"},
        truths={"beta": 2.0},
        bins=4,
    )

    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0
    assert captured["data"].shape == (3, 2)
    np.testing.assert_allclose(captured["data"][:, 0], [3.0, 2.0, 1.0])
    np.testing.assert_allclose(captured["data"][:, 1], [1.0, 2.0, 3.0])
    assert captured["labels"] == ["Beta", "Alpha"]
    assert captured["truths"] == [2.0, None]
    assert captured["kwargs"]["bins"] == 4
    np.testing.assert_allclose(captured["kwargs"]["levels"], 1.0 - np.exp(-0.5 * np.array([1.0, 2.0, 3.0]) ** 2))
    assert captured["kwargs"]["quantiles"] == (0.16, 0.5, 0.84)
    assert captured["kwargs"]["title_quantiles"] == (0.16, 0.5, 0.84)
    assert captured["kwargs"]["show_titles"] is True
    assert captured["kwargs"]["smooth"] == 1.0
    assert captured["kwargs"]["smooth1d"] == 1.0
    assert captured["kwargs"]["plot_datapoints"] is False
    assert captured["kwargs"]["fill_contours"] is True
    assert captured["kwargs"]["title_kwargs"] == {"fontsize": 8}
    assert captured["kwargs"]["title_fmt"] == ".3g"


def test_plot_corner_allows_default_overrides(monkeypatch):
    captured = {}
    fake_corner_module = types.ModuleType("corner")

    def _corner(data, labels=None, truths=None, **kwargs):
        captured["kwargs"] = kwargs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return fig

    fake_corner_module.corner = _corner
    monkeypatch.setitem(sys.modules, "corner", fake_corner_module)

    plot_corner(
        {"alpha": np.array([1.0, 2.0, 3.0])},
        levels=[0.5],
        quantiles=[0.25, 0.75],
        title_quantiles=[0.1, 0.5, 0.9],
        show_titles=False,
        smooth=0.5,
        smooth1d=None,
        plot_datapoints=True,
        fill_contours=False,
        title_kwargs={"fontsize": 6},
        title_fmt=".2f",
    )

    assert captured["kwargs"]["levels"] == [0.5]
    assert captured["kwargs"]["quantiles"] == [0.25, 0.75]
    assert captured["kwargs"]["title_quantiles"] == [0.1, 0.5, 0.9]
    assert captured["kwargs"]["show_titles"] is False
    assert captured["kwargs"]["smooth"] == 0.5
    assert captured["kwargs"]["smooth1d"] is None
    assert captured["kwargs"]["plot_datapoints"] is True
    assert captured["kwargs"]["fill_contours"] is False
    assert captured["kwargs"]["title_kwargs"] == {"fontsize": 6}
    assert captured["kwargs"]["title_fmt"] == ".2f"


def test_plot_corner_skips_constant_default_params(monkeypatch):
    captured = {}
    fake_corner_module = types.ModuleType("corner")

    def _corner(data, labels=None, truths=None, **kwargs):
        captured["data"] = data
        captured["labels"] = labels
        captured["truths"] = truths
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return fig

    fake_corner_module.corner = _corner
    monkeypatch.setitem(sys.modules, "corner", fake_corner_module)

    fig = plot_corner(
        {
            "constant": np.array([2.0, 2.0, 2.0]),
            "dynamic": np.array([1.0, 2.0, 3.0]),
        }
    )

    assert fig is not None
    assert captured["data"].shape == (3, 1)
    np.testing.assert_allclose(captured["data"][:, 0], [1.0, 2.0, 3.0])
    assert captured["labels"] == ["dynamic"]


def test_plot_corner_rejects_explicit_constant_param(monkeypatch):
    fake_corner_module = types.ModuleType("corner")
    fake_corner_module.corner = lambda *args, **kwargs: plt.figure()
    monkeypatch.setitem(sys.modules, "corner", fake_corner_module)

    with pytest.raises(ValueError, match="has no dynamic range"):
        plot_corner({"constant": np.array([2.0, 2.0, 2.0])}, params=["constant"])


def test_plot_corner_rejects_all_constant_default_params(monkeypatch):
    fake_corner_module = types.ModuleType("corner")
    fake_corner_module.corner = lambda *args, **kwargs: plt.figure()
    monkeypatch.setitem(sys.modules, "corner", fake_corner_module)

    with pytest.raises(RuntimeError, match="dynamic range"):
        plot_corner({"constant": np.array([2.0, 2.0, 2.0])})


def test_plot_trace_writes_grouped_chain_samples(tmp_path):
    class _MCMC:
        def __init__(self):
            self.grouped = None

        def get_samples(self, group_by_chain=False):
            self.grouped = group_by_chain
            return {
                "alpha": np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
                "vector_site": np.ones((2, 3, 2)),
                "beta": np.array([[3.0, 2.0, 1.0], [3.5, 2.5, 1.5]]),
            }

    class _Fitter:
        def __init__(self):
            self.mcmc = _MCMC()
            self.samples = {"alpha": np.array([-1.0])}
            self.nuts_result = {"mcmc": self.mcmc}

    fitter = _Fitter()
    output = tmp_path / "trace.pdf"
    fig = plot_trace(fitter, output_path=output, params=["beta", "alpha"])

    assert fitter.mcmc.grouped is True
    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0
    assert [ax.get_ylabel() for ax in fig.axes] == ["beta", "alpha"]
    assert len(fig.axes[0].lines) == 2
    assert fig.axes[0].yaxis.label.get_size() == 8
    assert fig.axes[-1].xaxis.label.get_size() == 9
    assert {tick.get_fontsize() for ax in fig.axes for tick in ax.get_xticklabels() + ax.get_yticklabels()} == {8.0}


def test_jaxsedfit_corner_and_trace_methods_delegate(monkeypatch):
    import jaxsedfit.plotting as plotting

    model = types.ModuleType("jaxsedfit.model")
    model.grahsp_photometric_model = lambda *args, **kwargs: None
    preload = types.ModuleType("jaxsedfit.preload")
    preload.ModelContext = object
    preload.build_model_context = lambda config: None
    monkeypatch.setitem(sys.modules, "jaxsedfit.model", model)
    monkeypatch.setitem(sys.modules, "jaxsedfit.preload", preload)
    sys.modules.pop("jaxsedfit.core", None)

    calls = {}

    def _plot_corner(fitter, **kwargs):
        calls["corner"] = (fitter, kwargs)
        return "corner"

    def _plot_trace(fitter, **kwargs):
        calls["trace"] = (fitter, kwargs)
        return "trace"

    try:
        from jaxsedfit.core import JAXSEDFit

        monkeypatch.setattr(plotting, "plot_corner", _plot_corner)
        monkeypatch.setattr(plotting, "plot_trace", _plot_trace)
        fitter = object.__new__(JAXSEDFit)

        assert fitter.plot_corner(output_path="corner.pdf", params=["alpha"]) == "corner"
        assert fitter.plot_trace(output_path="trace.pdf", params=["beta"]) == "trace"
        assert calls["corner"][0] is fitter
        assert calls["corner"][1]["output_path"] == "corner.pdf"
        assert calls["corner"][1]["params"] == ["alpha"]
        assert calls["trace"][0] is fitter
        assert calls["trace"][1]["output_path"] == "trace.pdf"
        assert calls["trace"][1]["params"] == ["beta"]
    finally:
        sys.modules.pop("jaxsedfit.core", None)
