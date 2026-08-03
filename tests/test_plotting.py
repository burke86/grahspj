from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jaxsedfit.plotting import (
    _COMPONENT_STYLE,
    _bridged_jaxsedfit_agn_lines,
    _grouped_trace_samples,
    _median_effective_variance,
    plot_corner,
    plot_fit_sed,
    plot_trace,
)


def test_joint_line_bridge_rescales_native_jaxsedfit_shape():
    pred = {
        "line_bl_obs_sed": np.array([[1.0, 2.0, 1.0]]),
        "line_nl_obs_sed": np.array([[0.5, 0.5, 0.5]]),
        "line_fluxes": np.array([[1.0, 1.0]]),
        "feii_fluxes": np.zeros((1, 2)),
        "jqf_line_photometry": np.array([[2.0, 4.0]]),
    }

    bridged = _bridged_jaxsedfit_agn_lines(pred)

    np.testing.assert_allclose(bridged, 3.0 * np.array([[1.5, 2.5, 1.5]]))


def test_joint_line_bridge_prefers_rendered_jaxqsofit_sed():
    rendered = np.array([[0.1, 0.4, 0.2]])
    pred = {
        "jqf_line_obs_sed": rendered,
        "line_bl_obs_sed": np.array([[10.0, 20.0, 10.0]]),
        "line_fluxes": np.array([[1.0, 1.0]]),
        "jqf_line_photometry": np.array([[100.0, 100.0]]),
    }

    bridged = _bridged_jaxsedfit_agn_lines(pred)

    np.testing.assert_array_equal(bridged, rendered)


def test_component_style_separates_feii_from_agn_lines():
    styles = {label: keys for keys, label, *_ in _COMPONENT_STYLE}

    assert "feii_obs_sed" not in styles["AGN lines"]
    assert styles["Fe II"] == ("feii_obs_sed",)


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
        context = type(
            "_Context",
            (),
            {
                "filters": [_Filter(1200.0), _Filter(2500.0), _Filter(6000.0)],
                "spec_wave_obs": np.array([1800.0, 2200.0, 2600.0]),
                "spec_fluxes": np.array([1.2, 1.4, 1.3]),
                "spec_mask": np.array([True, False, True]),
                "spec_spectrum_index": np.array([0, 0, 0]),
            },
        )()

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
                "spectrum_scale_fit": np.array([0.5]),
                "spec_wave_obs": np.array([[1800.0, 2200.0, 2600.0]]),
                "jqf_line_model_aperture": np.array([[0.2, 0.3, 0.4]]),
                "jqf_feii_model": np.array([[0.1, 0.1, 0.1]]),
                "jqf_feii_obs_sed": (0.06 * flux)[None, :],
                "jqf_balmer_obs_sed": (0.04 * flux)[None, :],
            }

    output = tmp_path / "sed_plot.png"
    fig = plot_fit_sed(_Fitter(), output_path=output)
    assert fig is not None
    spectrum_lines = [
        line
        for line in fig.axes[0].lines
        if line.get_label() == "Observed spectrum"
    ]
    assert len(spectrum_lines) == 1
    assert spectrum_lines[0].get_color() == "#c53030"
    assert np.array_equal(spectrum_lines[0].get_xdata(), np.array([1800.0, 2600.0]))
    assert np.array_equal(spectrum_lines[0].get_ydata(), np.array([2.4, 2.6]))
    agn_line_curves = [line for line in fig.axes[0].lines if line.get_label() == "AGN lines"]
    assert len(agn_line_curves) == 1
    assert np.array_equal(agn_line_curves[0].get_xdata(), wave)
    assert len([line for line in fig.axes[0].lines if line.get_label() == "Fe II"]) == 1
    assert len([line for line in fig.axes[0].lines if line.get_label() == "Balmer cont."]) == 1
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


def test_plot_fit_sed_uses_likelihood_photometry_and_saved_chi2():
    class _Filter:
        def __init__(self, wavelength):
            self.effective_wavelength = wavelength

    config = types.SimpleNamespace(
        observation=types.SimpleNamespace(object_id="dereddened"),
        photometry=types.SimpleNamespace(
            fluxes=[1.0, 1.0],
            errors=[0.1, 0.1],
            filter_names=["f1", "f2"],
        ),
    )
    context = types.SimpleNamespace(
        filters=[_Filter(2000.0), _Filter(4000.0)],
        fluxes=np.array([2.0, 4.0]),
        errors=np.array([0.2, 0.4]),
        upper_limits=np.array([False, False]),
    )
    prediction = {
        "obs_wave": np.array([[1000.0, 2000.0, 4000.0]]),
        "pred_fluxes": np.array([[1.5, 3.5], [1.7, 3.7]]),
        "total_obs_sed": np.array([[1.0, 2.0, 1.0]]),
        "sed_reduced_chi2": np.array([1.0, 3.0]),
    }
    fitter = types.SimpleNamespace(
        config=config,
        context=context,
        predict=lambda posterior="latest": prediction,
    )

    fig = plot_fit_sed(fitter)

    observed = next(
        container
        for container in fig.axes[0].containers
        if container.get_label() == "Observed photometry"
    )
    np.testing.assert_allclose(observed.lines[0].get_ydata(), context.fluxes)
    chi2_labels = [
        text for text in fig.axes[1].texts if r"\chi^2_\nu" in text.get_text()
    ]
    assert len(chi2_labels) == 1
    assert chi2_labels[0].get_text() == r"$\chi^2_\nu = 2.00$"
    assert chi2_labels[0].get_color() == "black"


def test_median_effective_variance_matches_nebular_and_lyman_terms():
    filters = [
        types.SimpleNamespace(effective_wavelength=1000.0),
        types.SimpleNamespace(effective_wavelength=2000.0),
    ]
    likelihood = types.SimpleNamespace(
        systematics_width=0.0,
        agn_systematics_width=0.0,
        variability_uncertainty=False,
        attenuation_model_uncertainty=False,
        lyman_break_uncertainty=True,
        local_nebular_line_uncertainty_dex=0.1,
    )
    fitter = types.SimpleNamespace(
        config=types.SimpleNamespace(
            likelihood=likelihood,
            photometry=types.SimpleNamespace(errors=[0.1, 0.1], fluxes=[1.0, 1.0]),
        ),
        context=types.SimpleNamespace(
            filters=filters,
            errors=np.array([0.1, 0.1]),
            fluxes=np.array([1.0, 1.0]),
        ),
    )
    pred = {
        "pred_fluxes": np.array([[1.0, 1.0]]),
        "nebular_lines_fluxes": np.array([[2.0, 2.0]]),
        "redshift_fit": np.array([0.0]),
    }

    variance = _median_effective_variance(fitter, pred)

    nebular_sigma = np.expm1(np.log(10.0) * 0.1)
    nebular_variance = (2.0 * nebular_sigma) ** 2
    np.testing.assert_allclose(
        variance,
        [0.1**2 + nebular_variance + 1.0e16, 0.1**2 + nebular_variance],
    )


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


def test_grouped_trace_samples_hide_internal_reparameterization_sites():
    class _MCMC:
        def get_samples(self, group_by_chain=False):
            assert group_by_chain is True
            return {
                "scale": np.ones((1, 3)),
                "scale_pivot": np.zeros((1, 3)),
            }

    fitter = types.SimpleNamespace(
        nuts_result={
            "mcmc": _MCMC(),
            "reparameterized_sites": {"scale": "scale_pivot"},
        },
        samples={"scale": np.ones(3)},
    )

    samples, grouped = _grouped_trace_samples(fitter)
    assert grouped is True
    assert set(samples) == {"scale"}


def test_jaxsedfit_corner_and_trace_methods_delegate(monkeypatch):
    import jaxsedfit.plotting as plotting

    missing = object()
    package = sys.modules["jaxsedfit"]
    original_core_module = sys.modules.get("jaxsedfit.core", missing)
    original_core_attribute = getattr(package, "core", missing)
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

    def _plot_fit_sed(fitter, **kwargs):
        calls["sed"] = (fitter, kwargs)
        return "sed"

    try:
        from jaxsedfit.core import JAXSEDFit
        from jaxsedfit.results import FitResult, PredictionResult

        monkeypatch.setattr(plotting, "plot_corner", _plot_corner)
        monkeypatch.setattr(plotting, "plot_trace", _plot_trace)
        monkeypatch.setattr(plotting, "plot_fit_sed", _plot_fit_sed)
        fitter = object.__new__(JAXSEDFit)
        fitter.predict = lambda **kwargs: {"pred_fluxes": np.array([[1.0], [3.0]])}

        assert fitter.plot_sed(output_path="sed.pdf") == "sed"
        assert fitter.plot_corner(output_path="corner.pdf", params=["alpha"]) == "corner"
        assert fitter.plot_trace(output_path="trace.pdf", params=["beta"]) == "trace"
        result = FitResult(fitter=fitter, samples={}, median={}, method="map")
        pred = result.predict()
        assert isinstance(pred, PredictionResult)
        np.testing.assert_allclose(pred.median["pred_fluxes"], [2.0])
        assert result.plot_corner(output_path="result_corner.pdf") == "corner"
        assert result.plot_trace(output_path="result_trace.pdf") == "trace"
        assert calls["sed"][0] is fitter
        assert calls["sed"][1]["output_path"] == "sed.pdf"
        assert calls["corner"][0] is fitter
        assert calls["corner"][1]["output_path"] == "result_corner.pdf"
        assert calls["trace"][0] is fitter
        assert calls["trace"][1]["output_path"] == "result_trace.pdf"
    finally:
        if original_core_module is missing:
            sys.modules.pop("jaxsedfit.core", None)
        else:
            sys.modules["jaxsedfit.core"] = original_core_module
        if original_core_attribute is missing:
            if hasattr(package, "core"):
                delattr(package, "core")
        else:
            package.core = original_core_attribute
