from pathlib import Path

import numpy as np

from grahspj.plotting import plot_fit_sed


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
