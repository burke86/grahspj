from types import SimpleNamespace

import numpy as np
import pytest

import jaxsedfit
from jaxsedfit.config import (
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    Observation,
    PhotometryData,
)
from jaxsedfit.core import JAXSEDFit
from jaxsedfit.results import FitResult


def _minimal_config() -> FitConfig:
    return FitConfig(
        observation=Observation(object_id="roundtrip", redshift=0.2),
        photometry=PhotometryData(filter_names=["toy"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(
            curves=[
                FilterCurve(
                    name="toy",
                    wave=[1000.0, 2000.0, 3000.0],
                    transmission=[0.0, 1.0, 0.0],
                )
            ]
        ),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=32, sfh_n_steps=8),
    )


def test_load_from_samples_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr("jaxsedfit.core.build_model_context", lambda config: SimpleNamespace(mw_ebv=0.03))
    fitter = JAXSEDFit(_minimal_config())
    fitter.samples = {"log_stellar_mass": np.array([10.0, 10.2, 10.4])}
    fitter.predictive = {"pred_fluxes": np.array([[0.9], [1.0], [1.1]])}

    saved_path = fitter.save(tmp_path)
    loaded = JAXSEDFit.load(saved_path)

    assert loaded.config.observation.object_id == "roundtrip"
    assert loaded.config.observation.redshift == 0.2
    assert loaded._loaded_posterior_path == saved_path
    np.testing.assert_allclose(loaded.samples["log_stellar_mass"], [10.0, 10.2, 10.4])
    np.testing.assert_allclose(loaded.predictive["pred_fluxes"], [[0.9], [1.0], [1.1]])


def test_top_level_load_from_samples_accepts_unique_directory(monkeypatch, tmp_path):
    monkeypatch.setattr("jaxsedfit.core.build_model_context", lambda config: SimpleNamespace(mw_ebv=0.0))
    fitter = JAXSEDFit(_minimal_config())
    fitter.samples = {"log_stellar_mass": np.array([9.9])}
    fitter.predictive = {"pred_fluxes": np.array([[1.0]])}
    fitter.save(tmp_path)

    loaded = jaxsedfit.load(tmp_path)

    assert isinstance(loaded, JAXSEDFit)
    np.testing.assert_allclose(loaded.samples["log_stellar_mass"], [9.9])


def test_load_result_wraps_loaded_fitter(monkeypatch, tmp_path):
    monkeypatch.setattr("jaxsedfit.core.build_model_context", lambda config: SimpleNamespace(mw_ebv=0.0))
    fitter = JAXSEDFit(_minimal_config())
    fitter.samples = {"log_stellar_mass": np.array([9.8, 10.0])}
    fitter.predictive = {"pred_fluxes": np.array([[1.0], [1.2]])}
    saved_path = fitter.save(tmp_path)

    result = JAXSEDFit.load_result(saved_path)

    assert isinstance(result, FitResult)
    assert isinstance(result.fitter, JAXSEDFit)
    assert result.path == saved_path
    np.testing.assert_allclose(result.samples["log_stellar_mass"], [9.8, 10.0])
    assert np.isclose(result.median["log_stellar_mass"], 9.9)


def test_fit_result_save_delegates_to_fitter(monkeypatch, tmp_path):
    monkeypatch.setattr("jaxsedfit.core.build_model_context", lambda config: SimpleNamespace(mw_ebv=0.0))
    fitter = JAXSEDFit(_minimal_config())
    fitter.samples = {"log_stellar_mass": np.array([9.8, 10.0])}
    fitter.predictive = {"pred_fluxes": np.array([[1.0], [1.2]])}
    result = fitter._make_result(method="map")

    saved_path = result.save(tmp_path)

    assert result.path == saved_path
    assert saved_path.exists()


def test_load_from_samples_requires_unique_posterior_file(monkeypatch, tmp_path):
    monkeypatch.setattr("jaxsedfit.core.build_model_context", lambda config: SimpleNamespace(mw_ebv=0.0))
    (tmp_path / "a_posterior.pkl").write_bytes(b"")
    (tmp_path / "b_posterior.pkl").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Multiple"):
        JAXSEDFit.load(tmp_path)
