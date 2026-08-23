from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

from jaxsedfit.core import JAXSEDFit
from jaxsedfit.results import _FitState


def _bare_fitter(samples, *, seed=3):
    fitter = object.__new__(JAXSEDFit)
    fitter.config = SimpleNamespace(inference=SimpleNamespace(seed=seed))
    fitter.context = SimpleNamespace()
    fitter._fit_state = _FitState(samples=samples)
    return fitter


def _small_model(context, *, include_components=False, force_component_fluxes=False):
    del context, include_components, force_component_fluxes
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))
    scatter = numpyro.sample("scatter", dist.Normal(theta, 0.1))
    numpyro.deterministic("pred_fluxes", jnp.asarray([theta, 2.0 * theta]))
    numpyro.deterministic("sed_chi2", theta**2)
    numpyro.deterministic("scattered_flux", scatter)


def test_streamed_predictive_matches_numpyro_bulk_prediction(monkeypatch):
    samples = {"theta": np.asarray([0.5, 1.5, -2.0])}
    fitter = _bare_fitter(samples, seed=11)
    monkeypatch.setattr("jaxsedfit.core.grahsp_photometric_model", _small_model)
    monkeypatch.setattr(
        fitter,
        "_predictive_return_sites",
        lambda kind: ["pred_fluxes", "sed_chi2", "scattered_flux"],
    )

    rng_key = jax.random.PRNGKey(fitter.config.inference.seed + 17)
    expected = Predictive(
        lambda: _small_model(None),
        posterior_samples=samples,
        return_sites=["pred_fluxes", "sed_chi2", "scattered_flux"],
    )(rng_key)

    actual = fitter.predict(kind="photometry")

    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(
            actual[key],
            np.asarray(expected[key]),
            rtol=1.0e-14,
            atol=0.0,
        )


def test_streamed_predictive_calls_numpyro_once_per_draw(monkeypatch):
    samples = {
        "theta": np.asarray([1.0, 2.0, 3.0]),
        "fixed": np.asarray(4.0),
    }
    fitter = _bare_fitter(samples, seed=7)
    calls = []

    class FakePredictive:
        def __init__(self, model, *, posterior_samples, return_sites):
            del model, return_sites
            self.posterior_samples = posterior_samples

        def __call__(self, rng_key):
            theta = np.asarray(self.posterior_samples["theta"])
            calls.append(
                {
                    "rng_key": np.asarray(rng_key),
                    "theta": theta.copy(),
                    "fixed": np.asarray(self.posterior_samples["fixed"]).copy(),
                }
            )
            return {
                "pred_fluxes": np.stack((theta, 2.0 * theta), axis=-1),
                "sed_chi2": theta**2,
            }

    monkeypatch.setattr("jaxsedfit.core.Predictive", FakePredictive)

    prediction = fitter.predict(kind="photometry")

    assert len(calls) == 3
    expected_keys = np.asarray(
        jax.random.split(jax.random.PRNGKey(fitter.config.inference.seed + 17), 3)
    )
    for index, call in enumerate(calls):
        assert call["theta"].shape == (1,)
        np.testing.assert_array_equal(call["theta"], [samples["theta"][index]])
        np.testing.assert_array_equal(call["fixed"], samples["fixed"])
        np.testing.assert_array_equal(call["rng_key"], expected_keys[index])
    np.testing.assert_array_equal(
        prediction["pred_fluxes"],
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
    )
    np.testing.assert_array_equal(prediction["sed_chi2"], [1.0, 4.0, 9.0])

    cached = fitter.predict(kind="photometry")
    assert len(calls) == 3
    np.testing.assert_array_equal(cached["pred_fluxes"], prediction["pred_fluxes"])


def test_single_draw_prediction_preserves_unsplit_rng_key(monkeypatch):
    fitter = _bare_fitter({"theta": np.asarray([2.0])}, seed=5)
    observed_keys = []

    class FakePredictive:
        def __init__(self, model, *, posterior_samples, return_sites):
            del model, return_sites
            self.posterior_samples = posterior_samples

        def __call__(self, rng_key):
            observed_keys.append(np.asarray(rng_key))
            return {"pred_fluxes": np.asarray(self.posterior_samples["theta"])[:, None]}

    monkeypatch.setattr("jaxsedfit.core.Predictive", FakePredictive)

    fitter.predict(kind="photometry")

    np.testing.assert_array_equal(
        observed_keys[0],
        np.asarray(jax.random.PRNGKey(fitter.config.inference.seed + 17)),
    )


def test_streamed_predictive_rejects_inconsistent_draw_dimensions():
    fitter = _bare_fitter(
        {
            "theta": np.asarray([1.0, 2.0]),
            "scale": np.asarray([1.0, 2.0, 3.0]),
        }
    )

    with pytest.raises(ValueError, match="share one leading draw dimension"):
        fitter.predict(kind="photometry")
