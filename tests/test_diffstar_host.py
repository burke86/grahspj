import numpy as np
from numpyro.handlers import seed, trace

from grahspj.config import (
    AGNConfig,
    EmissionLineTemplate,
    FeIITemplate,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    LikelihoodConfig,
    Observation,
    PhotometryData,
)
from grahspj.core import GRAHSPJ
from grahspj.model import _luminosity_distance_m_jax, grahsp_photometric_model
from grahspj.preload import build_model_context


def _mock_config():
    return FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, sfh_n_steps=16),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5]),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1],
                lumin_blagn=[1.0, 0.5],
                lumin_sy2=[0.2, 0.1],
                lumin_liner=[0.1, 0.05],
            ),
        ),
        likelihood=LikelihoodConfig(),
        inference=InferenceConfig(map_steps=2),
    )


def test_diffstar_host_model_exposes_log_stellar_mass(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert "log_stellar_mass" in tr
    assert "log_host_amp" not in tr
    assert np.all(np.isfinite(np.asarray(tr["host_age_weights"]["value"])))
    assert np.all(np.isfinite(np.asarray(tr["host_lgmet_weights"]["value"])))
    assert np.isfinite(float(np.asarray(tr["formed_stellar_mass"]["value"])))
    assert np.isfinite(float(np.asarray(tr["log_dust_luminosity_fit"]["value"])))
    assert np.all(np.isfinite(np.asarray(tr["host_absorbed_rest_sed"]["value"])))
    assert np.all(np.isfinite(np.asarray(tr["dust_rest_sed"]["value"])))
    assert np.all(np.asarray(tr["dust_rest_sed"]["value"]) >= 0.0)
    assert np.isfinite(float(np.asarray(tr["absolute_flux_scale_logprior"]["value"])))
    assert np.any(np.asarray(tr["line_bl_rest_sed"]["value"]) > 0.0)
    assert np.any(np.asarray(tr["line_nl_rest_sed"]["value"]) > 0.0)
    assert np.allclose(np.asarray(tr["line_liner_rest_sed"]["value"]), 0.0)


def test_agn_type_2_uses_sy2_narrow_lines_only(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.agn.agn_type = 2
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert np.allclose(np.asarray(tr["line_bl_rest_sed"]["value"]), 0.0)
    assert np.any(np.asarray(tr["line_nl_rest_sed"]["value"]) > 0.0)
    assert np.allclose(np.asarray(tr["line_liner_rest_sed"]["value"]), 0.0)
    assert np.allclose(np.asarray(tr["feii_rest_sed"]["value"]), 0.0)
    assert np.allclose(np.asarray(tr["balmer_rest_sed"]["value"]), 0.0)


def test_agn_type_3_uses_liner_lines_only(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.agn.agn_type = 3
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert np.allclose(np.asarray(tr["line_bl_rest_sed"]["value"]), 0.0)
    assert np.allclose(np.asarray(tr["line_nl_rest_sed"]["value"]), 0.0)
    assert np.any(np.asarray(tr["line_liner_rest_sed"]["value"]) > 0.0)
    assert np.allclose(np.asarray(tr["feii_rest_sed"]["value"]), 0.0)
    assert np.allclose(np.asarray(tr["balmer_rest_sed"]["value"]), 0.0)


def test_energy_balance_can_be_disabled(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.galaxy.use_energy_balance = False
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    dust_rest = np.asarray(tr["dust_rest_sed"]["value"])
    assert np.allclose(dust_rest, 0.0)
    assert float(np.asarray(tr["dust_alpha_fit"]["value"])) == cfg.galaxy.dust_alpha


def test_tabulated_redshift_pdf_prior_is_supported(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.observation.fit_redshift = True
    cfg.prior_config["redshift_pdf"] = {
        "z_grid": [0.05, 0.1, 0.2, 0.4],
        "pdf": [0.0, 1.0, 3.0, 0.0],
    }
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context), 0)).get_trace()

    redshift = float(np.asarray(tr["redshift"]["value"]))
    assert 0.05 <= redshift <= 0.4
    assert "redshift_pdf_prior" in tr
    prior_value = np.asarray(tr["redshift_pdf_prior"]["value"], dtype=float)
    assert np.all(np.isfinite(prior_value))


def test_luminosity_distance_jax_depends_on_redshift():
    d_lo = float(np.asarray(_luminosity_distance_m_jax(0.05, 70.0, 0.3)))
    d_hi = float(np.asarray(_luminosity_distance_m_jax(1.5, 70.0, 0.3)))

    assert np.isfinite(d_lo)
    assert np.isfinite(d_hi)
    assert d_hi > d_lo > 0.0


def test_summary_uses_log_stellar_mass_and_host_weights():
    fitter = GRAHSPJ.__new__(GRAHSPJ)
    fitter.samples = {
        "log_stellar_mass": np.array([10.2, 10.4]),
        "host_age_weights": np.array([[0.2, 0.8], [0.3, 0.7]]),
        "host_lgmet_weights": np.array([[0.6, 0.4], [0.5, 0.5]]),
        "gal_lgmet": np.array([-0.4, -0.3]),
        "gal_lgmet_scatter": np.array([0.1, 0.2]),
    }
    fitter.predictive = None
    fitter.context = type(
        "_Context",
        (),
        {
            "ssp_data": type(
                "_SSP",
                (),
                {
                    "ssp_lg_age_gyr": np.array([-1.0, 0.0]),
                    "ssp_lgmet": np.array([-1.0, 0.0]),
                },
            )()
        },
    )()
    summary = GRAHSPJ.summary(fitter)

    assert "log_stellar_mass_fit" in summary
    assert "host_age_weighted_gyr" in summary
    assert "host_lgmet_weighted" in summary
    assert summary["log_stellar_mass_fit"] > 0.0


def test_fit_dispatch_methods(monkeypatch):
    fitter = GRAHSPJ.__new__(GRAHSPJ)
    calls = []

    def _fit_map(self, **kwargs):
        calls.append(("optax", kwargs))
        return {"median": {"log_stellar_mass": 10.0}}

    def _fit_nuts(self, **kwargs):
        calls.append(("nuts", kwargs))
        return {"mcmc": "ok"}

    def _fit_ns(self, **kwargs):
        calls.append(("ns", kwargs))
        return {"nested": "ok"}

    monkeypatch.setattr(GRAHSPJ, "fit_map", _fit_map)
    monkeypatch.setattr(GRAHSPJ, "fit_nuts", _fit_nuts)
    monkeypatch.setattr(GRAHSPJ, "fit_ns", _fit_ns)

    out = GRAHSPJ.fit(fitter, fit_method="optax+nuts", progress_bar=True, steps=7, learning_rate=1e-2, num_warmup=3, num_samples=4)
    assert list(out) == ["map", "nuts"]
    assert calls[0][0] == "optax"
    assert calls[0][1]["steps"] == 7
    assert calls[0][1]["progress_bar"] is True
    assert calls[1][0] == "nuts"
    assert calls[1][1]["num_warmup"] == 3
    assert calls[1][1]["num_samples"] == 4
    assert calls[1][1]["progress_bar"] is True

    calls.clear()
    GRAHSPJ.fit(fitter, fit_method="optax", progress_bar=False, steps=2)
    assert calls == [("optax", {"steps": 2, "progress_bar": False})]

    calls.clear()
    GRAHSPJ.fit(fitter, fit_method="nuts", progress_bar=False, num_warmup=2)
    assert calls == [("nuts", {"num_warmup": 2, "progress_bar": False})]

    calls.clear()
    GRAHSPJ.fit(
        fitter,
        fit_method="ns",
        progress_bar=False,
        ns_live_points=25,
        ns_max_samples=200,
        ns_dlogz=0.1,
    )
    assert calls == [
        (
            "ns",
            {
                "num_live_points": 25,
                "max_samples": 200,
                "dlogz": 0.1,
                "progress_bar": False,
            },
        )
    ]


def test_fit_ns_populates_samples(monkeypatch):
    class _FakeNestedSampler:
        def __init__(self, model, *, constructor_kwargs=None, termination_kwargs=None):
            self.model = model
            self.constructor_kwargs = constructor_kwargs or {}
            self.termination_kwargs = termination_kwargs or {}
            self._results = {"status": "ok"}
            self.run_args = None

        def run(self, rng_key, *args, **kwargs):
            self.run_args = (rng_key, args, kwargs)

        def get_samples(self, rng_key, num_samples, *, group_by_chain=False):
            assert num_samples == 5
            assert group_by_chain is False
            return {
                "log_stellar_mass": np.linspace(10.0, 10.4, num_samples),
                "host_age_weights": np.tile(np.array([[0.2, 0.8]]), (num_samples, 1)),
                "host_lgmet_weights": np.tile(np.array([[0.6, 0.4]]), (num_samples, 1)),
            }

    monkeypatch.setattr("grahspj.core._get_nested_sampler_cls", lambda: _FakeNestedSampler)

    fitter = GRAHSPJ.__new__(GRAHSPJ)
    fitter.config = _mock_config()
    fitter.config.inference.num_samples = 5
    fitter.predictive = {"stale": True}
    fitter._model = lambda: None

    result = GRAHSPJ.fit_ns(
        fitter,
        num_live_points=17,
        max_samples=123,
        dlogz=0.05,
        progress_bar=False,
    )

    assert result["results"] == {"status": "ok"}
    assert result["constructor_kwargs"]["num_live_points"] == 17
    assert result["constructor_kwargs"]["max_samples"] == 123
    assert result["constructor_kwargs"]["verbose"] is False
    assert result["termination_kwargs"]["dlogZ"] == 0.05
    assert fitter.ns_result is result
    assert set(fitter.samples) == {"log_stellar_mass", "host_age_weights", "host_lgmet_weights"}
    assert fitter.samples["log_stellar_mass"].shape == (5,)
    assert fitter.predictive is None


def test_ns_samples_work_with_summary_and_predict(monkeypatch):
    fitter = GRAHSPJ.__new__(GRAHSPJ)
    fitter.samples = {
        "log_stellar_mass": np.array([10.2, 10.4]),
        "host_age_weights": np.array([[0.2, 0.8], [0.3, 0.7]]),
        "host_lgmet_weights": np.array([[0.6, 0.4], [0.5, 0.5]]),
    }
    fitter.predictive = None
    fitter.context = type(
        "_Context",
        (),
        {
            "ssp_data": type(
                "_SSP",
                (),
                {
                    "ssp_lg_age_gyr": np.array([-1.0, 0.0]),
                    "ssp_lgmet": np.array([-1.0, 0.0]),
                },
            )()
        },
    )()

    expected_predictive = {"pred_fluxes": np.array([[1.0, 2.0]])}
    monkeypatch.setattr(GRAHSPJ, "_compute_predictive", lambda self: expected_predictive)

    summary = GRAHSPJ.summary(fitter)
    pred = GRAHSPJ.predict(fitter)

    assert "log_stellar_mass_fit" in summary
    assert np.isclose(summary["log_stellar_mass_fit"], 10.3)
    assert pred is expected_predictive
