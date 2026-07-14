import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from jaxsedfit.config import (
    AGNConfig,
    EmissionLineTemplate,
    FeIITemplate,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    LikelihoodConfig,
    MassMetallicityPriorConfig,
    Observation,
    OutputConfig,
    PhotometryData,
    RedshiftPriorConfig,
)
from jaxsedfit.core import JAXSEDFit
from jaxsedfit.model import _default_gal_lgmet_loc, _mass_metallicity_relation_logprior, _luminosity_distance_m_jax, grahsp_photometric_model
from jaxsedfit.preload import build_model_context
from jaxsedfit.results import FitResult


def _mock_config():
    return FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", rest_wave_max=10000.0, n_wave=2048, sfh_n_steps=16),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1],
                lumin_blagn=[1.0, 0.5],
                lumin_sy2=[0.2, 0.1],
                lumin_liner=[0.1, 0.05],
                wavelength_unit="angstrom",
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

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.galaxy.host_sfh_model = "diffstar"
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
    assert np.any(np.asarray(tr["line_bl_rest_sed"]["value"]) > 0.0)
    assert np.any(np.asarray(tr["line_nl_rest_sed"]["value"]) > 0.0)
    assert np.allclose(np.asarray(tr["line_liner_rest_sed"]["value"]), 0.0)


def test_delayed_host_model_is_default(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-delayed.h5"
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert cfg.galaxy.host_sfh_model == "delayed"
    assert "log_sfh_age_gyr" in tr
    assert "log_sfh_tau_over_age" in tr
    assert "log_sfh_tau_gyr" in tr
    assert tr["log_sfh_tau_over_age"]["type"] == "sample"
    assert tr["log_sfh_tau_gyr"]["type"] == "deterministic"
    assert "u_lgmcrit" not in tr
    assert np.isfinite(float(np.asarray(tr["sfh_age_gyr_fit"]["value"])))
    assert np.isfinite(float(np.asarray(tr["sfh_tau_gyr_fit"]["value"])))
    assert np.all(np.isfinite(np.asarray(tr["gal_sfr_table"]["value"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(tr["gal_smh_table"]["value"], dtype=float)))


def test_delayed_host_priors_respect_physical_and_template_support(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-bounded-delayed.h5"
    cfg.prior_config.host.log_sfh_age_gyr = dist.Normal(0.0, 10.0)
    cfg.prior_config.host.log_sfh_tau_gyr = dist.Normal(0.0, 10.0)
    cfg.prior_config.host.gal_lgmet = dist.Normal(-1.0, 10.0)
    cfg.prior_config.host.dust_alpha = dist.Normal(2.0, 10.0)
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=False), 11)).get_trace()

    assert tr["log_sfh_tau_gyr"]["type"] == "sample"
    assert tr["log_sfh_tau_over_age"]["type"] == "deterministic"
    bounds = {
        "log_sfh_age_gyr": (np.log(cfg.galaxy.sfh_t_min_gyr), np.log(context.t_obs_gyr)),
        "log_sfh_tau_gyr": (np.log(0.03), np.log(30.0)),
        "gal_lgmet": (-2.0, -0.5),
        "dust_alpha": (float(np.min(context.templates.dust_alpha_grid)), float(np.max(context.templates.dust_alpha_grid))),
    }
    for name, (low, high) in bounds.items():
        support = tr[name]["fn"].support
        assert float(np.asarray(support.lower_bound)) == pytest.approx(low)
        assert float(np.asarray(support.upper_bound)) == pytest.approx(high)


def test_delayed_host_rejects_both_tau_prior_parameterizations():
    cfg = _mock_config()
    cfg.prior_config.host.log_sfh_tau_gyr = dist.Normal(0.0, 1.0)
    cfg.prior_config.host.log_sfh_tau_over_age = dist.Normal(0.0, 1.0)

    with pytest.raises(ValueError, match="Configure only one"):
        cfg.validate()


def test_agn_type_2_uses_sy2_narrow_lines_only(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
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

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
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

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.galaxy.use_energy_balance = False
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    dust_rest = np.asarray(tr["dust_rest_sed"]["value"])
    assert np.allclose(dust_rest, 0.0)
    assert float(np.asarray(tr["dust_alpha_fit"]["value"])) == cfg.galaxy.dust_alpha


def test_optional_mass_metallicity_prior_is_exposed(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.prior_config.mass_metallicity = MassMetallicityPriorConfig(
        configured=True,
        enabled=True,
        pivot_mass=10.0,
        pivot_logzsol=-0.2,
        slope=0.3,
        scale=0.2,
    )
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert "mass_metallicity_relation_prior" in tr
    assert np.all(np.isfinite(np.asarray(tr["mass_metallicity_relation_prior"]["value"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(tr["mass_metallicity_relation_logprior"]["value"], dtype=float)))


def test_mass_metallicity_prior_is_enabled_by_default(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert "mass_metallicity_relation_prior" in tr
    assert np.all(np.isfinite(np.asarray(tr["mass_metallicity_relation_prior"]["value"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(tr["mass_metallicity_relation_logprior"]["value"], dtype=float)))


def test_default_metallicity_prior_uses_dsps_absolute_lgmet_grid():
    ssp_lgmet = np.array([-4.34771165, -3.34771165, -2.34771165, -1.34771165])

    default_loc = float(np.asarray(_default_gal_lgmet_loc(ssp_lgmet)))
    assert np.isclose(default_loc, np.log10(0.019) - 0.3)

    low_mass_prior = _mass_metallicity_relation_logprior(
        8.0,
        np.log10(0.019) - 0.85,
        {},
        ssp_lgmet=ssp_lgmet,
    )
    old_convention_prior = _mass_metallicity_relation_logprior(
        8.0,
        -0.85,
        {},
        ssp_lgmet=ssp_lgmet,
    )

    assert np.isfinite(float(np.asarray(low_mass_prior)))
    assert float(np.asarray(low_mass_prior)) > float(np.asarray(old_convention_prior))


def test_mass_metallicity_prior_can_be_disabled(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.prior_config.mass_metallicity = MassMetallicityPriorConfig(configured=True, enabled=False)
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert "mass_metallicity_relation_prior" in tr
    assert np.allclose(np.asarray(tr["mass_metallicity_relation_prior"]["value"], dtype=float), 0.0)
    assert np.allclose(np.asarray(tr["mass_metallicity_relation_logprior"]["value"], dtype=float), 0.0)


def test_uniform_log_stellar_mass_prior_is_supported(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.prior_config.stellar_mass = dist.Uniform(6.0, 8.0)
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    log_stellar_mass = float(np.asarray(tr["log_stellar_mass"]["value"]))
    assert 6.0 <= log_stellar_mass <= 8.0


def test_tabulated_redshift_pdf_prior_is_supported(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    cfg.observation.redshift_mode = "fit"
    cfg.prior_config.redshift = RedshiftPriorConfig(
        z_grid=[0.05, 0.1, 0.2, 0.4],
        pdf=[0.0, 1.0, 3.0, 0.0],
    )
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
    fitter = JAXSEDFit.__new__(JAXSEDFit)
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
    summary = JAXSEDFit.summary(fitter)

    assert "log_stellar_mass_fit" in summary
    assert "host_age_weighted_gyr" in summary
    assert "host_lgmet_weighted" in summary
    assert summary["log_stellar_mass_fit"] > 0.0


def test_fit_dispatch_methods(monkeypatch):
    fitter = JAXSEDFit.__new__(JAXSEDFit)
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

    monkeypatch.setattr(JAXSEDFit, "fit_map", _fit_map)
    monkeypatch.setattr(JAXSEDFit, "fit_nuts", _fit_nuts)
    monkeypatch.setattr(JAXSEDFit, "fit_ns", _fit_ns)
    fitter.config = type("_Cfg", (), {"inference": InferenceConfig(method="optax+nuts"), "output": OutputConfig()})()
    fitter.config.inference.map_steps = 7
    fitter.config.inference.learning_rate = 1e-2
    fitter.config.inference.num_warmup = 3
    fitter.config.inference.num_samples = 4
    fitter.config.inference.dense_mass = True
    fitter.config.inference.max_tree_depth = 10

    out = JAXSEDFit.fit(fitter, progress_bar=True)
    assert isinstance(out, FitResult)
    assert out.method == "optax+nuts"
    assert calls[0][0] == "optax"
    assert calls[0][1]["steps"] == 7
    assert calls[0][1]["progress_bar"] is True
    assert calls[0][1]["staged"] is True
    assert calls[1][0] == "nuts"
    assert calls[1][1]["num_warmup"] == 3
    assert calls[1][1]["num_samples"] == 4
    assert calls[1][1]["dense_mass"] is True
    assert calls[1][1]["max_tree_depth"] == 10
    assert calls[1][1]["use_map_init"] is True
    assert calls[1][1]["progress_bar"] is True

    calls.clear()
    fitter.config.inference.method = "optax"
    fitter.config.inference.map_steps = 2
    out = JAXSEDFit.fit(fitter, progress_bar=False)
    assert isinstance(out, FitResult)
    assert out.method == "optax"
    assert calls == [("optax", {"steps": 2, "learning_rate": 1e-2, "progress_bar": False, "staged": True})]

    calls.clear()
    fitter.config.inference.method = "optax"
    fitter.config.inference.staged_map = False
    out = JAXSEDFit.fit(fitter, progress_bar=False)
    assert isinstance(out, FitResult)
    assert out.method == "optax"
    assert calls == [("optax", {"steps": 2, "learning_rate": 1e-2, "progress_bar": False, "staged": False})]

    calls.clear()
    fitter.config.inference.method = "nuts"
    fitter.config.inference.num_warmup = 2
    out = JAXSEDFit.fit(fitter, progress_bar=False)
    assert isinstance(out, FitResult)
    assert out.method == "nuts"
    assert calls == [
        (
            "nuts",
            {
                "num_warmup": 2,
                "num_samples": 4,
                "num_chains": 1,
                "target_accept_prob": 0.85,
                "dense_mass": True,
                "max_tree_depth": 10,
                "use_map_init": True,
                "progress_bar": False,
            },
        )
    ]

    calls.clear()
    fitter.config.inference.method = "ns"
    fitter.config.inference.ns_num_live_points = 25
    fitter.config.inference.ns_max_samples = 200
    fitter.config.inference.ns_dlogz = 0.1
    fitter.config.inference.ns_resamples = 30
    fitter.config.inference.ns_difficult_model = True
    fitter.config.inference.ns_parameter_estimation = True
    fitter.config.inference.ns_num_parallel_workers = 3
    fitter.config.inference.ns_init_efficiency_threshold = 0.2
    fitter.config.inference.ns_max_likelihood_evals = 5000
    fitter.config.inference.ns_efficiency_threshold = 0.001
    out = JAXSEDFit.fit(fitter, progress_bar=False)
    assert isinstance(out, FitResult)
    assert out.method == "ns"
    assert calls == [
        (
            "ns",
            {
                "num_live_points": 25,
                "max_samples": 200,
                "dlogz": 0.1,
                "num_resamples": 30,
                "difficult_model": True,
                "parameter_estimation": True,
                "num_parallel_workers": 3,
                "init_efficiency_threshold": 0.2,
                "max_likelihood_evals": 5000,
                "efficiency_threshold": 0.001,
                "progress_bar": False,
            },
        )
    ]


def test_fit_nuts_reads_sampler_settings_from_config(monkeypatch):
    captured = {}

    def _fake_nuts(model, **kwargs):
        captured["kernel_kwargs"] = kwargs
        return "kernel"

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured["mcmc_kernel"] = kernel
            captured["mcmc_kwargs"] = kwargs

        def run(self, rng_key):
            captured["rng_key"] = rng_key

        def get_samples(self):
            return {"log_stellar_mass": np.array([10.0, 10.2])}

    monkeypatch.setitem(JAXSEDFit.fit_nuts.__globals__, "NUTS", _fake_nuts)
    monkeypatch.setitem(JAXSEDFit.fit_nuts.__globals__, "MCMC", _FakeMCMC)

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = _mock_config()
    fitter.config.inference.num_warmup = 11
    fitter.config.inference.num_samples = 12
    fitter.config.inference.num_chains = 2
    fitter.config.inference.target_accept_prob = 0.9
    fitter.config.inference.dense_mass = True
    fitter.config.inference.max_tree_depth = 10
    fitter.map_result = None
    fitter.predictive = {"stale": True}
    fitter._model = lambda: None

    result = JAXSEDFit.fit_nuts(fitter, use_map_init=False, progress_bar=False)

    assert isinstance(result, FitResult)
    assert captured["kernel_kwargs"]["target_accept_prob"] == 0.9
    assert captured["kernel_kwargs"]["dense_mass"] is True
    assert captured["kernel_kwargs"]["max_tree_depth"] == 10
    assert captured["mcmc_kernel"] == "kernel"
    assert captured["mcmc_kwargs"]["num_warmup"] == 11
    assert captured["mcmc_kwargs"]["num_samples"] == 12
    assert captured["mcmc_kwargs"]["num_chains"] == 2
    assert captured["mcmc_kwargs"]["progress_bar"] is False
    assert fitter.predictive is None


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
            assert num_samples == 7
            assert group_by_chain is False
            return {
                "log_stellar_mass": np.linspace(10.0, 10.4, num_samples),
                "host_age_weights": np.tile(np.array([[0.2, 0.8]]), (num_samples, 1)),
                "host_lgmet_weights": np.tile(np.array([[0.6, 0.4]]), (num_samples, 1)),
            }

    monkeypatch.setitem(JAXSEDFit.fit_ns.__globals__, "_get_nested_sampler_cls", lambda: _FakeNestedSampler)

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = _mock_config()
    fitter.config.inference.num_samples = 5
    fitter.predictive = {"stale": True}
    fitter._model = lambda: None

    result = JAXSEDFit.fit_ns(
        fitter,
        num_live_points=17,
        max_samples=123,
        dlogz=0.05,
        ns_difficult_model=True,
        ns_parameter_estimation=True,
        ns_num_parallel_workers=2,
        ns_init_efficiency_threshold=0.15,
        ns_max_likelihood_evals=1000,
        ns_efficiency_threshold=0.01,
        ns_resamples=7,
        progress_bar=False,
    )

    assert isinstance(result, FitResult)
    assert result.method == "ns"
    assert result.samples is fitter.samples
    assert fitter.ns_result["results"] == {"status": "ok"}
    assert fitter.ns_result["constructor_kwargs"]["num_live_points"] == 17
    assert fitter.ns_result["constructor_kwargs"]["max_samples"] == 123
    assert fitter.ns_result["constructor_kwargs"]["verbose"] is False
    assert fitter.ns_result["constructor_kwargs"]["difficult_model"] is True
    assert fitter.ns_result["constructor_kwargs"]["parameter_estimation"] is True
    assert fitter.ns_result["constructor_kwargs"]["num_parallel_workers"] == 2
    assert fitter.ns_result["constructor_kwargs"]["init_efficiency_threshold"] == 0.15
    assert fitter.ns_result["termination_kwargs"]["dlogZ"] == 0.05
    assert fitter.ns_result["termination_kwargs"]["max_num_likelihood_evaluations"] == 1000
    assert fitter.ns_result["termination_kwargs"]["efficiency_threshold"] == 0.01
    assert fitter.ns_result["num_resamples"] == 7
    assert set(fitter.samples) == {"log_stellar_mass", "host_age_weights", "host_lgmet_weights"}
    assert fitter.samples["log_stellar_mass"].shape == (7,)
    assert fitter.predictive is None


def test_fit_ns_reads_sampler_settings_from_config(monkeypatch):
    class _FakeNestedSampler:
        def __init__(self, model, *, constructor_kwargs=None, termination_kwargs=None):
            self.model = model
            self.constructor_kwargs = constructor_kwargs or {}
            self.termination_kwargs = termination_kwargs or {}
            self._results = {"status": "ok"}

        def run(self, rng_key, *args, **kwargs):
            return None

        def get_samples(self, rng_key, num_samples, *, group_by_chain=False):
            assert num_samples == 9
            return {"log_stellar_mass": np.linspace(10.0, 10.4, num_samples)}

    monkeypatch.setitem(JAXSEDFit.fit_ns.__globals__, "_get_nested_sampler_cls", lambda: _FakeNestedSampler)

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = _mock_config()
    fitter.config.inference.ns_num_live_points = 21
    fitter.config.inference.ns_max_samples = 321
    fitter.config.inference.ns_dlogz = 0.07
    fitter.config.inference.ns_resamples = 9
    fitter.config.inference.ns_difficult_model = True
    fitter.config.inference.ns_parameter_estimation = True
    fitter.config.inference.ns_num_parallel_workers = 3
    fitter.config.inference.ns_init_efficiency_threshold = 0.2
    fitter.config.inference.ns_max_likelihood_evals = 2000
    fitter.config.inference.ns_efficiency_threshold = 0.02
    fitter.predictive = {"stale": True}
    fitter._model = lambda: None

    result = JAXSEDFit.fit_ns(fitter, progress_bar=False)

    assert isinstance(result, FitResult)
    assert fitter.ns_result["constructor_kwargs"]["num_live_points"] == 21
    assert fitter.ns_result["constructor_kwargs"]["max_samples"] == 321
    assert fitter.ns_result["constructor_kwargs"]["difficult_model"] is True
    assert fitter.ns_result["constructor_kwargs"]["parameter_estimation"] is True
    assert fitter.ns_result["constructor_kwargs"]["num_parallel_workers"] == 3
    assert fitter.ns_result["constructor_kwargs"]["init_efficiency_threshold"] == 0.2
    assert fitter.ns_result["termination_kwargs"]["dlogZ"] == 0.07
    assert fitter.ns_result["termination_kwargs"]["max_num_likelihood_evaluations"] == 2000
    assert fitter.ns_result["termination_kwargs"]["efficiency_threshold"] == 0.02
    assert fitter.ns_result["num_resamples"] == 9
    assert fitter.predictive is None


def test_fit_ns_passes_explicit_none_max_samples(monkeypatch):
    captured = {}

    class _FakeNestedSampler:
        def __init__(self, model, *, constructor_kwargs=None, termination_kwargs=None):
            captured["constructor_kwargs"] = constructor_kwargs or {}
            self._results = {"status": "ok"}

        def run(self, rng_key, *args, **kwargs):
            return None

        def get_samples(self, rng_key, num_samples, *, group_by_chain=False):
            return {"log_stellar_mass": np.linspace(10.0, 10.4, num_samples)}

    monkeypatch.setitem(JAXSEDFit.fit_ns.__globals__, "_get_nested_sampler_cls", lambda: _FakeNestedSampler)

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = _mock_config()
    fitter.config.inference.num_samples = 5
    fitter.predictive = None
    fitter._model = lambda: None

    JAXSEDFit.fit_ns(fitter, num_live_points=17, progress_bar=False)

    assert captured["constructor_kwargs"]["max_samples"] is None


def test_ns_samples_work_with_summary_and_predict(monkeypatch):
    fitter = JAXSEDFit.__new__(JAXSEDFit)
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
    monkeypatch.setattr(JAXSEDFit, "_compute_predictive", lambda self: expected_predictive)

    summary = JAXSEDFit.summary(fitter)
    pred = JAXSEDFit.predict(fitter)

    assert "log_stellar_mass_fit" in summary
    assert np.isclose(summary["log_stellar_mass_fit"], 10.3)
    assert pred is expected_predictive
