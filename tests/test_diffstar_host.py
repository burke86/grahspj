import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from diffstar import DEFAULT_DIFFSTAR_U_PARAMS, get_bounded_diffstar_params
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
    SpectroscopyData,
)
from jaxsedfit.core import (
    JAXSEDFit,
    _joint_dense_mass_blocks,
    _resolve_dense_mass_structure,
    _uses_spectral_feature_reparameterization,
)
from jaxsedfit.model import (
    _analytic_delayed_burst_age_weights,
    _analytic_delayed_ssp_weights,
    _cigale_delayed_burst_sfh_shape,
    _cigale_delayed_sfh_shape,
    _delayed_sfh_cumulative_mass,
    _default_gal_lgmet_loc,
    _diffstar_ssp_age_weights,
    _flat_lcdm_age_gyr_jax,
    _mass_metallicity_relation_logprior,
    _luminosity_distance_m_jax,
    _ssp_log_age_bin_edges,
    grahsp_photometric_model,
)
from jaxsedfit.preload import build_model_context
from jaxsedfit.results import FitResult, _FitState


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
    assert np.isfinite(float(np.asarray(tr["log_sfr_fit"]["value"])))
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
    assert tr["log_sfh_age_gyr"]["fn"].__class__.__name__ == "Uniform"
    assert tr["log_sfh_tau_gyr"]["type"] == "sample"
    assert tr["log_sfh_tau_gyr"]["fn"].__class__.__name__ == "Uniform"
    assert tr["log_sfh_tau_over_age"]["type"] == "deterministic"
    assert float(np.asarray(tr["log_sfh_age_gyr"]["fn"].support.lower_bound)) == pytest.approx(np.log(10.0**-0.8))
    assert float(np.asarray(tr["log_sfh_age_gyr"]["fn"].support.upper_bound)) == pytest.approx(np.log(min(10.0, context.t_obs_gyr)))
    assert float(np.asarray(tr["log_sfh_tau_gyr"]["fn"].support.lower_bound)) == pytest.approx(np.log(0.1))
    assert float(np.asarray(tr["log_sfh_tau_gyr"]["fn"].support.upper_bound)) == pytest.approx(np.log(10.0))
    assert "u_lgmcrit" not in tr
    assert np.isfinite(float(np.asarray(tr["sfh_age_gyr_fit"]["value"])))
    assert np.isfinite(float(np.asarray(tr["sfh_tau_gyr_fit"]["value"])))
    assert np.all(np.isfinite(np.asarray(tr["gal_sfr_table"]["value"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(tr["gal_smh_table"]["value"], dtype=float)))
    assert float(np.asarray(tr["log_sfr_fit"]["value"])) == pytest.approx(
        np.log10(float(np.asarray(tr["gal_sfr_table"]["value"])[-1]))
    )


def test_default_dale_alpha_prior_matches_grahsp_uniform_grid(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-dale-alpha.h5"
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=False), 12)).get_trace()

    prior = tr["dust_alpha"]["fn"]
    assert prior.__class__.__name__ == "Uniform"
    assert float(np.asarray(prior.support.lower_bound)) == pytest.approx(0.75)
    assert float(np.asarray(prior.support.upper_bound)) == pytest.approx(2.75)


def test_delayed_sfh_matches_cigale_v2025_1_static_reference():
    """Match the normalized no-burst output of CIGALE sfhdelayed.py."""
    elapsed_gyr = np.arange(5000, dtype=float) / 1000.0
    sfr = np.asarray(_cigale_delayed_sfh_shape(elapsed_gyr, 2.0, 5.0))
    normalized_sfr = sfr / (np.sum(sfr) * 1.0e6)
    sample_indices = np.asarray([0, 1, 100, 1000, 2000, 4999])
    expected = np.asarray(
        [
            0.0,
            3.5062740209461115e-13,
            3.336939071574288e-11,
            2.1277262922823895e-10,
            2.5810624634919064e-10,
            1.4402141727293028e-10,
        ]
    )

    np.testing.assert_allclose(normalized_sfr[sample_indices], expected, rtol=2.0e-12, atol=0.0)
    assert np.sum(normalized_sfr) * 1.0e6 == pytest.approx(1.0, rel=2.0e-12)
    assert float(_cigale_delayed_sfh_shape(5.001, 2.0, 5.0)) == 0.0


def test_delayed_sfh_analytic_integral_matches_dense_quadrature():
    elapsed = jnp.linspace(0.0, 3.7, 200_001)
    numerical = jnp.trapezoid(_cigale_delayed_sfh_shape(elapsed, 0.8, 3.7), elapsed)
    analytic = _delayed_sfh_cumulative_mass(3.7, 0.8)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-9, atol=1e-11)


def test_analytic_delayed_ssp_weights_are_normalized_and_differentiable():
    lg_age = jnp.linspace(-4.0, 1.1, 107)
    lgmet = jnp.linspace(-2.0, -1.0, 12)

    def mean_stellar_age(age, tau):
        weights, met_weights, age_weights = _analytic_delayed_ssp_weights(
            age, tau, -1.7, 0.1, lgmet, lg_age
        )
        assert weights.shape == (12, 107)
        return (
            jnp.sum(age_weights * 10.0**lg_age),
            (jnp.sum(weights), jnp.sum(met_weights), jnp.sum(age_weights)),
        )

    (value, sums), gradients = jax.value_and_grad(mean_stellar_age, argnums=(0, 1), has_aux=True)(
        3.7, 0.8
    )
    np.testing.assert_allclose(np.asarray(sums), 1.0, rtol=1e-12)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))


def test_smooth_delayed_burst_sfh_preserves_requested_formed_mass_fraction():
    """The smooth exponential component preserves CIGALE's mass fraction."""
    elapsed_gyr = np.linspace(0.0, 5.0, 5001)
    f_burst = 0.08
    main = np.asarray(_cigale_delayed_sfh_shape(elapsed_gyr, 2.0, 5.0))
    combined = np.asarray(
        _cigale_delayed_burst_sfh_shape(
            elapsed_gyr,
            2.0,
            5.0,
            f_burst,
            0.2,
            0.05,
        )
    )
    burst = combined - main
    recovered_fraction = np.trapezoid(burst, elapsed_gyr) / np.trapezoid(combined, elapsed_gyr)

    assert recovered_fraction == pytest.approx(f_burst, rel=2.0e-12)
    assert np.all(burst >= 0.0)
    assert np.max(burst[elapsed_gyr < 4.78]) < 5.0e-5 * np.max(burst)


def test_analytic_delayed_burst_weights_match_dense_bin_integrals():
    lg_age = jnp.linspace(-3.0, jnp.log10(5.5), 80)
    age, tau = 5.0, 2.0
    burst_fraction, burst_age, burst_tau = 0.08, 0.2, 0.05
    analytic = np.asarray(
        _analytic_delayed_burst_age_weights(
            age, tau, burst_fraction, burst_age, burst_tau, lg_age
        )
    )

    lg_age_np = np.asarray(lg_age)
    edges = np.concatenate(
        (
            [lg_age_np[0] - 0.5 * (lg_age_np[1] - lg_age_np[0])],
            0.5 * (lg_age_np[:-1] + lg_age_np[1:]),
            [lg_age_np[-1] + 0.5 * (lg_age_np[-1] - lg_age_np[-2])],
        )
    )
    elapsed = np.linspace(0.0, age, 500_001)
    sfh_elapsed = np.asarray(
        _cigale_delayed_burst_sfh_shape(
            elapsed, tau, age, burst_fraction, burst_age, burst_tau
        )
    )
    stellar_age = age - elapsed[::-1]
    sfh = sfh_elapsed[::-1]
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (sfh[1:] + sfh[:-1]) * np.diff(stellar_age)))
    )
    mass_at_edges = np.interp(np.clip(10.0**edges, 0.0, age), stellar_age, cumulative)
    numerical = np.diff(mass_at_edges)
    numerical /= numerical.sum()
    np.testing.assert_allclose(analytic, numerical, rtol=2e-3, atol=3e-7)

    gradient = jax.grad(
        lambda fraction: jnp.sum(
            _analytic_delayed_burst_age_weights(
                age, tau, fraction, burst_age, burst_tau, lg_age
            )
            * 10.0**lg_age
        )
    )(burst_fraction)
    assert np.isfinite(float(gradient))


def test_delayed_burst_age_gradient_is_smooth_across_ssp_bin_edge():
    lg_age = jnp.linspace(-3.0, jnp.log10(5.5), 80)
    stellar_ages = 10.0**lg_age
    edge_index = int(np.argmin(np.abs(np.asarray(stellar_ages) - 0.2)))
    edges = 10.0 ** _ssp_log_age_bin_edges(lg_age)
    burst_age_at_edge = edges[edge_index]

    def mean_stellar_age(burst_age):
        weights = _analytic_delayed_burst_age_weights(
            5.0, 2.0, 0.08, burst_age, 0.05, lg_age
        )
        return jnp.sum(weights * stellar_ages)

    gradient = jax.grad(mean_stellar_age)
    epsilon = 1.0e-7
    left = gradient(burst_age_at_edge - epsilon)
    center = gradient(burst_age_at_edge)
    right = gradient(burst_age_at_edge + epsilon)
    curvature = jax.grad(gradient)(burst_age_at_edge)

    assert np.all(np.isfinite(np.asarray([left, center, right, curvature])))
    np.testing.assert_allclose(left, right, rtol=2.0e-3, atol=1.0e-8)


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"u_lg_qt": -2.0, "u_qlglgdt": -2.0},
        {"u_lg_qt": 2.0, "u_qlglgdt": 2.0, "u_lg_drop": 2.0},
        {"u_lgmcrit": -2.0, "u_lgy_at_mcrit": 2.0},
    ],
)
def test_diffstar_ssp_bin_quadrature_matches_64_node_reference(updates):
    u_params = DEFAULT_DIFFSTAR_U_PARAMS._replace(**updates)
    bounded = get_bounded_diffstar_params(u_params)
    lg_age = jnp.linspace(-3.5, jnp.log10(13.5), 107)
    weights16, _ = _diffstar_ssp_age_weights(bounded, lg_age, 13.7)
    nodes64, quad64 = np.polynomial.legendre.leggauss(64)
    weights64, _ = _diffstar_ssp_age_weights(
        bounded, lg_age, 13.7, quad_nodes=nodes64, quad_weights=quad64
    )
    np.testing.assert_allclose(np.asarray(weights16).sum(), 1.0, rtol=1e-12)
    assert float(jnp.sum(jnp.abs(weights16 - weights64))) < 2.0e-4

    gradient = jax.grad(
        lambda lgmcrit: jnp.sum(
            _diffstar_ssp_age_weights(
                get_bounded_diffstar_params(u_params._replace(u_lgmcrit=lgmcrit)),
                lg_age,
                13.7,
            )[0]
            * 10.0**lg_age
        )
    )(u_params.u_lgmcrit)
    assert np.isfinite(float(gradient))


def test_delayed_burst_host_exposes_burst_parameters(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-delayed-burst.h5"
    cfg.galaxy.host_sfh_model = "delayed_burst"
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=False), 12)).get_trace()

    for name in (
        "log_sfh_burst_fraction",
        "log_sfh_burst_age_gyr",
        "log_sfh_burst_tau_gyr",
        "sfh_burst_fraction_fit",
        "sfh_burst_age_gyr_fit",
        "sfh_burst_tau_gyr_fit",
    ):
        assert name in tr
    assert 0.0 < float(np.asarray(tr["sfh_burst_fraction_fit"]["value"])) <= 0.2


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


def test_mass_metallicity_prior_is_disabled_by_default(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    cfg = _mock_config()
    cfg.galaxy.dsps_ssp_fn = "fake-diffstar.h5"
    assert cfg.prior_config.mass_metallicity.enabled is False
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert "mass_metallicity_relation_prior" in tr
    assert np.allclose(np.asarray(tr["mass_metallicity_relation_prior"]["value"], dtype=float), 0.0)
    assert np.allclose(np.asarray(tr["mass_metallicity_relation_logprior"]["value"], dtype=float), 0.0)

    missing_config_logprior = _mass_metallicity_relation_logprior(
        7.0,
        0.0,
        {},
        ssp_lgmet=_SSPData.ssp_lgmet,
        redshift=0.5,
    )
    assert float(np.asarray(missing_config_logprior)) == 0.0


def test_default_metallicity_prior_uses_dsps_absolute_lgmet_grid():
    ssp_lgmet = np.array([-4.34771165, -3.34771165, -2.34771165, -1.34771165])

    default_loc = float(np.asarray(_default_gal_lgmet_loc(ssp_lgmet)))
    assert np.isclose(default_loc, np.log10(0.019) - 0.3)

    low_mass_prior = _mass_metallicity_relation_logprior(
        8.0,
        np.log10(0.019) - 0.85,
        {"mass_metallicity_relation": {"enabled": True}},
        ssp_lgmet=ssp_lgmet,
    )
    old_convention_prior = _mass_metallicity_relation_logprior(
        8.0,
        -0.85,
        {"mass_metallicity_relation": {"enabled": True}},
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
    cfg.prior_config.mass_metallicity = MassMetallicityPriorConfig(enabled=False)
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
    cosmic_age = float(np.asarray(_flat_lcdm_age_gyr_jax(redshift, 70.0, 0.3)))
    age_upper = float(np.exp(np.asarray(tr["log_sfh_age_gyr"]["fn"].support.upper_bound)))
    assert age_upper == pytest.approx(min(10.0, cosmic_age), rel=1.0e-10)


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
    assert calls[0][1]["plot_init"] is False
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
    assert calls == [("optax", {"steps": 2, "learning_rate": 1e-2, "progress_bar": False, "staged": True, "plot_init": False})]

    calls.clear()
    fitter.config.inference.method = "optax"
    fitter.config.inference.staged_map = False
    out = JAXSEDFit.fit(fitter, progress_bar=False)
    assert isinstance(out, FitResult)
    assert out.method == "optax"
    assert calls == [("optax", {"steps": 2, "learning_rate": 1e-2, "progress_bar": False, "staged": False, "plot_init": False})]

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


def test_compact_map_warm_start_preserves_median_and_drops_svi_state(monkeypatch):
    cleared = []
    monkeypatch.setattr(jax, "clear_caches", lambda: cleared.append(True))

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter._fit_state = _FitState()
    median = {
        "scalar": jnp.asarray(1.25),
        "vector": jnp.asarray([2.0, 3.0]),
    }
    fitter.map_result = {
        "params": {"heavy_optimizer_state": jnp.ones((32, 32))},
        "median": median,
        "losses": np.asarray([3.0, 2.0, 1.0]),
        "staged": True,
        "stage1": {"params": {"heavy_stage1_state": jnp.ones((32, 32))}},
    }

    JAXSEDFit._compact_map_warm_start(fitter)

    assert set(fitter.map_result) == {"median", "losses", "staged"}
    assert fitter.map_result["staged"] is True
    np.testing.assert_array_equal(fitter.map_result["median"]["scalar"], 1.25)
    np.testing.assert_array_equal(fitter.map_result["median"]["vector"], [2.0, 3.0])
    np.testing.assert_array_equal(fitter.map_result["losses"], [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(fitter.samples["vector"], [[2.0, 3.0]])
    assert cleared == [True]


def test_fit_nuts_reads_sampler_settings_from_config(monkeypatch):
    captured = {}

    def _fake_nuts(model, **kwargs):
        captured["kernel_kwargs"] = kwargs
        return "kernel"

    class _FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured["mcmc_kernel"] = kernel
            captured["mcmc_kwargs"] = kwargs

        def run(self, rng_key, **kwargs):
            captured["rng_key"] = rng_key
            captured["run_kwargs"] = kwargs

        def print_summary(self):
            captured["print_summary_called"] = True

        def get_samples(self):
            return {"log_stellar_mass": np.array([10.0, 10.2])}

        def get_extra_fields(self, group_by_chain=False):
            assert group_by_chain is True
            return {
                "diverging": np.array([[False, False]]),
                "num_steps": np.array([[7, 15]]),
                "accept_prob": np.array([[0.85, 0.9]]),
                "potential_energy": np.array([[8.0, 8.25]]),
                "energy": np.array([[10.0, 10.5]]),
            }

    monkeypatch.setitem(JAXSEDFit.fit_nuts.__globals__, "NUTS", _fake_nuts)
    monkeypatch.setitem(JAXSEDFit.fit_nuts.__globals__, "MCMC", _FakeMCMC)

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = _mock_config()
    fitter.config.inference.num_warmup = 11
    fitter.config.inference.num_samples = 12
    fitter.config.inference.num_chains = 2
    fitter.config.inference.target_accept_prob = 0.9
    fitter.config.inference.dense_mass = True
    fitter.config.inference.max_tree_depth = 8
    fitter.config.inference.warmup_max_tree_depth = 11
    fitter.map_result = None
    fitter.predictive = {"stale": True}
    fitter._model = lambda: None

    result = JAXSEDFit.fit_nuts(fitter, use_map_init=False, progress_bar=False)

    assert isinstance(result, FitResult)
    assert captured["kernel_kwargs"]["target_accept_prob"] == 0.9
    assert captured["kernel_kwargs"]["dense_mass"] is True
    assert captured["kernel_kwargs"]["max_tree_depth"] == (11, 8)
    assert captured["kernel_kwargs"]["find_heuristic_step_size"] is True
    assert captured["kernel_kwargs"]["init_strategy"] is not None
    assert captured["mcmc_kernel"] == "kernel"
    assert captured["mcmc_kwargs"]["num_warmup"] == 11
    assert captured["mcmc_kwargs"]["num_samples"] == 12
    assert captured["mcmc_kwargs"]["num_chains"] == 2
    assert captured["mcmc_kwargs"]["progress_bar"] is False
    assert captured["run_kwargs"]["extra_fields"] == (
        "num_steps",
        "accept_prob",
        "potential_energy",
        "energy",
    )
    assert captured["print_summary_called"] is True
    assert fitter.predictive is None
    diagnostics = fitter.nuts_result["transition_diagnostics"]
    assert diagnostics["max_num_steps"] == 15
    assert diagnostics["n_max_num_steps"] == 0
    assert diagnostics["max_tree_depth"] == 8
    assert fitter.nuts_result["max_tree_depth"] == (11, 8)


def test_inference_defaults_to_block_dense_mass_adaptation():
    assert InferenceConfig().dense_mass == "blocks"


def test_spectral_feature_reparameterization_uses_sed_agn_flags():
    cfg = _mock_config()
    cfg.spectroscopy = SpectroscopyData(
        wave_obs=[4000.0],
        fluxes=[1.0],
        errors=[0.1],
    )

    assert not _uses_spectral_feature_reparameterization(cfg)

    cfg.agn.fit_feii = True
    assert _uses_spectral_feature_reparameterization(cfg)

    cfg.agn.fit_feii = False
    cfg.agn.fit_balmer_continuum = True
    assert _uses_spectral_feature_reparameterization(cfg)


def test_fit_map_plot_init_plots_both_staged_map_solutions(monkeypatch):
    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = type(
        "_Cfg",
        (),
        {
            "inference": InferenceConfig(map_steps=3, staged_map=True, plot_init=True),
            "output": OutputConfig(),
        },
    )()
    calls = []

    class _SVIResult:
        params = {"x": np.array(0.0)}
        losses = np.array([1.0])

    medians = iter(({"stage1": np.array(1.0)}, {"stage2": np.array(2.0)}))
    monkeypatch.setattr(
        fitter,
        "_run_map_svi",
        lambda *args, **kwargs: (_SVIResult(), next(medians)),
    )
    monkeypatch.setattr(
        fitter,
        "_plot_map_initialization",
        lambda median, **kwargs: calls.append((median, kwargs)),
    )

    fitter.fit_map(progress_bar=False)

    assert [call[1]["attr_prefix"] for call in calls] == ["init_stage1", "init_stage2"]
    assert calls[0][1]["include_sed_agn_features"] is True
    assert calls[0][1]["include_spectral_features"] is True
    assert calls[0][1]["include_spectral_lines"] is False
    assert calls[0][1]["include_spectral_bal"] is False
    assert calls[1][1]["include_sed_agn_features"] is True
    assert calls[1][1]["include_spectral_features"] is True
    assert calls[1][1]["include_spectral_lines"] is True
    assert calls[1][1]["include_spectral_bal"] is True


def test_stage1_map_model_disables_bal_only(monkeypatch):
    import jaxsedfit.core as core_module

    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.context = object()
    captured = {}

    def fake_model(context, **kwargs):
        captured["context"] = context
        captured.update(kwargs)
        return "stage1"

    monkeypatch.setattr(core_module, "grahsp_photometric_model", fake_model)

    assert fitter._continuum_init_model() == "stage1"
    assert captured["context"] is fitter.context
    assert captured["include_spectral_features"] is True
    assert captured["include_spectral_lines"] is False
    assert captured["include_spectral_bal"] is False


def test_joint_dense_mass_blocks_follow_active_sites():
    values = {
        "log_agn_amp": np.array(30.0),
        "pl_slope": np.array(-1.8),
        "log_spectrum_scale": np.array(0.0),
        "cool_lam": np.array(17.0),
        "cool_width": np.array(0.4),
        "log_stellar_mass": np.array(10.0),
        "log_sfh_age_gyr": np.array(0.5),
        "dust_alpha": np.array(2.0),
        "spectral_line_amp_group": np.ones(3),
        "spectral_line_sig_group": np.ones(3),
        "spectral_feii_norm": np.array(1.0),
        "spectral_feii_fwhm": np.array(3000.0),
        "unrelated": np.array(0.0),
    }

    blocks = _joint_dense_mass_blocks(values)

    assert ("spectral_line_amp_group", "spectral_line_sig_group") in blocks
    assert ("spectral_feii_fwhm", "spectral_feii_norm") in blocks
    assert {
        "log_agn_amp",
        "log_sfh_age_gyr",
        "log_stellar_mass",
        "pl_slope",
        "cool_lam",
        "cool_width",
        "dust_alpha",
    } == set(
        next(block for block in blocks if "log_agn_amp" in block)
    )
    assert all("log_spectrum_scale" not in block for block in blocks)
    assert all("unrelated" not in block for block in blocks)
    flattened = [name for block in blocks for name in block]
    assert len(flattened) == len(set(flattened))


def test_dense_mass_structure_accepts_block_and_explicit_modes():
    values = {"log_agn_amp": np.array(30.0), "pl_slope": np.array(-1.8)}
    assert _resolve_dense_mass_structure("blocks", values) == [("log_agn_amp", "pl_slope")]
    assert _resolve_dense_mass_structure("dense", values) is True
    assert _resolve_dense_mass_structure("diagonal", values) is False
    explicit = [("log_agn_amp", "pl_slope")]
    assert _resolve_dense_mass_structure(explicit, values) is explicit


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
    monkeypatch.setattr(
        JAXSEDFit,
        "_compute_predictive",
        lambda self, **kwargs: expected_predictive,
    )

    summary = JAXSEDFit.summary(fitter)
    pred = JAXSEDFit.predict(fitter)

    assert "log_stellar_mass_fit" in summary
    assert np.isclose(summary["log_stellar_mass_fit"], 10.3)
    assert pred is expected_predictive
