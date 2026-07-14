import numpy as np
import jax.numpy as jnp
import numpyro
import pytest
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.host import HostBasisJax, build_host_state, host_rest_on_basis


def _toy_host_basis(rest_scale=1.0):
    return HostBasisJax(
        ssp_lgmet=jnp.array([-1.5, -1.0, -0.5, 0.0], dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.log10(jnp.array([0.1, 0.5, 1.0], dtype=jnp.float64)),
        rest_llambda=rest_scale
        * jnp.arange(1, 1 + 4 * 3 * 5, dtype=jnp.float64).reshape(4, 3, 5),
        surviving_frac_by_age=jnp.array([0.9, 0.75, 0.6], dtype=jnp.float64),
        n_ly_per_msun=jnp.zeros((4, 3), dtype=jnp.float64),
        ly_lum_per_msun=jnp.zeros((4, 3), dtype=jnp.float64),
        gal_t_table=jnp.geomspace(0.01, 1.2, 16),
    )


def test_public_delayed_host_api_builds_physical_host_state():
    basis = _toy_host_basis()
    prior_config = {
        "log_stellar_mass": {"dist": "normal", "loc": 9.0, "scale": 0.1},
        "mass_metallicity_relation": {"enabled": True, "scale": 5.0},
    }

    def model():
        state = build_host_state(
            basis,
            prior_config,
            host_sfh_model="delayed",
            t_obs_gyr=1.2,
            redshift=0.3,
        )
        for key in [
            "host_rest",
            "host_ssp_weights",
            "host_age_weights",
            "host_lgmet_weights",
            "formed_mass",
        ]:
            numpyro.deterministic(f"public_{key}", state[key])

    tr = trace(
        seed(
            model,
            11,
        )
    ).get_trace()
    state = {
        "host_rest": tr["public_host_rest"]["value"],
        "host_ssp_weights": tr["public_host_ssp_weights"]["value"],
        "host_age_weights": tr["public_host_age_weights"]["value"],
        "host_lgmet_weights": tr["public_host_lgmet_weights"]["value"],
        "formed_mass": tr["public_formed_mass"]["value"],
    }

    assert state["host_rest"].shape == (5,)
    assert state["host_ssp_weights"].shape == (4, 3)
    assert state["host_age_weights"].shape == (3,)
    assert state["host_lgmet_weights"].shape == (4,)
    assert np.isfinite(np.asarray(state["host_rest"], dtype=float)).all()
    assert np.isfinite(float(np.asarray(state["formed_mass"], dtype=float)))
    assert np.isclose(float(np.asarray(jnp.sum(state["host_ssp_weights"]))), 1.0)
    assert "mass_metallicity_relation_prior" in tr
    assert "log_sfh_age_gyr" in tr
    assert "log_sfh_tau_over_age" in tr
    assert "log_sfh_tau_gyr" in tr
    assert tr["log_sfh_tau_over_age"]["type"] == "sample"
    assert tr["log_sfh_tau_gyr"]["type"] == "deterministic"


def test_public_host_rest_on_basis_reuses_sampled_weights():
    basis = _toy_host_basis(rest_scale=1.0)
    alternate_basis = _toy_host_basis(rest_scale=2.0)

    def model():
        state = build_host_state(
            basis,
            {"mass_metallicity_relation": {"enabled": False}},
            host_sfh_model="delayed",
            t_obs_gyr=1.2,
        )
        numpyro.deterministic("public_host_rest", state["host_rest"])
        numpyro.deterministic("public_host_rest_alternate", host_rest_on_basis(state, alternate_basis))

    tr = trace(seed(model, 13)).get_trace()

    host_rest = tr["public_host_rest_alternate"]["value"]

    assert host_rest.shape == (5,)
    assert np.isfinite(np.asarray(host_rest, dtype=float)).all()
    assert np.allclose(np.asarray(host_rest), 2.0 * np.asarray(tr["public_host_rest"]["value"]))


def _fixed_mass_host_trace(log_stellar_mass):
    basis = _toy_host_basis()

    def model():
        state = build_host_state(
            basis,
            {"mass_metallicity_relation": {"enabled": False}},
            host_sfh_model="delayed",
            t_obs_gyr=1.2,
        )
        for key in (
            "host_rest",
            "host_ssp_weights",
            "formed_mass",
            "surviving_mass_fraction",
            "gal_sfr_table",
            "gal_smh_table",
            "log_stellar_mass",
        ):
            numpyro.deterministic(f"mass_test_{key}", state[key])

    fixed = substitute(model, data={"log_stellar_mass": jnp.asarray(log_stellar_mass)})
    return trace(seed(fixed, 27)).get_trace()


def test_host_luminosity_and_histories_scale_linearly_with_stellar_mass():
    low = _fixed_mass_host_trace(8.0)
    high = _fixed_mass_host_trace(9.0)

    for key in ("host_rest", "formed_mass", "gal_sfr_table", "gal_smh_table"):
        low_value = np.asarray(low[f"mass_test_{key}"]["value"], dtype=float)
        high_value = np.asarray(high[f"mass_test_{key}"]["value"], dtype=float)
        np.testing.assert_allclose(high_value, 10.0 * low_value, rtol=2.0e-12, atol=1.0e-20)
    np.testing.assert_allclose(
        np.asarray(high["mass_test_host_ssp_weights"]["value"]),
        np.asarray(low["mass_test_host_ssp_weights"]["value"]),
        rtol=0.0,
        atol=0.0,
    )


def test_host_surviving_and_formed_mass_accounting_is_closed():
    tr = _fixed_mass_host_trace(9.25)
    formed = float(np.asarray(tr["mass_test_formed_mass"]["value"]))
    surviving_fraction = float(np.asarray(tr["mass_test_surviving_mass_fraction"]["value"]))
    target_surviving = 10.0 ** float(np.asarray(tr["mass_test_log_stellar_mass"]["value"]))
    final_history_mass = float(np.asarray(tr["mass_test_gal_smh_table"]["value"])[-1])

    assert formed * surviving_fraction == pytest.approx(target_surviving, rel=2.0e-12)
    assert final_history_mass == pytest.approx(formed, rel=2.0e-12)
