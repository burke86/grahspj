from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import reparam, scope, seed, substitute, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

from jaxsedfit.core import (
    _AdditivePivotReparam,
    _joint_dense_mass_blocks,
    _nuts_geometry_reparam_config,
    _physical_nuts_samples,
    _prepare_nuts_reparameterization,
    _remap_dense_mass_sites,
)
from jaxsedfit.inference import (
    nuts_metric_diagnostics,
    nuts_transition_diagnostics,
)
from jaxsedfit.config import InferenceConfig
from jaxsedfit.model import _spectrum_continuum_log_pivot


def _pivoted_scalar_model():
    offset = jnp.asarray(1.25)
    value = numpyro.sample(
        "scale",
        dist.Normal(-0.2, 0.7),
        infer={
            "jaxsedfit_additive_pivot": {
                "offset": offset,
                "auxiliary_name": "scale_pivot",
            }
        },
    )
    numpyro.factor("toy_likelihood", -0.5 * ((value - 0.4) / 0.3) ** 2)


def _dependent_pivot_model():
    continuum = numpyro.sample("continuum", dist.Normal(0.1, 0.8))
    offset = 0.3 * continuum + 0.2 * continuum**2
    scale = numpyro.sample(
        "scale",
        dist.Normal(-0.2, 0.7),
        infer={
            "jaxsedfit_additive_pivot": {
                "offset": offset,
                "auxiliary_name": "scale_pivot",
            }
        },
    )
    numpyro.factor("toy_likelihood", -0.5 * ((continuum + scale) / 0.4) ** 2)


def _embedded_jaxqsofit_feature_model():
    feii_norm = numpyro.sample(
        "jqf_feii_norm",
        dist.LogNormal(jnp.log(1.0e-3), 2.0),
        infer={
            "jaxqsofit_normal_lognormal_standardization": {
                "auxiliary_name": "jqf_feii_norm_std",
            }
        },
    )
    feii_shift = numpyro.sample(
        "jqf_feii_shift",
        dist.Normal(0.0, 0.01),
        infer={
            "jaxqsofit_normal_lognormal_standardization": {
                "auxiliary_name": "jqf_feii_shift_std",
            }
        },
    )
    numpyro.factor(
        "feature_likelihood",
        -0.5 * jnp.square(feii_norm + feii_shift),
    )


def test_spectrum_continuum_log_pivot_is_smooth_and_per_spectrum():
    continuum = jnp.asarray([2.0, 4.0, 3.0, 6.0])
    observed = jnp.asarray([1.0, 2.0, 1.0, 2.0])
    mask = jnp.asarray([True, True, True, True])
    spectrum_index = jnp.asarray([0, 0, 1, 1])

    pivot = _spectrum_continuum_log_pivot(
        continuum,
        observed,
        mask,
        spectrum_index,
        2,
    )
    np.testing.assert_allclose(pivot, np.log([2.0, 3.0]), rtol=1.0e-10)

    gradient = jax.jacrev(
        lambda values: _spectrum_continuum_log_pivot(
            values,
            observed,
            mask,
            spectrum_index,
            2,
        )
    )(continuum)
    assert gradient.shape == (2, 4)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(np.asarray(gradient[0, 2:]), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(gradient[1, :2]), 0.0, atol=1.0e-14)


def test_spectrum_continuum_log_pivot_masks_nonfinite_values_and_gradients():
    continuum = jnp.asarray([2.0, jnp.nan])
    observed = jnp.asarray([1.0, jnp.nan])
    mask = jnp.asarray([True, False])
    spectrum_index = jnp.asarray([0, 0])

    pivot = _spectrum_continuum_log_pivot(
        continuum,
        observed,
        mask,
        spectrum_index,
        1,
    )
    gradient = jax.grad(
        lambda values: _spectrum_continuum_log_pivot(
            values,
            observed,
            mask,
            spectrum_index,
            1,
        )
    )(continuum)

    np.testing.assert_allclose(pivot, np.log(2.0), rtol=1.0e-12)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(np.asarray(gradient), [0.5, 0.0], atol=1.0e-14)


def test_additive_pivot_reparam_preserves_density_and_maps_initial_value():
    physical_value = np.asarray(-0.45)
    pivot_value = physical_value + 1.25
    wrapped = reparam(_pivoted_scalar_model, config=_nuts_geometry_reparam_config)

    original_density, _ = log_density(
        _pivoted_scalar_model,
        (),
        {},
        {"scale": physical_value},
    )
    pivot_density, _ = log_density(
        wrapped,
        (),
        {},
        {"scale_pivot": pivot_value},
    )
    np.testing.assert_allclose(pivot_density, original_density, rtol=1.0e-12)

    prepared_model, prepared_init, replacements = _prepare_nuts_reparameterization(
        _pivoted_scalar_model,
        {"scale": physical_value},
        rng_seed=3,
    )
    assert "scale" not in prepared_init
    np.testing.assert_allclose(prepared_init["scale_pivot"], pivot_value)
    assert replacements == {"scale": "scale_pivot"}

    prepared_trace = trace(
        substitute(
            seed(prepared_model, jax.random.PRNGKey(4)),
            data=prepared_init,
        )
    ).get_trace()
    assert prepared_trace["scale_pivot"]["type"] == "sample"
    assert prepared_trace["scale"]["type"] == "deterministic"
    np.testing.assert_allclose(prepared_trace["scale"]["value"], physical_value)


def test_additive_pivot_preserves_density_with_latent_dependent_offset():
    continuum = jnp.asarray(0.35)
    physical_scale = jnp.asarray(-0.45)
    offset = 0.3 * continuum + 0.2 * continuum**2
    wrapped = reparam(_dependent_pivot_model, config=_nuts_geometry_reparam_config)

    original_density, _ = log_density(
        _dependent_pivot_model,
        (),
        {},
        {"continuum": continuum, "scale": physical_scale},
    )
    pivot_density, _ = log_density(
        wrapped,
        (),
        {},
        {"continuum": continuum, "scale_pivot": physical_scale + offset},
    )
    np.testing.assert_allclose(pivot_density, original_density, rtol=1.0e-12)

    _, no_init, replacements = _prepare_nuts_reparameterization(
        _dependent_pivot_model,
        None,
        rng_seed=7,
    )
    assert no_init is None
    assert replacements == {"scale": "scale_pivot"}


def test_additive_pivot_mcmc_returns_physical_and_auxiliary_samples():
    wrapped = reparam(_pivoted_scalar_model, config=_nuts_geometry_reparam_config)
    kernel = NUTS(wrapped, max_tree_depth=4)
    mcmc = MCMC(
        kernel,
        num_warmup=8,
        num_samples=8,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(5))

    samples = mcmc.get_samples()
    assert {"scale", "scale_pivot"} <= set(samples)
    np.testing.assert_allclose(samples["scale_pivot"] - samples["scale"], 1.25)
    physical = _physical_nuts_samples(
        mcmc,
        {"scale": "scale_pivot"},
        group_by_chain=False,
    )
    assert "scale" in physical
    assert "scale_pivot" not in physical


def test_embedded_jaxqsofit_feature_priors_are_prepared_and_can_be_disabled():
    assert InferenceConfig().reparameterize_jaxqsofit_features is True
    physical_init = {
        "jqf_feii_norm": np.asarray(2.0e-3),
        "jqf_feii_shift": np.asarray(0.02),
    }
    prepared_model, prepared_init, replacements = _prepare_nuts_reparameterization(
        _embedded_jaxqsofit_feature_model,
        physical_init,
        rng_seed=18,
        reparameterize_additive_pivots=False,
        reparameterize_jaxqsofit_features=True,
    )

    assert replacements == {
        "jqf_feii_norm": "jqf_feii_norm_std",
        "jqf_feii_shift": "jqf_feii_shift_std",
    }
    np.testing.assert_allclose(
        prepared_init["jqf_feii_norm_std"],
        0.5 * np.log(2.0),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        prepared_init["jqf_feii_shift_std"],
        2.0,
        rtol=1.0e-12,
    )
    prepared_trace = trace(
        substitute(
            seed(prepared_model, jax.random.PRNGKey(19)),
            data=prepared_init,
        )
    ).get_trace()
    for physical_name, auxiliary_name in replacements.items():
        assert prepared_trace[physical_name]["type"] == "deterministic"
        assert prepared_trace[auxiliary_name]["type"] == "sample"
        np.testing.assert_allclose(
            prepared_trace[physical_name]["value"],
            physical_init[physical_name],
            rtol=1.0e-12,
        )

    _, disabled_init, disabled_replacements = _prepare_nuts_reparameterization(
        _embedded_jaxqsofit_feature_model,
        physical_init,
        rng_seed=20,
        reparameterize_additive_pivots=False,
        reparameterize_jaxqsofit_features=False,
    )
    assert disabled_replacements == {}
    assert set(physical_init) <= set(disabled_init)


def test_explicit_dense_mass_sites_follow_reparameterized_names():
    blocks = [("scale", "continuum"), ("other",)]
    remapped = _remap_dense_mass_sites(blocks, {"scale": "scale_pivot"})
    assert remapped == [("scale_pivot", "continuum"), ("other",)]

    try:
        _remap_dense_mass_sites(
            [("scale",), ("scale_pivot",)],
            {"scale": "scale_pivot"},
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Duplicate remapped block sites must be rejected")


def test_joint_dense_blocks_include_line_leftovers_redshift_and_nebular_sites():
    values = {
        "redshift": np.asarray(0.2),
        "log_agn_amp": np.asarray(30.0),
        "fcov": np.asarray(0.4),
        "hot_fcov": np.asarray(1.0),
        "jqf_line_new_center": np.asarray(0.0),
        "jqf_line_new_width": np.asarray(1.0),
        "nebular_logu": np.asarray(-2.5),
        "log_nebular_amp": np.asarray(0.0),
        "unrelated": np.asarray(2.0),
    }

    blocks = _joint_dense_mass_blocks(values)
    assert ("fcov", "hot_fcov", "log_agn_amp", "redshift") in blocks
    assert ("jqf_line_new_center", "jqf_line_new_width") in blocks
    assert ("log_nebular_amp", "nebular_logu") in blocks
    assert all("unrelated" not in block for block in blocks)
    flattened = [name for block in blocks for name in block]
    assert len(flattened) == len(set(flattened))


def test_jaxqsofit_feature_block_remaps_to_standardized_nuts_sites():
    values = {
        "jqf_feii_norm": np.asarray(1.0e-3),
        "jqf_feii_fwhm": np.asarray(3000.0),
        "jqf_feii_shift": np.asarray(0.0),
        "jqf_balmer_norm": np.asarray(1.0e-3),
        "jqf_balmer_tau": np.asarray(1.0),
        "jqf_balmer_vel": np.asarray(3000.0),
    }
    physical_blocks = _joint_dense_mass_blocks(values)
    replacements = {
        name: f"{name}_std"
        for name in values
    }
    remapped = _remap_dense_mass_sites(physical_blocks, replacements)

    assert physical_blocks == [tuple(sorted(values))]
    assert remapped == [
        tuple(replacements[name] for name in sorted(values))
    ]


def test_joint_dense_blocks_group_fallback_line_and_sed_sites():
    values = {
        **{f"jqf_line_extra_{index}": np.asarray(0.0) for index in range(5)},
        "log_agn_amp": np.asarray(30.0),
        "fcov": np.asarray(0.4),
        "hot_fcov": np.asarray(1.0),
        "pl_slope": np.asarray(-2.0),
        "log_stellar_mass": np.asarray(10.0),
        "log_sfh_age_gyr": np.asarray(0.5),
    }

    blocks = _joint_dense_mass_blocks(values)
    line_names = {
        name for block in blocks for name in block if name.startswith("jqf_line_")
    }
    assert len(line_names) == 5
    assert any(
        {"log_agn_amp", "log_stellar_mass", "log_sfh_age_gyr"} <= set(block)
        for block in blocks
    )


class _TransitionMCMC:
    def get_extra_fields(self, group_by_chain=False):
        assert group_by_chain is True
        return {
            "diverging": np.asarray([[False, False, True, False]]),
            "num_steps": np.asarray([[1, 7, 8, 15]]),
            "accept_prob": np.asarray([[0.8, 0.9, 0.7, 1.0]]),
            "potential_energy": np.asarray([[2.0, 2.5, 2.2, 2.1]]),
            "energy": np.asarray([[0.0, 1.0, 0.0, 1.0]]),
        }


def test_transition_diagnostics_distinguish_final_level_from_full_trajectory():
    diagnostics = nuts_transition_diagnostics(_TransitionMCMC(), max_tree_depth=4)

    assert diagnostics["n_transitions"] == 4
    assert diagnostics["n_divergent"] == 1
    assert diagnostics["final_tree_level_fraction"] == 0.5
    assert diagnostics["max_num_steps_fraction"] == 0.25
    assert diagnostics["full_trajectory_fraction"] == 0.25
    assert diagnostics["median_tree_depth_lower_bound"] == 3.5
    np.testing.assert_array_equal(
        diagnostics["extra_fields"]["num_steps"],
        [[1, 7, 8, 15]],
    )
    assert np.isfinite(diagnostics["bfmi"][0])


def test_metric_diagnostics_flag_nonpositive_and_nonfinite_eigenvalues():
    inverse_mass = {
        ("x", "y"): np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        ("z",): np.asarray([np.nan]),
        ("a", "b"): np.asarray([[np.nan, 0.2], [0.2, 1.0]]),
    }
    mcmc = SimpleNamespace(
        num_chains=1,
        last_state=SimpleNamespace(
            adapt_state=SimpleNamespace(
                inverse_mass_matrix=inverse_mass,
                step_size=np.asarray(0.02),
            )
        ),
    )

    diagnostics = nuts_metric_diagnostics(mcmc)
    blocks = {tuple(block["sites"]): block for block in diagnostics["blocks"]}
    assert blocks[("x", "y")]["n_nonpositive_eigenvalues"] == 1
    assert np.isinf(blocks[("x", "y")]["condition_number"])
    assert blocks[("z",)]["n_nonfinite_eigenvalues"] == 1
    assert np.isinf(blocks[("z",)]["condition_number"])
    assert blocks[("a", "b")]["n_nonfinite_eigenvalues"] == 2
    assert np.isinf(blocks[("a", "b")]["condition_number"])
    np.testing.assert_allclose(diagnostics["adapted_step_size"], 0.02)


def test_additive_pivot_reparam_rejects_observed_sites():
    reparameterizer = _AdditivePivotReparam(1.0, "pivot")
    try:
        reparameterizer("value", dist.Normal(0.0, 1.0), np.asarray(0.0))
    except ValueError as exc:
        assert "latent site" in str(exc)
    else:
        raise AssertionError("Observed sites must not be additively reparameterized")
