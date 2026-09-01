import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.spectroscopy import (
    SpectralComponentConfig,
    evaluate_joint_spectral_components,
    render_joint_feature_state,
    make_custom_component,
    make_custom_line_component,
)
from jaxsedfit.spectral_components import _apply_bal_absorption
from jaxsedfit.spectral_defaults import build_default_bal_components
from jaxsedfit.spectroscopy import (
    NORMAL_LOGNORMAL_STANDARDIZATION,
    build_spectral_prior_config as build_default_prior_config,
)


def _constant_custom_component(wave, params, metadata):
    return jnp.ones_like(wave) * params["amplitude"] * metadata.get("scale", 1.0)


def _constant_bal_optical_depth(wave, params, metadata):
    del metadata
    return jnp.ones_like(wave) * params["tau_peak"]


def _fixed_bal_component(name, tau_peak, covering):
    return make_custom_component(
        name,
        {
            "tau_peak": dist.Delta(float(tau_peak)),
            "covering": dist.Delta(float(covering)),
        },
        _constant_bal_optical_depth,
        metadata={"component_type": "bal_absorption"},
    )


def test_joint_bal_identity_partial_covering_and_negative_decrement():
    wave_obs = np.linspace(1400.0, 1600.0, 9)
    continuum = np.full_like(wave_obs, 7.0)
    disk = np.full_like(wave_obs, 3.0)

    identity = seed(evaluate_joint_spectral_components, jax.random.PRNGKey(100))(
        wave_obs,
        0.0,
        continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            custom_components=(_fixed_bal_component("bal_zero", 0.0, 0.4),),
        ),
        bal_continuum_mjy=disk,
    )
    np.testing.assert_allclose(identity["bal_transmission"], 1.0)
    np.testing.assert_allclose(identity["bal_decrement"], 0.0)
    np.testing.assert_allclose(identity["total"], continuum)

    absorbed = seed(evaluate_joint_spectral_components, jax.random.PRNGKey(101))(
        wave_obs,
        0.0,
        continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            custom_components=(
                _fixed_bal_component("bal_partial", np.log(2.0), 0.4),
            ),
        ),
        bal_continuum_mjy=disk,
    )
    expected_transmission = 0.8
    np.testing.assert_allclose(absorbed["bal_transmission"], expected_transmission)
    np.testing.assert_allclose(absorbed["bal_decrement"], -0.6)
    np.testing.assert_allclose(absorbed["custom"]["bal_partial"], -0.6)
    np.testing.assert_allclose(absorbed["total"], 6.4)


def test_joint_multiple_bal_transmissions_multiply():
    wave_obs = np.linspace(1400.0, 1600.0, 9)
    disk = np.full_like(wave_obs, 3.0)
    cfg = SpectralComponentConfig(
        use_lines=False,
        custom_components=(
            _fixed_bal_component("bal_a", np.log(2.0), 0.4),
            _fixed_bal_component("bal_b", np.log(3.0), 0.5),
        ),
    )
    result = seed(evaluate_joint_spectral_components, jax.random.PRNGKey(102))(
        wave_obs,
        0.0,
        np.full_like(wave_obs, 7.0),
        config=cfg,
        bal_continuum_mjy=disk,
    )

    expected_transmission = 0.8 * (2.0 / 3.0)
    np.testing.assert_allclose(result["bal_transmission"], expected_transmission)
    np.testing.assert_allclose(result["bal_decrement"], 3.0 * (expected_transmission - 1.0))
    np.testing.assert_allclose(
        result["custom"]["bal_a"] + result["custom"]["bal_b"],
        result["bal_decrement"],
    )
    np.testing.assert_allclose(result["total"], 7.0 + 3.0 * (expected_transmission - 1.0))


def test_bal_absorption_leaves_host_torus_and_narrow_components_unchanged():
    wave = jnp.arange(4.0)
    bal = _fixed_bal_component("bal", np.log(2.0), 0.5)
    additive = make_custom_component(
        "additive", {"amplitude": dist.Delta(1.0)}, _constant_custom_component
    )
    broad = make_custom_line_component(
        "custom_broad",
        {"amplitude": dist.Delta(3.0)},
        _constant_custom_component,
        line_kind="broad",
    )
    narrow = make_custom_line_component(
        "custom_narrow",
        {"amplitude": dist.Delta(5.0)},
        _constant_custom_component,
        line_kind="narrow",
    )
    cfg = SpectralComponentConfig(
        use_lines=False,
        custom_components=(bal, additive),
        custom_line_components=(broad, narrow),
    )
    state = {
        f"custom:{bal.prefix}": {"tau_peak": np.log(2.0), "covering": 0.5},
        f"custom:{additive.prefix}": {"amplitude": 1.0},
        f"custom:{broad.prefix}": {"amplitude": 3.0},
        f"custom:{narrow.prefix}": {"amplitude": 5.0},
    }
    raw_custom = {
        bal.output_name: jnp.full_like(wave, np.log(2.0)),
        additive.output_name: jnp.ones_like(wave),
        broad.output_name: jnp.full_like(wave, 3.0),
        narrow.output_name: jnp.full_like(wave, 5.0),
    }
    result = _apply_bal_absorption(
        wave,
        raw_custom,
        state,
        cfg,
        line_broad=jnp.full_like(wave, 2.0),
        line_narrow=jnp.full_like(wave, 4.0),
        feii=jnp.full_like(wave, 6.0),
        balmer=jnp.full_like(wave, 8.0),
        bal_continuum_mjy=jnp.full_like(wave, 10.0),
    )

    np.testing.assert_allclose(result["bal_transmission"], 0.75)
    np.testing.assert_allclose(result["line_broad"], (2.0 + 3.0) * 0.75)
    np.testing.assert_allclose(result["line_narrow"], 4.0 + 5.0)
    np.testing.assert_allclose(result["feii"], 6.0 * 0.75)
    np.testing.assert_allclose(result["balmer"], 8.0 * 0.75)
    np.testing.assert_allclose(result["custom"]["additive"], 0.75)
    np.testing.assert_allclose(result["custom"]["custom_narrow"], 5.0)
    # Host and torus are part of the external continuum but absent from the
    # compact ``bal_continuum_mjy`` reference, so this decrement cannot touch them.
    np.testing.assert_allclose(result["custom_continuum"], 0.75 - 2.5)


def test_joint_default_bal_components_use_one_shared_physical_site_set():
    components = build_default_bal_components(np.ones(3))
    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(103))).get_trace(
        wave_obs=np.linspace(1100.0, 1650.0, 64),
        redshift=0.0,
        continuum_mjy=np.ones(64),
        config=SpectralComponentConfig(use_lines=False, custom_components=components),
    )

    for name in (
        "spectral_custom_bal_v_out",
        "spectral_custom_bal_tau_peak",
        "spectral_custom_bal_covering",
        "spectral_custom_bal_fwhm_kms",
    ):
        assert name in tr
        assert tr[name]["type"] == "sample"
    for transition in ("nv", "siiv", "civ"):
        assert f"spectral_custom_bal_{transition}_shape_power" in tr
        assert f"spectral_custom_bal_{transition}_v_out" not in tr
        assert f"spectral_custom_bal_{transition}_tau_peak" not in tr
        assert f"spectral_custom_bal_{transition}_covering" not in tr
        assert f"spectral_custom_bal_{transition}_fwhm_kms" not in tr


def test_evaluate_joint_spectral_components_uses_external_continuum():
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(3))).get_trace(
        wave_obs=wave_obs,
        redshift=0.1,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            multiplicative_tilt=False,
        ),
    )

    assert np.allclose(np.asarray(tr["spectral_total_model"]["value"]), continuum)
    assert np.allclose(np.asarray(tr["spectral_line_model"]["value"]), 0.0)


def test_joint_custom_components_are_sampled_once_and_can_be_rerendered():
    wave_obs = np.linspace(4000.0, 8000.0, 32)
    continuum = np.ones_like(wave_obs)
    custom = make_custom_component(
        "extra continuum",
        {"amplitude": dist.Delta(0.5)},
        _constant_custom_component,
        metadata={"scale": 2.0},
    )
    custom_line = make_custom_line_component(
        "extra narrow line",
        {"amplitude": dist.Delta(0.25)},
        _constant_custom_component,
        line_kind="narrow",
    )
    cfg = SpectralComponentConfig(
        use_lines=False,
        custom_components=(custom,),
        custom_line_components=(custom_line,),
    )

    result = seed(evaluate_joint_spectral_components, jax.random.PRNGKey(31))(
        wave_obs=wave_obs,
        redshift=0.2,
        continuum_mjy=continuum,
        config=cfg,
    )
    rerendered = render_joint_feature_state(
        np.linspace(5000.0, 6000.0, 7),
        0.2,
        result["state"],
        config=cfg,
    )

    assert np.allclose(np.asarray(result["custom_continuum"]), 1.0)
    assert np.allclose(np.asarray(result["line_narrow"]), 0.25)
    assert np.allclose(np.asarray(result["total"]), 2.25)
    assert np.allclose(np.asarray(rerendered["custom_continuum"]), 1.0)
    assert np.allclose(np.asarray(rerendered["line_narrow"]), 0.25)


def test_evaluate_joint_spectral_components_adds_line_sites():
    wave_obs = np.linspace(4500.0, 7500.0, 64)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(4))).get_trace(
        wave_obs=wave_obs,
        redshift=0.1,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_centers_rest=(4861.33,),
            line_names=("Hbeta",),
            broad_line_names=("Hbeta",),
        ),
    )

    assert "spectral_line_amp_Hbeta" in tr
    assert "spectral_line_fwhm_Hbeta" in tr
    assert "spectral_line_velocity_Hbeta" in tr
    assert np.asarray(tr["spectral_total_model"]["value"]).shape == wave_obs.shape


def test_joint_feature_amplitudes_can_use_observed_spectrum_coordinates():
    wave_obs = np.linspace(3000.0, 7000.0, 128)
    template_wave = np.linspace(2000.0, 8000.0, 256)
    template_flux = np.ones_like(template_wave)
    cfg = SpectralComponentConfig(
        use_lines=True,
        tied_lines=False,
        line_centers_rest=(4861.33,),
        line_names=("Hbeta",),
        broad_line_names=("Hbeta",),
        use_feii=True,
        use_balmer_continuum=True,
        broadening_convolution="direct",
    )
    params = {
        "spectral_line_amp_Hbeta": 0.2,
        "spectral_line_fwhm_Hbeta": 3000.0,
        "spectral_line_velocity_Hbeta": 0.0,
        "spectral_feii_norm": 0.1,
        "spectral_feii_fwhm": 1000.0,
        "spectral_feii_shift": 0.0,
        "spectral_balmer_norm": 0.1,
        "spectral_balmer_tau": 1.0,
        "spectral_balmer_vel": 3000.0,
    }

    def evaluate(scale):
        fn = substitute(
            seed(evaluate_joint_spectral_components, jax.random.PRNGKey(17)),
            data=params,
        )
        return fn(
            wave_obs=wave_obs,
            redshift=0.0,
            continuum_mjy=np.zeros_like(wave_obs),
            config=cfg,
            feii_template_wave_rest=template_wave,
            feii_template_flux=template_flux,
            feature_amplitude_scale=scale,
        )

    unit = evaluate(1.0)
    doubled = evaluate(2.0)
    for name in ("lines", "feii", "balmer"):
        np.testing.assert_allclose(
            np.asarray(doubled[name]),
            0.5 * np.asarray(unit[name]),
            rtol=1e-10,
            atol=1e-12,
        )


def test_joint_feii_balmer_sites_advertise_nuts_standardization_only():
    wave_obs = np.linspace(3000.0, 7000.0, 32)
    template_wave = np.linspace(2000.0, 8000.0, 64)
    tr = trace(
        seed(evaluate_joint_spectral_components, jax.random.PRNGKey(18))
    ).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=np.ones_like(wave_obs),
        config=SpectralComponentConfig(
            use_lines=False,
            use_feii=True,
            use_balmer_continuum=True,
            broadening_convolution="direct",
        ),
        feii_template_wave_rest=template_wave,
        feii_template_flux=np.ones_like(template_wave),
    )
    expected = {
        "spectral_feii_norm",
        "spectral_feii_fwhm",
        "spectral_feii_shift",
        "spectral_balmer_norm",
        "spectral_balmer_tau",
        "spectral_balmer_vel",
    }
    advertised = {
        name
        for name, site in tr.items()
        if NORMAL_LOGNORMAL_STANDARDIZATION in (site.get("infer") or {})
    }

    assert advertised == expected
    for name in expected:
        assert tr[name]["type"] == "sample"
        assert (
            tr[name]["infer"][NORMAL_LOGNORMAL_STANDARDIZATION][
                "auxiliary_name"
            ]
            == f"{name}_std"
        )


def test_evaluate_joint_spectral_components_uses_default_tied_lines():
    wave_obs = np.linspace(4700.0, 5100.0, 96)
    continuum = np.full_like(wave_obs, 2.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(5))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_flux_scale_mjy=2.0,
        ),
    )

    assert "spectral_line_dmu_independent_group_std" in tr
    assert "spectral_line_log_fwhm_delta_group_std" in tr
    assert "spectral_line_amp_group" in tr
    assert tr["spectral_line_dmu_group"]["type"] == "deterministic"
    assert tr["spectral_line_sig_group"]["type"] == "deterministic"
    assert tr["spectral_line_amp_group"]["type"] == "deterministic"
    assert any(
        name.startswith("spectral_line_amp_") and name != "spectral_line_amp_group"
        for name in tr
    )
    assert "spectral_line_amp_per_component" in tr
    assert "spectral_line_model_broad" in tr
    assert "spectral_line_model_narrow" in tr
    assert np.asarray(tr["spectral_total_model"]["value"]).shape == wave_obs.shape


def test_joint_feature_state_renders_same_tied_lines_on_another_grid():
    wave_obs = np.linspace(4700.0, 5100.0, 96)
    cfg = SpectralComponentConfig(
        use_lines=True,
        tied_lines=True,
        use_feii=False,
        use_balmer_continuum=False,
        line_flux_scale_mjy=2.0,
    )
    fn = seed(evaluate_joint_spectral_components, jax.random.PRNGKey(15))
    result = fn(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=np.zeros_like(wave_obs),
        config=cfg,
    )

    rendered = render_joint_feature_state(wave_obs, 0.0, result["state"], config=cfg)

    np.testing.assert_allclose(np.asarray(rendered["lines"]), np.asarray(result["lines"]), rtol=1e-12, atol=1e-12)
    assert set(("line_amp_per_component", "line_mu_per_component", "line_sig_per_component", "line_broad_mask_per_component")) <= set(result["state"])


def test_evaluate_joint_spectral_components_filters_tied_lines_to_coverage():
    wave_obs = np.linspace(4850.0, 5010.0, 64)
    continuum = np.full_like(wave_obs, 1.0)
    line_table = [
        {
            "lambda": 4862.68,
            "linename": "Hb",
            "compname": "Hb",
            "inisca": 0.1,
            "minsca": 1.0e-4,
            "maxsca": 1.0,
            "inisig": 1.0e-3,
            "minsig": 1.0e-4,
            "maxsig": 1.0e-2,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        },
        {
            "lambda": 6564.61,
            "linename": "Ha",
            "compname": "Ha",
            "inisca": 0.1,
            "minsca": 1.0e-4,
            "maxsca": 1.0,
            "inisig": 1.0e-3,
            "minsig": 1.0e-4,
            "maxsig": 1.0e-2,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        },
    ]

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(8))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_table=line_table,
            line_coverage_rest=(4800.0, 5050.0),
        ),
    )

    assert np.asarray(tr["spectral_line_amp_per_component"]["value"]).shape == (1,)


def test_evaluate_joint_spectral_components_accepts_prior_config_object_as_line_prior_config():
    wave_obs = np.linspace(4700.0, 5100.0, 96)
    continuum = np.full_like(wave_obs, 2.0)
    prior_config = build_default_prior_config(
        continuum,
        include_elg_narrow_lines=False,
        include_high_ionization_lines=False,
    )

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(6))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            tied_lines=True,
            use_feii=False,
            use_balmer_continuum=False,
            line_prior_config=prior_config,
        ),
    )

    assert "spectral_line_dmu_independent_group_std" in tr
    assert tr["spectral_line_dmu_group"]["type"] == "deterministic"
    assert "spectral_line_amp_per_component" in tr
    assert np.asarray(tr["spectral_total_model"]["value"]).shape == wave_obs.shape


def test_evaluate_joint_spectral_components_reports_fixed_narrow_line_controls():
    wave_obs = np.linspace(4990.0, 5010.0, 96)
    continuum = np.full_like(wave_obs, 1.0)

    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(6))).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=True,
            tied_lines=False,
            use_feii=False,
            use_balmer_continuum=False,
            line_centers_rest=(5000.0,),
            line_names=("OIII5007c",),
            fixed_narrow_fwhm_kms=321.0,
            fixed_narrow_amp_scale=2.5,
        ),
    )

    assert tr["spectral_line_narrow_fwhm_kms"]["value"] == 321.0
    assert tr["spectral_line_narrow_amp_scale"]["value"] == 2.5
    assert np.nanmax(np.asarray(tr["spectral_line_model_narrow"]["value"])) > 0.0


def test_evaluate_joint_spectral_components_converts_feii_template_to_fnu_shape():
    wave_obs = np.array([2000.0, 3000.0, 4000.0])
    continuum = np.zeros_like(wave_obs)
    template_wave = np.array([1000.0, 5000.0])
    template_flux = np.ones_like(template_wave)
    fn = substitute(
        seed(evaluate_joint_spectral_components, jax.random.PRNGKey(7)),
        data={
            "spectral_feii_norm": 1.0,
            "spectral_feii_fwhm": 1.0,
            "spectral_feii_shift": 0.0,
        },
    )

    tr = trace(fn).get_trace(
        wave_obs=wave_obs,
        redshift=0.0,
        continuum_mjy=continuum,
        config=SpectralComponentConfig(
            use_lines=False,
            use_feii=True,
            use_balmer_continuum=False,
            feii_fnu_pivot_rest=3000.0,
        ),
        feii_template_wave_rest=template_wave,
        feii_template_flux=template_flux,
    )
    feii = np.asarray(tr["spectral_feii_model"]["value"])

    assert feii[2] / feii[0] > 3.9
    assert np.isclose(feii[1] / feii[0], (3000.0 / 2000.0) ** 2, rtol=0.05)


def test_feii_template_arrays_can_be_traced_under_jit():
    wave_obs = np.linspace(2000.0, 4000.0, 32)
    template_wave = np.linspace(1000.0, 5000.0, 64)
    template_flux = np.ones_like(template_wave)
    cfg = SpectralComponentConfig(
        use_lines=False,
        use_feii=True,
        use_balmer_continuum=False,
        broadening_convolution="direct",
        feii_fnu_pivot_rest=3000.0,
    )

    def render(twave, tflux):
        fn = substitute(
            seed(evaluate_joint_spectral_components, jax.random.PRNGKey(21)),
            data={"spectral_feii_norm": 1.0, "spectral_feii_fwhm": 1000.0, "spectral_feii_shift": 0.0},
        )
        return fn(
            wave_obs=wave_obs,
            redshift=0.0,
            continuum_mjy=np.zeros_like(wave_obs),
            config=cfg,
            feii_template_wave_rest=twave,
            feii_template_flux=tflux,
        )["feii"]

    result = jax.jit(render)(template_wave, template_flux)

    assert np.asarray(result).shape == wave_obs.shape
    assert np.all(np.isfinite(np.asarray(result)))
