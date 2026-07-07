import numpy as np
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.config import (
    AGNConfig,
    EmissionLineTemplate,
    FeIITemplate,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    JaxQSOFitConfig,
    LikelihoodConfig,
    NebularConfig,
    Observation,
    PhotometryData,
    PriorConfig,
    SpectroscopyConfig,
    SpectroscopyData,
)
from jaxsedfit.model import GRAHSP_PL_BEND_LOC_A, GRAHSP_PL_BEND_WIDTH, GRAHSP_PL_CUTOFF_A, _project_filters, _redshift_to_obs, evaluate_photometric_state, grahsp_photometric_model
from jaxsedfit.preload import build_model_context


def _patch_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._HOST_BASIS_CACHE", {})


def _cfg(
    *,
    fit_host=True,
    fit_agn=True,
    fit_host_kinematics=False,
    fit_feii_broadening=False,
    fit_balmer_continuum=False,
    rest_wave_max=3.0e6,
    n_wave=512,
    spectroscopy_enabled=False,
    aperture_diameter_arcsec=None,
):
    return FitConfig(
        observation=Observation(object_id="assembly", redshift=0.05),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
            aperture_diameter_arcsec=aperture_diameter_arcsec,
        ),
        filters=FilterSet(
            curves=[FilterCurve(name="f1", wave=[1500.0, 2000.0, 2500.0], transmission=[0.0, 1.0, 0.0])],
        ),
        galaxy=GalaxyConfig(
            fit_host=fit_host,
            dsps_ssp_fn="fake-assembly.h5",
            fit_host_kinematics=fit_host_kinematics,
            rest_wave_min=100.0,
            rest_wave_max=rest_wave_max,
            n_wave=n_wave,
            sfh_n_steps=16,
            use_energy_balance=True,
            dust_alpha=2.0,
        ),
        agn=AGNConfig(
            fit_agn=fit_agn,
            fit_feii_broadening=fit_feii_broadening,
            fit_balmer_continuum=fit_balmer_continuum,
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0, 3000.0], lumin=[0.0, 1.0, 0.0]),
            emission_line_template=EmissionLineTemplate(
                wave=[486.1, 656.3],
                lumin_blagn=[1.0, 0.5],
                lumin_sy2=[0.2, 0.1],
                lumin_liner=[0.1, 0.05],
            ),
        ),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
            use_absolute_flux_scale_prior=False,
            use_host_capture_model=False,
        ),
        spectroscopy=(
            SpectroscopyData(
                wave_obs=[1200.0, 1500.0, 1800.0],
                fluxes=[1.0, 1.0, 1.0],
                errors=[0.1, 0.1, 0.1],
            )
            if spectroscopy_enabled
            else None
        ),
        spectroscopy_config=SpectroscopyConfig(enabled=spectroscopy_enabled),
        nebular=NebularConfig(enabled=True, f_esc=0.0, f_dust=0.2, zgas=0.02, lines_width=300.0),
        inference=InferenceConfig(map_steps=2),
        prior_config=PriorConfig(stellar_mass=dist.Normal(8.0, 1.0e-6)),
    )


def _deterministic_trace(context, data=None):
    data = {} if data is None else data
    model = substitute(lambda: grahsp_photometric_model(context, include_components=True), data=data)
    return trace(seed(model, 0)).get_trace()


def _deterministic_likelihood_trace(context, data=None):
    data = {} if data is None else data
    model = substitute(lambda: grahsp_photometric_model(context, include_components=False), data=data)
    return trace(seed(model, 0)).get_trace()


def _site(tr, key):
    return np.asarray(tr[key]["value"], dtype=float)


def _log_positive(value):
    return np.array(np.log(max(float(value), 1.0e-12)))


def _weighted_std(x, weight):
    weight = np.clip(np.asarray(weight, dtype=float), 0.0, None)
    mean = np.sum(x * weight) / np.maximum(np.sum(weight), 1.0e-300)
    var = np.sum(weight * (x - mean) ** 2) / np.maximum(np.sum(weight), 1.0e-300)
    return np.sqrt(var)


def _fixed_component_data():
    return {
        "log_ebv_gal": _log_positive(0.2),
        "log_ebv_agn": _log_positive(0.1),
        "dust_alpha": np.array(2.0),
        "log_agn_amp": np.array(np.log(1.0e34)),
        "uv_slope": np.array(0.0),
        "pl_slope": np.array(-1.0),
        "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
        "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
        "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
        "log_fcov": _log_positive(0.2),
        "si": np.array(0.0),
        "cool_lam": np.array(17.0),
        "cool_width": np.array(0.45),
        "hot_lam": np.array(2.0),
        "hot_width": np.array(0.5),
        "log_hot_fcov": _log_positive(0.1),
        "broad_lines_strength": np.array(1.0),
        "narrow_lines_strength": np.array(1.0),
        "log_broad_line_width_kms": np.array(np.log(3000.0)),
        "log_narrow_line_width_kms": np.array(np.log(500.0)),
        "feii_norm": np.array(1.0),
        "feii_fwhm": np.array(3000.0),
        "feii_shift": np.array(0.0),
        "balmer_norm": np.array(0.2),
        "balmer_tau": np.array(1.0),
        "balmer_vel": np.array(3000.0),
    }


def test_native_agn_lines_use_distinct_broad_and_narrow_widths(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _cfg(fit_host=False, n_wave=4096, rest_wave_max=1000.0)
    cfg.nebular.enabled = False
    cfg.agn.fit_balmer_continuum = False
    context = build_model_context(cfg)

    data = _fixed_component_data()
    data["log_broad_line_width_kms"] = np.array(np.log(3000.0))
    data["log_narrow_line_width_kms"] = np.array(np.log(300.0))
    data["feii_norm"] = np.array(0.0)
    tr = _deterministic_trace(context, data)

    wave = _site(tr, "rest_wave")
    near_hbeta = (wave > 470.0) & (wave < 505.0)
    broad_std = _weighted_std(wave[near_hbeta], _site(tr, "line_bl_rest_sed")[near_hbeta])
    narrow_std = _weighted_std(wave[near_hbeta], _site(tr, "line_nl_rest_sed")[near_hbeta])

    assert broad_std > 5.0 * narrow_std


def test_systematics_width_default_is_tight_log_prior(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _cfg()
    cfg.likelihood.fit_systematics_width = True
    context = build_model_context(cfg)

    tr = _deterministic_likelihood_trace(
        context,
        {
            **_fixed_component_data(),
            "log_systematics_width": _log_positive(0.10),
        },
    )

    assert "log_systematics_width" in tr
    assert "systematics_width" in tr
    assert tr["log_systematics_width"]["type"] == "sample"
    assert tr["systematics_width"]["type"] == "deterministic"
    fn = tr["log_systematics_width"]["fn"]
    assert fn.__class__.__name__ == "TwoSidedTruncatedDistribution"
    assert np.isclose(np.asarray(fn.base_dist.loc), np.log(0.10))
    assert np.isclose(np.asarray(fn.base_dist.scale), 0.05)
    assert np.isclose(np.asarray(fn.low), np.log(0.07))
    assert np.isclose(np.asarray(fn.high), np.log(0.15))
    assert np.isclose(_site(tr, "systematics_width"), 0.10)
    assert "photometry_loglike" in tr


def test_systematics_width_can_be_sampled_with_exponential_prior(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _cfg()
    cfg.likelihood.fit_systematics_width = True
    cfg.prior_config.likelihood.systematics_width = dist.Exponential(20.0)
    context = build_model_context(cfg)

    tr = _deterministic_likelihood_trace(
        context,
        {
            **_fixed_component_data(),
            "systematics_width": np.array(0.02),
        },
    )

    assert "systematics_width" in tr
    assert tr["systematics_width"]["type"] == "sample"
    assert np.isclose(np.asarray(tr["systematics_width"]["fn"].rate), 20.0)
    assert np.isclose(_site(tr, "systematics_width"), 0.02)
    assert "photometry_loglike" in tr


def test_systematics_width_can_use_log_normal_override(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _cfg()
    cfg.likelihood.fit_systematics_width = True
    cfg.prior_config.likelihood.log_systematics_width = dist.Normal(np.log(0.02), 0.2)
    context = build_model_context(cfg)

    tr = _deterministic_likelihood_trace(
        context,
        {
            **_fixed_component_data(),
            "log_systematics_width": _log_positive(0.02),
        },
    )

    assert "log_systematics_width" in tr
    assert "systematics_width" in tr
    assert tr["log_systematics_width"]["type"] == "sample"
    assert tr["systematics_width"]["type"] == "deterministic"
    assert np.isclose(_site(tr, "systematics_width"), 0.02)


def test_systematics_width_can_use_physical_lognormal_prior(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _cfg()
    cfg.likelihood.fit_systematics_width = True
    cfg.prior_config.likelihood.systematics_width = dist.LogNormal(np.log(0.02), 0.2)
    context = build_model_context(cfg)

    tr = _deterministic_likelihood_trace(
        context,
        {
            **_fixed_component_data(),
            "systematics_width": np.array(0.02),
        },
    )

    assert "systematics_width" in tr
    assert tr["systematics_width"]["type"] == "sample"
    assert tr["systematics_width"]["fn"].__class__.__name__ == "LogNormal"
    assert np.isclose(_site(tr, "systematics_width"), 0.02)


def test_positive_geometry_parameters_are_sampled_in_log_space(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())

    tr = _deterministic_trace(context, _fixed_component_data())

    for log_key, value_key in (
        ("log_ebv_gal", "ebv_gal"),
        ("log_ebv_agn", "ebv_agn"),
        ("log_fcov", "fcov"),
        ("log_hot_fcov", "hot_fcov"),
        ("log_gal_lgmet_scatter", "gal_lgmet_scatter"),
    ):
        assert log_key in tr
        assert value_key in tr
        assert tr[log_key]["type"] == "sample"
        assert tr[value_key]["type"] == "deterministic"
        np.testing.assert_allclose(_site(tr, value_key), np.exp(_site(tr, log_key)))


def test_component_rest_and_observed_seds_sum_to_total(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(fit_balmer_continuum=True))
    tr = _deterministic_trace(
        context,
        {
            "log_ebv_gal": _log_positive(0.2),
            "log_ebv_agn": _log_positive(0.1),
            "dust_alpha": np.array(2.0),
            "log_agn_amp": np.array(np.log(1.0e34)),
            "uv_slope": np.array(0.0),
            "pl_slope": np.array(-1.0),
            "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
            "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
            "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
            "log_fcov": _log_positive(0.2),
            "si": np.array(0.0),
            "cool_lam": np.array(17.0),
            "cool_width": np.array(0.45),
            "hot_lam": np.array(2.0),
            "hot_width": np.array(0.5),
            "log_hot_fcov": _log_positive(0.1),
            "broad_lines_strength": np.array(1.0),
            "narrow_lines_strength": np.array(1.0),
            "log_broad_line_width_kms": np.array(np.log(3000.0)),
            "log_narrow_line_width_kms": np.array(np.log(500.0)),
            "feii_norm": np.array(1.0),
            "feii_fwhm": np.array(3000.0),
            "feii_shift": np.array(0.0),
            "balmer_norm": np.array(0.2),
            "balmer_tau": np.array(1.0),
            "balmer_vel": np.array(3000.0),
        },
    )

    agn_parts = _site(tr, "disk_rest_sed") + _site(tr, "torus_rest_sed") + _site(tr, "feii_rest_sed") + _site(tr, "line_rest_sed") + _site(tr, "balmer_rest_sed")
    host_parts = _site(tr, "host_rest_sed") + _site(tr, "nebular_rest_sed")
    total_parts = _site(tr, "host_total_rest_sed") + _site(tr, "dust_rest_sed") + _site(tr, "agn_rest_sed")

    assert np.allclose(_site(tr, "agn_rest_sed"), agn_parts, rtol=2.0e-10, atol=1.0e-20)
    assert np.allclose(_site(tr, "host_total_rest_sed"), host_parts, rtol=2.0e-10, atol=1.0e-20)
    assert np.allclose(_site(tr, "total_rest_sed"), total_parts, rtol=2.0e-10, atol=1.0e-20)

    agn_obs_parts = _site(tr, "disk_obs_sed") + _site(tr, "torus_obs_sed") + _site(tr, "feii_obs_sed") + _site(tr, "line_obs_sed") + _site(tr, "balmer_obs_sed")
    host_obs_parts = _site(tr, "host_obs_sed") + _site(tr, "nebular_obs_sed")
    total_obs_parts = _site(tr, "host_total_obs_sed") + _site(tr, "dust_obs_sed") + _site(tr, "agn_obs_sed")

    assert np.allclose(_site(tr, "agn_obs_sed"), agn_obs_parts, rtol=2.0e-10, atol=1.0e-40)
    assert np.allclose(_site(tr, "host_total_obs_sed"), host_obs_parts, rtol=2.0e-10, atol=1.0e-40)
    assert np.allclose(_site(tr, "total_obs_sed"), total_obs_parts, rtol=2.0e-10, atol=1.0e-40)


def test_torus_component_is_not_foreground_attenuated(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())
    low_ebv = _fixed_component_data()
    high_ebv = _fixed_component_data()
    low_ebv["log_ebv_gal"] = _log_positive(0.0)
    low_ebv["log_ebv_agn"] = _log_positive(0.0)
    high_ebv["log_ebv_gal"] = _log_positive(0.5)
    high_ebv["log_ebv_agn"] = _log_positive(0.5)

    tr_low = _deterministic_trace(context, low_ebv)
    tr_high = _deterministic_trace(context, high_ebv)

    np.testing.assert_allclose(_site(tr_high, "torus_rest_sed"), _site(tr_low, "torus_rest_sed"))
    assert np.sum(_site(tr_high, "disk_rest_sed")) < np.sum(_site(tr_low, "disk_rest_sed"))


def test_evaluate_photometric_state_matches_deterministic_sites(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())
    data = {"log_ebv_gal": _log_positive(0.2), "log_ebv_agn": _log_positive(0.1), "dust_alpha": np.array(2.0)}
    model = substitute(lambda: evaluate_photometric_state(context, include_components=True), data=data)
    trace_handler = trace(seed(model, 0))
    state = trace_handler()
    tr = trace_handler.trace

    for key in ("pred_fluxes", "agn_fluxes", "host_fluxes", "dust_fluxes", "nebular_fluxes", "total_rest_sed"):
        np.testing.assert_allclose(np.asarray(state[key], dtype=float), _site(tr, key))


def test_evaluate_photometric_state_can_return_component_fluxes_without_full_components(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())
    data = {"log_ebv_gal": _log_positive(0.2), "log_ebv_agn": _log_positive(0.1), "dust_alpha": np.array(2.0)}
    model = substitute(
        lambda: evaluate_photometric_state(
            context,
            include_components=False,
            force_component_fluxes=True,
        ),
        data=data,
    )
    state = trace(seed(model, 0))()

    assert "total_rest_sed" not in state
    assert np.all(np.isfinite(np.asarray(state["pred_fluxes"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(state["agn_fluxes"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(state["host_fluxes"], dtype=float)))


def test_energy_balance_dust_sed_integrates_to_absorbed_luminosity(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(rest_wave_max=2.3e9, n_wave=4096))
    tr = _deterministic_trace(context, {"log_ebv_gal": _log_positive(0.5), "log_ebv_agn": _log_positive(0.0), "dust_alpha": np.array(2.0)})

    rest_wave = _site(tr, "rest_wave")
    dust_luminosity = 10.0 ** float(_site(tr, "log_dust_luminosity_fit"))
    emitted_dust_luminosity = float(np.trapezoid(_site(tr, "dust_rest_sed"), x=rest_wave))

    assert dust_luminosity > 0.0
    np.testing.assert_allclose(emitted_dust_luminosity, dust_luminosity, rtol=2.0e-2, atol=0.0)


def test_host_capture_scales_energy_balance_dust(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _cfg(
        fit_agn=False,
        rest_wave_max=2.3e9,
        n_wave=4096,
        aperture_diameter_arcsec=[0.5],
    )
    cfg.likelihood.use_host_capture_model = True
    context = build_model_context(cfg)
    tr = _deterministic_trace(
        context,
        {
            "log_ebv_gal": _log_positive(0.5),
            "dust_alpha": np.array(2.0),
            "log_host_capture_scale_arcsec": np.log(3.0),
            "log_host_capture_slope": np.log(2.0),
        },
    )

    capture = _site(tr, "host_capture_fraction_fluxes")
    assert np.all(capture < 1.0)
    uncaptured_host_source = _site(tr, "host_total_fluxes") + _site(tr, "dust_fluxes")
    captured_host_source = _site(tr, "host_capture_source_fluxes") * capture

    np.testing.assert_allclose(_site(tr, "host_capture_source_fluxes"), uncaptured_host_source)
    np.testing.assert_allclose(_site(tr, "pred_fluxes"), captured_host_source)
    assert np.all(_site(tr, "pred_fluxes") < uncaptured_host_source)


def test_agn_off_mode_has_zero_agn_components_and_no_total_leak(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(fit_agn=False))
    tr = _deterministic_trace(context, {"log_ebv_gal": _log_positive(0.2), "dust_alpha": np.array(2.0)})

    for key in ("agn_rest_sed", "disk_rest_sed", "torus_rest_sed", "feii_rest_sed", "line_rest_sed", "balmer_rest_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    for key in ("agn_obs_sed", "disk_obs_sed", "torus_obs_sed", "feii_obs_sed", "line_obs_sed", "balmer_obs_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    assert np.allclose(_site(tr, "total_rest_sed"), _site(tr, "host_total_rest_sed") + _site(tr, "dust_rest_sed"))
    assert np.allclose(_site(tr, "pred_fluxes"), _site(tr, "host_total_fluxes") + _site(tr, "dust_fluxes"))


def test_host_off_mode_has_zero_host_components_and_no_total_leak(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(fit_host=False))
    tr = _deterministic_trace(context, {"log_agn_amp": np.array(np.log(1.0e34)), "log_fcov": _log_positive(0.2), "si": np.array(0.0)})

    for key in ("host_rest_sed", "host_total_rest_sed", "host_absorbed_rest_sed", "dust_rest_sed", "nebular_rest_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    for key in ("host_obs_sed", "host_total_obs_sed", "dust_obs_sed", "nebular_obs_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    assert np.allclose(_site(tr, "total_rest_sed"), _site(tr, "agn_rest_sed"))
    assert np.allclose(_site(tr, "pred_fluxes"), _site(tr, "agn_fluxes"))


def test_agn_slope_ordering_uses_positive_delta_without_hard_factor(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(fit_host=False))
    tr = _deterministic_trace(context, {"log_agn_amp": np.array(np.log(1.0e34)), "log_fcov": _log_positive(0.2), "si": np.array(0.0)})

    assert "uv_slope_gt_pl_slope" not in tr
    assert "uv_slope_delta" in tr
    assert _site(tr, "uv_slope_delta") > 0.0
    assert _site(tr, "uv_slope") > _site(tr, "pl_slope")


def test_host_kinematics_default_off_skips_broadening_call(monkeypatch):
    _patch_ssp(monkeypatch)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("host broadening should be skipped when fit_host_kinematics=False")

    monkeypatch.setattr("jaxsedfit.model._shift_and_broaden_single_spectrum_lnlam", _raise_if_called)
    context = build_model_context(_cfg(fit_agn=False))
    tr = _deterministic_trace(context, {"log_ebv_gal": _log_positive(0.2), "dust_alpha": np.array(2.0)})

    assert "gal_v_kms" not in tr
    assert "gal_sigma_kms" not in tr
    assert np.all(np.isfinite(_site(tr, "pred_fluxes")))


def test_host_kinematics_flag_ignored_for_photometry_only(monkeypatch):
    _patch_ssp(monkeypatch)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("host broadening should be skipped for photometry-only SED fits")

    monkeypatch.setattr("jaxsedfit.model._shift_and_broaden_single_spectrum_lnlam", _raise_if_called)
    context = build_model_context(_cfg(fit_agn=False, fit_host_kinematics=True))
    tr = _deterministic_trace(context, {"log_ebv_gal": _log_positive(0.2), "dust_alpha": np.array(2.0)})

    assert "gal_v_kms" not in tr
    assert "gal_sigma_kms" not in tr
    assert np.all(np.isfinite(_site(tr, "pred_fluxes")))


def test_host_kinematics_enabled_with_spectroscopy_samples_and_broadens(monkeypatch):
    _patch_ssp(monkeypatch)
    calls = {"n": 0}

    def _identity_broaden(lnwave, spectrum, v_kms, sigma_kms):
        calls["n"] += 1
        return spectrum

    monkeypatch.setattr("jaxsedfit.model._shift_and_broaden_single_spectrum_lnlam", _identity_broaden)
    context = build_model_context(_cfg(fit_agn=False, fit_host_kinematics=True, spectroscopy_enabled=True))
    tr = _deterministic_trace(
        context,
        {
            "gal_v_kms": np.array(0.0),
            "gal_sigma_kms": np.array(150.0),
            "log_ebv_gal": _log_positive(0.2),
            "dust_alpha": np.array(2.0),
        },
    )

    assert calls["n"] == 1
    assert "gal_v_kms" in tr
    assert "gal_sigma_kms" in tr


def test_agn_only_context_skips_host_ssp_loading(monkeypatch):
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._HOST_BASIS_CACHE", {})

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("AGN-only contexts should not load host SSP templates")

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", _raise_if_called)
    context = build_model_context(_cfg(fit_host=False))

    assert context.ssp_data.ssp_flux.shape == (1, 1, 1)
    assert context.host_basis.rest_llambda.shape[-1] == context.rest_wave.size
    assert np.allclose(context.host_basis.rest_llambda, 0.0)


def test_host_only_context_skips_agn_template_loading(monkeypatch):
    _patch_ssp(monkeypatch)
    monkeypatch.setattr("jaxsedfit.preload._TEMPLATE_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._REST_TEMPLATE_CACHE", {})

    def _raise_loadtxt(*args, **kwargs):
        raise AssertionError("Host-only contexts should not load FeII or AGN emission-line templates")

    monkeypatch.setattr("jaxsedfit.preload.np.loadtxt", _raise_loadtxt)
    context = build_model_context(_cfg(fit_agn=False))

    assert np.allclose(np.asarray(context.feii_template_on_rest_jax, dtype=float), 0.0)
    assert np.asarray(context.templates.line_wave, dtype=float).size == 1


def test_disabled_balmer_continuum_skips_balmer_kernel(monkeypatch):
    _patch_ssp(monkeypatch)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("Balmer continuum should be skipped unless fit_balmer_continuum=True")

    monkeypatch.setattr("jaxsedfit.model._balmer_continuum_jax", _raise_if_called)
    context = build_model_context(_cfg(fit_balmer_continuum=False))
    tr = _deterministic_trace(
        context,
        {
            **_fixed_component_data(),
            "balmer_norm": np.array(0.2),
            "balmer_tau": np.array(1.0),
            "balmer_vel": np.array(3000.0),
        },
    )

    assert "balmer_norm" not in tr
    assert "balmer_tau" not in tr
    assert "balmer_vel" not in tr
    assert np.allclose(_site(tr, "balmer_rest_sed"), 0.0)
    assert np.allclose(_site(tr, "balmer_obs_sed"), 0.0)


def test_feii_broadening_default_off_uses_direct_template(monkeypatch):
    _patch_ssp(monkeypatch)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("FeII broadening should be skipped unless fit_feii_broadening=True")

    monkeypatch.setattr("jaxsedfit.model._feii_component", _raise_if_called)
    context = build_model_context(_cfg(fit_feii_broadening=False))
    tr = _deterministic_trace(
        context,
        {
            **_fixed_component_data(),
            "feii_norm": np.array(1.0),
            "feii_fwhm": np.array(3000.0),
            "feii_shift": np.array(0.0),
        },
    )

    assert "feii_norm" in tr
    assert "feii_fwhm" not in tr
    assert "feii_shift" not in tr
    assert np.any(_site(tr, "feii_rest_sed") > 0.0)


def test_feii_broadening_enabled_samples_and_calls_kernel(monkeypatch):
    _patch_ssp(monkeypatch)
    calls = {"n": 0}

    def _identity_feii(wave, template_flux_on_wave, norm, fwhm_kms, shift_frac):
        calls["n"] += 1
        return norm * template_flux_on_wave

    monkeypatch.setattr("jaxsedfit.model._feii_component", _identity_feii)
    context = build_model_context(_cfg(fit_feii_broadening=True))
    tr = _deterministic_trace(context, _fixed_component_data())

    assert calls["n"] == 1
    assert "feii_fwhm" in tr
    assert "feii_shift" in tr


def test_jaxqsofit_backend_owns_feii_and_balmer_components(monkeypatch):
    _patch_ssp(monkeypatch)

    def _raise_native_feii(*args, **kwargs):
        raise AssertionError("Native jaxsedfit FeII should be skipped when jaxqsofit owns spectral FeII")

    def _raise_native_balmer(*args, **kwargs):
        raise AssertionError("Native jaxsedfit Balmer continuum should be skipped when jaxqsofit owns spectral Balmer")

    def _stub_jaxqsofit_backend(wave_obs, redshift, continuum_mjy, cfg, *args, **kwargs):
        assert cfg.spectroscopy_config.jaxqsofit.use_spectral_feii is True
        assert cfg.spectroscopy_config.jaxqsofit.use_spectral_balmer_continuum is True
        return {
            "total": continuum_mjy,
            "line_broad": np.zeros_like(np.asarray(wave_obs, dtype=float)),
            "line_narrow": np.zeros_like(np.asarray(wave_obs, dtype=float)),
        }

    monkeypatch.setattr("jaxsedfit.model._feii_component", _raise_native_feii)
    monkeypatch.setattr("jaxsedfit.model._balmer_continuum_jax", _raise_native_balmer)
    monkeypatch.setattr("jaxsedfit.model._evaluate_jaxqsofit_backend", _stub_jaxqsofit_backend)

    cfg = _cfg(
        spectroscopy_enabled=True,
        fit_feii_broadening=True,
        fit_balmer_continuum=True,
    )
    cfg.spectroscopy_config = SpectroscopyConfig(
        enabled=True,
        backend="jaxqsofit",
        fit_scale=False,
        jaxqsofit=JaxQSOFitConfig(
            use_spectral_lines=False,
            use_spectral_feii=True,
            use_spectral_balmer_continuum=True,
            use_spectral_smart_priors=False,
            use_line_strength_priors=False,
        ),
    )
    context = build_model_context(cfg)
    tr = _deterministic_trace(context, _fixed_component_data())

    assert "feii_norm" not in tr
    assert "feii_fwhm" not in tr
    assert "feii_shift" not in tr
    assert "balmer_norm" not in tr
    assert "balmer_tau" not in tr
    assert "balmer_vel" not in tr
    assert np.allclose(_site(tr, "feii_rest_sed"), 0.0)
    assert np.allclose(_site(tr, "feii_obs_sed"), 0.0)
    assert np.allclose(_site(tr, "balmer_rest_sed"), 0.0)
    assert np.allclose(_site(tr, "balmer_obs_sed"), 0.0)


def test_plotted_component_sites_are_attenuated_likelihood_components(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())
    tr = _deterministic_trace(context, {"log_ebv_gal": _log_positive(0.2), "log_ebv_agn": _log_positive(0.1), "dust_alpha": np.array(2.0)})

    rest_wave = _site(tr, "rest_wave")
    obs_wave = _site(tr, "obs_wave")
    redshift = float(_site(tr, "redshift_fit"))
    igm = np.asarray(context.fixed_igm_jax, dtype=float)
    d_l = float(np.asarray(context.fixed_luminosity_distance_m_jax))
    for rest_key, obs_key in (
        ("host_rest_sed", "host_obs_sed"),
        ("dust_rest_sed", "dust_obs_sed"),
        ("disk_rest_sed", "disk_obs_sed"),
        ("torus_rest_sed", "torus_obs_sed"),
        ("feii_rest_sed", "feii_obs_sed"),
        ("line_rest_sed", "line_obs_sed"),
        ("balmer_rest_sed", "balmer_obs_sed"),
        ("agn_rest_sed", "agn_obs_sed"),
        ("total_rest_sed", "total_obs_sed"),
    ):
        expected = np.asarray(_redshift_to_obs(rest_wave, _site(tr, rest_key) * igm, obs_wave, redshift, d_l))
        assert np.allclose(_site(tr, obs_key), expected, rtol=2.0e-10, atol=1.0e-40)

    projected_total = np.asarray(_project_filters(_site(tr, "total_obs_sed"), context.packed_filters_jax))
    projected_coarse_nebular_lines = np.asarray(_project_filters(_site(tr, "nebular_lines_obs_sed"), context.packed_filters_jax))
    corrected_total = projected_total - projected_coarse_nebular_lines + _site(tr, "nebular_lines_fluxes")
    assert np.allclose(_site(tr, "pred_fluxes"), corrected_total, rtol=2.0e-10, atol=1.0e-30)
    assert np.asarray(_site(tr, "nebular_lines_local_obs_wave")).size > np.asarray(_site(tr, "nebular_lines_obs_sed")).size


def test_fast_fixed_filter_projection_matches_legacy_photometry(monkeypatch):
    _patch_ssp(monkeypatch)
    fast_cfg = _cfg(n_wave=256)
    slow_cfg = _cfg(n_wave=256)
    fast_cfg.likelihood.use_fast_photometry_projection = True
    fast_cfg.likelihood.use_local_line_photometry = False
    slow_cfg.likelihood.use_fast_photometry_projection = False
    slow_cfg.likelihood.use_local_line_photometry = False
    fast_context = build_model_context(fast_cfg)
    slow_context = build_model_context(slow_cfg)

    fast_tr = _deterministic_likelihood_trace(fast_context, _fixed_component_data())
    slow_tr = _deterministic_likelihood_trace(slow_context, _fixed_component_data())

    np.testing.assert_allclose(_site(fast_tr, "pred_fluxes"), _site(slow_tr, "pred_fluxes"), rtol=2.0e-12, atol=1.0e-30)


def test_local_line_photometry_improves_coarse_grid_line_projection(monkeypatch):
    _patch_ssp(monkeypatch)

    def _line_cfg(n_wave, *, local_lines):
        cfg = _cfg(fit_host=False, n_wave=n_wave, rest_wave_max=2000.0)
        cfg.photometry = PhotometryData(filter_names=["ha"], fluxes=[1.0], errors=[0.1])
        cfg.filters = FilterSet(
            curves=[
                FilterCurve(
                    name="ha",
                    wave=[650.0, 690.0, 730.0],
                    transmission=[0.0, 1.0, 0.0],
                )
            ],
        )
        cfg.likelihood.use_fast_photometry_projection = True
        cfg.likelihood.use_local_line_photometry = local_lines
        cfg.likelihood.variability_uncertainty = False
        cfg.nebular.enabled = False
        cfg.agn.fit_balmer_continuum = False
        return cfg

    data = _fixed_component_data()
    data["log_broad_line_width_kms"] = np.array(np.log(1200.0))
    data["log_narrow_line_width_kms"] = np.array(np.log(1200.0))
    data["feii_norm"] = np.array(0.0)

    coarse_legacy = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(64, local_lines=False)), data),
        "pred_fluxes",
    )
    coarse_local = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(64, local_lines=True)), data),
        "pred_fluxes",
    )
    fine_reference = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(4096, local_lines=False)), data),
        "pred_fluxes",
    )

    legacy_error = np.abs(coarse_legacy - fine_reference)
    local_error = np.abs(coarse_local - fine_reference)

    assert not np.allclose(coarse_local, coarse_legacy)
    assert np.all(local_error < legacy_error)


def test_component_prediction_uses_local_agn_line_photometry(monkeypatch):
    _patch_ssp(monkeypatch)

    cfg = _cfg(fit_host=False, n_wave=64, rest_wave_max=2000.0)
    cfg.photometry = PhotometryData(filter_names=["ha"], fluxes=[1.0], errors=[0.1])
    cfg.filters = FilterSet(
        curves=[
            FilterCurve(
                name="ha",
                wave=[650.0, 690.0, 730.0],
                transmission=[0.0, 1.0, 0.0],
            )
        ],
    )
    cfg.likelihood.use_fast_photometry_projection = False
    cfg.likelihood.use_local_line_photometry = True
    cfg.likelihood.variability_uncertainty = False
    cfg.nebular.enabled = False
    cfg.agn.fit_balmer_continuum = False
    context = build_model_context(cfg)

    data = _fixed_component_data()
    data["log_broad_line_width_kms"] = np.array(np.log(1200.0))
    data["log_narrow_line_width_kms"] = np.array(np.log(1200.0))
    data["feii_norm"] = np.array(0.0)
    predictive = _deterministic_trace(context, data)
    likelihood = _deterministic_likelihood_trace(context, data)

    np.testing.assert_allclose(_site(predictive, "pred_fluxes"), _site(likelihood, "pred_fluxes"), rtol=2.0e-10, atol=1.0e-30)
    np.testing.assert_allclose(_site(predictive, "agn_fluxes"), _site(likelihood, "pred_fluxes"), rtol=2.0e-10, atol=1.0e-30)


def test_fixed_local_line_cache_matches_exact_local_line_projection(monkeypatch):
    _patch_ssp(monkeypatch)

    def _line_cfg(*, use_cache):
        cfg = _cfg(fit_host=False, n_wave=64, rest_wave_max=2000.0)
        cfg.photometry = PhotometryData(filter_names=["ha"], fluxes=[1.0], errors=[0.1])
        cfg.filters = FilterSet(
            curves=[
                FilterCurve(
                    name="ha",
                    wave=[650.0, 690.0, 730.0],
                    transmission=[0.0, 1.0, 0.0],
                )
            ],
        )
        cfg.likelihood.use_fast_photometry_projection = True
        cfg.likelihood.use_local_line_photometry = True
        cfg.likelihood.use_fixed_local_line_cache = use_cache
        cfg.likelihood.variability_uncertainty = False
        cfg.nebular.enabled = False
        cfg.agn.fit_balmer_continuum = False
        return cfg

    data = _fixed_component_data()
    data["log_broad_line_width_kms"] = np.array(np.log(1200.0))
    data["log_narrow_line_width_kms"] = np.array(np.log(1200.0))
    data["feii_norm"] = np.array(0.0)

    cached = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(use_cache=True)), data),
        "pred_fluxes",
    )
    exact = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(use_cache=False)), data),
        "pred_fluxes",
    )

    np.testing.assert_allclose(cached, exact, rtol=5.0e-4, atol=1.0e-30)


def test_local_line_photometry_improves_redshift_fit_line_projection(monkeypatch):
    _patch_ssp(monkeypatch)

    def _line_cfg(n_wave, *, local_lines):
        cfg = _cfg(fit_host=False, n_wave=n_wave, rest_wave_max=2000.0)
        cfg.observation.redshift_mode = "fit"
        cfg.observation.redshift_err = 0.01
        cfg.photometry = PhotometryData(filter_names=["ha"], fluxes=[1.0], errors=[0.1])
        cfg.filters = FilterSet(
            curves=[
                FilterCurve(
                    name="ha",
                    wave=[650.0, 690.0, 730.0],
                    transmission=[0.0, 1.0, 0.0],
                )
            ],
        )
        cfg.likelihood.use_fast_photometry_projection = True
        cfg.likelihood.use_local_line_photometry = local_lines
        cfg.likelihood.variability_uncertainty = False
        cfg.nebular.enabled = False
        cfg.agn.fit_balmer_continuum = False
        return cfg

    data = _fixed_component_data()
    data["redshift"] = np.array(0.05)
    data["log_broad_line_width_kms"] = np.array(np.log(1200.0))
    data["log_narrow_line_width_kms"] = np.array(np.log(1200.0))
    data["feii_norm"] = np.array(0.0)

    coarse_legacy = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(64, local_lines=False)), data),
        "pred_fluxes",
    )
    coarse_local = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(64, local_lines=True)), data),
        "pred_fluxes",
    )
    fine_reference = _site(
        _deterministic_likelihood_trace(build_model_context(_line_cfg(4096, local_lines=False)), data),
        "pred_fluxes",
    )

    legacy_error = np.abs(coarse_legacy - fine_reference)
    local_error = np.abs(coarse_local - fine_reference)

    assert not np.allclose(coarse_local, coarse_legacy)
    assert np.all(local_error < legacy_error)
