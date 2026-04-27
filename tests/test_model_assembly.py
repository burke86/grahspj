import numpy as np
from numpyro.handlers import seed, substitute, trace

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
    NebularConfig,
    Observation,
    PhotometryData,
)
from grahspj.model import GRAHSP_PL_BEND_LOC_A, GRAHSP_PL_BEND_WIDTH, GRAHSP_PL_CUTOFF_A, _project_filters, _redshift_to_obs, grahsp_photometric_model
from grahspj.preload import build_model_context


def _patch_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("grahspj.preload._HOST_BASIS_CACHE", {})


def _cfg(*, fit_host=True, fit_agn=True, fit_host_kinematics=False, rest_wave_max=3.0e6, n_wave=512):
    return FitConfig(
        observation=Observation(object_id="assembly", redshift=0.05),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(
            curves=[FilterCurve(name="f1", wave=[1500.0, 2000.0, 2500.0], transmission=[0.0, 1.0, 0.0])],
            use_grahsp_database=False,
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
            balmer_continuum_default=0.2,
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0, 3000.0], lumin=[0.0, 1.0, 0.0]),
            emission_line_template=EmissionLineTemplate(
                wave=[486.1, 656.3],
                lumin_blagn=[1.0, 0.5],
                lumin_sy2=[0.2, 0.1],
                lumin_liner=[0.1, 0.05],
            ),
        ),
        likelihood=LikelihoodConfig(
            fit_intrinsic_scatter=False,
            variability_uncertainty=False,
            use_absolute_flux_scale_prior=False,
            use_host_capture_model=False,
        ),
        nebular=NebularConfig(enabled=True, f_esc=0.0, f_dust=0.2, zgas=0.02, lines_width=300.0),
        inference=InferenceConfig(map_steps=2),
        prior_config={"log_stellar_mass": {"dist": "uniform", "low": 8.0, "high": 8.0}},
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


def _fixed_component_data():
    return {
        "ebv_gal": np.array(0.2),
        "ebv_agn": np.array(0.1),
        "dust_alpha": np.array(2.0),
        "log_agn_amp": np.array(np.log(1.0e34)),
        "uv_slope": np.array(0.0),
        "pl_slope": np.array(-1.0),
        "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
        "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
        "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
        "fcov": np.array(0.2),
        "si": np.array(0.0),
        "cool_lam": np.array(17.0),
        "cool_width": np.array(0.45),
        "hot_lam": np.array(2.0),
        "hot_width": np.array(0.5),
        "hot_fcov": np.array(0.1),
        "lines_strength": np.array(1.0),
        "line_width_kms": np.array(3000.0),
        "feii_norm": np.array(1.0),
        "feii_fwhm": np.array(3000.0),
        "feii_shift": np.array(0.0),
        "balmer_norm": np.array(0.2),
        "balmer_tau": np.array(1.0),
        "balmer_vel": np.array(3000.0),
    }


def test_component_rest_and_observed_seds_sum_to_total(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())
    tr = _deterministic_trace(
        context,
        {
            "ebv_gal": np.array(0.2),
            "ebv_agn": np.array(0.1),
            "dust_alpha": np.array(2.0),
            "log_agn_amp": np.array(np.log(1.0e34)),
            "uv_slope": np.array(0.0),
            "pl_slope": np.array(-1.0),
            "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
            "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
            "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
            "fcov": np.array(0.2),
            "si": np.array(0.0),
            "cool_lam": np.array(17.0),
            "cool_width": np.array(0.45),
            "hot_lam": np.array(2.0),
            "hot_width": np.array(0.5),
            "hot_fcov": np.array(0.1),
            "lines_strength": np.array(1.0),
            "line_width_kms": np.array(3000.0),
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


def test_energy_balance_dust_sed_integrates_to_absorbed_luminosity(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(rest_wave_max=2.3e9, n_wave=4096))
    tr = _deterministic_trace(context, {"ebv_gal": np.array(0.5), "ebv_agn": np.array(0.0), "dust_alpha": np.array(2.0)})

    rest_wave = _site(tr, "rest_wave")
    dust_luminosity = 10.0 ** float(_site(tr, "log_dust_luminosity_fit"))
    emitted_dust_luminosity = float(np.trapezoid(_site(tr, "dust_rest_sed"), x=rest_wave))

    assert dust_luminosity > 0.0
    np.testing.assert_allclose(emitted_dust_luminosity, dust_luminosity, rtol=2.0e-2, atol=0.0)


def test_agn_off_mode_has_zero_agn_components_and_no_total_leak(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(fit_agn=False))
    tr = _deterministic_trace(context, {"ebv_gal": np.array(0.2), "dust_alpha": np.array(2.0)})

    for key in ("agn_rest_sed", "disk_rest_sed", "torus_rest_sed", "feii_rest_sed", "line_rest_sed", "balmer_rest_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    for key in ("agn_obs_sed", "disk_obs_sed", "torus_obs_sed", "feii_obs_sed", "line_obs_sed", "balmer_obs_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    assert np.allclose(_site(tr, "total_rest_sed"), _site(tr, "host_total_rest_sed") + _site(tr, "dust_rest_sed"))
    assert np.allclose(_site(tr, "pred_fluxes"), _site(tr, "host_total_fluxes") + _site(tr, "dust_fluxes"))


def test_host_off_mode_has_zero_host_components_and_no_total_leak(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg(fit_host=False))
    tr = _deterministic_trace(context, {"log_agn_amp": np.array(np.log(1.0e34)), "fcov": np.array(0.2), "si": np.array(0.0)})

    for key in ("host_rest_sed", "host_total_rest_sed", "host_absorbed_rest_sed", "dust_rest_sed", "nebular_rest_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    for key in ("host_obs_sed", "host_total_obs_sed", "dust_obs_sed", "nebular_obs_sed"):
        assert np.allclose(_site(tr, key), 0.0)
    assert np.allclose(_site(tr, "total_rest_sed"), _site(tr, "agn_rest_sed"))
    assert np.allclose(_site(tr, "pred_fluxes"), _site(tr, "agn_fluxes"))


def test_host_kinematics_default_off_skips_broadening_call(monkeypatch):
    _patch_ssp(monkeypatch)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("host broadening should be skipped when fit_host_kinematics=False")

    monkeypatch.setattr("grahspj.model._shift_and_broaden_single_spectrum_lnlam", _raise_if_called)
    context = build_model_context(_cfg(fit_agn=False))
    tr = _deterministic_trace(context, {"ebv_gal": np.array(0.2), "dust_alpha": np.array(2.0)})

    assert "gal_v_kms" not in tr
    assert "gal_sigma_kms" not in tr
    assert np.all(np.isfinite(_site(tr, "pred_fluxes")))


def test_host_kinematics_enabled_samples_and_broadens(monkeypatch):
    _patch_ssp(monkeypatch)
    calls = {"n": 0}

    def _identity_broaden(lnwave, spectrum, v_kms, sigma_kms):
        calls["n"] += 1
        return spectrum

    monkeypatch.setattr("grahspj.model._shift_and_broaden_single_spectrum_lnlam", _identity_broaden)
    context = build_model_context(_cfg(fit_agn=False, fit_host_kinematics=True))
    tr = _deterministic_trace(
        context,
        {
            "gal_v_kms": np.array(0.0),
            "gal_sigma_kms": np.array(150.0),
            "ebv_gal": np.array(0.2),
            "dust_alpha": np.array(2.0),
        },
    )

    assert calls["n"] == 1
    assert "gal_v_kms" in tr
    assert "gal_sigma_kms" in tr


def test_plotted_component_sites_are_attenuated_likelihood_components(monkeypatch):
    _patch_ssp(monkeypatch)
    context = build_model_context(_cfg())
    tr = _deterministic_trace(context, {"ebv_gal": np.array(0.2), "ebv_agn": np.array(0.1), "dust_alpha": np.array(2.0)})

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
    assert np.allclose(_site(tr, "pred_fluxes"), projected_total, rtol=2.0e-10, atol=1.0e-30)


def test_fast_fixed_filter_projection_matches_legacy_photometry(monkeypatch):
    _patch_ssp(monkeypatch)
    fast_cfg = _cfg(n_wave=256)
    slow_cfg = _cfg(n_wave=256)
    fast_cfg.likelihood.use_fast_photometry_projection = True
    slow_cfg.likelihood.use_fast_photometry_projection = False
    fast_context = build_model_context(fast_cfg)
    slow_context = build_model_context(slow_cfg)

    fast_tr = _deterministic_likelihood_trace(fast_context, _fixed_component_data())
    slow_tr = _deterministic_likelihood_trace(slow_context, _fixed_component_data())

    np.testing.assert_allclose(_site(fast_tr, "pred_fluxes"), _site(slow_tr, "pred_fluxes"), rtol=2.0e-12, atol=1.0e-30)
