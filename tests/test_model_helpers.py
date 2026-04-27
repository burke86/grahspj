import numpy as np
import pytest
import jax
from types import SimpleNamespace
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
    JaxQSOFitConfig,
    LikelihoodConfig,
    Observation,
    PhotometryData,
    SpectroscopyConfig,
    SpectroscopyData,
    _coerce_spectroscopy_config,
)
from grahspj.core import GRAHSPJ
from grahspj.model import (
    GRAHSP_BIATTENUATION_BREAK_A,
    GRAHSP_SI_ABS_LAM_A,
    GRAHSP_SI_ABS_WIDTH_A,
    GRAHSP_SI_EM_LAM_A,
    GRAHSP_SI_EM_WIDTH_A,
    GRAHSP_TORUS_NORM_A,
    _attenuation_curve,
    _apply_biattenuation,
    _balmer_continuum_jax,
    _feii_component,
    _flux_conserving_line_gaussians,
    _host_dust_emission,
    _igm_transmission,
    _line_gaussians,
    _powerlaw_jax,
    _project_filters,
    _redshift_to_obs,
    _torus_component,
    grahsp_photometric_model,
)
from grahspj.preload import _build_fixed_igm_jax, _build_igm_cache_jax, build_model_context
from grahspj.preload import (
    ModelContext,
    PackedFiltersJax,
    SSPData,
    _DALE2014_CACHE,
    _HOST_BASIS_CACHE,
    _build_host_basis,
    _lnu_lsun_per_hz_to_llambda_w_per_a_np,
    _load_vendored_dale2014_templates,
)


def test_likelihood_defaults_include_absolute_flux_scale_prior():
    cfg = LikelihoodConfig()
    assert cfg.use_absolute_flux_scale_prior is True
    assert cfg.absolute_flux_scale_prior_sigma_dex > 0.0


def test_agn_disk_is_normalized_at_5100_angstrom():
    wave = np.asarray([2500.0, 5100.0, 10000.0])
    disk = np.asarray(_powerlaw_jax(wave, 2.0, 0.0, -1.0, 5100.0, 1000.0, 10.0, 0.0))

    assert disk[1] == pytest.approx(2.0)
    assert np.all(np.isfinite(disk))
    assert np.all(disk > 0.0)


def test_agn_disk_powerlaw_slopes_and_cutoff_are_in_wavelength_space():
    blue_wave = np.asarray([2500.0, 5000.0])
    red_wave = np.asarray([10000.0, 20000.0])
    blue = np.asarray(_powerlaw_jax(blue_wave, 1.0, -0.5, -2.0, 5100.0, 7000.0, 1000.0, 0.0))
    red = np.asarray(_powerlaw_jax(red_wave, 1.0, -0.5, -2.0, 5100.0, 7000.0, 1000.0, 0.0))
    cutoff = np.asarray(_powerlaw_jax(np.asarray([500.0, 5000.0]), 1.0, -0.5, -2.0, 5100.0, 7000.0, 1000.0, 1000.0))
    no_cutoff = np.asarray(_powerlaw_jax(np.asarray([500.0, 5000.0]), 1.0, -0.5, -2.0, 5100.0, 7000.0, 1000.0, 0.0))

    assert blue[1] < blue[0]
    assert red[1] < red[0]
    assert cutoff[0] > no_cutoff[0] * 0.8
    assert cutoff[1] < no_cutoff[1]


def test_flux_conserving_lines_preserve_integrated_luminosity():
    wave = np.linspace(4500.0, 5500.0, 20001)
    line = np.asarray(_flux_conserving_line_gaussians(wave, np.asarray([5000.0]), np.asarray([3.0]), 300.0))

    assert np.trapezoid(line, x=wave) == pytest.approx(3.0, rel=2.0e-4)


def test_native_agn_lines_use_5100_scaled_normalization():
    wave = np.linspace(4900.0, 5100.0, 20001)
    line_wave = np.asarray([5000.0])
    line_lumin = np.asarray([2.0])
    line = np.asarray(_line_gaussians(wave, line_wave, line_lumin, 300.0))

    assert wave[np.argmax(line)] == pytest.approx(5000.0, abs=0.05)
    assert np.trapezoid(line, x=wave) == pytest.approx(np.sqrt(2.0) * 5100.0 * line_lumin[0], rel=2.0e-4)


def test_biattenuation_break_is_grahsp_nm_value_converted_to_angstrom():
    wave = np.asarray([5500.0, GRAHSP_BIATTENUATION_BREAK_A, 22000.0])
    curve = np.asarray(_attenuation_curve(wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A))

    assert GRAHSP_BIATTENUATION_BREAK_A == 11000.0
    assert curve[1] == pytest.approx(1.2)
    assert curve[0] > curve[1] > curve[2]


def test_biattenuation_routes_host_and_agn_extinction_and_dust_budget():
    wave = np.asarray([1000.0, 2000.0, 4000.0])
    host = np.asarray([3.0, 3.0, 3.0])
    agn = np.asarray([5.0, 5.0, 5.0])
    ebv_gal = 0.2
    ebv_agn = 0.3
    gal_att, agn_att, host_absorbed, dust_luminosity = _apply_biattenuation(
        wave,
        host,
        agn,
        ebv_gal,
        ebv_agn,
        -1.2,
        -3.0,
        1.2,
        GRAHSP_BIATTENUATION_BREAK_A,
    )
    curve = np.asarray(_attenuation_curve(wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A))
    expected_host = host * 10 ** (ebv_gal * curve / -2.5)
    expected_agn = agn * 10 ** ((ebv_gal + ebv_agn) * curve / -2.5)

    assert np.allclose(np.asarray(gal_att), expected_host)
    assert np.allclose(np.asarray(agn_att), expected_agn)
    assert np.allclose(np.asarray(host_absorbed), host - expected_host)
    assert float(dust_luminosity) == pytest.approx(np.trapezoid(host - expected_host, x=wave))


def test_dale2014_host_dust_matches_cigale_v2025_1_reference():
    _DALE2014_CACHE.clear()
    alpha_grid, wave_a, lumin_per_a = _load_vendored_dale2014_templates()

    with np.load("tests/fixtures/cigale_v2025_1_dale2014_reference.npz") as ref:
        assert str(ref["cigale_version"]) == "2025.1"
        assert np.array_equal(alpha_grid, ref["alpha_grid"])
        assert np.allclose(wave_a, ref["wave_a"], rtol=0.0, atol=1.0e-10)
        assert np.allclose(wave_a[ref["wave_indices"]], ref["wave_targets_a"], rtol=0.0, atol=1.0e-10)
        assert np.allclose(
            lumin_per_a[np.ix_(ref["alpha_indices"], ref["wave_indices"])],
            ref["lumin_samples"],
            rtol=2.0e-7,
            atol=0.0,
        )
        assert np.allclose(np.trapezoid(lumin_per_a, x=wave_a, axis=1), ref["integrals"], rtol=2.0e-7, atol=1.0e-12)

    assert np.allclose(np.trapezoid(lumin_per_a, x=wave_a, axis=1), 1.0, rtol=2.0e-7, atol=1.0e-12)
    assert np.all(lumin_per_a[:, wave_a < 20000.0] == 0.0)


def test_host_dust_emission_integrates_to_absorbed_luminosity_on_broad_grid():
    _DALE2014_CACHE.clear()
    alpha_grid, wave_a, lumin_per_a = _load_vendored_dale2014_templates()
    context = SimpleNamespace(
        dust_alpha_grid_jax=np.asarray(alpha_grid),
        dust_lumin_rest_jax=np.asarray(lumin_per_a),
    )

    dust = np.asarray(_host_dust_emission(context, 7.5, 2.0))

    assert np.trapezoid(dust, x=wave_a) == pytest.approx(7.5, rel=2.0e-7)


def test_host_stellar_basis_lnu_to_llambda_units_and_interpolation():
    _HOST_BASIS_CACHE.clear()
    ssp_wave = np.asarray([1000.0, 2000.0, 4000.0])
    ssp_flux = np.asarray([[[1.0, 2.0, 4.0], [0.5, 1.0, 2.0]]])
    ssp_data = SSPData(
        ssp_lgmet=np.asarray([0.0]),
        ssp_lg_age_gyr=np.asarray([-3.0, -1.0]),
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
    )
    rest_wave = np.asarray([1500.0, 3000.0])

    basis = _build_host_basis(rest_wave, ssp_data)
    expected_native = _lnu_lsun_per_hz_to_llambda_w_per_a_np(ssp_wave[None, None, :], ssp_flux)
    expected_rest = np.asarray(
        [
            [
                np.interp(rest_wave, ssp_wave, expected_native[0, 0], left=0.0, right=0.0),
                np.interp(rest_wave, ssp_wave, expected_native[0, 1], left=0.0, right=0.0),
            ]
        ]
    )

    assert np.allclose(basis.rest_llambda, expected_rest, rtol=1.0e-12, atol=0.0)
    assert np.all(basis.n_ly_per_msun == 0.0)
    assert np.all(basis.ly_lum_per_msun == 0.0)
    assert basis.surviving_frac_by_age.shape == ssp_data.ssp_lg_age_gyr.shape


def test_torus_component_wavelengths_are_angstrom_converted_to_micron():
    wave = np.asarray([2000.0, 20000.0, GRAHSP_TORUS_NORM_A, 170000.0])
    torus = np.asarray(
        _torus_component(
            wave,
            fcov=0.2,
            si=0.0,
            cool_lam=17.0,
            cool_width=0.45,
            hot_lam=2.0,
            hot_width=0.2,
            hot_fcov=1.0,
            si_ratio=0.29,
            si_em_lam=9841.0,
            si_abs_lam=14224.0,
            si_em_width=1025.3,
            si_abs_width=1163.5,
            l_agn=1.0,
        )
    )

    assert torus[1] > 100.0 * torus[0]
    assert torus[2] > 100.0 * torus[0]
    assert torus[3] > 100.0 * torus[0]


def test_torus_normalization_scales_with_covering_fraction_and_agn_luminosity():
    wave = np.asarray([GRAHSP_TORUS_NORM_A])
    torus = np.asarray(
        _torus_component(
            wave,
            fcov=0.2,
            si=0.0,
            cool_lam=17.0,
            cool_width=0.45,
            hot_lam=2.0,
            hot_width=0.2,
            hot_fcov=0.1,
            si_ratio=0.29,
            si_em_lam=GRAHSP_SI_EM_LAM_A,
            si_abs_lam=GRAHSP_SI_ABS_LAM_A,
            si_em_width=GRAHSP_SI_EM_WIDTH_A,
            si_abs_width=GRAHSP_SI_ABS_WIDTH_A,
            l_agn=10.0,
        )
    )

    assert torus[0] == pytest.approx(2.5 * 10.0 * 0.2 / GRAHSP_TORUS_NORM_A)


def test_torus_hot_and_cool_components_peak_in_micron_space():
    wave = np.logspace(np.log10(10000.0), np.log10(300000.0), 2000)
    cool_only = np.asarray(
        _torus_component(wave, 0.2, 0.0, 17.0, 0.15, 2.0, 0.1, 0.0, 0.29, GRAHSP_SI_EM_LAM_A, GRAHSP_SI_ABS_LAM_A, GRAHSP_SI_EM_WIDTH_A, GRAHSP_SI_ABS_WIDTH_A, 1.0)
    )
    hot_only = np.asarray(
        _torus_component(wave, 0.2, 0.0, 17.0, 0.15, 2.0, 0.1, 1.0, 0.29, GRAHSP_SI_EM_LAM_A, GRAHSP_SI_ABS_LAM_A, GRAHSP_SI_EM_WIDTH_A, GRAHSP_SI_ABS_WIDTH_A, 1.0)
    ) - cool_only

    assert wave[np.argmax(cool_only)] == pytest.approx(170000.0, rel=0.02)
    assert wave[np.argmax(hot_only)] == pytest.approx(20000.0, rel=0.02)


def test_torus_silicate_features_are_in_mid_ir_angstroms():
    wave = np.asarray([9841.0, GRAHSP_SI_EM_LAM_A, GRAHSP_SI_ABS_LAM_A])
    torus = np.asarray(
        _torus_component(
            wave,
            fcov=0.2,
            si=1.0,
            cool_lam=17.0,
            cool_width=0.45,
            hot_lam=2.0,
            hot_width=0.2,
            hot_fcov=0.0,
            si_ratio=0.29,
            si_em_lam=GRAHSP_SI_EM_LAM_A,
            si_abs_lam=GRAHSP_SI_ABS_LAM_A,
            si_em_width=GRAHSP_SI_EM_WIDTH_A,
            si_abs_width=GRAHSP_SI_ABS_WIDTH_A,
            l_agn=1.0,
        )
    )

    assert torus[1] > torus[0]
    assert torus[1] > torus[2]


def test_feii_velocity_shift_moves_template_feature_by_fractional_wavelength():
    wave = np.linspace(2400.0, 2800.0, 4001)
    template = np.exp(-0.5 * ((wave - 2600.0) / 5.0) ** 2)
    shifted = np.asarray(_feii_component(wave, template, norm=1.0, fwhm_kms=10.0, shift_frac=0.01))
    unshifted = np.asarray(_feii_component(wave, template, norm=1.0, fwhm_kms=10.0, shift_frac=0.0))

    assert wave[np.argmax(unshifted)] == pytest.approx(2600.0, abs=0.5)
    assert wave[np.argmax(shifted)] == pytest.approx(2600.0 * 1.01, abs=1.0)
    assert np.trapezoid(shifted, x=wave) == pytest.approx(np.trapezoid(unshifted, x=wave), rel=0.05)


def test_balmer_continuum_has_3646_angstrom_edge_and_blueward_emission():
    wave = np.linspace(2500.0, 4500.0, 2001)
    balmer = np.asarray(_balmer_continuum_jax(wave, balmer_norm=2.0, balmer_te=15000.0, balmer_tau=1.0, balmer_vel=10.0))

    assert np.nanmax(balmer[wave > 3800.0]) < 1.0e-4 * np.nanmax(balmer)
    assert balmer[np.argmin(np.abs(wave - 3646.0))] > 0.0
    assert np.all(np.isfinite(balmer))


def test_redshift_projection_uses_luminosity_distance_and_one_plus_z():
    rest_wave = np.asarray([1000.0, 2000.0, 3000.0])
    rest_lum = np.asarray([4.0, 4.0, 4.0])
    obs_wave = rest_wave * 2.0
    d_l = 10.0
    obs = np.asarray(_redshift_to_obs(rest_wave, rest_lum, obs_wave, redshift=1.0, luminosity_distance_m=d_l))

    assert np.allclose(obs, rest_lum / (4.0 * np.pi * d_l**2 * 2.0))


def test_filter_projection_flat_flambda_to_mjy_units():
    packed = PackedFiltersJax(
        interp_indices=np.asarray([[0, 1, 2]], dtype=np.int32),
        interp_weight=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        transmission=np.asarray([[1.0, 1.0, 1.0]], dtype=float),
        work_wave=np.asarray([[4000.0, 5000.0, 6000.0]], dtype=float),
        effective_wavelength=np.asarray([5000.0], dtype=float),
        valid_mask=np.asarray([[True, True, True]], dtype=bool),
    )
    obs_flux = np.asarray([2.0e-20, 2.0e-20, 2.0e-20, 2.0e-20])
    projected = np.asarray(_project_filters(obs_flux, packed))
    expected = 1.0e-10 / 299792458.0 * 1.0e29 * 5000.0**2 * 2.0e-20

    assert projected[0] == pytest.approx(expected)


def test_build_context_with_inline_templates(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
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
        spectroscopy=SpectroscopyData(
            wave_obs=[3500.0, 4500.0, 5500.0],
            fluxes=[0.1, 0.2, 0.15],
            errors=[0.01, 0.02, 0.015],
            mask=[True, False, True],
            instrument="test",
        ),
        spectroscopy_config=SpectroscopyConfig(enabled=True),
        inference=InferenceConfig(map_steps=2),
    )
    context = build_model_context(cfg)
    assert context.ssp_data.ssp_flux.shape == (2, 2, 4)
    assert context.gal_t_table.shape == (cfg.galaxy.sfh_n_steps,)
    assert context.t_obs_gyr > 0.0
    assert len(context.filters) == 1
    assert context.filters[0].name == "inline-f1"
    assert context.templates.feii_wave.shape[0] == 2
    assert context.templates.dust_alpha_grid.size > 0
    assert context.templates.dust_wave.size > 0
    assert context.templates.dust_lumin.ndim == 2
    assert context.spec_wave_obs.tolist() == [3500.0, 4500.0, 5500.0]
    assert context.spec_mask.tolist() == [True, False, True]
    assert context.spec_spectrum_index.tolist() == [0, 0, 0]
    assert context.spec_instruments == ("test",)


def test_context_accepts_multiple_spectra(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
            aperture_diameter_arcsec=[2.0],
        ),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(use_host_capture_model=True),
        spectroscopy=[
            SpectroscopyData(
                wave_obs=[5000.0, 4000.0],
                fluxes=[0.2, 0.1],
                errors=[0.02, 0.01],
                instrument="sdss",
                aperture_diameter_arcsec=3.0,
            ),
            SpectroscopyData(
                wave_obs=[7000.0],
                fluxes=[0.3],
                errors=[0.03],
                instrument="desi",
                psf_fwhm_arcsec=1.5,
            ),
        ],
        spectroscopy_config=SpectroscopyConfig(enabled=True),
        inference=InferenceConfig(map_steps=2),
    )

    context = build_model_context(cfg)

    assert context.spec_wave_obs.tolist() == [4000.0, 5000.0, 7000.0]
    assert context.spec_spectrum_index.tolist() == [0, 0, 1]
    assert context.spec_effective_spatial_scale_arcsec.tolist() == [3.0, 1.5]
    assert context.spec_aperture_diameter_arcsec.tolist()[0] == 3.0
    assert np.isnan(context.spec_aperture_diameter_arcsec.tolist()[1])
    assert context.spec_instruments == ("sdss", "desi")


def test_spectroscopy_config_migrates_legacy_jaxqsofit_flags():
    cfg = _coerce_spectroscopy_config(
        {
            "enabled": True,
            "backend": "jaxqsofit",
            "jaxqsofit_use_lines": False,
            "jaxqsofit_use_feii": True,
            "jaxqsofit": {
                "use_balmer_continuum": True,
                "line_flux_scale_mjy": 0.1,
            },
        }
    )

    assert cfg.jaxqsofit.use_spectral_lines is False
    assert cfg.jaxqsofit.use_spectral_feii is True
    assert cfg.jaxqsofit.use_spectral_balmer_continuum is True
    assert cfg.jaxqsofit.line_flux_scale_mjy == 0.1


def test_jaxqsofit_joint_backend_builds_flux_scaled_smart_priors(monkeypatch):
    pytest.importorskip("jaxqsofit.defaults")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(),
        spectroscopy=SpectroscopyData(
            wave_obs=[5000.0, 5100.0, 5200.0],
            fluxes=[2.0, 4.0, 100.0],
            errors=[0.1, 0.1, 0.1],
            mask=[True, True, False],
            instrument="sdss",
        ),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            jaxqsofit=JaxQSOFitConfig(line_flux_scale_mjy=0.01),
        ),
        inference=InferenceConfig(map_steps=2),
    )

    context = build_model_context(cfg)

    prior = context.jaxqsofit_prior_config
    assert prior is not None
    assert prior["log_cont_norm"]["loc"] == pytest.approx(np.log(3.0))
    line_table = prior["line"]["table"]
    assert line_table
    assert min(float(row["minsca"]) for row in line_table) >= 3.0e-4


def test_grahspj_model_can_call_jaxqsofit_backend(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
            aperture_diameter_arcsec=[2.0],
        ),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[0.0, 0.0]),
            emission_line_template=EmissionLineTemplate(
                wave=[486.1],
                lumin_blagn=[0.0],
                lumin_sy2=[0.0],
                lumin_liner=[0.0],
            ),
        ),
        likelihood=LikelihoodConfig(
            use_host_capture_model=True,
            fit_intrinsic_scatter=False,
            variability_uncertainty=False,
            use_absolute_flux_scale_prior=False,
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=[5200.0, 5400.0, 5600.0],
            fluxes=[0.1, 0.12, 0.11],
            errors=[0.02, 0.02, 0.02],
            instrument="sdss",
            aperture_diameter_arcsec=3.0,
        ),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            fit_scale=False,
            jaxqsofit=JaxQSOFitConfig(
                use_spectral_lines=False,
                use_spectral_feii=False,
                use_spectral_balmer_continuum=False,
            ),
        ),
        inference=InferenceConfig(map_steps=2),
    )
    context = build_model_context(cfg)

    params = {
        "log_agn_amp": np.array(30.0),
        "uv_slope": np.array(0.0),
        "pl_slope": np.array(-1.0),
        "pl_bend_loc": np.array(100.0),
        "pl_bend_width": np.array(10.0),
        "pl_cutoff": np.array(10000.0),
        "fcov": np.array(0.1),
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
        "ebv_agn": np.array(0.0),
    }
    tr = trace(substitute(seed(grahsp_photometric_model, jax.random.PRNGKey(1)), data=params)).get_trace(
        context,
        include_components=False,
    )

    assert "jqf_total_model" in tr
    assert "jqf_line_model" in tr
    assert np.asarray(tr["pred_spectrum_fluxes"]["value"]).shape == (3,)


def test_grahspj_jaxqsofit_backend_uses_nested_tied_line_config(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.0),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
        ),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(
            fit_intrinsic_scatter=False,
            variability_uncertainty=False,
            use_absolute_flux_scale_prior=False,
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=[4800.0, 4900.0, 5000.0],
            fluxes=[0.1, 0.12, 0.11],
            errors=[0.02, 0.02, 0.02],
            instrument="sdss",
        ),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            fit_scale=False,
            jaxqsofit=JaxQSOFitConfig(
                use_spectral_lines=True,
                use_tied_lines=True,
                use_spectral_feii=False,
                use_spectral_balmer_continuum=False,
                line_flux_scale_mjy=0.1,
            ),
        ),
        inference=InferenceConfig(map_steps=2),
    )
    context = build_model_context(cfg)

    params = {
        "log_agn_amp": np.array(30.0),
        "uv_slope": np.array(0.0),
        "pl_slope": np.array(-1.0),
        "pl_bend_loc": np.array(100.0),
        "pl_bend_width": np.array(10.0),
        "pl_cutoff": np.array(10000.0),
        "fcov": np.array(0.1),
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
        "ebv_agn": np.array(0.0),
    }
    tr = trace(substitute(seed(grahsp_photometric_model, jax.random.PRNGKey(2)), data=params)).get_trace(
        context,
        include_components=False,
    )

    assert "jqf_line_dmu_group" in tr
    assert "jqf_line_sig_group" in tr
    assert "jqf_line_amp_group" in tr
    assert "jqf_line_model_broad" in tr
    assert np.asarray(tr["pred_spectrum_fluxes"]["value"]).shape == (3,)


def test_grahspj_jaxqsofit_tied_line_backend_runs_svi_jit(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())

    line_table = [
        {
            "lambda": 5008.24,
            "linename": "OIII5007",
            "compname": "Hb",
            "ngauss": 1,
            "inisca": 0.01,
            "minsca": 1.0e-6,
            "maxsca": 1.0,
            "inisig": 1.0e-3,
            "minsig": 1.0e-4,
            "maxsig": 1.0e-2,
            "voff": 0.01,
            "vindex": 1,
            "windex": 1,
            "findex": 1,
            "fvalue": 1.0,
        }
    ]
    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.0),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(
            fit_intrinsic_scatter=False,
            variability_uncertainty=False,
            use_absolute_flux_scale_prior=False,
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=[4900.0, 5000.0, 5100.0],
            fluxes=[0.1, 0.12, 0.11],
            errors=[0.02, 0.02, 0.02],
            instrument="sdss",
        ),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            fit_scale=False,
            jaxqsofit=JaxQSOFitConfig(
                use_spectral_lines=True,
                use_tied_lines=True,
                line_table=line_table,
                line_flux_scale_mjy=0.1,
            ),
        ),
        inference=InferenceConfig(map_steps=1, learning_rate=1.0e-3),
    )

    fitter = GRAHSPJ(cfg)
    result = fitter.fit_map(steps=1, progress_bar=False)

    assert np.asarray(result["losses"]).shape == (1,)
    assert np.isfinite(float(np.asarray(result["losses"])[0]))


def test_plot_jaxqsofit_spectrum_adapts_joint_predictive(monkeypatch):
    jaxqsofit = pytest.importorskip("jaxqsofit")
    captured = {}

    def _fake_plot_fig(self, **kwargs):
        captured["wave"] = np.asarray(self.wave)
        captured["flux"] = np.asarray(self.flux)
        captured["model_total"] = np.asarray(self.model_total)
        captured["host"] = np.asarray(self.host)
        captured["line"] = np.asarray(self.f_line_model)
        captured["custom_components"] = dict(self.custom_components)
        captured["kwargs"] = kwargs
        self.fig = "fig"

    monkeypatch.setattr(jaxqsofit.QSOFit, "plot_fig", _fake_plot_fig)

    fitter = GRAHSPJ.__new__(GRAHSPJ)
    fitter.config = SimpleNamespace(
        observation=SimpleNamespace(object_id="obj", redshift=1.0),
        spectroscopy_config=SimpleNamespace(backend="jaxqsofit"),
    )
    fitter.context = SimpleNamespace(
        spec_wave_obs=np.asarray([4000.0, 5000.0, 6000.0]),
        spec_fluxes=np.asarray([1.0, 2.0, 3.0]),
        spec_errors=np.asarray([0.1, 0.2, 0.3]),
        spec_mask=np.asarray([True, True, True]),
        spec_spectrum_index=np.asarray([0, 0, 0]),
    )
    fitter.predict = lambda posterior="latest": {
        "obs_wave": np.asarray([[4000.0, 5000.0, 6000.0], [4000.0, 5000.0, 6000.0]]),
        "pred_spectrum_fluxes": np.asarray([[1.1, 2.2, 3.3], [1.3, 2.4, 3.5]]),
        "jqf_continuum_model": np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "jqf_line_model": np.asarray([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]),
        "spectrum_scale_fit": np.asarray([2.0, 2.0]),
        "spectrum_host_capture_fraction": np.asarray([0.5, 0.5]),
        "host_obs_sed": np.asarray([[1.0e-20, 2.0e-20, 3.0e-20], [1.0e-20, 2.0e-20, 3.0e-20]]),
        "disk_obs_sed": np.asarray([[2.0e-20, 2.0e-20, 2.0e-20], [2.0e-20, 2.0e-20, 2.0e-20]]),
        "torus_obs_sed": np.asarray([[0.5e-20, 0.5e-20, 0.5e-20], [0.5e-20, 0.5e-20, 0.5e-20]]),
        "dust_obs_sed": np.asarray([[0.1e-20, 0.1e-20, 0.1e-20], [0.1e-20, 0.1e-20, 0.1e-20]]),
    }

    fig = fitter.plot_jaxqsofit_spectrum(show_plot=False, plot_residual=False)

    assert fig == "fig"
    assert captured["wave"].tolist() == [2000.0, 2500.0, 3000.0]
    assert np.all(captured["model_total"] > captured["flux"] * 0.0)
    assert np.all(captured["host"] > 0.0)
    assert np.all(captured["line"] > 0.0)
    assert "grahspj_torus" in captured["custom_components"]
    assert "grahspj_host_dust" in captured["custom_components"]
    assert captured["kwargs"]["show_plot"] is False
    assert captured["kwargs"]["plot_residual"] is False


def test_config_rejects_invalid_redshift_pdf():
    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1, fit_redshift=True),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(),
        inference=InferenceConfig(map_steps=2),
        prior_config={
            "redshift_pdf": {
                "z_grid": [0.3, 0.2, 0.4],
                "pdf": [0.2, 0.5, 0.3],
            }
        },
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        cfg.validate()


def test_igm_transmission_on_rest_grid_is_near_unity_redward_of_lyman_alpha():
    rest_wave = np.array([150.0, 121.6, 100.0, 91.2, 80.0], dtype=float)
    cache = _build_igm_cache_jax(rest_wave)
    transmission = np.asarray(_build_fixed_igm_jax(cache, 1.0), dtype=float)

    assert transmission[0] > 0.99
    assert transmission[1] > 0.95
    assert transmission[2] < 1.0
    assert transmission[3] < transmission[2]
    assert transmission[4] < transmission[3]


def test_dynamic_and_fixed_igm_evaluators_match():
    rest_wave = np.array([150.0, 121.6, 100.0, 91.2, 80.0], dtype=float)
    cache = _build_igm_cache_jax(rest_wave)

    fixed = np.asarray(_build_fixed_igm_jax(cache, 2.0), dtype=float)
    dynamic = np.asarray(_igm_transmission(cache, 2.0), dtype=float)

    assert np.allclose(dynamic, fixed)
