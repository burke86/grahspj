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
from grahspj.model import _igm_transmission, _torus_component, grahsp_photometric_model
from grahspj.preload import _build_fixed_igm_jax, _build_igm_cache_jax, build_model_context


def test_likelihood_defaults_include_absolute_flux_scale_prior():
    cfg = LikelihoodConfig()
    assert cfg.use_absolute_flux_scale_prior is True
    assert cfg.absolute_flux_scale_prior_sigma_dex > 0.0


def test_torus_component_wavelengths_are_angstrom_converted_to_micron():
    wave = np.asarray([2000.0, 20000.0, 170000.0])
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
