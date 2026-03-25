import numpy as np
import pytest

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
from grahspj.preload import build_model_context


def test_likelihood_defaults_include_absolute_flux_scale_prior():
    cfg = LikelihoodConfig()
    assert cfg.use_absolute_flux_scale_prior is True
    assert cfg.absolute_flux_scale_prior_sigma_dex > 0.0


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
