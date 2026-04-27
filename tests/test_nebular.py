import numpy as np
from numpyro.handlers import seed, trace

from grahspj.config import (
    AGNConfig,
    EmissionLineTemplate,
    FeIITemplate,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    NebularConfig,
    Observation,
    PhotometryData,
)
from grahspj.model import _cigale_nebular_correction, grahsp_photometric_model
from grahspj.preload import build_model_context


def _mock_config():
    return FitConfig(
        observation=Observation(object_id="obj", redshift=0.01),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])], use_grahsp_database=False),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", rest_wave_min=100.0, rest_wave_max=10000.0, n_wave=96, sfh_n_steps=16),
        agn=AGNConfig(
            fit_agn=False,
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5]),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1],
                lumin_blagn=[1.0, 0.5],
                lumin_sy2=[0.2, 0.1],
                lumin_liner=[0.1, 0.05],
            ),
        ),
        inference=InferenceConfig(map_steps=2),
    )


def _patch_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    monkeypatch.setattr("grahspj.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("grahspj.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("grahspj.preload._HOST_BASIS_CACHE", {})


def test_nebular_config_validates_escape_and_dust_fraction():
    NebularConfig(enabled=True, f_esc=0.2, f_dust=0.3).validate()
    for kwargs in ({"f_esc": -0.1}, {"f_dust": 1.2}, {"f_esc": 0.7, "f_dust": 0.4}):
        cfg = NebularConfig(enabled=True, **kwargs)
        try:
            cfg.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"NebularConfig accepted invalid values: {kwargs}")


def test_cigale_nebular_correction_limits():
    assert np.isclose(float(_cigale_nebular_correction(0.0, 0.0)), 1.0)
    assert np.isclose(float(_cigale_nebular_correction(1.0, 0.0)), 0.0)
    assert 0.0 < float(_cigale_nebular_correction(0.2, 0.1)) < 1.0


def test_host_basis_lyman_rates_are_finite(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _mock_config()
    context = build_model_context(cfg)

    assert np.all(np.isfinite(context.host_basis.n_ly_per_msun))
    assert np.all(context.host_basis.n_ly_per_msun >= 0.0)
    assert np.any(context.host_basis.n_ly_per_msun > 0.0)


def test_nebular_disabled_outputs_are_zero(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _mock_config()
    cfg.nebular.enabled = False
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 0)).get_trace()

    assert np.allclose(np.asarray(tr["nebular_rest_sed"]["value"]), 0.0)
    assert np.allclose(np.asarray(tr["nebular_fluxes"]["value"]), 0.0)


def test_nebular_enabled_adds_finite_component(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _mock_config()
    cfg.nebular = NebularConfig(enabled=True, f_esc=0.0, f_dust=0.0, zgas=0.02)
    cfg.prior_config["nebular_logU"] = {"loc": -2.0, "scale": 0.01}
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 1)).get_trace()

    nebular_rest = np.asarray(tr["nebular_rest_sed"]["value"])
    assert np.all(np.isfinite(nebular_rest))
    assert np.any(nebular_rest > 0.0)
    assert "nebular_logU" in tr
    assert "nebular_logU_fit" in tr


def test_nebular_escape_fraction_one_suppresses_emission(monkeypatch):
    _patch_ssp(monkeypatch)
    cfg = _mock_config()
    cfg.nebular = NebularConfig(enabled=True, f_esc=1.0, f_dust=0.0, zgas=0.02)
    context = build_model_context(cfg)
    tr = trace(seed(lambda: grahsp_photometric_model(context, include_components=True), 2)).get_trace()

    assert np.allclose(np.asarray(tr["nebular_rest_sed"]["value"]), 0.0)
