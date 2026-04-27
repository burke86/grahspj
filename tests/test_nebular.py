import numpy as np
from numpyro.handlers import seed, trace
from pathlib import Path

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
from grahspj.preload import _load_nebular_templates_jax, build_model_context


REFERENCE = Path(__file__).parent / "fixtures" / "cigale_v2025_1_nebular_reference.npz"


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


def test_vendored_nebular_resources_are_cigale_v2025_1():
    with np.load("src/grahspj/resources/nebular/nebular_lines.npz") as lines, np.load(
        "src/grahspj/resources/nebular/nebular_continuum.npz"
    ) as cont:
        assert str(lines["cigale_version"]) == "2025.1"
        assert str(cont["cigale_version"]) == "2025.1"
        assert str(lines["cigale_git_tag"]) == "v2025.1"
        assert str(lines["cigale_git_commit"]) == "29cb909fe2636800b4acdb1dfc7129d8c8494a24"
        assert np.array_equal(lines["z_grid"], cont["z_grid"])
        assert np.array_equal(lines["logu_grid"], cont["logu_grid"])
        assert np.array_equal(lines["ne_grid"], cont["ne_grid"])


def test_nebular_resources_match_cigale_v2025_1_static_reference():
    with np.load(REFERENCE) as ref, np.load("src/grahspj/resources/nebular/nebular_lines.npz") as lines, np.load(
        "src/grahspj/resources/nebular/nebular_continuum.npz"
    ) as cont:
        z_idx = int(np.where(np.isclose(lines["z_grid"], 0.02))[0][0])
        u_idx = int(np.where(np.isclose(lines["logu_grid"], -2.0))[0][0])
        ne_idx = int(np.where(np.isclose(lines["ne_grid"], 100.0))[0][0])

        assert str(ref["cigale_version"]) == "2025.1"
        assert np.array_equal(lines["z_grid"], ref["z_grid"])
        assert np.array_equal(lines["logu_grid"], ref["logu_grid"])
        assert np.array_equal(lines["ne_grid"], ref["ne_grid"])
        assert np.array_equal(lines["line_name"][ref["line_indices"]], ref["line_names"])
        assert np.allclose(lines["line_wave_a"][ref["line_indices"]], ref["line_wave_a"], rtol=0.0, atol=1.0e-10)
        assert np.allclose(
            lines["line_lumin_per_photon"][z_idx, u_idx, ne_idx, ref["line_indices"]],
            ref["line_lumin_z002_logu_m2_ne100"],
            rtol=2.0e-7,
            atol=0.0,
        )
        assert np.allclose(cont["continuum_wave_a"][ref["continuum_indices"]], ref["continuum_wave_a"], rtol=0.0, atol=1.0e-10)
        assert np.allclose(
            cont["continuum_lumin_per_a_per_photon"][z_idx, u_idx, ne_idx, ref["continuum_indices"]],
            ref["continuum_lumin_z002_logu_m2_ne100"],
            rtol=2.0e-7,
            atol=0.0,
        )


def test_nebular_template_loader_uses_v2025_1_grid():
    templates = _load_nebular_templates_jax(True)

    assert np.isclose(np.asarray(templates.z_grid), 0.02).any()
    assert np.isclose(np.asarray(templates.logu_grid), -2.0).any()
    assert np.isclose(np.asarray(templates.ne_grid), 100.0).any()
    assert templates.line_lumin_per_photon.shape == (26, 31, 3, 129)
    assert templates.continuum_lumin_per_a_per_photon.shape == (26, 31, 3, 1600)


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
