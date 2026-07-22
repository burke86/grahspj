import numpy as np
import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from types import SimpleNamespace
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
    RedshiftPriorConfig,
    MassMetallicityPriorConfig,
    SpectroscopyConfig,
    SpectroscopyData,
    fit_config_from_mapping,
)
from jaxsedfit.core import JAXSEDFit, _joint_dense_mass_blocks
from jaxsedfit.model import (
    GRAHSP_BIATTENUATION_BREAK_A,
    GRAHSP_PL_BEND_LOC_A,
    GRAHSP_PL_BEND_WIDTH,
    GRAHSP_PL_CUTOFF_A,
    GRAHSP_SI_ABS_LAM_A,
    GRAHSP_SI_ABS_WIDTH_A,
    GRAHSP_SI_EM_LAM_A,
    GRAHSP_SI_EM_WIDTH_A,
    GRAHSP_TORUS_NORM_A,
    C_MS,
    _band_transmitted_fraction,
    _absorbed_line_luminosity,
    _attenuation_transmitted_fraction,
    _attenuation_curve,
    _apply_biattenuation,
    _apply_extended_capture,
    _balmer_continuum_jax,
    _chi2_upper_limit,
    _default_log_agn_amp_prior,
    _feii_component,
    _flux_conserving_line_gaussians,
    _gal_lgmet_to_absolute_z,
    _host_dust_emission,
    _igm_transmission,
    _line_gaussians,
    _local_flux_conserving_line_grid,
    _evaluate_jaxqsofit_backend,
    _powerlaw_jax,
    _project_local_nebular_line_filters,
    _project_jaxqsofit_line_state_filters,
    _project_filters,
    _redshift_to_obs,
    _sample_bounded_normal,
    _torus_component,
    evaluate_sed_model,
    grahsp_photometric_model,
    photometric_log_likelihood,
    photometric_loglike,
    sed_numpyro_model,
    spectroscopic_likelihood_weight,
    spectroscopic_log_likelihood,
)
from jaxsedfit.preload import _build_fixed_igm_jax, _build_igm_cache_jax, _surviving_fraction_for_imf, build_model_context
from jaxsedfit.filters import load_filter_curve
from jaxsedfit.preload import (
    ModelContext,
    PackedFilterCurvesJax,
    PackedFilters,
    PackedFiltersJax,
    SSPData,
    _DALE2014_CACHE,
    _HOST_BASIS_CACHE,
    _build_filter_projection_matrices_for_redshift,
    _build_fixed_filter_projection_matrices,
    _build_host_basis,
    _lnu_lsun_per_hz_to_llambda_w_per_a_np,
    _load_filter_responses,
    _load_templates,
    _mw_band_attenuation_factor,
    _mw_pixel_attenuation_factor,
    _load_vendored_dale2014_templates,
)


def test_likelihood_defaults_include_local_line_photometry():
    cfg = LikelihoodConfig()
    assert not hasattr(cfg, "use_absolute_flux_scale_prior")
    assert not hasattr(cfg, "absolute_flux_scale_prior_sigma_dex")
    assert cfg.use_local_line_photometry is True
    assert cfg.local_nebular_line_uncertainty_dex == pytest.approx(0.3)
    assert cfg.fit_agn_systematics_width is True
    assert cfg.agn_systematics_width_prior_scale == pytest.approx(0.20)
    assert cfg.likelihood_family == "student_t"
    assert cfg.student_t_df == pytest.approx(5.0)


@pytest.mark.parametrize("redshift", [-0.1, np.nan, np.inf])
def test_observation_rejects_invalid_redshift(redshift):
    with pytest.raises(ValueError, match="finite and non-negative"):
        Observation(redshift=redshift).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rest_wave_min": 0.0},
        {"rest_wave_min": 1000.0, "rest_wave_max": 1000.0},
        {"n_wave": 1},
        {"sfh_n_steps": 1},
        {"stellar_metallicity": 0.0},
        {"stellar_metallicity_scatter": -0.1},
    ],
)
def test_galaxy_rejects_invalid_internal_grids(kwargs):
    with pytest.raises(ValueError):
        GalaxyConfig(**kwargs).validate()


def test_inline_templates_require_explicit_wavelength_units():
    cfg = AGNConfig(
        feii_template=FeIITemplate(wave=[1000.0, 2000.0], lumin=[1.0, 1.0]),
        emission_line_template=EmissionLineTemplate(
            wave=[5000.0], lumin_blagn=[1.0], lumin_sy2=[1.0], lumin_liner=[1.0]
        ),
    )
    with pytest.raises(ValueError, match="wavelength_unit is required"):
        cfg.validate()


def test_photometry_method_is_normalized_metadata_only():
    phot = PhotometryData(
        filter_names=["u", "g", "r", "i"],
        fluxes=[1.0, 2.0, 3.0, 4.0],
        errors=[0.1, 0.1, 0.1, 0.1],
        photometry_method=["PSF", " profile ", " catalog ", None],
    )
    phot.validate()

    assert phot.photometry_method == ["psf", "profile", "catalog", None]

    bad = PhotometryData(
        filter_names=["u"],
        fluxes=[1.0],
        errors=[0.1],
        photometry_method=["adaptive-secret-method"],
    )
    with pytest.raises(ValueError, match="Unknown photometry_method"):
        bad.validate()


def test_total_flux_photometry_bypasses_host_capture_scale(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    methods = ["psf", "aperture", "fiber", "profile", "auto", "model", "cmodel", "petrosian", "catalog"]
    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=[f"f{i}" for i in range(len(methods))],
            fluxes=np.ones(len(methods)),
            errors=np.full(len(methods), 0.1),
            psf_fwhm_arcsec=np.full(len(methods), 2.0),
            photometry_method=methods,
        ),
        filters=FilterSet(
            curves=[
                FilterCurve(name=f"f{i}", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])
                for i in range(len(methods))
            ]
        ),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(use_host_capture_model=True),
        inference=InferenceConfig(map_steps=2),
    )

    context = build_model_context(cfg)

    assert context.photometry_total_capture.tolist() == [
        False, False, False, True, True, True, True, True, False
    ]


def _local_projection_context(filter_wave, filter_trans):
    filter_wave = np.asarray(filter_wave, dtype=float)
    filter_trans = np.asarray(filter_trans, dtype=float)
    denom = np.trapezoid(filter_trans, filter_wave)
    eff_wave = np.trapezoid(filter_wave * filter_trans, filter_wave) / denom
    return SimpleNamespace(
        rest_wave_jax=jax.numpy.asarray(np.linspace(1000.0, 10000.0, 512), dtype=jax.numpy.float64),
        packed_filter_curves_jax=PackedFilterCurvesJax(
            wave=jax.numpy.asarray(filter_wave[None, :], dtype=jax.numpy.float64),
            transmission=jax.numpy.asarray(filter_trans[None, :], dtype=jax.numpy.float64),
            denom=jax.numpy.asarray([denom], dtype=jax.numpy.float64),
            valid_mask=jax.numpy.asarray(np.ones((1, filter_wave.size), dtype=bool)),
        ),
        filter_effective_wavelength_jax=jax.numpy.asarray([eff_wave], dtype=jax.numpy.float64),
    )


def test_jaxqsofit_line_state_filter_projection_conserves_flux():
    wave = np.linspace(4800.0, 5200.0, 401)
    transmission = np.ones_like(wave)
    context = SimpleNamespace(
        packed_filter_curves_jax=PackedFilterCurvesJax(
            wave=jnp.asarray(wave[None, :]),
            transmission=jnp.asarray(transmission[None, :]),
            denom=jnp.asarray([np.trapezoid(transmission, wave)]),
            valid_mask=jnp.ones((1, wave.size), dtype=bool),
        ),
        filter_effective_wavelength_jax=jnp.asarray([5000.0]),
    )
    sigma_ln = 0.002
    state = {
        "line_amp_per_component": jnp.asarray([1.0]),
        "line_mu_per_component": jnp.asarray([np.log(5000.0)]),
        "line_sig_per_component": jnp.asarray([sigma_ln]),
        "line_broad_mask_per_component": jnp.asarray([1.0]),
    }

    projected = float(_project_jaxqsofit_line_state_filters(context, state, 0.0)[0])
    dense_wave = np.linspace(4700.0, 5300.0, 200_001)
    fnu = np.exp(-0.5 * ((np.log(dense_wave) - np.log(5000.0)) / sigma_ln) ** 2)
    conversion = 1.0e-10 / 299792458.0 * 1.0e29
    flambda = fnu / (conversion * dense_wave**2)
    inside = (dense_wave >= wave[0]) & (dense_wave <= wave[-1])
    expected = conversion * 5000.0**2 * np.trapezoid(flambda[inside], dense_wave[inside]) / (wave[-1] - wave[0])

    assert projected == pytest.approx(expected, rel=2.0e-3)


def _dense_nebular_line_filter_flux(context, *, line_wave, line_lumin, width_kms, luminosity_distance_m=1.0e20):
    line_wave = float(line_wave)
    line_lumin = float(line_lumin)
    width_kms = float(width_kms)
    fwhm_wave = max(line_wave * width_kms / 299792.458, 1.0e-8)
    sigma = fwhm_wave / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    rest_wave = line_wave + np.linspace(-8.0, 8.0, 20001) * fwhm_wave
    z = (rest_wave - line_wave) / sigma
    rest_lumin = line_lumin * np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    filt_wave = np.asarray(context.packed_filter_curves_jax.wave[0])
    filt_trans = np.asarray(context.packed_filter_curves_jax.transmission[0])
    trans = np.interp(rest_wave, filt_wave, filt_trans, left=0.0, right=0.0)
    distance_scale = 4.0 * np.pi * luminosity_distance_m**2
    numer = np.trapezoid(trans * rest_lumin / distance_scale, rest_wave)
    f_lambda = numer / float(context.packed_filter_curves_jax.denom[0])
    eff_wave = float(context.filter_effective_wavelength_jax[0])
    return 1.0e-10 / 299792458.0 * 1.0e29 * eff_wave * eff_wave * f_lambda


@pytest.mark.parametrize(
    ("case", "filter_wave", "filter_trans", "line_wave", "line_lumin", "width_kms", "rtol"),
    [
        (
            "very narrow filter",
            np.linspace(4998.0, 5002.0, 401),
            np.ones(401),
            5000.0,
            100.0,
            500.0,
            3.0e-2,
        ),
        (
            "line near filter edge",
            np.linspace(5000.0, 5100.0, 501),
            np.ones(501),
            5002.0,
            100.0,
            300.0,
            3.0e-2,
        ),
        (
            "high equivalent width scaling",
            np.linspace(4900.0, 5100.0, 501),
            np.maximum(1.0 - np.abs(np.linspace(4900.0, 5100.0, 501) - 5000.0) / 100.0, 0.0),
            5000.0,
            1.0e6,
            1000.0,
            3.0e-2,
        ),
        (
            "very narrow line width",
            np.linspace(4990.0, 5010.0, 1001),
            np.ones(1001),
            5000.0,
            100.0,
            20.0,
            3.0e-2,
        ),
    ],
)
def test_local_nebular_line_projection_matches_dense_edge_cases(
    case,
    filter_wave,
    filter_trans,
    line_wave,
    line_lumin,
    width_kms,
    rtol,
):
    context = _local_projection_context(filter_wave, filter_trans)
    local = float(
        _project_local_nebular_line_filters(
            context,
            jax.numpy.asarray([line_wave], dtype=jax.numpy.float64),
            jax.numpy.asarray([line_lumin], dtype=jax.numpy.float64),
            width_kms,
            0.0,
            0.0,
            1.0e20,
            jax.numpy.ones_like(context.rest_wave_jax),
        )[0]
    )
    dense = _dense_nebular_line_filter_flux(
        context,
        line_wave=line_wave,
        line_lumin=line_lumin,
        width_kms=width_kms,
    )

    assert local == pytest.approx(dense, rel=rtol), case


def test_jaxqsofit_config_broadening_convolution_default_and_validation():
    assert JaxQSOFitConfig().broadening_convolution == "fft"
    assert JaxQSOFitConfig().photometric_feature_grid_size == 2048
    assert JaxQSOFitConfig(broadening_convolution="direct").broadening_convolution == "direct"
    with pytest.raises(ValueError, match="broadening_convolution"):
        JaxQSOFitConfig(broadening_convolution="scipy")
    with pytest.raises(ValueError, match="photometric_feature_grid_size"):
        JaxQSOFitConfig(photometric_feature_grid_size=128)


def test_photometry_method_is_normalized_metadata_only():
    phot = PhotometryData(
        filter_names=["u", "g", "r", "i"],
        fluxes=[1.0, 2.0, 3.0, 4.0],
        errors=[0.1, 0.1, 0.1, 0.1],
        photometry_method=["PSF", " profile ", " catalog ", None],
    )
    phot.validate()

    assert phot.photometry_method == ["psf", "profile", "catalog", None]

    bad = PhotometryData(
        filter_names=["u"],
        fluxes=[1.0],
        errors=[0.1],
        photometry_method=["adaptive-secret-method"],
    )
    with pytest.raises(ValueError, match="Unknown photometry_method"):
        bad.validate()


def test_public_model_names_delegate_to_legacy_implementations(monkeypatch):
    import jaxsedfit.model as modelmod

    calls = {}

    def _legacy_model(*args, **kwargs):
        calls["model"] = (args, kwargs)
        return "model"

    def _legacy_eval(*args, **kwargs):
        calls["eval"] = (args, kwargs)
        return "state"

    def _phot_like(*args, **kwargs):
        calls["phot"] = (args, kwargs)
        return "phot"

    def _spec_like(*args, **kwargs):
        calls["spec"] = (args, kwargs)
        return "spec"

    monkeypatch.setattr(modelmod, "grahsp_photometric_model", _legacy_model)
    monkeypatch.setattr(modelmod, "evaluate_photometric_state", _legacy_eval)
    monkeypatch.setattr(modelmod, "photometric_loglike", _phot_like)
    monkeypatch.setattr(modelmod, "spectroscopic_loglike", _spec_like)

    assert sed_numpyro_model("ctx", include_components=True) == "model"
    assert evaluate_sed_model("ctx", return_state=False) == "state"
    assert photometric_log_likelihood("pred", obs_fluxes="obs") == "phot"
    assert spectroscopic_log_likelihood("pred", obs_fluxes="obs") == "spec"
    assert calls["model"][0] == ("ctx",)
    assert calls["model"][1]["include_components"] is True
    assert calls["eval"][1]["return_state"] is False
    assert calls["phot"][1]["obs_fluxes"] == "obs"
    assert calls["spec"][1]["obs_fluxes"] == "obs"


def test_prior_config_object_exposes_flat_mapping():
    prior = PriorConfig(
        redshift=RedshiftPriorConfig(z_grid=[0.1, 0.2, 0.3], pdf=[0.2, 0.6, 0.2]),
        stellar_mass=dist.Uniform(8.0, 12.0),
        mass_metallicity=MassMetallicityPriorConfig(enabled=False),
    )
    prior.agn.log_amp = dist.Normal(44.0, 1.0)
    prior.agn.log_broad_line_width_kms = dist.TruncatedNormal(
        np.log(3000.0),
        0.4,
        low=np.log(1000.0),
        high=np.log(15000.0),
    )
    mapping = prior.to_mapping()

    assert "redshift_pdf" in mapping
    assert isinstance(mapping["log_stellar_mass"], dist.Uniform)
    assert mapping["mass_metallicity_relation"]["enabled"] is False
    assert isinstance(mapping["log_agn_amp"], dist.Normal)
    assert float(mapping["log_agn_amp"].loc) == pytest.approx(44.0)
    width_prior = mapping["log_broad_line_width_kms"]
    assert width_prior.__class__.__name__ == "TwoSidedTruncatedDistribution"
    assert float(width_prior.base_dist.loc) == pytest.approx(np.log(3000.0))
    assert float(width_prior.base_dist.scale) == pytest.approx(0.4)
    assert float(width_prior.low) == pytest.approx(np.log(1000.0))
    assert float(width_prior.high) == pytest.approx(np.log(15000.0))


def test_agn_disk_is_normalized_at_5100_angstrom():
    wave = np.asarray([2500.0, 5100.0, 10000.0])
    disk = np.asarray(_powerlaw_jax(wave, 2.0, 0.0, -1.0, 5100.0, 1000.0, 10.0, 0.0))

    assert disk[1] == pytest.approx(2.0)
    assert np.all(np.isfinite(disk))
    assert np.all(disk > 0.0)


def test_default_agn_amplitude_prior_allows_weak_agn():
    context = SimpleNamespace(
        fluxes=np.asarray([1.0, 5.0, 2.0]),
        positive_detected_mask=np.asarray([True, True, True]),
        filters=[
            SimpleNamespace(effective_wavelength=2500.0),
            SimpleNamespace(effective_wavelength=5600.0),
            SimpleNamespace(effective_wavelength=22000.0),
        ],
        luminosity_distance_m=1.0e26,
    )
    prior = _default_log_agn_amp_prior(context, redshift=0.1)

    seeded_loc = np.log(4.0 * np.pi * context.luminosity_distance_m**2 * (C_MS / (5600.0e-10)) * 5.0e-29)
    assert float(prior.loc) == pytest.approx(seeded_loc - 4.0)
    assert float(prior.scale) == pytest.approx(3.0)


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


def _grahsp_activatepl_sbpl_nm(wave_nm, norm, lam1, lam2, x0_nm, xbrk_nm, bend_width_nm):
    q = np.log(wave_nm / xbrk_nm) / bend_width_nm
    qpiv = np.log(x0_nm / xbrk_nm) / bend_width_nm
    return (
        norm
        * (wave_nm / x0_nm) ** ((lam1 + lam2 + 2.0) / 2.0)
        * ((np.exp(q) + np.exp(-q)) / (np.exp(qpiv) + np.exp(-qpiv))) ** ((lam2 - lam1) / 2.0 * bend_width_nm)
        * (x0_nm / wave_nm)
    )


def test_agn_disk_grahsp_default_wavelength_parameters_are_converted_to_angstroms():
    assert GRAHSP_PL_BEND_LOC_A == pytest.approx(100.0 * 10.0)
    assert GRAHSP_PL_BEND_WIDTH == pytest.approx(10.0)
    assert GRAHSP_PL_CUTOFF_A == pytest.approx(10000.0 * 10.0)

    wave_a = np.asarray([700.0, 1000.0, 2500.0, 5100.0, 20000.0, 100000.0])
    wave_nm = wave_a / 10.0
    grahsp_nm = _grahsp_activatepl_sbpl_nm(
        wave_nm,
        norm=2.0,
        lam1=0.0,
        lam2=-1.0,
        x0_nm=510.0,
        xbrk_nm=100.0,
        bend_width_nm=10.0,
    )
    grahsp_nm *= -np.expm1(-(10000.0 / wave_nm))
    jaxsedfit_a = np.asarray(
        _powerlaw_jax(
            wave_a,
            norm=2.0,
            lam1=0.0,
            lam2=-1.0,
            x0=5100.0,
            xbrk=GRAHSP_PL_BEND_LOC_A,
            bend_width=GRAHSP_PL_BEND_WIDTH,
            cutoff=GRAHSP_PL_CUTOFF_A,
        )
    )

    np.testing.assert_allclose(jaxsedfit_a, grahsp_nm, rtol=1.0e-12, atol=0.0)


def test_flux_conserving_lines_preserve_integrated_luminosity():
    wave = np.linspace(4500.0, 5500.0, 20001)
    line = np.asarray(_flux_conserving_line_gaussians(wave, np.asarray([5000.0]), np.asarray([3.0]), 300.0))

    assert np.trapezoid(line, x=wave) == pytest.approx(3.0, rel=2.0e-4)


def test_absorbed_line_luminosity_uses_conserved_integrated_energy():
    line_wave = np.asarray([1000.0, 5000.0])
    line_lumin = np.asarray([2.0, 3.0])
    ebv = 0.2
    curve = np.asarray(_attenuation_curve(line_wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A))
    expected = np.sum(line_lumin * (1.0 - 10 ** (ebv * curve / -2.5)))

    absorbed = _absorbed_line_luminosity(
        line_wave,
        line_lumin,
        ebv,
        -1.2,
        -3.0,
        1.2,
        GRAHSP_BIATTENUATION_BREAK_A,
    )

    assert float(absorbed) == pytest.approx(expected, rel=1.0e-12)


def test_bounded_physical_priors_have_supported_nonzero_edge_gradients():
    specs = {
        "sfh_age": (0.0, 1.0, -4.0, 2.0),
        "sfh_tau": (0.0, 1.0, np.log(0.03), np.log(30.0)),
        "stellar_metallicity": (-1.0, 0.5, -2.0, -0.5),
        "dust_alpha": (2.0, 0.4, 0.5, 4.0),
        "nebular_logu": (-2.0, 0.3, -4.0, -1.0),
        "nebular_z": (0.02, 0.01, 1.0e-4, 0.05),
        "nebular_ne": (100.0, 100.0, 10.0, 1000.0),
        "line_width": (300.0, 100.0, 1.0, 1.0e5),
    }
    prior_config = {
        name: {
            "dist": "TruncatedNormal",
            "loc": loc,
            "scale": scale,
            "low": low - abs(high - low),
            "high": high + abs(high - low),
        }
        for name, (loc, scale, low, high) in specs.items()
    }

    def model():
        for name, (loc, scale, low, high) in specs.items():
            _sample_bounded_normal(prior_config, name, loc, scale, low, high)

    tr = trace(seed(model, 123)).get_trace()
    for name, (_, _, low, high) in specs.items():
        fn = tr[name]["fn"]
        assert float(np.asarray(fn.support.lower_bound)) == pytest.approx(low)
        assert float(np.asarray(fn.support.upper_bound)) == pytest.approx(high)
        span = high - low
        for value in (low + 0.01 * span, high - 0.01 * span):
            gradient = jax.grad(lambda x: fn.log_prob(x))(value)
            assert np.isfinite(float(gradient))
            assert abs(float(gradient)) > 0.0


def test_local_nebular_line_grid_converges_at_filter_edge():
    """Bound sparse-quadrature error where a line crosses a sharp band edge."""
    line_wave = 5000.0
    line_lumin = np.asarray([1.0])
    width_kms = 300.0
    fwhm = line_wave * width_kms / 299792.458
    edge_offsets = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])

    sparse_wave, sparse_lumin = _local_flux_conserving_line_grid(
        np.asarray([line_wave]), line_lumin, width_kms
    )
    dense_wave, dense_lumin = _local_flux_conserving_line_grid(
        np.asarray([line_wave]), line_lumin, width_kms, n_local=4097
    )
    sparse_wave = np.asarray(sparse_wave[0])
    sparse_lumin = np.asarray(sparse_lumin[0])
    dense_wave = np.asarray(dense_wave[0])
    dense_lumin = np.asarray(dense_lumin[0])

    sparse_flux = []
    dense_flux = []
    for offset in edge_offsets:
        edge_center = line_wave + offset * fwhm
        # A deliberately severe edge that ramps from zero to unity over two
        # line FWHM; ordinary broadband edges are generally much smoother.
        sparse_trans = np.clip((sparse_wave - edge_center) / (2.0 * fwhm) + 0.5, 0.0, 1.0)
        dense_trans = np.clip((dense_wave - edge_center) / (2.0 * fwhm) + 0.5, 0.0, 1.0)
        sparse_flux.append(np.trapezoid(sparse_lumin * sparse_trans, x=sparse_wave))
        dense_flux.append(np.trapezoid(dense_lumin * dense_trans, x=dense_wave))

    np.testing.assert_allclose(sparse_flux, dense_flux, rtol=0.0, atol=0.04)


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


def test_gal_lgmet_to_absolute_z_respects_ssp_metallicity_convention():
    assert float(_gal_lgmet_to_absolute_z(np.log10(0.019), metallicity_coordinate="absolute_log10_z")) == pytest.approx(0.019)
    assert float(_gal_lgmet_to_absolute_z(0.0, metallicity_coordinate="log10_z_over_zsun")) == pytest.approx(0.019)
    assert float(_gal_lgmet_to_absolute_z(-0.3, metallicity_coordinate="log10_z_over_zsun")) == pytest.approx(0.019 * 10.0**-0.3)


@pytest.mark.parametrize("field", ["age_grid_gyr", "logzsol_grid", "imf_type", "zcontinuous", "sfh"])
def test_removed_fsps_generation_fields_are_not_galaxy_runtime_options(field):
    with pytest.raises(TypeError):
        GalaxyConfig(**{field: 1})


@pytest.mark.parametrize("ssp_imf", ["chabrier_2003", "salpeter_1955", "kroupa_2001", "van_dokkum_2008"])
def test_galaxy_accepts_supported_ssp_imf_provenance(ssp_imf):
    GalaxyConfig(ssp_imf=ssp_imf).validate()


def test_default_ssp_imf_matches_public_dsps_fsps_artifact():
    assert GalaxyConfig().ssp_imf == "kroupa_2001"


def test_galaxy_rejects_ambiguous_ssp_provenance():
    with pytest.raises(ValueError, match="ssp_imf"):
        GalaxyConfig(ssp_imf="unknown").validate()
    with pytest.raises(ValueError, match="ssp_metallicity_coordinate"):
        GalaxyConfig(ssp_metallicity_coordinate="guess").validate()


def test_surviving_mass_fraction_uses_declared_ssp_imf():
    ages = np.asarray([-1.0, 0.0, 1.0])
    chabrier = _surviving_fraction_for_imf(ages, "chabrier_2003")
    salpeter = _surviving_fraction_for_imf(ages, "salpeter_1955")

    assert np.all(np.isfinite(chabrier))
    assert np.all(np.isfinite(salpeter))
    assert not np.allclose(chabrier, salpeter)


def test_attenuation_transmitted_fraction_uses_only_direct_light():
    direct_intrinsic = np.asarray([10.0, 10.0, 0.0])
    direct_attenuated = np.asarray([2.0, 8.0, 0.0])
    reemitted_dust_or_torus = np.asarray([100.0, 100.0, 100.0])

    frac = np.asarray(_attenuation_transmitted_fraction(direct_attenuated, direct_intrinsic))
    total_emergent_fraction = np.clip(
        (direct_attenuated + reemitted_dust_or_torus)
        / np.maximum(direct_intrinsic + reemitted_dust_or_torus, 1.0e-30),
        1.0e-4,
        1.0,
    )

    np.testing.assert_allclose(frac[:2], [0.2, 0.8])
    assert frac[2] == pytest.approx(1.0e-4)
    assert np.all(total_emergent_fraction[:2] > frac[:2])


def test_band_transmitted_fraction_is_dimensionless_ratio():
    fraction = np.asarray(_band_transmitted_fraction([2.0, 8.0, 0.0], [10.0, 10.0, 0.0]))
    np.testing.assert_allclose(fraction, [0.2, 0.8, 1.0])


def test_extended_capture_leaves_compact_flux_unchanged_and_scales_extended_flux():
    total = np.asarray([10.0, 20.0, 30.0])
    extended = np.asarray([2.0, 5.0, 10.0])
    capture = np.asarray([0.5, 0.2, 1.0])

    captured = np.asarray(_apply_extended_capture(total, extended, capture))

    np.testing.assert_allclose(captured, np.asarray([9.0, 16.0, 30.0]))


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


def test_torus_silicate_absorption_cannot_make_negative_flux():
    wave = np.linspace(GRAHSP_SI_EM_LAM_A, GRAHSP_SI_ABS_LAM_A, 128)
    torus = np.asarray(
        _torus_component(
            wave,
            fcov=0.2,
            si=100.0,
            cool_lam=17.0,
            cool_width=0.45,
            hot_lam=2.0,
            hot_width=0.2,
            hot_fcov=0.1,
            si_ratio=10.0,
            si_em_lam=GRAHSP_SI_EM_LAM_A,
            si_abs_lam=GRAHSP_SI_ABS_LAM_A,
            si_em_width=GRAHSP_SI_EM_WIDTH_A,
            si_abs_width=GRAHSP_SI_ABS_WIDTH_A,
            l_agn=10.0,
        )
    )

    assert np.min(torus) >= 0.0
    assert np.min(torus) == pytest.approx(0.0, abs=1.0e-30)


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


def test_balmer_continuum_planck_factor_uses_grahsp_angstrom_kelvin_constant():
    wave = np.linspace(1000.0, 3646.0, 2647)
    balmer = np.asarray(
        _balmer_continuum_jax(
            wave,
            balmer_norm=1.0,
            balmer_te=15000.0,
            balmer_tau=1.0,
            balmer_vel=3000.0,
        )
    )

    idx_1500 = np.argmin(np.abs(wave - 1500.0))
    idx_2000 = np.argmin(np.abs(wave - 2000.0))
    h_c_per_k_B_angstrom = 1.439e8
    ratio_wave = np.asarray([wave[idx_1500], wave[idx_2000]])
    tau = (ratio_wave / 3646.0) ** 3
    bb = (ratio_wave**-5) / np.expm1(h_c_per_k_B_angstrom / (15000.0 * ratio_wave))
    bb0 = (3646.0**-5) / np.expm1(h_c_per_k_B_angstrom / (15000.0 * 3646.0))
    expected = (1.0 - np.exp(-tau)) * bb / bb0 / (1.0 - np.exp(-1.0))

    assert balmer[idx_1500] / balmer[idx_2000] == pytest.approx(expected[0] / expected[1], rel=1.0e-6)


def test_redshift_projection_matches_cigale_v2025_1_static_reference():
    """Match CIGALE redshifting.py's wavelength and energy-density scaling."""
    rest_wave = np.asarray([1000.0, 2000.0, 5000.0])
    rest_lum = np.asarray([2.0, 5.0, 11.0])
    obs_wave = np.asarray([3000.0, 6000.0, 15000.0])
    expected = np.asarray([9.431404035075278e-4, 2.3578510087688197e-3, 5.187272219291403e-3])

    obs = np.asarray(
        _redshift_to_obs(
            rest_wave,
            rest_lum,
            obs_wave,
            redshift=2.0,
            luminosity_distance_m=7.5,
        )
    )

    np.testing.assert_allclose(obs, expected, rtol=2.0e-12, atol=0.0)


def test_upper_limit_likelihood_uses_standard_deviation_not_variance():
    limit = np.asarray([1.0])
    model = np.asarray([1.2])
    sigma = np.asarray([0.1])

    chi2 = np.asarray(_chi2_upper_limit(limit, model, sigma**2))

    flux_scale = 1.0e3
    chi2_scaled = np.asarray(_chi2_upper_limit(limit * flux_scale, model * flux_scale, (sigma * flux_scale) ** 2))

    assert chi2_scaled == pytest.approx(chi2, rel=1.0e-12)


def test_lyman_break_uncertainty_threshold_uses_angstroms():
    kwargs = dict(
        pred_fluxes=np.asarray([100.0]),
        obs_fluxes=np.asarray([1.0]),
        obs_errors=np.asarray([0.1]),
        upper_limits=np.asarray([False]),
        data_mask=np.asarray([True]),
        systematics_width=0.0,
        likelihood_family="gaussian",
        student_t_df=5.0,
        agn_component=np.asarray([0.0]),
        agn_bol_lum_w=1.0e38,
        agn_nev=0.1,
        variability_uncertainty=False,
        attenuation_model_uncertainty=False,
        transmitted_fraction=np.asarray([1.0]),
        filter_wavelength=np.asarray([1400.0]),
        redshift=0.0,
    )

    logl_without_uncertainty = float(photometric_loglike(**kwargs, lyman_break_uncertainty=False))
    logl_with_uncertainty = float(photometric_loglike(**kwargs, lyman_break_uncertainty=True))

    assert logl_without_uncertainty < -1.0e5
    assert logl_with_uncertainty > -100.0


def test_local_nebular_line_uncertainty_regularizes_only_line_component():
    kwargs = dict(
        pred_fluxes=np.asarray([10.0]),
        obs_fluxes=np.asarray([5.0]),
        obs_errors=np.asarray([0.1]),
        upper_limits=np.asarray([False]),
        data_mask=np.asarray([True]),
        systematics_width=0.0,
        likelihood_family="gaussian",
        student_t_df=5.0,
        agn_component=np.asarray([0.0]),
        agn_bol_lum_w=1.0e38,
        agn_nev=0.1,
        variability_uncertainty=False,
        attenuation_model_uncertainty=False,
        transmitted_fraction=np.asarray([1.0]),
        lyman_break_uncertainty=False,
        filter_wavelength=np.asarray([5000.0]),
        redshift=0.0,
        nebular_line_component=np.asarray([5.0]),
    )

    exact = float(photometric_loglike(**kwargs, local_nebular_line_uncertainty_dex=0.0))
    regularized = float(photometric_loglike(**kwargs, local_nebular_line_uncertainty_dex=0.3))

    assert exact < -1.0e3
    assert regularized > exact

    no_line_component = float(
        photometric_loglike(
            **{**kwargs, "nebular_line_component": np.asarray([0.0])},
            local_nebular_line_uncertainty_dex=0.3,
        )
    )
    assert no_line_component == pytest.approx(exact)


def test_photometric_systematic_variance_matches_grahsp_error_model():
    kwargs = _minimal_photometric_loglike_kwargs()
    kwargs.update(
        obs_fluxes=np.asarray([8.0]),
        obs_errors=np.asarray([0.5]),
        systematics_width=0.1,
        agn_systematics_width=0.02,
        agn_component=np.asarray([5.0]),
    )
    pred_fluxes = np.asarray([9.0])

    actual = float(photometric_loglike(pred_fluxes=pred_fluxes, **kwargs))
    sigma = np.sqrt(0.5**2 + (0.1 * 8.0) ** 2 + (0.02 * 5.0) ** 2)
    expected = -0.5 * (((8.0 - 9.0) / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))

    assert actual == pytest.approx(expected)


def _minimal_photometric_loglike_kwargs():
    return dict(
        obs_fluxes=np.asarray([1.0]),
        obs_errors=np.asarray([0.1]),
        upper_limits=np.asarray([False]),
        data_mask=np.asarray([True]),
        systematics_width=0.0,
        likelihood_family="gaussian",
        student_t_df=5.0,
        agn_component=np.asarray([0.0]),
        agn_bol_lum_w=1.0e38,
        agn_nev=0.1,
        variability_uncertainty=False,
        attenuation_model_uncertainty=False,
        transmitted_fraction=np.asarray([1.0]),
        lyman_break_uncertainty=False,
        filter_wavelength=np.asarray([5000.0]),
        redshift=0.0,
    )


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, -np.inf])
def test_photometric_likelihood_penalizes_nonfinite_active_predictions(invalid_prediction):
    logl = float(
        photometric_loglike(
            pred_fluxes=np.asarray([invalid_prediction]),
            **_minimal_photometric_loglike_kwargs(),
        )
    )

    assert np.isfinite(logl)
    assert logl < -9.0e5


def test_photometric_likelihood_ignores_nonfinite_masked_predictions():
    kwargs = _minimal_photometric_loglike_kwargs()
    kwargs["data_mask"] = np.asarray([False])
    logl = float(photometric_loglike(pred_fluxes=np.asarray([np.nan]), **kwargs))

    assert logl == pytest.approx(0.0)


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, -np.inf])
def test_spectroscopic_likelihood_penalizes_nonfinite_active_predictions(invalid_prediction):
    logl = float(
        spectroscopic_log_likelihood(
            np.asarray([invalid_prediction]),
            np.asarray([1.0]),
            np.asarray([0.1]),
            np.asarray([True]),
            0.0,
            5.0,
        )
    )

    assert np.isfinite(logl)
    assert logl < -9.0e5


def test_spectroscopic_likelihood_ignores_nonfinite_masked_predictions():
    logl = float(
        spectroscopic_log_likelihood(
            np.asarray([np.nan]),
            np.asarray([1.0]),
            np.asarray([0.1]),
            np.asarray([False]),
            0.0,
            5.0,
        )
    )

    assert logl == pytest.approx(0.0)


def test_spectroscopic_likelihood_weight_uses_resolution_elements():
    wave = np.arange(5000.0, 5010.0, 1.0)
    mask = np.ones_like(wave, dtype=bool)
    spectrum_index = np.zeros_like(wave, dtype=int)

    weight = float(
        spectroscopic_likelihood_weight(
            wave,
            mask,
            spectrum_index,
            likelihood_weight_mode="resolution_elements",
            resolving_power=2000.0,
        )
    )

    assert weight == pytest.approx(0.36, rel=1.0e-2)
    assert float(spectroscopic_likelihood_weight(wave, mask, spectrum_index, "pixels", 2000.0)) == 1.0


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


def test_filter_projection_padded_rows_do_not_add_spurious_trapezoid_segment():
    work_wave = np.asarray([[4000.0, 5000.0, 6000.0, 6000.0]], dtype=float)
    transmission = np.asarray([[1.0, 1.0, 0.1, 0.0]], dtype=float)
    valid_mask = np.asarray([[True, True, True, False]], dtype=bool)
    interp_indices = np.asarray([[0, 1, 2, 2]], dtype=np.int32)
    interp_weight = np.zeros_like(work_wave)
    effective_wavelength = np.asarray([5000.0], dtype=float)
    obs_flux = 1.0e-20 * (np.asarray([4000.0, 5000.0, 6000.0, 7000.0]) / 5000.0) ** 2
    packed_jax = PackedFiltersJax(
        interp_indices=interp_indices,
        interp_weight=interp_weight,
        transmission=transmission,
        work_wave=work_wave,
        effective_wavelength=effective_wavelength,
        valid_mask=valid_mask,
    )
    packed_np = PackedFilters(
        interp_indices=interp_indices,
        interp_weight=interp_weight,
        transmission=transmission,
        work_wave=work_wave,
        effective_wavelength=effective_wavelength,
        valid_mask=valid_mask,
    )

    real_wave = work_wave[0, valid_mask[0]]
    real_trans = transmission[0, valid_mask[0]]
    real_flux = obs_flux[: real_wave.size]
    f_lambda = np.trapezoid(real_flux * real_trans, real_wave) / np.trapezoid(real_trans, real_wave)
    expected = 1.0e-10 / 299792458.0 * 1.0e29 * effective_wavelength[0] ** 2 * f_lambda

    projected = np.asarray(_project_filters(obs_flux, packed_jax))
    _, fixed_scalar_matrix = _build_fixed_filter_projection_matrices(
        rest_wave=np.asarray([4000.0, 5000.0, 6000.0, 7000.0]),
        packed_filters=packed_np,
        fixed_igm=np.ones(4),
        luminosity_distance_m=1.0,
        redshift=0.0,
    )
    _, dynamic_scalar_matrix = _build_filter_projection_matrices_for_redshift(
        rest_wave=np.asarray([4000.0, 5000.0, 6000.0, 7000.0]),
        packed_filters=packed_np,
        igm=np.ones(4),
        luminosity_distance_m=1.0,
        redshift=0.0,
    )

    assert projected[0] == pytest.approx(expected)
    assert (fixed_scalar_matrix @ obs_flux)[0] == pytest.approx(expected)
    assert (dynamic_scalar_matrix @ obs_flux)[0] == pytest.approx(expected)


def test_ukidss_dr11plus_vendored_filters_load_in_angstroms():
    expected_ranges = {
        "ukirt.wfcam.Y": (9000.0, 12000.0),
        "ukirt.wfcam.J": (11000.0, 14000.0),
        "ukirt.wfcam.H": (14000.0, 19000.0),
        "ukirt.wfcam.K": (19000.0, 25000.0),
    }
    for name, (lo, hi) in expected_ranges.items():
        curve = load_filter_curve(name)
        wave = np.asarray(curve.wave, dtype=float)
        trans = np.asarray(curve.transmission, dtype=float)
        assert wave.ndim == 1
        assert wave.size == trans.size
        assert lo < wave[np.argmax(trans)] < hi
        assert np.nanmax(trans) > 0.0


def test_legacy_filter_aliases_resolve_to_vendored_curves():
    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=["u_sdss", "J_2mass", "W1"],
            fluxes=[1.0, 1.0, 1.0],
            errors=[0.1, 0.1, 0.1],
        ),
        filters=FilterSet(),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5"),
        inference=InferenceConfig(map_steps=2),
    )

    curves = _load_filter_responses(cfg)

    assert [curve.name for curve in curves] == ["u_sdss", "J_2mass", "W1"]
    for curve in curves:
        wave = np.asarray(curve.wave, dtype=float)
        trans = np.asarray(curve.transmission, dtype=float)
        assert wave.ndim == 1
        assert wave.size == trans.size
        assert wave.size > 3
        assert np.nanmax(trans) > 0.0
        assert np.isfinite(curve.effective_wavelength)


def test_build_context_with_inline_templates(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1],
                lumin_blagn=[1.0, 0.5],
                lumin_sy2=[0.2, 0.1],
                lumin_liner=[0.1, 0.05],
                wavelength_unit="nm",
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
    assert context.filters[0].name == "f1"
    assert context.templates.feii_wave.shape[0] == 2
    np.testing.assert_allclose(context.templates.line_wave, [1216.0, 4861.0])
    assert context.templates.dust_alpha_grid.size > 0
    assert context.templates.dust_wave.size > 0
    assert context.templates.dust_lumin.ndim == 2
    assert context.spec_wave_obs.tolist() == [3500.0, 4500.0, 5500.0]
    assert context.spec_mask.tolist() == [True, False, True]
    assert context.spec_spectrum_index.tolist() == [0, 0, 0]
    assert context.spec_instruments == ("test",)


def test_mw_dereddening_applies_to_photometry_and_spectra(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._get_sfd_query", lambda: (lambda coord: 0.1))

    filt_wave = np.asarray([1000.0, 2000.0, 3000.0])
    filt_trans = np.asarray([0.0, 1.0, 0.0])
    spec_wave = np.asarray([3500.0, 4500.0, 5500.0])
    spec_flux = np.asarray([0.1, 0.2, 0.15])
    spec_err = np.asarray([0.01, 0.02, 0.015])
    cfg = FitConfig(
        observation=Observation(
            object_id="obj",
            redshift=0.1,
            ra=180.0,
            dec=0.0,
            apply_mw_deredden=True,
        ),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=filt_wave, transmission=filt_trans)]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(),
        spectroscopy=SpectroscopyData(
            wave_obs=spec_wave,
            fluxes=spec_flux,
            errors=spec_err,
            instrument="test",
        ),
        spectroscopy_config=SpectroscopyConfig(enabled=True),
        inference=InferenceConfig(map_steps=2),
    )

    context = build_model_context(cfg)

    loaded_filter = context.filters[0]
    band_factor = _mw_band_attenuation_factor(loaded_filter.work_wave, loaded_filter.transmission, 0.1)
    spec_factors = _mw_pixel_attenuation_factor(spec_wave, 0.1)
    assert context.mw_ebv == pytest.approx(0.1)
    np.testing.assert_allclose(context.fluxes, np.asarray([1.0]) / band_factor)
    np.testing.assert_allclose(context.errors, np.asarray([0.1]) / band_factor)
    np.testing.assert_allclose(context.spec_fluxes, spec_flux / spec_factors)
    np.testing.assert_allclose(context.spec_errors, spec_err / spec_factors)


def test_context_accepts_multiple_spectra(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
            aperture_diameter_arcsec=[2.0],
        ),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
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
    np.testing.assert_allclose(
        context.spec_effective_spatial_scale_arcsec,
        [1.5, np.sqrt(2.0) * 1.5 / 2.354820045],
    )
    assert context.spec_aperture_diameter_arcsec.tolist()[0] == 3.0
    assert np.isnan(context.spec_aperture_diameter_arcsec.tolist()[1])
    assert context.spec_instruments == ("sdss", "desi")


def test_spectroscopy_config_uses_nested_jaxqsofit_section():
    cfg = fit_config_from_mapping(
        {
            "observation": {"object_id": "obj", "redshift": 0.1},
            "photometry": {"filter_names": ["f1"], "fluxes": [1.0], "errors": [0.1]},
            "spectroscopy_config": {
                "enabled": True,
                "backend": "jaxqsofit",
                "jaxqsofit": {
                    "use_spectral_lines": False,
                    "use_spectral_feii": True,
                    "use_spectral_balmer_continuum": True,
                    "line_flux_scale_mjy": 0.1,
                },
            },
        }
    ).spectroscopy_config

    assert cfg.jaxqsofit.use_spectral_lines is False
    assert cfg.jaxqsofit.use_spectral_feii is True
    assert cfg.jaxqsofit.use_spectral_balmer_continuum is True
    assert cfg.jaxqsofit.line_flux_scale_mjy == 0.1


def test_jaxqsofit_fixed_narrow_width_component_reports_tied_width():
    pytest.importorskip("jaxqsofit.components")
    from jaxqsofit.components import SpectralComponentConfig, evaluate_joint_spectral_components

    cfg = SpectralComponentConfig(
        use_lines=True,
        use_tied_lines=False,
        line_centers_rest=[5000.0],
        line_names=["OIII5007c"],
        fixed_narrow_fwhm_kms=321.0,
        fixed_narrow_amp_scale=2.5,
    )
    tr = trace(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(1))).get_trace(
        np.asarray([4995.0, 5000.0, 5005.0]),
        0.0,
        np.ones(3),
        config=cfg,
    )

    assert tr["jqf_line_narrow_fwhm_kms"]["value"] == pytest.approx(321.0)
    assert tr["jqf_line_narrow_amp_scale"]["value"] == pytest.approx(2.5)


def test_jaxqsofit_backend_does_not_fix_narrow_width_without_explicit_override(monkeypatch):
    jqf_components = pytest.importorskip("jaxqsofit.components")
    captured = {}

    def fake_evaluate_joint_spectral_components(wave_obs, redshift, continuum_mjy, *, config, **kwargs):
        captured["config"] = config
        zeros = np.zeros_like(np.asarray(wave_obs, dtype=float))
        return {
            "total": np.asarray(continuum_mjy, dtype=float),
            "continuum": np.asarray(continuum_mjy, dtype=float),
            "lines": zeros,
            "line_broad": zeros,
            "line_narrow": zeros,
            "feii": zeros,
            "balmer": zeros,
        }

    monkeypatch.setattr(jqf_components, "evaluate_joint_spectral_components", fake_evaluate_joint_spectral_components)
    cfg = FitConfig(
        observation=Observation(redshift=0.0),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            jaxqsofit=JaxQSOFitConfig(
                use_spectral_lines=True,
                use_tied_lines=False,
                line_table=None,
                use_spectral_feii=False,
                use_spectral_balmer_continuum=False,
            ),
        ),
    )
    wave = np.asarray([4995.0, 5000.0, 5005.0], dtype=float)
    out = _evaluate_jaxqsofit_backend(
        wave,
        0.0,
        np.ones_like(wave),
        cfg,
        {"line": {"table": []}},
        wave,
        np.zeros_like(wave),
    )

    assert np.allclose(out["total"], 1.0)
    assert captured["config"].fixed_narrow_fwhm_kms is None
    assert captured["config"].fixed_narrow_amp_scale is None
    assert captured["config"].narrow_fwhm_kms_default == pytest.approx(500.0)


def test_fit_config_mapping_coerces_agn_template_config():
    cfg = fit_config_from_mapping(
        {
            "observation": {"object_id": "obj", "redshift": 0.1},
            "photometry": {"filter_names": ["f1"], "fluxes": [1.0], "errors": [0.1]},
            "agn": {
                "fit_feii_broadening": True,
                "fit_balmer_continuum": True,
                "feii_template": {"name": "custom"},
            },
        }
    )

    assert cfg.agn.fit_feii_broadening is True
    assert cfg.agn.fit_balmer_continuum is True
    assert cfg.agn.feii_template.name == "custom"


def test_fit_config_mapping_rejects_legacy_agn_defaults():
    with pytest.raises(TypeError, match="broad_line_width_kms_default"):
        fit_config_from_mapping(
            {
                "observation": {"object_id": "obj", "redshift": 0.1},
                "photometry": {"filter_names": ["f1"], "fluxes": [1.0], "errors": [0.1]},
                "agn": {"broad_line_width_kms_default": 4000.0},
            }
        )


def test_fit_config_mapping_rejects_flat_prior_config():
    with pytest.raises(TypeError, match="log_stellar_mass"):
        fit_config_from_mapping(
            {
                "observation": {"object_id": "obj", "redshift": 0.1},
                "photometry": {"filter_names": ["f1"], "fluxes": [1.0], "errors": [0.1]},
                "prior_config": {"log_stellar_mass": {"loc": 10.0, "scale": 1.0}},
            }
        )


def test_jaxqsofit_joint_backend_builds_flux_scaled_smart_priors(monkeypatch):
    jqf_config = pytest.importorskip("jaxqsofit.config")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
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
    if hasattr(prior, "to_mapping"):
        prior = prior.to_mapping()
    expected = jqf_config.PriorConfig.from_spectrum(
        flux=np.asarray([2.0, 4.0]),
        redshift=0.1,
    ).to_mapping()
    assert float(prior["log_cont_norm"].loc) == pytest.approx(float(expected["log_cont_norm"].loc))
    line_table = prior["line"]["table"]
    assert line_table
    assert min(float(row["minsca"]) for row in line_table) >= 3.0e-4


def test_context_builds_spectrum_resolution_host_basis(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=16, fit_host=True),
        agn=AGNConfig(),
        spectroscopy=SpectroscopyData(
            wave_obs=[5000.0, 5100.0, 5200.0],
            fluxes=[2.0, 4.0, 5.0],
            errors=[0.1, 0.1, 0.1],
            instrument="sdss",
        ),
        spectroscopy_config=SpectroscopyConfig(
            enabled=True,
            backend="jaxqsofit",
            jaxqsofit=JaxQSOFitConfig(use_spectral_smart_priors=False),
        ),
        inference=InferenceConfig(map_steps=2),
    )

    context = build_model_context(cfg)

    assert context.spec_host_basis_jax is not None
    assert np.asarray(context.spec_rest_wave_jax).tolist() == pytest.approx(np.asarray(cfg.spectroscopy.wave_obs) / 1.1)
    assert context.spec_host_basis_jax.rest_llambda.shape[-1] == len(cfg.spectroscopy.wave_obs)


def test_context_skips_spectrum_resolution_host_basis_when_backend_does_not_need_it(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=16, fit_host=True),
        agn=AGNConfig(),
        spectroscopy=SpectroscopyData(
            wave_obs=[5000.0, 5100.0, 5200.0],
            fluxes=[2.0, 4.0, 5.0],
            errors=[0.1, 0.1, 0.1],
            instrument="sdss",
        ),
        spectroscopy_config=SpectroscopyConfig(enabled=True, backend="jaxsedfit"),
        inference=InferenceConfig(map_steps=2),
    )

    context = build_model_context(cfg)

    assert context.spec_host_basis_jax is None
    assert np.asarray(context.spec_rest_wave_jax).size == 0


def test_jaxqsofit_spectrum_resolution_host_basis_uses_host_kinematics(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    broadened_grids = []

    def _identity_broaden(lnwave, spectrum, v_kms, sigma_kms):
        broadened_grids.append(np.asarray(lnwave).size)
        return spectrum

    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._HOST_BASIS_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.model._shift_and_broaden_single_spectrum_lnlam", _identity_broaden)

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(
            dsps_ssp_fn="fake.h5",
            n_wave=16,
            fit_host=True,
            fit_host_kinematics=True,
        ),
        agn=AGNConfig(fit_agn=False),
        nebular=NebularConfig(enabled=False),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=[5000.0, 5100.0, 5200.0],
            fluxes=[2.0, 4.0, 5.0],
            errors=[0.1, 0.1, 0.1],
            instrument="sdss",
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
        prior_config=PriorConfig(stellar_mass=dist.Normal(8.0, 1.0e-6)),
    )
    context = build_model_context(cfg)

    tr = trace(
        substitute(
            seed(grahsp_photometric_model, jax.random.PRNGKey(3)),
            data={
                "gal_v_kms": np.array(0.0),
                "gal_sigma_kms": np.array(120.0),
                "log_ebv_gal": np.array(np.log(1.0e-12)),
                "dust_alpha": np.array(2.0),
            },
        )
    ).get_trace(context, include_components=False)

    assert context.spec_host_basis_jax is not None
    assert sorted(broadened_grids) == [len(cfg.spectroscopy.wave_obs), cfg.galaxy.n_wave]
    assert np.asarray(tr["pred_spectrum_fluxes"]["value"]).shape == (3,)


def test_jaxqsofit_backend_keeps_nebular_width_fixed_without_nebular_prior(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.0),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=True),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[0.0, 0.0], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[5000.0],
                lumin_blagn=[0.0],
                lumin_sy2=[0.0],
                lumin_liner=[0.0],
                wavelength_unit="angstrom",
            ),
        ),
        nebular=NebularConfig(enabled=True, zgas=0.02),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=[4995.0, 5000.0, 5005.0],
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
                use_spectral_feii=False,
                use_spectral_balmer_continuum=False,
                use_spectral_smart_priors=False,
            ),
        ),
        inference=InferenceConfig(map_steps=2),
    )
    context = build_model_context(cfg)

    assert context.fixed_nebular_line_profile_jax is not None
    assert context.nebular_rest_templates_jax.line_profile_per_photon is not None


def test_jaxsedfit_model_can_call_jaxqsofit_backend(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
            aperture_diameter_arcsec=[2.0],
        ),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[0.0, 0.0], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[486.1],
                lumin_blagn=[0.0],
                lumin_sy2=[0.0],
                lumin_liner=[0.0],
                wavelength_unit="angstrom",
            ),
        ),
        likelihood=LikelihoodConfig(
            use_host_capture_model=True,
            variability_uncertainty=False,
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
        "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
        "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
        "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
        "fcov": np.array(0.1),
        "si": np.array(0.0),
        "cool_lam": np.array(17.0),
        "cool_width": np.array(0.45),
        "hot_lam": np.array(2.0),
        "hot_width": np.array(0.5),
        "log_hot_fcov": np.array(np.log(0.1)),
        "broad_lines_strength": np.array(1.0),
        "narrow_lines_strength": np.array(1.0),
        "log_broad_line_width_kms": np.array(np.log(3000.0)),
        "log_narrow_line_width_kms": np.array(np.log(500.0)),
        "feii_norm": np.array(1.0),
        "feii_fwhm": np.array(3000.0),
        "feii_shift": np.array(0.0),
        "log_ebv_agn": np.array(np.log(1.0e-12)),
    }
    tr = trace(substitute(seed(grahsp_photometric_model, jax.random.PRNGKey(1)), data=params)).get_trace(
        context,
        include_components=False,
    )

    assert "jqf_total_model" in tr
    assert "jqf_line_model" in tr
    assert np.asarray(tr["pred_spectrum_fluxes"]["value"]).shape == (3,)


def test_jaxsedfit_jaxqsofit_backend_uses_nested_tied_line_config(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.0),
        photometry=PhotometryData(
            filter_names=["f1"],
            fluxes=[1.0],
            errors=[0.1],
        ),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
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
        "pl_bend_loc": np.array(GRAHSP_PL_BEND_LOC_A),
        "pl_bend_width": np.array(GRAHSP_PL_BEND_WIDTH),
        "pl_cutoff": np.array(GRAHSP_PL_CUTOFF_A),
        "fcov": np.array(0.1),
        "si": np.array(0.0),
        "cool_lam": np.array(17.0),
        "cool_width": np.array(0.45),
        "hot_lam": np.array(2.0),
        "hot_width": np.array(0.5),
        "log_hot_fcov": np.array(np.log(0.1)),
        "broad_lines_strength": np.array(1.0),
        "narrow_lines_strength": np.array(1.0),
        "log_broad_line_width_kms": np.array(np.log(3000.0)),
        "log_narrow_line_width_kms": np.array(np.log(500.0)),
        "feii_norm": np.array(1.0),
        "feii_fwhm": np.array(3000.0),
        "feii_shift": np.array(0.0),
        "log_ebv_agn": np.array(np.log(1.0e-12)),
    }
    tr = trace(substitute(seed(grahsp_photometric_model, jax.random.PRNGKey(2)), data=params)).get_trace(
        context,
        include_components=False,
    )

    assert "jqf_line_dmu_group" in tr
    assert "jqf_line_sig_group" in tr
    assert "jqf_line_amp_group" in tr
    assert "jqf_line_model_broad" in tr
    assert "jqf_broad_to_sed_broad_line_prior" not in tr
    assert "jqf_narrow_to_sed_narrow_line_prior" not in tr
    assert np.asarray(tr["jqf_line_photometry"]["value"]).shape == (1,)
    assert np.asarray(tr["jqf_extrapolated_broad_photometry"]["value"]).shape == (1,)
    assert np.asarray(tr["jqf_extrapolated_narrow_photometry"]["value"]).shape == (1,)
    assert np.asarray(tr["pred_spectrum_fluxes"]["value"]).shape == (3,)
    assert context.jaxqsofit_prior_config["standardize_active_priors"] is True
    standardized_amplitudes = [
        name
        for name, site in tr.items()
        if name.startswith("jqf_line_amp_")
        and name.endswith("_std")
        and site.get("type") == "sample"
    ]
    assert standardized_amplitudes
    latent_values = {
        name: np.asarray(site["value"])
        for name, site in tr.items()
        if site.get("type") == "sample" and not site.get("is_observed", False)
    }
    blocks = _joint_dense_mass_blocks(latent_values, context=context)
    line_blocks = [block for block in blocks if any(name.startswith("jqf_line_") for name in block)]
    assert (
        "jqf_line_amp_Hb_std",
        "jqf_line_broad_center_0_std",
        "jqf_line_broad_relative_offsets_0_std",
        "jqf_line_ordered_width_logits_Hb_std",
    ) in line_blocks
    assert all(set(block) <= set(latent_values) for block in line_blocks)


def test_jaxsedfit_jaxqsofit_tied_line_backend_runs_svi_jit(monkeypatch):
    pytest.importorskip("jaxqsofit.components")

    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())

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
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64, fit_host=False),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
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
                    use_spectral_feii=True,
                    line_table=line_table,
                line_flux_scale_mjy=0.1,
            ),
        ),
        inference=InferenceConfig(map_steps=1, learning_rate=1.0e-3),
    )

    fitter = JAXSEDFit(cfg)
    result = fitter.fit_map(steps=1, progress_bar=False)

    assert result.samples is fitter.samples
    assert result.method == "map"
    assert "log_agn_amp" in result.median
    assert np.asarray(fitter.map_result["losses"]).shape == (1,)
    assert np.isfinite(float(np.asarray(fitter.map_result["losses"])[0]))


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
        captured["pred_bands"] = self.pred_bands
        captured["kwargs"] = kwargs
        self.fig = "fig"

    monkeypatch.setattr(jaxqsofit.JAXQSOFit, "plot_fig", _fake_plot_fig)

    fitter = JAXSEDFit.__new__(JAXSEDFit)
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
        "spec_host_model_fluxes": np.asarray([[0.2, 0.3, 0.4], [0.25, 0.35, 0.45]]),
        "spec_disk_model_fluxes": np.asarray([[0.5, 0.6, 0.7], [0.55, 0.65, 0.75]]),
        "spec_torus_model_fluxes": np.asarray([[0.05, 0.06, 0.07], [0.055, 0.065, 0.075]]),
        "jqf_continuum_model": np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        "jqf_line_model": np.asarray([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]),
        "jqf_line_model_aperture": np.asarray([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]),
        "spectrum_scale_fit": np.asarray([2.0, 2.0]),
        "spectrum_host_capture_fraction": np.asarray([0.5, 0.5]),
        "host_obs_sed": np.asarray([[1.0e-20, 2.0e-20, 3.0e-20], [1.0e-20, 2.0e-20, 3.0e-20]]),
        "disk_obs_sed": np.asarray([[2.0e-20, 2.0e-20, 2.0e-20], [2.0e-20, 2.0e-20, 2.0e-20]]),
        "torus_obs_sed": np.asarray([[0.5e-20, 0.5e-20, 0.5e-20], [0.5e-20, 0.5e-20, 0.5e-20]]),
        "dust_obs_sed": np.asarray([[0.1e-20, 0.1e-20, 0.1e-20], [0.1e-20, 0.1e-20, 0.1e-20]]),
        "line_obs_sed": np.zeros((2, 3)),
        "feii_obs_sed": np.zeros((2, 3)),
        "nebular_lines_obs_sed": np.asarray([[0.2e-20, 0.3e-20, 0.2e-20], [0.2e-20, 0.3e-20, 0.2e-20]]),
    }

    fig = fitter.plot_jaxqsofit_spectrum(show_plot=False, plot_residual=False)

    assert fig == "fig"
    assert captured["wave"].tolist() == [2000.0, 2500.0, 3000.0]
    assert np.all(captured["model_total"] > captured["flux"] * 0.0)
    assert np.all(captured["host"] > 0.0)
    np.testing.assert_allclose(
        captured["line"],
        JAXSEDFit._mjy_to_rest_flambda_1e17(
            np.asarray([4000.0, 5000.0, 6000.0]),
            2.0 * np.asarray([2.0, 2.0, 2.0]),
            1.0,
        ),
    )
    assert "jaxsedfit_torus" in captured["custom_components"]
    assert "jaxsedfit_host_dust" in captured["custom_components"]
    assert "jaxsedfit_sed_lines" not in captured["custom_components"]
    assert "jaxsedfit_nebular_lines" not in captured["custom_components"]
    assert "total_model" in captured["pred_bands"]
    assert "host" in captured["pred_bands"]
    assert "PL" in captured["pred_bands"]
    assert "jaxsedfit_torus" in captured["pred_bands"]
    lo, hi = captured["pred_bands"]["total_model"]
    assert lo.shape == captured["wave"].shape
    assert hi.shape == captured["wave"].shape
    assert np.all(hi >= lo)
    assert captured["kwargs"]["show_plot"] is False
    assert captured["kwargs"]["plot_residual"] is False

    fig = fitter.plot_jaxqsofit_spectrum(show_plot=False, plot_residual=False, show_nebular_lines=True)
    assert fig == "fig"
    assert "jaxsedfit_nebular_lines" in captured["custom_components"]
    assert "jaxsedfit_sed_lines" not in captured["custom_components"]
    assert np.nanmax(captured["custom_components"]["jaxsedfit_nebular_lines"]) > 0.0
    assert "jaxsedfit_nebular_lines" in captured["pred_bands"]


def test_config_rejects_invalid_redshift_pdf():
    cfg = FitConfig(
        observation=Observation(object_id="obj", redshift=0.1, redshift_mode="fit"),
        photometry=PhotometryData(filter_names=["f1"], fluxes=[1.0], errors=[0.1]),
        filters=FilterSet(curves=[FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0])]),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=64),
        agn=AGNConfig(),
        likelihood=LikelihoodConfig(),
        inference=InferenceConfig(map_steps=2),
        prior_config=PriorConfig(
            redshift=RedshiftPriorConfig(
                z_grid=[0.3, 0.2, 0.4],
                pdf=[0.2, 0.5, 0.3],
            )
        ),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        cfg.validate()


def test_igm_transmission_on_rest_grid_is_near_unity_redward_of_lyman_alpha():
    rest_wave = np.array([1500.0, 1216.0, 1000.0, 912.0, 800.0], dtype=float)
    cache = _build_igm_cache_jax(rest_wave)
    transmission = np.asarray(_build_fixed_igm_jax(cache, 1.0), dtype=float)

    assert transmission[0] > 0.99
    assert transmission[1] > 0.95
    assert transmission[2] < 1.0
    assert transmission[3] < transmission[2]
    assert transmission[4] < transmission[3]


def test_dynamic_and_fixed_igm_evaluators_match():
    rest_wave = np.array([1500.0, 1216.0, 1000.0, 912.0, 800.0], dtype=float)
    cache = _build_igm_cache_jax(rest_wave)

    fixed = np.asarray(_build_fixed_igm_jax(cache, 2.0), dtype=float)
    dynamic = np.asarray(_igm_transmission(cache, 2.0), dtype=float)

    assert np.allclose(dynamic, fixed)


def test_cigale_igm_has_finite_redshift_gradient_away_from_breaks():
    rest_wave = np.array([400.0, 700.0, 850.0, 1000.0, 1100.0], dtype=float)
    cache = _build_igm_cache_jax(rest_wave)

    gradient = jax.grad(lambda z: jnp.sum(_igm_transmission(cache, z)))(2.3)

    assert np.isfinite(float(gradient))
    assert float(gradient) != 0.0


@pytest.mark.parametrize(
    ("redshift", "expected"),
    [
        (0.0, [1.0] * 14),
        (
            1.0,
            [
                0.9894695081807627, 0.8047814563025332, 0.469797051674372,
                0.5006698438923702, 0.6263263010411965, 0.9170078562913831,
                0.9712357914069667, 0.9779976377529525, 0.9794058532748684,
                0.978099815015404, 0.9812548237094142, 0.9730320130680872,
                1.0, 1.0,
            ],
        ),
        (
            3.0,
            [
                0.8393095579193087, 0.07801575335662445, 0.023072199153607205,
                0.03328977636059566, 0.09183669868503967, 0.497414730477263,
                0.6402215246125254, 0.7209904907133703, 0.7439335942798158,
                0.7309904047874979, 0.781980671598943, 0.7009704533516865,
                1.0, 1.0,
            ],
        ),
        (
            5.0,
            [
                0.4360492092375524, 0.0007002535973785085, 3.2500325606542065e-05,
                7.992945118685922e-05, 0.000935517126255898, 0.052563092065358925,
                0.0955437058706739, 0.19508756202041233, 0.23544159675752477,
                0.21462597115802082, 0.30954036864292406, 0.15970475051041755,
                1.0, 1.0,
            ],
        ),
        (
            8.0,
            [
                0.02234772395609677, 1.891743779078237e-11, 3.654617375483278e-15,
                2.042103572067685e-14, 5.037730646286129e-12, 4.23528236670826e-08,
                1.6112720239855764e-07, 2.5450367506636165e-05, 9.484117175650355e-05,
                5.3359327785828654e-05, 0.000695519421210653, 1.149091850708478e-05,
                1.0, 1.0,
            ],
        ),
    ],
)
def test_igm_matches_cigale_v2025_1_static_reference(redshift, expected):
    """Regression values generated by official CIGALE v2025.1 redshifting.py."""
    rest_wave = np.array(
        [100.0, 300.0, 500.0, 700.0, 800.0, 900.0, 912.0, 950.0,
         1000.0, 1025.0, 1100.0, 1215.0, 1217.0, 1500.0],
        dtype=float,
    )
    cache = _build_igm_cache_jax(rest_wave)
    transmission = np.asarray(_build_fixed_igm_jax(cache, redshift), dtype=float)

    np.testing.assert_allclose(transmission, expected, rtol=2.0e-12, atol=1.0e-15)
