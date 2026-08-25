import numpy as np
import numpyro.distributions as dist

from jaxsedfit.spectral_contract import (
    LineDefinition,
    fwhm_kms_from_sigma_ln,
    gaussian_flambda_flux_erg_s_cm2,
    line_component_metadata,
    normalize_line_definitions,
)
from jaxsedfit.spectral_custom_components import (
    SpectralComponentSpec,
    split_spectral_components,
)
from jaxsedfit.spectral_model import build_tied_line_meta_from_linelist
from jaxsedfit import FitConfig, Observation, SpectroscopyData, SpectrumConfig


def _constant_component(wave, params, metadata):
    return np.zeros_like(np.asarray(wave)) + params["amplitude"]


def _line(name="Hb_br", components=2):
    return LineDefinition(
        name=name,
        rest_wavelength_angstrom=4862.68,
        component="Hb",
        components=components,
        amplitude_initial=1.0,
        sigma_ln_initial=0.01,
        sigma_ln_minimum=0.004,
        sigma_ln_maximum=0.05,
        max_center_offset_ln=0.01,
    )


def test_line_definition_is_accepted_by_the_shared_metadata_builder():
    definition = _line()

    rows = normalize_line_definitions([definition])
    metadata = build_tied_line_meta_from_linelist(
        [definition], np.linspace(4700.0, 5000.0, 20)
    )

    assert rows[0]["linename"] == "Hb_br"
    assert rows[0]["ngauss"] == 2
    assert metadata["names"] == ["Hb_br_1", "Hb_br_2"]


def test_line_component_metadata_owns_public_component_naming():
    metadata = {
        "names": ["Hb_br_1", "Hb_br_2", "OIII_5007_1"],
        "line_lambda": [4862.68, 4862.68, 5008.24],
        "broad_mask": [True, True, False],
    }

    components, groups = line_component_metadata(metadata)

    assert [item.public_name for item in components] == [
        "Hb_br_1",
        "Hb_br_2",
        "OIII_5007",
    ]
    assert groups[0].component_names == ("Hb_br_1", "Hb_br_2")


def test_shared_line_unit_conversions_are_draw_preserving():
    sigma = np.array([0.01, 0.02])
    center = np.log(np.array([4862.68, 4863.0]))

    widths = fwhm_kms_from_sigma_ln(sigma)
    fluxes = gaussian_flambda_flux_erg_s_cm2([1.0, 2.0], center, sigma)

    assert widths.shape == (2,)
    assert fluxes.shape == (2,)
    assert np.all(widths > 0.0)
    assert np.all(fluxes > 0.0)


def test_unified_component_spec_partitions_only_at_the_model_boundary():
    continuum = SpectralComponentSpec(
        name="continuum_extra",
        parameter_priors={"amplitude": dist.Normal(0.0, 1.0)},
        evaluate=_constant_component,
    )
    line = SpectralComponentSpec(
        name="line_extra",
        kind="broad_line",
        parameter_priors={"amplitude": dist.Normal(0.0, 1.0)},
        evaluate=_constant_component,
    )

    continua, lines = split_spectral_components([continuum, line])

    assert [item.name for item in continua] == ["continuum_extra"]
    assert [item.name for item in lines] == ["line_extra"]
    assert lines[0].line_kind == "broad"


def test_shared_spectrum_config_drives_joint_fit_feature_switches():
    config = FitConfig(
        observation=Observation(redshift=0.1),
        spectroscopy=SpectroscopyData(
            wave_obs=[5000.0, 5001.0],
            fluxes=[1.0, 1.0],
            errors=[0.1, 0.1],
        ),
        spectrum=SpectrumConfig(
            host_enabled=False,
            lines_enabled=False,
            feii_enabled=True,
            broadening_convolution="direct",
        ),
    )

    assert config.galaxy.fit_host is False
    assert config.agn.fit_lines is False
    assert config.agn.fit_feii is True
    assert config.agn.broadening_convolution == "direct"
