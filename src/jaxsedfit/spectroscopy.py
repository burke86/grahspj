"""Supported public interface to the detailed quasar spectral engine.

Joint SED+spectrum fitting and spectrum-focused interfaces
should import through this module instead of depending on implementation
helpers spread across the ``spectral_*`` modules.
"""

from __future__ import annotations

from .spectral_components import (
    SpectralComponentConfig,
    build_joint_tied_line_meta,
    evaluate_joint_spectral_components,
    render_joint_feature_state,
)
from .spectral_geometry import line_complex_dense_mass_blocks
from .spectral_custom_components import (
    CustomComponentSpec,
    CustomLineComponentSpec,
    make_custom_component,
    make_custom_line_component,
    make_template_component,
)
from .spectral_reparameterization import (
    NORMAL_LOGNORMAL_STANDARDIZATION,
    NormalLogNormalStandardizeReparam,
    normal_lognormal_standardization_reparam,
    standardized_prior_site,
)


def build_spectral_prior_config(
    flux,
    *,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
):
    """Build scale-aware default priors for the detailed spectral engine."""
    from .spectral_defaults import _build_default_prior_config

    return _build_default_prior_config(
        flux,
        include_elg_narrow_lines=include_elg_narrow_lines,
        include_high_ionization_lines=include_high_ionization_lines,
    )


__all__ = [
    "NORMAL_LOGNORMAL_STANDARDIZATION",
    "NormalLogNormalStandardizeReparam",
    "SpectralComponentConfig",
    "CustomComponentSpec",
    "CustomLineComponentSpec",
    "build_joint_tied_line_meta",
    "build_spectral_prior_config",
    "evaluate_joint_spectral_components",
    "line_complex_dense_mass_blocks",
    "make_custom_component",
    "make_custom_line_component",
    "make_template_component",
    "normal_lognormal_standardization_reparam",
    "render_joint_feature_state",
    "standardized_prior_site",
]
