from .config import (
    AGNConfig,
    EmissionLineTemplate,
    FeIITemplate,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    AGNPriorConfig,
    HostPriorConfig,
    InferenceConfig,
    LikelihoodConfig,
    LikelihoodPriorConfig,
    NebularConfig,
    NebularPriorConfig,
    Observation,
    OutputConfig,
    PhotometryData,
    PriorConfig,
    RedshiftPriorConfig,
    MassMetallicityPriorConfig,
    SpectroscopyData,
)

__all__ = [
    "AGNConfig",
    "build_chimera_fit_config",
    "CHIMERA_FILTER_NAMES",
    "EmissionLineTemplate",
    "FeIITemplate",
    "FilterCurve",
    "FilterSet",
    "FitConfig",
    "FitResult",
    "GalaxyConfig",
    "AGNPriorConfig",
    "HostPriorConfig",
    "HostBasisJax",
    "JAXSEDFit",
    "InferenceConfig",
    "LikelihoodConfig",
    "LikelihoodPriorConfig",
    "NebularConfig",
    "NebularPriorConfig",
    "Observation",
    "OutputConfig",
    "PhotometryData",
    "PredictionResult",
    "LineComponentResult",
    "LineGroupResult",
    "SpectralResult",
    "SpectrumObservationResult",
    "PriorConfig",
    "RedshiftPriorConfig",
    "MassMetallicityPriorConfig",
    "SpectroscopyData",
    "CustomComponentSpec",
    "CustomLineComponentSpec",
    "make_custom_component",
    "make_custom_line_component",
    "make_template_component",
    "build_host_basis_jax",
    "build_host_state",
    "host_rest_on_basis",
    "load_from_samples",
    "load",
    "load_result",
    "load_filter_curve",
    "load_filter_curves",
    "plot_corner",
    "plot_fit_sed",
    "plot_trace",
    "style_path",
    "load_chimera_benchmark_dataset",
    "run_chimera_mass_benchmark",
    "select_chimera_subset",
]


def __getattr__(name):
    """Lazily expose heavier public objects and helpers on first access.

    Parameters
    ----------
    name : object
        name value.
    """
    if name == "JAXSEDFit":
        from .core import JAXSEDFit

        return JAXSEDFit
    if name in {
        "CustomComponentSpec",
        "CustomLineComponentSpec",
        "make_custom_component",
        "make_custom_line_component",
        "make_template_component",
    }:
        from . import spectroscopy as _spectroscopy

        return getattr(_spectroscopy, name)
    if name in {"FitResult", "PredictionResult"}:
        from . import results as _results

        return getattr(_results, name)
    if name in {
        "LineComponentResult",
        "LineGroupResult",
        "SpectralResult",
        "SpectrumObservationResult",
    }:
        from . import spectral_results as _spectral_results

        return getattr(_spectral_results, name)
    if name in {"load_from_samples", "load"}:
        from .core import JAXSEDFit

        return JAXSEDFit.load
    if name == "load_result":
        from .core import JAXSEDFit

        return JAXSEDFit.load_result
    if name == "plot_fit_sed":
        from .plotting import plot_fit_sed

        return plot_fit_sed
    if name == "plot_corner":
        from .plotting import plot_corner

        return plot_corner
    if name == "plot_trace":
        from .plotting import plot_trace

        return plot_trace
    if name == "style_path":
        from .mplstyle import style_path

        return style_path
    if name in {
        "CHIMERA_FILTER_NAMES",
        "build_chimera_fit_config",
        "load_chimera_benchmark_dataset",
        "run_chimera_mass_benchmark",
        "select_chimera_subset",
    }:
        from . import benchmark as _benchmark

        return getattr(_benchmark, name)
    if name in {"HostBasisJax", "build_host_basis_jax", "build_host_state", "host_rest_on_basis"}:
        from . import host as _host

        return getattr(_host, name)
    if name in {"load_filter_curve", "load_filter_curves"}:
        from . import filters as _filters

        return getattr(_filters, name)
    raise AttributeError(name)
