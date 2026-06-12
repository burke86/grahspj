from .config import (
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
    SpectroscopyConfig,
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
    "GalaxyConfig",
    "HostBasisJax",
    "JAXSEDFit",
    "InferenceConfig",
    "JaxQSOFitConfig",
    "LikelihoodConfig",
    "NebularConfig",
    "Observation",
    "PhotometryData",
    "SpectroscopyConfig",
    "SpectroscopyData",
    "build_host_basis_jax",
    "build_host_state",
    "host_rest_on_basis",
    "plot_corner",
    "plot_fit_sed",
    "plot_trace",
    "style_path",
    "load_chimera_benchmark_dataset",
    "run_chimera_mass_benchmark",
    "select_chimera_subset",
]


def __getattr__(name):
    """Lazily expose heavier public objects and helpers on first access."""
    if name == "JAXSEDFit":
        from .core import JAXSEDFit

        return JAXSEDFit
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
    raise AttributeError(name)
