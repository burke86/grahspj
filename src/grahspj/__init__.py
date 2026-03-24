from .config import (
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
    "GRAHSPJ",
    "InferenceConfig",
    "LikelihoodConfig",
    "Observation",
    "PhotometryData",
    "plot_fit_sed",
    "style_path",
    "load_chimera_benchmark_dataset",
    "run_chimera_mass_benchmark",
    "select_chimera_subset",
]


def __getattr__(name):
    """Lazily expose heavier public objects and helpers on first access."""
    if name == "GRAHSPJ":
        from .core import GRAHSPJ

        return GRAHSPJ
    if name == "plot_fit_sed":
        from .plotting import plot_fit_sed

        return plot_fit_sed
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
    raise AttributeError(name)
