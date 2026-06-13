from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import FilterCurve

FILTER_NAME_ALIASES = {
    "FUV_galex": "galex.FUV",
    "NUV_galex": "galex.NUV",
    "B_johnson": "generic.johnson.B",
    "V_johnson": "generic.johnson.V",
    "u_sdss": "sloan.sdss.u",
    "g_sdss": "sloan.sdss.g",
    "r_sdss": "sloan.sdss.r",
    "i_sdss": "sloan.sdss.i",
    "z_sdss": "sloan.sdss.z",
    "J_2mass": "2mass.J",
    "H_2mass": "2mass.H",
    "Ks_2mass": "2mass.Ks",
    "W1": "wise.W1",
    "W2": "wise.W2",
    "W3": "wise.W3",
    "W4": "wise.W4",
}


def _package_resource_path(relpath: str) -> Path:
    """Return an absolute path to a packaged jaxsedfit resource."""
    return Path(str(resources.files("jaxsedfit").joinpath(relpath)))


@lru_cache(maxsize=1)
def vendored_filter_registry() -> dict[str, str]:
    """Return the packaged filter registry as ``filter_name -> resource path``."""
    registry_path = _package_resource_path("resources/filters/filter_registry.txt")
    data = np.loadtxt(registry_path, dtype=str, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return dict(zip(data[:, 0].tolist(), data[:, 1].tolist()))


def resolve_filter_name(filter_name: str) -> str:
    """Return the canonical vendored filter name for a public alias."""
    return FILTER_NAME_ALIASES.get(str(filter_name), str(filter_name))


def filter_effective_wavelength(wave: Sequence[float], transmission: Sequence[float]) -> float:
    """Compute the effective wavelength convention used by jaxsedfit filters."""
    wave_arr = np.asarray(wave, dtype=float)
    trans_arr = np.asarray(transmission, dtype=float)
    denom = float(np.trapezoid(trans_arr, wave_arr))
    if denom <= 0.0:
        return float(np.nanmean(wave_arr))
    return float(np.trapezoid(wave_arr * trans_arr, wave_arr) / denom)


def normalize_filter_curve(curve: FilterCurve, name: str | None = None) -> FilterCurve:
    """Return a sorted, unique, non-negative filter curve."""
    wave = np.asarray(curve.wave, dtype=float)
    trans = np.clip(np.asarray(curve.transmission, dtype=float), 0.0, None)
    if wave.ndim != 1 or trans.ndim != 1 or wave.size != trans.size:
        raise ValueError(f"Filter curve {curve.name!r} must have 1D wave/transmission arrays of equal length.")
    if wave.size < 3:
        raise ValueError(f"Filter curve {curve.name!r} must have at least 3 wavelength samples.")
    finite = np.isfinite(wave) & np.isfinite(trans)
    wave = wave[finite]
    trans = trans[finite]
    if wave.size < 3:
        raise ValueError(f"Filter curve {curve.name!r} must have at least 3 finite wavelength samples.")
    order = np.argsort(wave, kind="stable")
    wave, trans = wave[order], trans[order]
    unique = np.concatenate(([True], np.diff(wave) > 0))
    wave, trans = wave[unique], trans[unique]
    if wave.size < 3:
        raise ValueError(f"Filter curve {curve.name!r} must have at least 3 unique wavelength samples.")
    wave = wave.copy()
    trans = trans.copy()
    trans[0] = 0.0
    trans[-1] = 0.0
    return FilterCurve(
        name=name if name is not None else curve.name,
        wave=wave,
        transmission=trans,
        effective_wavelength=filter_effective_wavelength(wave, trans),
    )


def load_filter_curve(filter_name: str) -> FilterCurve:
    """Load one packaged jaxsedfit filter curve by canonical name or alias."""
    resolved_name = resolve_filter_name(filter_name)
    relpath = vendored_filter_registry().get(resolved_name)
    if relpath is None:
        raise ValueError(f"Filter {filter_name!r} is not available in the vendored jaxsedfit filter registry.")

    path = _package_resource_path(relpath)
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Vendored filter file {path} does not contain two-column transmission data.")

    wave = np.array(data[:, 0], dtype=float)
    trans = np.array(data[:, 1], dtype=float)
    if _load_filter_type(path) == "photon":
        trans *= wave
    return normalize_filter_curve(FilterCurve(name=str(filter_name), wave=wave, transmission=trans), name=str(filter_name))


def load_filter_curves(filter_names: Sequence[str]) -> list[FilterCurve]:
    """Load multiple packaged jaxsedfit filter curves in order."""
    return [load_filter_curve(name) for name in filter_names]


def _load_filter_type(path: Path) -> str:
    """Read the # photon / # energy header line from a .dat filter file."""
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                stripped = line.lstrip("#").strip()
                if stripped in ("energy", "photon"):
                    return stripped
    return "energy"
