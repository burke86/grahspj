from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def median_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return posterior medians for every value in a sample-like mapping."""
    out: dict[str, Any] = {}
    for key, value in (values or {}).items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = arr.item()
        elif arr.size == 0:
            out[key] = arr
        else:
            out[key] = np.nanmedian(arr, axis=0)
    return out


@dataclass
class PredictionResult:
    """Dict-like posterior predictive result with lazy median summaries."""

    data: Mapping[str, Any]
    fitter: Any
    _median: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def median(self) -> dict[str, Any]:
        """Median predictive values over the leading posterior axis."""
        if self._median is None:
            self._median = median_mapping(self.data)
        return self._median

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class FitResult:
    """High-level result object returned by ``jaxsedfit`` fit methods."""

    fitter: Any
    samples: Mapping[str, Any] | None
    median: Mapping[str, Any]
    method: str
    summary: Mapping[str, Any] | None = None
    path: Path | None = None
    figure: Any = None

    def predict(self, **kwargs) -> PredictionResult:
        """Run or return posterior predictive products for this fit."""
        return PredictionResult(self.fitter.predict(**kwargs), fitter=self.fitter)

    def save(self, path: str | Path | None = None, **kwargs) -> Path:
        """Save the result with the fitter's native persistence format."""
        output_path = Path("." if path is None else path)
        self.path = Path(self.fitter.save(output_path, **kwargs))
        return self.path

    def plot_corner(self, **kwargs):
        """Plot posterior samples with the fitter's corner-plot helper."""
        return self.fitter.plot_corner(**kwargs)

    def plot_trace(self, **kwargs):
        """Plot posterior samples with the fitter's trace-plot helper."""
        return self.fitter.plot_trace(**kwargs)
