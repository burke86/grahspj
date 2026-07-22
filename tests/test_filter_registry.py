import numpy as np
import pytest

from jaxsedfit.filters import load_filter_curve, vendored_filter_registry


NEW_FILTER_NAMES = (
    "spitzer.mips.24mu",
    "spitzer.mips.70mu",
    "spitzer.mips.160mu",
    "herschel.pacs.blue",
    "herschel.pacs.green",
    "herschel.pacs.red",
    "herschel.spire.PSW",
    "herschel.spire.PMW",
    "herschel.spire.PLW",
    "herschel.spire.PSW_ext",
    "herschel.spire.PMW_ext",
    "herschel.spire.PLW_ext",
    "subaru.hsc.g",
    "subaru.hsc.r",
    "subaru.hsc.i",
    "subaru.hsc.z",
    "subaru.hsc.Y",
)


@pytest.mark.parametrize("filter_name", NEW_FILTER_NAMES)
def test_new_registry_filters_load(filter_name):
    assert filter_name in vendored_filter_registry()
    curve = load_filter_curve(filter_name)
    assert curve.name == filter_name
    assert len(curve.wave) >= 3
    assert np.all(np.isfinite(curve.wave))
    assert np.all(np.isfinite(curve.transmission))
    assert curve.effective_wavelength > 0.0
