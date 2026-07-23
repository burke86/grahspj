from __future__ import annotations

import copy
from typing import Any, Dict, List, NamedTuple

import numpy as np
import numpyro.distributions as dist

from .spectral_config import PriorConfig
from .spectral_custom_components import CustomComponentSpec, make_custom_component
from .spectral_model import gaussian_bal_optical_depth_component

MINSCA_DEFAULT = 0.0
MAXSCA_DEFAULT = 1e10
AMPLITUDE_FLOOR = 1e-32
ROBUST_FLUX_HIGH_PERCENTILE = 99.5

class LineWidthPrior(NamedTuple):
    """Initial, minimum, and maximum Gaussian widths in ln-wavelength."""

    initial: float
    minimum: float
    maximum: float


class LineTies(NamedTuple):
    """Velocity, width, and amplitude tie-group indices."""

    velocity: int = 0
    width: int = 0
    amplitude: int = 0


BROAD_LINE_WIDTH = LineWidthPrior(initial=5e-3, minimum=0.004, maximum=0.05)
NARROW_LINE_WIDTH = LineWidthPrior(initial=1e-3, minimum=2.3e-4, maximum=0.00169)
RELAXED_NARROW_LINE_WIDTH = LineWidthPrior(
    initial=1e-3, minimum=5e-4, maximum=NARROW_LINE_WIDTH.maximum
)
UV_NARROW_LINE_WIDTH = LineWidthPrior(
    initial=1e-3, minimum=3.333e-4, maximum=NARROW_LINE_WIDTH.maximum
)
OIII_WING_LINE_WIDTH = LineWidthPrior(
    initial=3e-3, minimum=NARROW_LINE_WIDTH.minimum, maximum=0.004
)
UV_BROAD_LINE_WIDTH = LineWidthPrior(initial=5e-3, minimum=0.002, maximum=0.05)
INTERMEDIATE_UV_LINE_WIDTH = LineWidthPrior(
    initial=2e-3, minimum=0.001, maximum=0.01
)
RELAXED_UV_NARROW_LINE_WIDTH = LineWidthPrior(
    initial=RELAXED_NARROW_LINE_WIDTH.initial,
    minimum=RELAXED_NARROW_LINE_WIDTH.minimum,
    maximum=0.002,
)
EXTENDED_INTERMEDIATE_UV_LINE_WIDTH = LineWidthPrior(
    initial=INTERMEDIATE_UV_LINE_WIDTH.initial,
    minimum=INTERMEDIATE_UV_LINE_WIDTH.minimum,
    maximum=0.015,
)
CIV_BROAD_LINE_WIDTH = LineWidthPrior(
    initial=UV_BROAD_LINE_WIDTH.initial,
    minimum=0.001,
    maximum=UV_BROAD_LINE_WIDTH.maximum,
)
UV_SEMIBROAD_LINE_WIDTH = LineWidthPrior(initial=5e-3, minimum=0.0025, maximum=0.02)

BROAD_MAX_LOG_SHIFT = 0.015
BALMER_BROAD_MAX_LOG_SHIFT = 0.01
NARROW_MAX_LOG_SHIFT = 0.01
TIGHT_NARROW_MAX_LOG_SHIFT = 5e-3
UV_BROAD_MAX_LOG_SHIFT = 0.015
LYA_MAX_LOG_SHIFT = 0.02
NV_MAX_LOG_SHIFT = 0.005
ELG_MAX_LOG_SHIFT = 0.01
RED_ELG_MAX_LOG_SHIFT = 0.008


def _line(
    lam: float,
    component: str,
    name: str,
    width: LineWidthPrior,
    max_log_shift: float,
    initial_amplitude: float,
    ties: LineTies = LineTies(),
    ngauss: int = 1,
) -> PriorConfig:
    """Build one row in the public line-table dictionary schema."""
    return {
        "lambda": lam,
        "compname": component,
        "linename": name,
        "ngauss": ngauss,
        "inisca": 0.0,
        "minsca": MINSCA_DEFAULT,
        "maxsca": MAXSCA_DEFAULT,
        "inisig": width.initial,
        "minsig": width.minimum,
        "maxsig": width.maximum,
        "voff": max_log_shift,
        "vindex": ties.velocity,
        "windex": ties.width,
        "findex": ties.amplitude,
        "fvalue": initial_amplitude,
        "vary": 1,
    }


def _lnlam_peak_ratio_for_flux_ratio(
    flux_ratio: float,
    numerator_lam: float,
    denominator_lam: float,
) -> float:
    """Convert an integrated-flux ratio to a tied peak-amplitude ratio.

    Line ties are applied to Gaussian peak amplitudes in ln-lambda space. For
    equal ln-lambda widths, integrated flux scales as peak * rest wavelength.

    Parameters
    ----------
    flux_ratio : object
        flux_ratio value.
    numerator_lam : object
        numerator_lam value.
    denominator_lam : object
        denominator_lam value.
    """
    return flux_ratio * denominator_lam / numerator_lam


"""
Default line-prior table.

Each row below defines one emission-line prior in the same plain-dict schema
accepted by notebook line configs. The table is converted into NumPy/JAX
metadata by ``build_tied_line_meta_from_linelist`` before sampling.

Coordinate system
-----------------
``lambda`` is the rest-frame vacuum wavelength in Angstroms. The Gaussian
model itself is evaluated in ln(lambda), not linear wavelength. Consequently,
``inisig``, ``minsig``, and ``maxsig`` are Gaussian widths in ln(lambda). For
small widths, these are approximately velocity widths divided by c. For
example, ``sigma_ln_lambda = 0.001`` corresponds to roughly 300 km/s Gaussian
sigma, or about 700 km/s FWHM.

Amplitude and width fields
--------------------------
``inisca``, ``minsca``, and ``maxsca`` are priors on the Gaussian peak
amplitude. They are not integrated line fluxes. The integrated flux of a
single Gaussian in linear wavelength scales approximately as
peak_amplitude * sigma_ln_lambda * lambda0. This matters for fixed doublet
ratios: the helper ``_lnlam_peak_ratio_for_flux_ratio`` converts an intended
integrated-flux ratio into the peak-amplitude ratio required by the ln-lambda
Gaussian model, assuming the tied components share the same width.

``inisig``, ``minsig``, and ``maxsig`` define the prior for the Gaussian
ln-lambda width group. If multiple rows share a nonzero ``windex`` within the
same component complex, they share one sampled width. If ``windex`` is zero,
the row is not tied to other rows by width.

Velocity offsets: ``voff`` and ``vindex``
----------------------------------------
``voff`` is the allowed absolute center shift in ln(lambda): the sampled
center offset is constrained to ``[-voff, +voff]`` around ``log(lambda)``.
For small offsets this is approximately a velocity range of ``voff * c``.

``vindex`` controls tied velocity shifts. Rows with the same positive
``vindex`` within the same component complex share one sampled center offset.
Rows with ``vindex=0`` are independent. This follows the PyQSOFit convention
that only nonzero tie indices are constraints, but jaxqsofit additionally
scopes the tie by component complex to avoid accidental cross-complex tying
when the same integer is reused in different wavelength regions.

Width ties: ``windex``
---------------------
``windex`` works like ``vindex``, but for Gaussian width. Rows with the same
positive ``windex`` within the same component complex share one sampled
``sigma_ln_lambda``. Rows with ``windex=0`` are independent. This is commonly
used for physically related doublets or narrow-line complexes whose velocity
widths should match.

Amplitude/flux-ratio ties: ``findex`` and ``fvalue``
---------------------------------------------------
PyQSOFit documents this rule as: entries with the same nonzero ``findex`` have
constrained flux ratios. In this implementation, the same convention is used
with two precise details:

1. Only positive ``findex`` values tie rows together. ``findex=0`` means the
   row gets its own independent amplitude group.
2. Ties are local to the component complex, represented internally by
   ``compname``. The same positive ``findex`` can therefore be reused in
   different complexes without coupling unrelated lines such as Ha and Hb.

Within each tied amplitude group, one sampled peak-amplitude parameter is
created. Each component's peak amplitude is then
``line_amp_group[fgroup] * flux_ratio``, where ``flux_ratio`` is derived from
that row's ``fvalue`` relative to the first row in the group. Thus ``fvalue``
sets the fixed relative peak amplitude inside a tied group. For equal
ln-lambda widths, choosing ``fvalue`` with
``_lnlam_peak_ratio_for_flux_ratio`` enforces the desired integrated-flux
ratio.

For untied rows with ``findex=0``, ``fvalue`` is not a fixed flux ratio. It is
only the initial/default amplitude scale used to seed that independent
amplitude group's prior.

Narrow-line centroid pooling
----------------------------
By default, narrow cores are pooled into low-ionization, high-ionization, and
coronal kinematic families. All lines in a family share one exact centroid and
one exact FWHM across complexes. No complex-specific offsets or wavelength-
calibration error terms are added. Broad components and explicitly identified
wings or outflows are excluded. Set
``LineConfig.pool_narrow_centroids=False`` to restore the line-table centroid
and width ties without cross-complex family pooling.

Multiple Gaussians: ``ngauss``
-----------------------------
``ngauss`` expands one row into multiple Gaussian components with names like
``CIV_br_1``, ``CIV_br_2``, etc. Each expanded Gaussian is intentionally given
an independent internal tie label. For broad-line rows with ``ngauss > 1``,
their widths are sampled in strictly increasing order to remove equivalent
label-switched posterior modes. Their centroids use a shared broad-line shift
plus zero-sum relative offsets. Peak amplitudes remain independent. If a
genuinely tied multi-component structure is needed, write the components as
explicit rows with shared positive tie indices.

Line naming and plotting
------------------------
``linename`` is the output component name used in model metadata and plots.
The current plotting convention draws names containing ``"_br"`` and [O III]
wing names ending in ``"w"`` with broad-component styling; other built-in line
names use narrow-component styling.
``compname`` is used for grouping/tie scoping by line complex; it is not just a
display label.
"""
# Default line table in plain dict rows (same schema as notebook line config).
# Columns: wavelength, complex, output name, width prior, maximum center shift,
# initial amplitude, followed by optional ties and Gaussian count.
DEFAULT_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    # Halpha complex
    _line(6564.61, 'Ha', 'Ha_br', BROAD_LINE_WIDTH, BROAD_MAX_LOG_SHIFT, 0.05, ngauss=2),
    _line(6564.61, 'Ha', 'Ha_na', RELAXED_NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    _line(6549.85, 'Ha', 'NII6549', NARROW_LINE_WIDTH, TIGHT_NARROW_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=1, width=1, amplitude=1)),
    _line(6585.28, 'Ha', 'NII6585', NARROW_LINE_WIDTH, TIGHT_NARROW_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(3.0, 6585.28, 6549.85), ties=LineTies(velocity=1, width=1, amplitude=1)),
    _line(6718.29, 'Ha', 'SII6718', NARROW_LINE_WIDTH, TIGHT_NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1, amplitude=2)),
    _line(6732.67, 'Ha', 'SII6732', NARROW_LINE_WIDTH, TIGHT_NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1, amplitude=2)),
    # Hbeta / [OIII]
    _line(4862.68, 'Hb', 'Hb_br', BROAD_LINE_WIDTH, BALMER_BROAD_MAX_LOG_SHIFT, 0.01, ngauss=2),
    _line(4862.68, 'Hb', 'Hb_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    _line(4960.30, 'Hb', 'OIII4959c', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=1, width=1, amplitude=3)),
    _line(5008.24, 'Hb', 'OIII5007c', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(2.98, 5008.24, 4960.30), ties=LineTies(velocity=1, width=1, amplitude=3)),
    _line(4960.30, 'Hb', 'OIII4959w', OIII_WING_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=2, width=2, amplitude=4)),
    _line(5008.24, 'Hb', 'OIII5007w', OIII_WING_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(2.98, 5008.24, 4960.30), ties=LineTies(velocity=2, width=2, amplitude=4)),
    # Higher-order Balmer
    _line(4341.68, 'Hg', 'Hg_br', BROAD_LINE_WIDTH, BALMER_BROAD_MAX_LOG_SHIFT, 0.01),
    _line(4341.68, 'Hg', 'Hg_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    _line(4102.89, 'Hd', 'Hd_br', BROAD_LINE_WIDTH, BALMER_BROAD_MAX_LOG_SHIFT, 0.01),
    _line(4102.89, 'Hd', 'Hd_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    # Other optical/UV
    _line(3728.48, 'OII', 'OII3728', UV_NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    _line(3426.84, 'NeV', 'NeV3426', UV_NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001),
    # Principal Paschen lines. Each row has findex=0 so the broad- and
    # narrow-line amplitudes, and the amplitudes of different transitions,
    # remain independent rather than imposing Case-B ratios on the AGN BLR.
    _line(9548.59, 'Pae', 'Pae_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    _line(10052.13, 'Pad', 'Pad_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    _line(10941.09, 'Pag', 'Pag_br', BROAD_LINE_WIDTH, BALMER_BROAD_MAX_LOG_SHIFT, 0.01),
    _line(10941.09, 'Pag', 'Pag_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    _line(10833.31, 'HeI10830', 'HeI10830_br', BROAD_LINE_WIDTH, BROAD_MAX_LOG_SHIFT, 0.01),
    _line(10833.31, 'HeI10830', 'HeI10830_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    _line(12821.67, 'Pab', 'Pab_br', BROAD_LINE_WIDTH, BALMER_BROAD_MAX_LOG_SHIFT, 0.02),
    _line(12821.67, 'Pab', 'Pab_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    _line(18756.13, 'Paa', 'Paa_br', BROAD_LINE_WIDTH, BALMER_BROAD_MAX_LOG_SHIFT, 0.02),
    _line(18756.13, 'Paa', 'Paa_na', NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=1, width=1)),
    # Mg II complex
    _line(2798.75, 'MgII', 'MgII_br', BROAD_LINE_WIDTH, BROAD_MAX_LOG_SHIFT, 0.05, ngauss=2),
    _line(2798.75, 'MgII', 'MgII_na', RELAXED_NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    # CIII complex
    _line(1908.73, 'CIII', 'CIII_br', UV_BROAD_LINE_WIDTH, UV_BROAD_MAX_LOG_SHIFT, 0.01, ties=LineTies(velocity=3), ngauss=2),
    _line(1908.73, 'CIII', 'CIII_na', RELAXED_UV_NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=4, width=4)),
    _line(1892.03, 'CIII', 'SiIII1892', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, 0.003, 0.005, ties=LineTies(velocity=1, width=1)),
    _line(1857.40, 'CIII', 'AlIII1857', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, 0.003, 0.005, ties=LineTies(velocity=1, width=1)),
    _line(1816.98, 'CIII', 'SiII1816', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.0002, ties=LineTies(velocity=2, width=2)),
    _line(1750.26, 'CIII', 'NIII1750', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=2, width=2)),
    _line(1718.55, 'CIII', 'NIV1718', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=2, width=2)),
    # CIV complex
    _line(1549.06, 'CIV', 'CIV_br', CIV_BROAD_LINE_WIDTH, UV_BROAD_MAX_LOG_SHIFT, 0.05, ngauss=3),
    _line(1549.06, 'CIV', 'CIV_na', RELAXED_UV_NARROW_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    _line(1663.48, 'CIV', 'OIII1663', RELAXED_UV_NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    _line(1663.48, 'CIV', 'OIII1663_br', UV_SEMIBROAD_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=2, width=2)),
    _line(1640.42, 'CIV', 'HeII1640', RELAXED_UV_NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=1, width=1)),
    _line(1640.42, 'CIV', 'HeII1640_br', UV_SEMIBROAD_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.002, ties=LineTies(velocity=2, width=2)),
    # SiIV complex
    _line(1402.06, 'SiIV', 'SiIV_OIV1_br', UV_BROAD_LINE_WIDTH, UV_BROAD_MAX_LOG_SHIFT, 0.05, ties=LineTies(velocity=1, width=1)),
    _line(1396.76, 'SiIV', 'SiIV_OIV2_br', UV_BROAD_LINE_WIDTH, UV_BROAD_MAX_LOG_SHIFT, 0.05, ties=LineTies(velocity=1, width=1)),
    _line(1335.30, 'SiIV', 'CII1335', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=2, width=2)),
    _line(1304.35, 'SiIV', 'OI1304', EXTENDED_INTERMEDIATE_UV_LINE_WIDTH, NARROW_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=2, width=2)),
    # Lya complex
    _line(1215.67, 'Lya', 'Lya_br', UV_BROAD_LINE_WIDTH, LYA_MAX_LOG_SHIFT, 0.05, ngauss=3),
    _line(1240.14, 'Lya', 'NV1240_br', INTERMEDIATE_UV_LINE_WIDTH, NV_MAX_LOG_SHIFT, 0.002),
]

DEFAULT_LINE_CONFIG: Dict[str, Any] = {
    "line_dmu_scale_mult": 0.25,
    "line_sig_scale_mult": 0.25,
    "line_amp_scale_mult": 0.25,
    # Suppress unsupported second-and-later Gaussians while allowing the data
    # to retain them when a broad-line profile genuinely needs extra structure.
    "line_extra_amp_scale_mult": 0.5,
    "line": {"table": DEFAULT_LINE_PRIOR_ROWS},
}

# Additional narrow lines commonly used for emission-line galaxies (ELGs).
# These can be appended to the default line list via
# _build_default_prior_config(..., include_elg_narrow_lines=True).
DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    _line(3726.03, 'OII', 'OII3726', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=31)),
    _line(3728.82, 'OII', 'OII3729', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=31)),
    _line(3869.86, 'NeIII', 'NeIII3869', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(3968.59, 'NeIII', 'NeIII3968', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(4102.89, 'Hd', 'Hd_na_elg', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(4341.68, 'Hg', 'Hg_na_elg', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(4364.44, 'OIII', 'OIII4363', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(4862.68, 'Hb', 'Hb_na_elg', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(4687.02, 'HeII', 'HeII4686', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(4960.30, 'OIII', 'OIII4959', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=32)),
    _line(5008.24, 'OIII', 'OIII5007', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(2.98, 5008.24, 4960.30), ties=LineTies(velocity=11, width=11, amplitude=32)),
    _line(5877.25, 'HeI', 'HeI5876', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(6302.05, 'OI', 'OI6300', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(3.05, 6302.05, 6365.54), ties=LineTies(velocity=11, width=11, amplitude=33)),
    _line(6365.54, 'OI', 'OI6363', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=33)),
    _line(6549.85, 'NII', 'NII6548', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=34)),
    _line(6564.61, 'Ha', 'Ha_na_elg', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(6585.28, 'NII', 'NII6583', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(3.0, 6585.28, 6549.85), ties=LineTies(velocity=11, width=11, amplitude=34)),
    _line(6718.29, 'SII', 'SII6716', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=35)),
    _line(6732.67, 'SII', 'SII6731', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=35)),
    # Red optical / far-red forbidden + He I
    _line(7067.17, 'HeI', 'HeI7065', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(7137.77, 'ArIII', 'ArIII7138', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    _line(7322.19, 'OII', 'OII7320', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11, amplitude=22)),
    _line(7332.97, 'OII', 'OII7330', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11, amplitude=22)),
    _line(7753.19, 'ArIII', 'ArIII7751', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=11, width=11)),
    # Higher-order Paschen series (vacuum wavelengths, narrow by default).
    # The stronger Pa-epsilon through Pa-alpha lines are part of the default
    # AGN table above.
    _line(8752.87, 'Paschen', 'Pa12', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
    _line(8865.22, 'Paschen', 'Pa11', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
    _line(9017.38, 'Paschen', 'Pa10', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
    _line(9231.55, 'Paschen', 'Pa9', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
    # Strong red/NIR forbidden lines
    _line(9071.09, 'SIII', 'SIII9069', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=11, width=11, amplitude=23)),
    _line(9533.20, 'SIII', 'SIII9531', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(2.5, 9533.20, 9071.09), ties=LineTies(velocity=11, width=11, amplitude=23)),
]

# Optional high-ionization/coronal narrow-line set.
DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS: List[Dict[str, Any]] = [
    _line(3346.79, 'NeV', 'NeV3346', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, 1.0, ties=LineTies(velocity=12, width=12, amplitude=41)),
    _line(3426.84, 'NeV', 'NeV3426_hi', NARROW_LINE_WIDTH, ELG_MAX_LOG_SHIFT, _lnlam_peak_ratio_for_flux_ratio(2.7, 3426.84, 3346.79), ties=LineTies(velocity=12, width=12, amplitude=41)),
    _line(5721.0, 'FeVII', 'FeVII5721', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
    _line(6087.0, 'FeVII', 'FeVII6087', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
    _line(6374.0, 'FeX', 'FeX6374', NARROW_LINE_WIDTH, RED_ELG_MAX_LOG_SHIFT, 0.001, ties=LineTies(velocity=12, width=12)),
]


def _apply_robust_line_scale_priors(
    line_rows: List[Dict[str, Any]],
    fscale: float,
    fmax: float,
) -> List[Dict[str, Any]]:
    """Apply flux-aware robust bounds/initialization to line-scale priors.

    Parameters
    ----------
    line_rows : object
        line_rows value.
    fscale : object
        fscale value.
    fmax : object
        fmax value.
    """
    if len(line_rows) == 0:
        return line_rows

    # Keep dynamic range positive even for nearly flat/noisy spectra.
    delta = max(float(fmax - fscale), 0.1 * float(fscale), AMPLITUDE_FLOOR)

    for row in line_rows:
        linename = str(row.get("linename", "")).lower()
        is_broad = linename.endswith("_br") or ("_br" in linename)

        maxsca = float(row.get("maxsca", np.inf))
        minsca = float(row.get("minsca", 0.0))
        inisca = float(row.get("inisca", 0.0))

        # Broad lines get a tighter cap than narrow lines by default.
        if is_broad:
            max_cap = 1.0 * delta
        else:
            max_cap = 1.2 * delta
        maxsca = min(maxsca, max_cap)

        # Keep scales strictly positive and ordered.  Default qsopar rows use
        # ``inisca=0`` as a sentinel, but initializing a bounded amplitude at
        # the resulting lower floor leaves its unconstrained coordinate deep
        # in the transform tail.  Optax can then fail to move an otherwise
        # strong line away from zero.  Put sentinel/non-finite starts safely
        # inside the data-scaled interval while preserving explicit positive
        # user initializations.
        mins_floor = max(minsca, 1e-4 * float(fscale), AMPLITUDE_FLOOR)
        maxsca = max(maxsca, 1.01 * mins_floor)
        init_fraction = 0.05 if is_broad else 0.02
        if not np.isfinite(inisca) or inisca <= mins_floor:
            inisca = mins_floor + init_fraction * (maxsca - mins_floor)
        init_margin = max(1e-6 * (maxsca - mins_floor), AMPLITUDE_FLOOR)
        inisca = float(np.clip(inisca, mins_floor + init_margin, maxsca - init_margin))

        row["minsca"] = mins_floor
        row["maxsca"] = maxsca
        row["inisca"] = inisca

    return line_rows


def _append_unique_by_wavelength(
    base_rows: List[Dict[str, Any]],
    extra_rows: List[Dict[str, Any]],
    atol_angstrom: float = 1.0,
) -> List[Dict[str, Any]]:
    """Append rows from `extra_rows` only if no near-duplicate wavelength exists.

    Parameters
    ----------
    base_rows : object
        base_rows value.
    extra_rows : object
        extra_rows value.
    atol_angstrom : object
        atol_angstrom value.
    """
    out = list(base_rows)
    for row in extra_rows:
        lam_new = float(row.get("lambda", np.nan))
        if not np.isfinite(lam_new):
            continue
        exists = False
        for old in out:
            lam_old = float(old.get("lambda", np.nan))
            if np.isfinite(lam_old) and abs(lam_old - lam_new) <= float(atol_angstrom):
                exists = True
                break
        if not exists:
            out.append(row)
    return out


def append_optional_line_rows(
    prior_config: Dict[str, Any],
    flux: np.ndarray,
    *,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
) -> Dict[str, Any]:
    """Append optional built-in line sets selected by ``LineConfig``.

    Existing rows win when an optional row has the same wavelength, so this
    preserves user-provided line definitions and avoids duplicate components.
    Newly appended rows receive the same data-scaled amplitude initialization
    and bounds as rows constructed by the default-prior builder.
    """
    line_config = prior_config.get("line", {})
    if not isinstance(line_config, dict):
        return prior_config
    table = line_config.get("table")
    if not isinstance(table, list):
        return prior_config

    extras: List[Dict[str, Any]] = []
    if include_elg_narrow_lines:
        extras.extend(copy.deepcopy(DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS))
    if include_high_ionization_lines:
        extras.extend(copy.deepcopy(DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS))
    if not extras:
        return prior_config

    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    fmax = (
        float(np.nanpercentile(np.abs(f[finite]), ROBUST_FLUX_HIGH_PERCENTILE))
        if np.any(finite)
        else fscale
    )
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = fscale
    extras = _apply_robust_line_scale_priors(extras, fscale=fscale, fmax=fmax)
    line_config["table"] = _append_unique_by_wavelength(
        list(table), extras, atol_angstrom=1.0
    )
    return prior_config


def build_default_bal_components(
    flux: np.ndarray,
    *,
    tau_scale: float = 0.25,
    covering_loc: float = 0.15,
    covering_scale: float = 0.12,
    covering_high: float = 0.70,
    fwhm_kms_loc: float = 8000.0,
    fwhm_kms_scale: float = 2500.0,
    fwhm_kms_low: float = 2000.0,
    fwhm_kms_high: float = 15000.0,
) -> tuple[CustomComponentSpec, ...]:
    """Return built-in BAL custom components with conservative depth priors.

    Parameters
    ----------
    flux : object
        flux value.
    tau_scale : object
        tau_scale value.
    covering_loc : object
        covering_loc value.
    covering_scale : object
        covering_scale value.
    covering_high : object
        covering_high value.
    fwhm_kms_loc : object
        fwhm_kms_loc value.
    fwhm_kms_scale : object
        fwhm_kms_scale value.
    fwhm_kms_low : object
        fwhm_kms_low value.
    fwhm_kms_high : object
        fwhm_kms_high value.
    """
    def _bal_component(
        name: str,
        tau_scale: float,
        line_lambda: float,
        v_out_loc: float,
        v_out_scale: float,
        v_out_low: float,
        v_out_high: float,
    ):
        """Build one multiplicative BAL optical-depth component spec.

        Parameters
        ----------
        name : object
            name value.
        tau_scale : object
            tau_scale value.
        line_lambda : object
            line_lambda value.
        v_out_loc : object
            v_out_loc value.
        v_out_scale : object
            v_out_scale value.
        v_out_low : object
            v_out_low value.
        v_out_high : object
            v_out_high value.
        """
        return make_custom_component(
            name=name,
            parameter_priors={
                "tau_peak": dist.HalfNormal(float(max(tau_scale, 1.0e-6))),
                "covering": dist.TruncatedNormal(
                    float(covering_loc), float(max(covering_scale, 1.0e-6)),
                    low=0.0, high=float(covering_high),
                ),
                "v_out": dist.TruncatedNormal(
                    # The center is computed as lambda0 * (1 - v_out / c), so
                    # positive v_out values force absorption blueward of the
                    # associated transition.
                    float(v_out_loc), float(v_out_scale),
                    low=float(v_out_low), high=float(v_out_high),
                ),
                "fwhm_kms": dist.TruncatedNormal(
                    float(fwhm_kms_loc), float(max(fwhm_kms_scale, 1.0e-6)),
                    low=float(fwhm_kms_low), high=float(fwhm_kms_high),
                ),
                "shape_power": dist.TruncatedNormal(2.0, 1.5, low=2.0, high=12.0),
            },
            evaluate=gaussian_bal_optical_depth_component,
            metadata={
                "component_type": "bal_absorption",
                "line_lambda": float(line_lambda),
                "shared_parameter_sites": {
                    "v_out": "custom_bal_v_out",
                    "tau_peak": "custom_bal_tau_peak",
                    "covering": "custom_bal_covering",
                    "fwhm_kms": "custom_bal_fwhm_kms",
                },
            },
        )

    # Trump et al. (2006)
    return (
        _bal_component("bal_nv", tau_scale=tau_scale, line_lambda=1240.14, v_out_loc=6000.0, v_out_scale=2500.0, v_out_low=3000.0, v_out_high=12000.0),
        # _bal_component("bal_nv_2", depth_frac=0.025, center=1160.0, scale=90.0, low=1100.0, high=1240.0, sigma=40.0),
        _bal_component("bal_siiv", tau_scale=tau_scale, line_lambda=1396.76, v_out_loc=6000.0, v_out_scale=2500.0, v_out_low=3000.0, v_out_high=12000.0),
        # _bal_component("bal_siiv_2", depth_frac=0.025, center=1320.0, scale=90.0, low=1260.0, high=1397.0, sigma=40.0),
        _bal_component("bal_civ", tau_scale=tau_scale, line_lambda=1549.06, v_out_loc=6000.0, v_out_scale=2500.0, v_out_low=3000.0, v_out_high=12000.0),
        # _bal_component("bal_civ_2", depth_frac=0.03, center=1450.0, scale=100.0, low=1350.0, high=1549.0, sigma=45.0),
        # not common, often blended with other lines
        # _bal_component("bal_ciii", tau_scale=0.8, line_lambda=1908.73, v_out_loc=9200.0, v_out_scale=8000.0, v_out_low=300.0, v_out_high=25000.0, sigma=30.0),
        # _bal_component("bal_ciii_2", depth_frac=0.02, center=1800.0, scale=100.0, low=1700.0, high=1909.0, sigma=50.0),
        # Fe absorption, not common
        # _bal_component("bal_fe1", tau_scale=0.8, line_lambda=2050.0, v_out_loc=7300.0, v_out_scale=8000.0, v_out_low=300.0, v_out_high=15000.0, sigma=30.0),
        # _bal_component("bal_fe2", tau_scale=0.8, line_lambda=2250.0, v_out_loc=6600.0, v_out_scale=8000.0, v_out_low=300.0, v_out_high=13000.0, sigma=30.0),
        # not common
        # _bal_component("bal_mgii", tau_scale=0.8, line_lambda=2798.75, v_out_loc=5000.0, v_out_scale=7000.0, v_out_low=300.0, v_out_high=5200.0, sigma=40.0),
        # _bal_component("bal_mgii_2", depth_frac=0.02, center=2760.0, scale=120.0, low=2700.0, high=2798.0, sigma=55.0),
    )


def _build_default_prior_config(
    flux: np.ndarray,
    line_config: Dict[str, Any] | None = None,
    include_elg_narrow_lines: bool = False,
    include_high_ionization_lines: bool = False,
    pl_pivot: float | None = None,
) -> Dict[str, Any]:
    """Build a full PriorConfig with sane defaults from data flux scale.

    Parameters
    ----------
    flux : ndarray
        Input flux array used to set data-scale-aware defaults.
    line_config : dict or None, optional
        Optional line configuration override. If None, default line config is used.
    include_elg_narrow_lines : bool, optional
        If True, append additional narrow ELG lines from
        ``DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS`` to the active line table.
    include_high_ionization_lines : bool, optional
        If True, append additional high-ionization lines from
        ``DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS`` to the active line table.
    pl_pivot : float or None, optional
        Optional manual override for the power-law continuum pivot wavelength in
        Angstrom. If ``None``, the model uses the midpoint of the fitted rest-frame
        wavelength coverage.

    Notes
    -----
    ``log_ebv`` controls the amplitude of the built-in SMC-like attenuation
    curve in log space and is literal :math:`E(B-V)=A_B-A_V`. Legacy
    ``log_reddening_a2500`` prior dictionaries remain supported.
    """
    f = np.asarray(flux, dtype=float)
    finite = np.isfinite(f)
    fscale = float(np.nanmedian(np.abs(f[finite]))) if np.any(finite) else 1.0
    fmax = (
        float(np.nanpercentile(np.abs(f[finite]), ROBUST_FLUX_HIGH_PERCENTILE))
        if np.any(finite)
        else fscale
    )
    if not np.isfinite(fscale) or fscale <= 0:
        fscale = 1.0
    if not np.isfinite(fmax) or fmax <= 0:
        fmax = fscale

    cfg: Dict[str, Any] = {
        "cont_norm": dist.LogNormal(np.log(max(fscale, AMPLITUDE_FLOOR)), 0.3),
        "PL_norm": dist.HalfNormal(max(0.5 * fscale, AMPLITUDE_FLOOR)),
        "PL_slope": dist.Normal(-1.5, 0.4),
        "PL_pivot": None if pl_pivot is None else float(pl_pivot),
        "poly_pivot": None,
        # This corresponds to the historical median A(2500)=0.1 mag.
        "log_ebv": dist.Normal(np.log(0.1 * ((4400.0 / 2500.0) ** -1.2 - (5500.0 / 2500.0) ** -1.2)), 0.6),
        "reddening_uv_ref": 2500.0,
        "reddening_alpha": 1.2,
        "residualize_reddening_geometry": True,
        "log_frac_host": dist.StudentT(df=3.0, loc=0.0, scale=2.0),
        "host_redshift_prior": {
            "enabled": False,
            "z_mid": 1.0,
            "width": 0.2,
            "lowz_loc_offset": 0.0,
            "highz_loc_offset": -8.0,
            "lowz_scale_mult": 1.0,
            "highz_scale_mult": 0.05,
            "lowz_df": 3.0,
            "highz_df": 20.0,
        },
        "tau_host": dist.HalfNormal(1.0),
        "raw_w": dist.Normal(-0.5, 1.0),
        "host_template_age_prior": {
            "type": "prefer_old",
            "pivot_gyr": 1.0,
            "strength": 1.0,
            "min_logit": -3.0,
            "max_logit": 2.0,
        },
        "log_stellar_mass": dist.TruncatedNormal(9.0, 0.75, low=7.0, high=12.0),
        "log_host_aperture_scale": dist.Normal(0.0, 0.5),
        "log_sfh_age_gyr": dist.Normal(np.log(3.0), 1.0),
        "log_sfh_tau_over_age": dist.Normal(0.0, 0.5),
        "gal_lgmet": dist.Normal(0.0, 0.5),
        "log_gal_lgmet_scatter": dist.Normal(np.log(0.15), 0.7),
        "mass_metallicity_relation": {
            "enabled": False,
            "pivot_mass": 10.0,
            "pivot_logzsol": -0.15,
            "slope": 0.35,
            "scale": 0.25,
            "min": -1.5,
            "max": 0.3,
        },
        "gal_v_kms": dist.Normal(0.0, 120.0),
        "log_gal_sigma_kms": dist.TruncatedNormal(np.log(150.0), 0.4, low=np.log(30.0), high=np.log(500.0)),
        "Fe_uv_norm": dist.LogNormal(np.log(max(0.03 * fscale, 1e-12)), 1.0),
        "log_Fe_op_over_uv": dist.Normal(0.0, 1.0),
        "Fe_FWHM": dist.LogNormal(np.log(3000.0), 0.5),
        "Fe_shift": dist.Normal(0.0, 1e-3),
        "Balmer_norm": dist.LogNormal(np.log(max(1e-3 * fscale, AMPLITUDE_FLOOR)), 0.5),
        "Balmer_Tau": dist.LogNormal(np.log(0.5), 0.25),
        "log_Balmer_vel": dist.TruncatedNormal(np.log(3000.0), 0.3, low=np.log(1000.0), high=np.log(15000.0)),
        "poly_c2": dist.Normal(0.0, 0.03),
        "poly_c3": dist.Normal(0.0, 0.03),
        "poly_c4": dist.Normal(0.0, 0.03),
        "poly_c5": dist.Normal(0.0, 0.03),
        "poly_c6": dist.Normal(0.0, 0.03),
        "frac_jitter": dist.HalfNormal(0.02),
        "frac_fe_jitter": {"dist": "Delta", "value": 0.20},
        "add_jitter": {"dist": "Delta", "value": 0.0},
        "student_t_df": 3.0,
        "out_params": {
            "cont_loc": [1350.0, 2500.0, 3000.0, 4200.0, 5100.0],
        },
    }

    lc = copy.deepcopy(DEFAULT_LINE_CONFIG if line_config is None else line_config)
    if isinstance(lc, dict):
        line_cfg = lc.get("line", {})
        if isinstance(line_cfg, dict):
            table = line_cfg.get("table", None)
            if isinstance(table, list):
                if include_elg_narrow_lines:
                    table = _append_unique_by_wavelength(
                        list(table),
                        copy.deepcopy(DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS),
                        atol_angstrom=1.0,
                    )
                if include_high_ionization_lines:
                    table = _append_unique_by_wavelength(
                        list(table),
                        copy.deepcopy(DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS),
                        atol_angstrom=1.0,
                    )
                line_cfg["table"] = _apply_robust_line_scale_priors(table, fscale=fscale, fmax=fmax)
    cfg.update(lc)
    return PriorConfig._from_model_priors(cfg)
