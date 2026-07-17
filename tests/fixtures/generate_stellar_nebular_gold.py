"""Generate the compact external stellar/nebular gold-test fixture.

This script is intentionally not run by pytest.  It requires a checkout of
CIGALE v2025.1 and the public DSPS FSPS-v3.2 continuum SSP file documented in
jaxqsofit.  Keeping the generator beside the resulting fixture makes every
number in the gold tests reproducible.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


CIGALE_COMMIT = "29cb909fe2636800b4acdb1dfc7129d8c8494a24"
FSPS_URL = "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5"


class _CigaleRecord:
    """Minimal stand-in used to unpickle CIGALE's attribute-only records."""


class _CigaleUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("pcigale.data"):
            return _CigaleRecord
        return super().find_class(module, name)


def _read_pickle(path: Path):
    with path.open("rb") as stream:
        return _CigaleUnpickler(stream).load()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cigale", type=Path, required=True, help="CIGALE v2025.1 source checkout")
    parser.add_argument("--fsps", type=Path, required=True, help="DSPS FSPS-v3.2 SSP HDF5 file")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stellar_nebular_gold_v1.npz"))
    args = parser.parse_args()

    line_path = args.cigale / "pcigale/data/nebular_lines/Z=0.02_logU=-2.0_ne=100.0.pickle"
    continuum_path = args.cigale / "pcigale/data/nebular_continuum/Z=0.02_logU=-2.0_ne=100.0.pickle"
    lines = _read_pickle(line_path)
    continuum = _read_pickle(continuum_path)

    with h5py.File(args.fsps, "r") as handle:
        wave = np.asarray(handle["ssp_wave"])
        ages = np.asarray(handle["ssp_lg_age_gyr"])
        metallicities = np.asarray(handle["ssp_lgmet"])
        # Young, intermediate, and old SSPs at approximately solar abundance.
        metal_index = int(np.argmin(np.abs(metallicities)))
        target_ages = np.asarray([-3.0, -2.0, -1.0])
        age_indices = np.asarray([np.argmin(np.abs(ages - age)) for age in target_ages])
        ly_mask = wave < 912.0
        fsps_full_lnu = np.asarray(handle["ssp_flux"])[metal_index, age_indices]
        delayed_age_indices = np.unique(np.linspace(0, ages.size - 1, 25).round().astype(int))
        delayed_full_lnu = np.asarray(handle["ssp_flux"])[metal_index, delayed_age_indices]
        fsps_wave = wave[ly_mask]
        fsps_lnu = fsps_full_lnu[:, ly_mask]

    wave_m = fsps_wave * 1.0e-10
    llambda = fsps_lnu * 3.828e26 * 2.99792458e8 / wave_m**2 * 1.0e-10
    photon_kernel = wave_m / (6.62607015e-34 * 2.99792458e8)
    n_ly = np.trapezoid(llambda * photon_kernel, x=fsps_wave, axis=-1)
    ly_lum = np.trapezoid(llambda, x=fsps_wave, axis=-1)

    np.savez_compressed(
        args.output,
        fixture_version="1",
        cigale_version="2025.1",
        cigale_git_commit=CIGALE_COMMIT,
        fsps_version="3.2",
        fsps_source_url=FSPS_URL,
        fsps_lsun_w=3.828e26,
        fsps_wave_a=fsps_wave,
        fsps_lnu_lsun_per_hz=fsps_lnu,
        fsps_full_wave_a=wave,
        fsps_full_lnu_lsun_per_hz=fsps_full_lnu,
        delayed_fsps_lg_age_gyr=ages[delayed_age_indices],
        delayed_fsps_full_lnu_lsun_per_hz=delayed_full_lnu,
        fsps_lgmet=metallicities[metal_index],
        fsps_lg_age_gyr=ages[age_indices],
        fsps_n_ly_per_msun=n_ly,
        fsps_ly_lum_w_per_msun=ly_lum,
        line_name=np.asarray(lines.name),
        line_wave_a=np.asarray(lines.wl) * 10.0,
        line_lumin_w_per_photon=np.asarray(lines.spec),
        continuum_wave_a=np.asarray(continuum.wl) * 10.0,
        continuum_lumin_w_per_a_per_photon=np.asarray(continuum.spec) / 10.0,
        zgas=0.02,
        logu=-2.0,
        ne=100.0,
    )


if __name__ == "__main__":
    main()
