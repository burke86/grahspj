"""Generate CIGALE v2025.1 IGM transmission reference curves.

Run this with CIGALE v2025.1 importable, for example::

    PYTHONPATH=/path/to/cigale-v2025.1 python generate_cigale_igm_gold.py

CIGALE applies ``igm_transmission`` after shifting its wavelength grid, so the
function receives observed-frame wavelengths in nm.  The fixture stores both
those wavelengths and the corresponding rest-frame Angstrom grid used by
jaxsedfit.
"""

from pathlib import Path

import numpy as np

from pcigale.sed_modules.redshifting import igm_transmission


CIGALE_COMMIT = "29cb909fe2636800b4acdb1dfc7129d8c8494a24"


def main() -> None:
    output = Path(__file__).with_name("cigale_v2025_1_igm_reference.npz")
    # Include the Lyman limit and all important Lyman-series structure while
    # retaining a red optical control region where transmission must be one.
    rest_wave_a = np.unique(
        np.concatenate(
            [
                np.geomspace(300.13, 849.87, 220),
                # Offset the regular grid so no pixel lies exactly on a
                # discontinuous Lyman-series threshold. Which side owns the
                # exact floating-point equality is not physically meaningful.
                np.linspace(850.13, 1299.87, 901),
                np.geomspace(1300.13, 10000.13, 280),
            ]
        )
    )
    redshift = np.asarray([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 4.01, 5.0, 6.0])
    transmission = np.stack(
        [igm_transmission(rest_wave_a * (1.0 + z) / 10.0, z) for z in redshift]
    )
    np.savez_compressed(
        output,
        cigale_version="2025.1",
        cigale_git_tag="v2025.1",
        cigale_git_commit=CIGALE_COMMIT,
        wavelength_convention="observed_nm",
        rest_wave_a=rest_wave_a,
        redshift=redshift,
        transmission=transmission,
    )


if __name__ == "__main__":
    main()
