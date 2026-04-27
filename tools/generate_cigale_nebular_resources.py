from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.constants as cst
from astropy.table import Table


VERSION = "2025.1"
TAG = "v2025.1"
COMMIT = "29cb909fe2636800b4acdb1dfc7129d8c8494a24"
SOURCE = "https://gitlab.lam.fr/cigale/cigale/-/archive/v2025.1/cigale-v2025.1.tar.gz"


def _read_continuum(path: Path, z_grid: np.ndarray, logu_grid: np.ndarray, ne_grid: np.ndarray):
    blocks: list[np.ndarray] = []
    current: list[list[float]] = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("###"):
            if current:
                blocks.append(np.asarray(current, dtype=np.float64))
                current = []
            continue
        current.append([float(x) for x in stripped.split()])
    if current:
        blocks.append(np.asarray(current, dtype=np.float64))
    if len(blocks) != z_grid.size:
        raise ValueError(f"Expected {z_grid.size} continuum metallicity blocks, found {len(blocks)}.")

    cont = np.stack(blocks, axis=0)
    wave_a = cont[0, :, 0].astype(np.float64)
    if not np.allclose(cont[:, :, 0], wave_a[None, :]):
        raise ValueError("Continuum wavelength grids differ across metallicities.")

    lumin = cont[:, :, 1:]
    lumin = np.reshape(lumin, (z_grid.size, wave_a.size, logu_grid.size, ne_grid.size))
    lumin = np.moveaxis(lumin, (0, 1, 2, 3), (0, 3, 1, 2))

    wave_nm = wave_a * 0.1
    lumin_per_nm = lumin * 1.0e-7 * cst.c * 1.0e9 / wave_nm**2
    return wave_a, (lumin_per_nm / 10.0).astype(np.float32)


def convert_nebular_tables(source_dir: Path, output_dir: Path) -> None:
    table = Table.read(source_dir / "line_wavelengths.dat", format="ascii")
    line_wave_a = np.asarray(table["col1"], dtype=np.float64)
    line_name = np.asarray(table["col2"], dtype="<U64")
    raw_lines = np.genfromtxt(source_dir / "lines.dat")

    z_grid = np.unique(raw_lines[:, 1]).astype(np.float64)
    logu_grid = np.around(np.arange(-4.0, -0.9, 0.1), 1).astype(np.float64)
    ne_grid = np.asarray([10.0, 100.0, 1000.0], dtype=np.float64)

    line_lumin = raw_lines[:, 2:]
    line_lumin = np.reshape(line_lumin, (line_wave_a.size, z_grid.size, logu_grid.size, ne_grid.size))
    line_lumin = np.moveaxis(line_lumin, (0, 1, 2, 3), (3, 0, 1, 2))
    line_lumin = (10.0**line_lumin).astype(np.float32)

    continuum_wave_a, continuum_lumin = _read_continuum(source_dir / "continuum.dat", z_grid, logu_grid, ne_grid)

    metadata = {
        "cigale_version": np.asarray(VERSION),
        "cigale_git_tag": np.asarray(TAG),
        "cigale_git_commit": np.asarray(COMMIT),
        "source": np.asarray(SOURCE),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "nebular_lines.npz",
        z_grid=z_grid,
        logu_grid=logu_grid,
        ne_grid=ne_grid,
        line_wave_a=line_wave_a,
        line_name=line_name,
        line_lumin_per_photon=line_lumin,
        **metadata,
    )
    np.savez_compressed(
        output_dir / "nebular_continuum.npz",
        z_grid=z_grid,
        logu_grid=logu_grid,
        ne_grid=ne_grid,
        continuum_wave_a=continuum_wave_a,
        continuum_lumin_per_a_per_photon=continuum_lumin,
        **metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CIGALE v2025.1 nebular source tables to grahspj resources.")
    parser.add_argument("source_dir", type=Path, help="Path to CIGALE database_builder/nebular/data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/grahspj/resources/nebular"),
        help="Output directory for compact grahspj .npz resources.",
    )
    args = parser.parse_args()
    convert_nebular_tables(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
