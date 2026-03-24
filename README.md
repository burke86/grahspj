# grahspj

`grahspj` is an experimental JAX-based port of pieces of `CIGALE` and `GRAHSP`.

The current Python package and install name are:

- package: `grahspj`
- install name: `grahspj`

This codebase is not a full line-by-line reimplementation of either upstream project. It is a research-oriented, single-source fitting framework that ports selected `GRAHSP`/`pcigale` model components into JAX/NumPyro and combines them with a JAX-native galaxy path based on `Diffstar` + `DSPS`.

At a high level, `grahspj` currently includes:

- a JAX/NumPyro fitting engine
- `Diffstar` + `DSPS` host-galaxy modeling
- JAX ports of selected `GRAHSP` AGN, attenuation, redshifting, and dust-emission components
- `pcigale`-style SED plotting
- a Chimera benchmark for stellar-mass recovery

Important scope note:

- this is an experimental port of `CIGALE` and `GRAHSP`, not strict full-module parity
- some parts are intentionally simplified or redesigned for JAX-based inference
- model behavior should be treated as under active development

Filter curves are handled through `speclite`. Built-in `speclite` filters are used when available, inline curves are wrapped into `speclite.filters.FilterResponse`, and vendored package resources are used for the supported non-`speclite` filters/templates needed by the default benchmark path.

The package also ships a Matplotlib style file used by the built-in plotting helpers and benchmark plots:

```python
import matplotlib.pyplot as plt
from grahspj.mplstyle import style_path

plt.style.use(style_path())
```

## Install

`grahspj` requires Python 3.10 or newer.

```bash
cd /Users/colinburke/research/grahspj
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
curl -O https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5
```

This installs the package in editable mode. You will also need a DSPS SSP template file such as `ssp_data_fsps_v3.2_lgmet_age.h5`, downloaded above, and then referenced from your configuration via `cfg.galaxy.dsps_ssp_fn` or passed directly to `fit(...)` via `dsps_ssp_fn`.

The install provides two console commands:

- `grahspj`
- `grahspj-benchmark`

## SED plotting

`grahspj` includes a `pcigale`-style component SED plot that overlays:

- observed photometry with uncertainties
- model photometry
- host galaxy spectrum
- AGN disk
- torus
- Fe II
- emission lines
- Balmer continuum
- total AGN
- total model

From Python:

```python
from grahspj.core import GRAHSPJ

fitter = GRAHSPJ(cfg)
fitter.fit(
    fit_method="optax+nuts",
    optax_steps=600,
    optax_lr=1e-2,
    nuts_warmup=50,
    nuts_samples=50,
    nuts_chains=1,
    plot_fig=False,
    save_fig=True,
    save_result=True,
    output_dir="fit_outputs",
)
```

or with the standalone helper:

```python
from grahspj.plotting import plot_fit_sed

plot_fit_sed(fitter, output_path="sed_fit.png")
```

This uses the lazy predictive path, so the component spectra are generated when you first call `plot_sed()` or `plot_fit_sed(...)`.

## Fitting API

The primary fitting interface is a single `fit(...)` method with selectable inference backends.

Use the single `fit(...)` entrypoint with one of:

- `"optax"`: run the Optax/NumPyro MAP fit
- `"nuts"`: run NUTS directly
- `"optax+nuts"`: run MAP first, then initialize NUTS from the MAP result

Example:

```python
fitter = GRAHSPJ(cfg)
result = fitter.fit(
    fit_method="optax+nuts",
    prior_config=prior_config,
    dsps_ssp_fn="../tempdata.h5",
    optax_steps=600,
    optax_lr=1e-2,
    nuts_warmup=50,
    nuts_samples=50,
    nuts_chains=1,
    plot_fig=True,
    save_fig=False,
    save_result=False,
    progress_bar=True,
)
```

Common convenience options accepted directly by `fit(...)`:

- `prior_config`
- `dsps_ssp_fn`
- `optax_steps`
- `optax_lr`
- `nuts_warmup`
- `nuts_samples`
- `nuts_chains`
- `plot_fig`
- `save_fig`
- `save_result`
- `output_dir`

The returned dictionary contains:

- `fit`
- `summary`
- `figure`
- `figure_path`
- `result_path`

The lower-level methods remain available:

- `fit_map(...)`
- `fit_nuts(...)`

Both the SVI/Optax and NUTS paths now expose progress bars when `progress_bar=True`.

## Example notebook

A worked single-object tutorial is available in:

- [notebooks/01_example.ipynb](/Users/colinburke/research/grahspj/notebooks/01_example.ipynb)

It shows how to:

- load one Chimera example SED
- build a fit configuration
- run `GRAHSPJ.fit(...)`
- inspect summary outputs
- make the component SED plot

## License and provenance

`grahspj` is an experimental port of parts of `CIGALE` and `GRAHSP`.

Some model logic and several bundled resource files are derived from or closely based on `GRAHSP` / `pcigale`, which is distributed under the `CeCILL v2` license.

See:

- [LICENSES/CeCILL-v2.txt](/Users/colinburke/research/grahspj/LICENSES/CeCILL-v2.txt)
- [THIRD_PARTY_NOTICES.md](/Users/colinburke/research/grahspj/THIRD_PARTY_NOTICES.md)

Bundled third-party resources under [src/grahspj/resources](/Users/colinburke/research/grahspj/src/grahspj/resources) include per-directory provenance notes.

## Filters

`grahspj` routes all filter handling through `speclite`.

- Built-in mappings cover common names such as `u_sdss -> sdss2010-u` and `J_2mass -> twomass-J`
- You can override any mapping with `filters.speclite_names`
- Vendored IRAC benchmark filters are bundled with the package and wrapped into `speclite` objects automatically
- If a filter is not available in `speclite` or in the vendored package resources, the optional GRAHSP database fallback is still supported, but it is no longer required for the benchmark/default supported subset
- Inline curves are also wrapped into `speclite` objects before synthetic photometry is computed

## Chimera benchmark

The Chimera benchmark is intended as a regression and calibration tool for this experimental port, not as a finalized scientific validation of full `GRAHSP`/`CIGALE` parity.

`dsps_ssp_fn` must point to a valid DSPS SSP HDF5 file. Additional SPS template files, including variants with nebular grids, are available at `https://halos.as.arizona.edu/suchethacooray/dsps_ssp/`. At present, `grahspj` cannot vary nebular parameters independently beyond whatever is baked into the selected DSPS template, but this is expected to be sufficient for most broad-band fitting use cases.

The stellar-mass recovery benchmark uses:

- `data/chimeras-2023-10-11/chimeras-grahsp.fits` for input photometry
- `data/chimeras-2023-10-11/chimeras-fullinfo.fits` for truth labels
- `MASS_MED_GAL` as the truth stellar mass
- `resample_weight` for weighted metrics
- `data/chimeras-2023-10-11/benchmark_subset_ids.txt` as the deterministic subset

### Run from the CLI

```bash
grahspj-benchmark --output-dir benchmark_outputs --dsps-ssp-fn tempdata.h5
```

You can also run it without installing the script entry point:

```bash
python -m grahspj.benchmark --output-dir benchmark_outputs --dsps-ssp-fn tempdata.h5
```

Optional thresholds:

```bash
grahspj-benchmark \
  --output-dir benchmark_outputs \
  --dsps-ssp-fn tempdata.h5 \
  --max-weighted-mae 3.0 \
  --max-abs-weighted-bias 2.0 \
  --min-finite-fraction 0.95
```

To run only a small deterministic prefix of the benchmark subset:

```bash
grahspj-benchmark \
  --output-dir benchmark_outputs_small \
  --dsps-ssp-fn tempdata.h5 \
  --limit 5
```

### Run from Python

```python
from grahspj.benchmark import run_chimera_mass_benchmark

result = run_chimera_mass_benchmark(
    output_dir="benchmark_outputs",
    dsps_ssp_fn="tempdata.h5",
    limit=5,
)
print(result["passed"])
print(result["metrics"])
```

### Outputs

The benchmark returns a dictionary with:

- `metrics`
- `by_redshift_bin`
- `by_chimera_qso_weight`
- `rows`
- `passed`
- `thresholds`

The output directory contains:

- `chimera_mass_recovery_rows.csv`
- `chimera_mass_recovery_metrics.json`
- `chimera_mass_scatter.png`
- `chimera_mass_residual_vs_qso_weight.png`

### Success criteria

The benchmark passes when all three conditions are met:

- `weighted_mae <= max_weighted_mae`
- `abs(weighted_bias) <= max_abs_weighted_bias`
- `finite_fit_fraction >= min_finite_fraction`

### Notes

- The benchmark runs the `MAP` fitting path, not `NUTS`.
- You need a working `jax` / `numpyro` / `optax` / `dsps` environment.
- `dsps_ssp_fn` must point to a valid DSPS SSP HDF5 file.
- Import the package as `grahspj`.
