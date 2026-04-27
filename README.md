# GRAHSP-J

`GRAHSP-J` is a Bayesian SED fitting code for AGN and galaxies. It is an experimental JAX-based implementation of `CIGALE` and `GRAHSP`. It ports `GRAHSP`/`pcigale` model components into JAX/NumPyro and combines them with a JAX-native galaxy models based on `Diffstar` + `DSPS`.

At a high level, `grahspj` currently includes:

- a JAX/NumPyro fitting engine
- `Diffstar` + `DSPS` host-galaxy modeling
- JAX ports of selected `GRAHSP` AGN, attenuation, redshifting, and dust-emission components
- `pcigale`-style SED plotting
- a Chimera benchmark for stellar-mass recovery

## Install

`grahspj` requires Python 3.10 or newer. First, clone this repository. Then:

```bash
python -m pip install .
curl -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5
```
`grahspj` now also requires `jax_cosmo` and `setuptools` in the runtime environment so the redshift-dependent luminosity-distance path stays JAX-native during inference.

You will also need a DSPS SSP template file such as `ssp_data_fsps_v3.2_lgmet_age.h5`, downloaded above, and then referenced from your configuration via `cfg.galaxy.dsps_ssp_fn` or passed directly to `fit(...)` via `dsps_ssp_fn`.

This repo assumes `dustmaps` is already configured and SFD maps are available.

Typical one-time setup:

```
python setup.py fetch --map-name=sfd
```

After fetching, make sure `dustmaps` is configured to use the directory containing the SFD maps.

## Example notebook

A worked single-object tutorial is available in:

- [notebooks/01_example.ipynb](notebooks/01_example.ipynb)
- [notebooks/02_vizier_fairall9.ipynb](notebooks/02_vizier_fairall9.ipynb)

It shows how to:

- load one Chimera example SED
- build a fit configuration
- run `GRAHSPJ.fit(...)`
- inspect summary outputs
- make the component SED plot

The Fairall 9 notebook shows how to:

- query broadband photometry from the VizieR SED service
- map supported survey filters into `grahspj`
- build a manual `FitConfig`
- fit and plot the resulting AGN SED


## Usage

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

Nested sampling is also available through NumPyro's `jaxns` wrapper:

```python
fitter.fit(
    fit_method="ns",
    ns_live_points=200,
    ns_dlogz=0.1,
)
```

or with the standalone helper:

```python
from grahspj.plotting import plot_fit_sed

plot_fit_sed(fitter, output_path="sed_fit.png")
```

This uses the lazy predictive path, so the component spectra are generated when you first call `plot_sed()` or `plot_fit_sed(...)`.


## License and provenance

`grahspj` is an experimental port of parts of `CIGALE` and `GRAHSP`.

Some model logic and several bundled resource files are derived from or closely based on `GRAHSP` / `pcigale`, which is distributed under the `CeCILL v2` license.

See:

- [LICENSES/CeCILL-v2.txt](/Users/colinburke/research/grahspj/LICENSES/CeCILL-v2.txt)
- [LICENSES/THIRD_PARTY_NOTICES.md](/Users/colinburke/research/grahspj/LICENSES/THIRD_PARTY_NOTICES.md)

Bundled third-party resources under [src/grahspj/resources](/Users/colinburke/research/grahspj/src/grahspj/resources) include per-directory provenance notes.

## Filters

`grahspj` routes all filter handling through `speclite`.

- Built-in mappings cover common names such as `u_sdss -> sdss2010-u` and `J_2mass -> twomass-J`
- You can override any mapping with `filters.speclite_names`
- Vendored IRAC benchmark filters are bundled with the package and wrapped into `speclite` objects automatically
- If a filter is not available in `speclite` or in the vendored package resources, the optional GRAHSP database fallback is still supported, but it is no longer required for the benchmark/default supported subset
- Inline curves are also wrapped into `speclite` objects before synthetic photometry is computed

## Survey PSF Sizes In The Likelihood

Broad-band catalogs do not all measure the same physical light profile. A
GALEX, SDSS, 2MASS, WISE, or IRAC point has a different effective angular
resolution, and aperture photometry can capture a different fraction of extended
host-galaxy light than PSF-like photometry. `grahspj` can account for this with
the optional host-capture likelihood model.

Pass one value per photometric point through `PhotometryData.psf_fwhm_arcsec`.
If an aperture diameter is known, pass `PhotometryData.aperture_diameter_arcsec`
as well. During context construction, `grahspj` defines the effective spatial
scale for each band as:

```python
effective_scale = aperture_diameter_arcsec if finite else psf_fwhm_arcsec
```

When `LikelihoodConfig(use_host_capture_model=True)` and host fitting are both
enabled, the model fits a smooth capture fraction for the host component as a
function of that effective scale. Internally this is a sigmoid in
`log(effective_scale)` with two sampled parameters:

- `log_host_capture_scale_arcsec`, the turnover scale, default prior centered near `log(3 arcsec)`
- `host_capture_slope`, the transition sharpness, default prior centered near `2`

The AGN point-source component is not scaled by this factor. The raw model is
first projected through each filter; then only the host contribution is adjusted:

```python
model_flux = total_flux - host_flux + host_capture_fraction * host_flux
```

The likelihood then compares this PSF-aware model flux to the observed fluxes
using the usual Student-t photometric likelihood, including measurement errors,
fractional model systematics, optional intrinsic scatter, and optional AGN
variability variance. If no finite PSF/aperture sizes are provided, or
`use_host_capture_model=False`, every band uses `host_capture_fraction = 1` and
the fit reduces to the standard integrated-flux likelihood.

## Chimera benchmark

The Chimera benchmark is intended as a regression and calibration tool for this experimental port, not as a finalized scientific validation of full `GRAHSP`/`CIGALE` parity.

`dsps_ssp_fn` must point to a valid DSPS SSP HDF5 file. Additional SPS template files, including variants with nebular grids, are available at `https://halos.as.arizona.edu/suchethacooray/dsps_ssp/`. At present, `grahspj` cannot vary nebular parameters independently beyond whatever is baked into the selected DSPS template, but this is expected to be sufficient for most broad-band fitting use cases.

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
