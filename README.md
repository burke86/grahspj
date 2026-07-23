# JAXSEDFit

`JAXSEDFit` is a Bayesian SED fitting code for AGN and galaxies. It is an experimental JAX-based implementation of `CIGALE` and `GRAHSP`. It ports `GRAHSP`/`pcigale` model components into JAX/NumPyro and combines them with a JAX-native galaxy models based on `Diffstar` + `DSPS`.

JAXSEDFit also owns the shared differentiable quasar spectral engine used by
joint and standalone spectrum fits: tied emission lines, Fe II, Balmer
continuum, spectral priors, custom spectral components, and NumPyro geometry.
The `jaxqsofit` package provides the spectrum-focused interface on top of this
engine; JAXSEDFit itself does not depend on `jaxqsofit`.

Reusable spectral integrations should import the supported
`jaxsedfit.spectroscopy` API rather than private helpers from the underlying
`spectral_*` implementation modules.

Documentation: [https://jaxsedfit.readthedocs.io/](https://jaxsedfit.readthedocs.io/)

At a high level, `jaxsedfit` currently includes:

- a JAX/NumPyro fitting engine
- `Diffstar` + `DSPS` host-galaxy modeling
- JAX ports of selected `GRAHSP` AGN, attenuation, redshifting, and dust-emission components
- `pcigale`-style SED plotting
- a Chimera benchmark for stellar-mass recovery

## Install

`jaxsedfit` requires Python 3.10 or newer. First, clone this repository. Then:

```bash
python -m pip install .
curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5
```
`jaxsedfit` now also requires `jax_cosmo` and `setuptools` in the runtime environment so the redshift-dependent luminosity-distance path stays JAX-native during inference.

You will also need a continuum-only DSPS SSP template file such as `ssp_data_continuum_fsps_v3.2_lgmet_age.h5`, downloaded above, and then referenced from your configuration via `cfg.galaxy.dsps_ssp_fn`. The continuum-only template is preferred because `jaxsedfit` models nebular emission lines separately.

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
- run `JAXSEDFit.fit(...)`
- inspect summary outputs
- make the component SED plot

The Fairall 9 notebook shows how to:

- query broadband photometry from the VizieR SED service
- map supported survey filters into `jaxsedfit`
- build a manual `FitConfig`
- fit and plot the resulting AGN SED


## Usage

`jaxsedfit` includes a `pcigale`-style component SED plot that overlays:

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
from jaxsedfit.core import JAXSEDFit

cfg.inference.method = "optax+nuts"
cfg.inference.map_steps = 600
cfg.inference.learning_rate = 1e-2
cfg.inference.num_warmup = 50
cfg.inference.num_samples = 50
cfg.inference.num_chains = 1
cfg.inference.dense_mass = "blocks"
cfg.inference.max_tree_depth = 8
cfg.output.plot_fig = False
cfg.output.save_fig = True
cfg.output.save_result = True
cfg.output.output_dir = "fit_outputs"

fitter = JAXSEDFit(cfg)
fitter.fit()
```

Nested sampling is also available through NumPyro's `jaxns` wrapper:

```python
cfg.inference.method = "ns"
cfg.inference.ns_num_live_points = 200
cfg.inference.ns_dlogz = 0.1

fitter = JAXSEDFit(cfg)
fitter.fit()
```

The public API keeps run settings on `FitConfig`, especially under
`cfg.inference` and `cfg.output`.

or with the standalone helper:

```python
from jaxsedfit.plotting import plot_fit_sed

plot_fit_sed(fitter, output_path="sed_fit.png")
```

This uses the lazy predictive path, so the component spectra are generated when you first call `plot_sed()` or `plot_fit_sed(...)`.


## License and provenance

`jaxsedfit` is an experimental port of parts of `CIGALE` and `GRAHSP`.

Some model logic and several bundled resource files are derived from or closely based on `GRAHSP` / `pcigale`, which is distributed under the `CeCILL v2` license.

See:

- [LICENSES/CeCILL-v2.txt](LICENSES/CeCILL-v2.txt)
- [LICENSES/THIRD_PARTY_NOTICES.md](LICENSES/THIRD_PARTY_NOTICES.md)

Bundled third-party resources under [src/jaxsedfit/resources](src/jaxsedfit/resources) include per-directory provenance notes.

## Filters

`jaxsedfit` uses vendored GRAHSP/pcigale-style filter curves for synthetic photometry.

- Built-in aliases cover common legacy names such as `u_sdss -> sloan.sdss.u`, `J_2mass -> 2mass.J`, and `W1 -> wise.W1`
- Vendored photon-response filters are converted to the internal energy-response convention before projection
- Filters must be available inline or in the vendored package resources
- Inline curves are used directly as internal filter curves before synthetic photometry is computed

## Survey PSF Sizes In The Likelihood

Broad-band catalogs do not all measure the same physical light profile. A
GALEX, SDSS, 2MASS, WISE, or IRAC point has a different effective angular
resolution, and aperture photometry can capture a different fraction of extended
host-galaxy light than PSF-like photometry. `jaxsedfit` can account for this with
the optional host-capture likelihood model.

Pass one value per photometric point through `PhotometryData.psf_fwhm_arcsec`.
If an aperture diameter is known, pass `PhotometryData.aperture_diameter_arcsec`
as well. During context construction, `jaxsedfit` defines the effective spatial
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

`dsps_ssp_fn` must point to a valid DSPS SSP HDF5 file. Additional SPS template files, including variants with nebular grids, are available at `https://halos.as.arizona.edu/suchethacooray/dsps_ssp/`. At present, `jaxsedfit` cannot vary nebular parameters independently beyond whatever is baked into the selected DSPS template, but this is expected to be sufficient for most broad-band fitting use cases.

### Run from the CLI

```bash
jaxsedfit-benchmark --output-dir benchmark_outputs --dsps-ssp-fn tempdata.h5
```

You can also run it without installing the script entry point:

```bash
python -m jaxsedfit.benchmark --output-dir benchmark_outputs --dsps-ssp-fn tempdata.h5
```

Optional thresholds:

```bash
jaxsedfit-benchmark \
  --output-dir benchmark_outputs \
  --dsps-ssp-fn tempdata.h5 \
  --max-weighted-mae 3.0 \
  --max-abs-weighted-bias 2.0 \
  --min-finite-fraction 0.95
```

To run only a small deterministic prefix of the benchmark subset:

```bash
jaxsedfit-benchmark \
  --output-dir benchmark_outputs_small \
  --dsps-ssp-fn tempdata.h5 \
  --limit 5
```

### Run from Python

```python
from jaxsedfit.benchmark import run_chimera_mass_benchmark

result = run_chimera_mass_benchmark(
    output_dir="benchmark_outputs",
    dsps_ssp_fn="tempdata.h5",
    limit=5,
)
print(result["passed"])
print(result["metrics"])
```
