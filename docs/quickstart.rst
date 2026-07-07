Quickstart
==========

``jaxsedfit`` is configured through a single :class:`jaxsedfit.FitConfig`
object. The top-level config groups observation metadata, broadband
photometry, optional spectroscopy, filter definitions, galaxy and AGN model
options, likelihood settings, inference settings, output behavior, and
optional prior overrides. Build the config first, pass it to
:class:`jaxsedfit.JAXSEDFit`, and then call
:meth:`jaxsedfit.JAXSEDFit.fit`.

Minimal SED-only example
------------------------

Photometric ``fluxes`` and ``errors`` are flux densities in mJy. The
``filter_names`` must either match packaged ``jaxsedfit`` filter names or
be supplied explicitly through :class:`jaxsedfit.FilterSet`.

.. code-block:: python

   from jaxsedfit import (
       FitConfig,
       GalaxyConfig,
       InferenceConfig,
       JAXSEDFit,
       Observation,
       OutputConfig,
       PhotometryData,
   )

   cfg = FitConfig(
       observation=Observation(object_id="demo", redshift=0.1),
       photometry=PhotometryData(
           filter_names=["u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss"],
           fluxes=[0.22, 0.48, 0.73, 0.86, 0.91],
           errors=[0.03, 0.04, 0.05, 0.06, 0.07],
       ),
       galaxy=GalaxyConfig(dsps_ssp_fn="tempdata.h5"),
       inference=InferenceConfig(
           method="optax+nuts",
           map_steps=600,
           num_warmup=100,
           num_samples=100,
           num_chains=1,
           dense_mass=False,
           max_tree_depth=8,
       ),
       output=OutputConfig(
           output_dir="fit_outputs",
           plot_fig=False,
           save_fig=True,
           save_result=True,
       ),
   )

   fitter = JAXSEDFit(cfg)
   result = fitter.fit()
   result.summary

``dsps_ssp_fn`` must point to a valid DSPS SSP HDF5 file. A missing SSP file
is the most common setup failure. For quick smoke tests, use
``cfg.inference.method = "optax"`` to run MAP optimization without posterior
sampling.

Photometry inputs
-----------------

The photometry payload carries both fluxes and aperture metadata:

.. code-block:: python

   cfg.photometry = PhotometryData(
       filter_names=["g_sdss", "r_sdss", "i_sdss", "W1", "W2"],
       fluxes=[0.48, 0.73, 0.86, 1.9, 1.6],
       errors=[0.04, 0.05, 0.06, 0.08, 0.08],
       is_upper_limit=[False, False, False, False, False],
       psf_fwhm_arcsec=[1.4, 1.3, 1.3, 6.1, 6.8],
       aperture_diameter_arcsec=[None, None, None, None, None],
       photometry_method=["psf", "psf", "psf", "catalog", "catalog"],
   )

``is_upper_limit``
   Marks one-sided photometric constraints.

``psf_fwhm_arcsec``
   Records the effective PSF size for each photometric point. This is useful
   when comparing compact and extended components or when using photometry
   gathered from heterogeneous surveys.

``aperture_diameter_arcsec``
   Records explicit aperture diameters when catalog fluxes come from a known
   circular aperture.

``photometry_method``
   Records provenance such as ``"psf"``, ``"aperture"``, ``"model"``, or
   ``"catalog"``. It is metadata for now, but it makes plots and downstream
   checks easier to interpret.

Custom filter curves can be supplied directly:

.. code-block:: python

   from jaxsedfit import FilterCurve, FilterSet

   cfg.filters = FilterSet(
       curves=[
           FilterCurve(
               name="my_filter",
               wave=[4000.0, 5000.0, 6000.0],
               transmission=[0.0, 1.0, 0.0],
           )
       ]
   )

Galaxy and SPS options
----------------------

The host-galaxy model is controlled by :class:`jaxsedfit.GalaxyConfig`.
The default star-formation history is ``host_sfh_model="delayed"``. It is
low-dimensional and usually the most stable choice for broadband SED fitting.

.. code-block:: python

   from jaxsedfit import GalaxyConfig

   cfg.galaxy = GalaxyConfig(
       fit_host=True,
       host_sfh_model="delayed",
       dsps_ssp_fn="tempdata.h5",
       rest_wave_min=100.0,
       rest_wave_max=3.0e6,
       n_wave=1024,
   )

Use ``host_sfh_model="diffstar"`` when you want the Diffstar-based SFH
parameterization:

.. code-block:: python

   cfg.galaxy.host_sfh_model = "diffstar"

The host model samples stellar mass, metallicity, dust, and SFH parameters.
Useful controls include:

``fit_host``
   Disable only for AGN-only experiments. At least one of ``fit_host`` or
   ``agn.fit_agn`` must be true.

``fit_host_kinematics``
   Enables host velocity and velocity-dispersion parameters for spectroscopic
   fits.

``n_wave``
   Sets the internal rest-frame wavelength grid size. Larger values are more
   faithful for detailed spectra but slower for NUTS.

``use_energy_balance`` and ``dust_alpha``
   Control the host-dust energy-balance component. Disable dust only for
   debugging or intentionally dust-free model comparisons.

Nebular emission
----------------

Host-galaxy nebular emission is configured separately from the stellar
continuum:

.. code-block:: python

   from jaxsedfit import NebularConfig

   cfg.nebular = NebularConfig(
       enabled=True,
       emission=True,
       logU=-2.0,
       zgas=None,
       ne=100.0,
       f_esc=0.0,
       f_dust=0.0,
       lines_width=300.0,
   )

By default, ``logU`` and ``ne`` are fixed to common values rather than sampled.
If ``zgas=None``, the gas metallicity follows the host metallicity proxy. Set
``zgas`` explicitly to hold the gas metallicity fixed. Only add priors for
``logU``, ``zgas``, or ``ne`` when the data can actually constrain them. The
packaged nebular templates are interpolated over metallicity, ionization
parameter, and density, so fixed or sampled off-grid values remain smooth for
gradient-based inference.

.. code-block:: python

   from numpyro import distributions as dist

   cfg.prior_config.nebular.logU = dist.Normal(-2.0, 0.3)
   cfg.prior_config.nebular.zgas = dist.TruncatedNormal(
       loc=0.02,
       scale=0.01,
       low=1.0e-4,
       high=0.1,
   )

``cfg.likelihood.use_local_line_photometry`` controls robust local
high-resolution line projections for photometric line corrections. Keep it
enabled for science runs. If you are diagnosing sampler geometry or running a
very fast smoke test, disabling it removes detailed local line corrections and
can make the model easier to sample.

AGN components
--------------

The AGN continuum, torus, broad/narrow AGN line corrections, Fe II, and Balmer
continuum are controlled by :class:`jaxsedfit.AGNConfig` and AGN prior fields.

.. code-block:: python

   from jaxsedfit import AGNConfig

   cfg.agn = AGNConfig(
       fit_agn=True,
       use_powerlaw_disk=True,
       agn_type=1,
       fit_feii_broadening=False,
       fit_balmer_continuum=False,
   )

For ordinary broadband SED fitting, the native ``jaxsedfit`` AGN components
provide SED-scale disk, torus, and broad/narrow line corrections. For joint
spectroscopic fitting, prefer the ``jaxqsofit`` backend for spectral Fe II,
Balmer continuum, and emission-line structure, while keeping the broadband
AGN continuum, torus, and dust emission in ``jaxsedfit``.

Likelihood and model-error settings
-----------------------------------

The likelihood defaults include a fractional systematics width to avoid
forcing broadband photometry to constrain a model more tightly than the
templates justify:

.. code-block:: python

   cfg.likelihood.likelihood_family = "gaussian"
   cfg.likelihood.systematics_width = 0.10
   cfg.likelihood.fit_systematics_width = True
   cfg.likelihood.use_absolute_flux_scale_prior = True

Useful options:

``likelihood_family``
   Use ``"gaussian"`` for the default normal likelihood or ``"student_t"`` for
   a heavier-tailed likelihood with ``student_t_df`` degrees of freedom.

``systematics_width``
   Fractional model-error floor. Fix it for simpler geometry, or sample it
   with ``fit_systematics_width=True`` when the data warrant it.

``use_absolute_flux_scale_prior``
   Adds a broad prior on absolute scale so mass, AGN amplitude, and systematic
   scatter do not drift into implausible combinations.

``variability_uncertainty``
   Adds an AGN variability term to photometric uncertainty when AGN emission is
   present.

If NUTS repeatedly reaches maximum tree depth, first try a simpler geometry:
fix ``systematics_width`` at a reasonable value, keep ``dense_mass=False``,
use ``max_tree_depth=8``, and run an ``optax`` fit to verify the MAP solution.

Inference recipes
-----------------

Fast MAP-only smoke test:

.. code-block:: python

   cfg.inference.method = "optax"
   cfg.inference.map_steps = 600
   cfg.inference.learning_rate = 5.0e-3

Standard posterior run:

.. code-block:: python

   cfg.inference.method = "optax+nuts"
   cfg.inference.use_map_init = True
   cfg.inference.num_warmup = 200
   cfg.inference.num_samples = 200
   cfg.inference.num_chains = 1
   cfg.inference.target_accept_prob = 0.85
   cfg.inference.dense_mass = False
   cfg.inference.max_tree_depth = 8

Nested sampling:

.. code-block:: python

   cfg.inference.method = "ns"
   cfg.inference.ns_num_live_points = 200
   cfg.inference.ns_dlogz = 0.1

Use fewer warmup and sample draws for notebooks and CI. Increase them for
published posterior summaries.

Joint spectroscopy and photometry
---------------------------------

Add spectroscopy with :class:`jaxsedfit.SpectroscopyData`. Spectroscopic
``fluxes`` and ``errors`` are observed-frame flux densities in mJy on the
``wave_obs`` grid.

.. code-block:: python

   import numpy as np
   from jaxsedfit import SpectroscopyConfig, SpectroscopyData

   cfg.spectroscopy = SpectroscopyData(
       wave_obs=np.linspace(3800.0, 9200.0, 2000),
       fluxes=np.full(2000, 1.0),
       errors=np.full(2000, 0.05),
       instrument="sdss",
       aperture_diameter_arcsec=3.0,
   )
   cfg.spectroscopy_config = SpectroscopyConfig(
       enabled=True,
       backend="jaxsedfit",
       fit_scale=True,
       likelihood_weight_mode="pixels",
   )

Use ``backend="jaxsedfit"`` for a self-contained SED/spectrum model. Use
``backend="jaxqsofit"`` when the spectrum needs quasar-style Fe II, Balmer
continuum, and tied emission-line modeling:

.. code-block:: python

   from jaxsedfit import JaxQSOFitConfig

   cfg.spectroscopy_config.backend = "jaxqsofit"
   cfg.spectroscopy_config.likelihood_weight_mode = "resolving_power"
   cfg.spectroscopy_config.resolving_power = 2000.0
   cfg.spectroscopy_config.jaxqsofit = JaxQSOFitConfig(
       use_spectral_lines=True,
       use_spectral_feii=True,
       use_spectral_balmer_continuum=True,
       include_elg_narrow_lines=True,
   )

When the spectrum comes from a fiber or slit, set
``aperture_diameter_arcsec`` or ``psf_fwhm_arcsec`` on the spectroscopy
payload and enable the host-capture model:

.. code-block:: python

   cfg.likelihood.use_host_capture_model = True
   cfg.spectroscopy_config.fit_scale = True

This allows the spectroscopic host contribution to differ from broadband
photometry while keeping compact AGN light and extended host light treated
consistently.

Prior configuration
-------------------

Priors use NumPyro distribution objects through :class:`jaxsedfit.PriorConfig`.
Only fields you set are added to the model; defaults are used otherwise.

.. code-block:: python

   import numpy as np
   from numpyro import distributions as dist

   cfg.prior_config.stellar_mass = dist.Normal(10.5, 0.5)
   cfg.prior_config.host.log_ebv_gal = dist.TruncatedNormal(
       loc=np.log(0.08),
       scale=0.5,
       low=np.log(1.0e-4),
       high=np.log(2.0),
   )
   cfg.prior_config.agn.log_fcov = dist.TruncatedNormal(
       loc=np.log(0.3),
       scale=0.5,
       low=np.log(0.01),
       high=np.log(0.95),
   )
   cfg.prior_config.likelihood.log_systematics_width = dist.TruncatedNormal(
       loc=np.log(0.10),
       scale=0.15,
       low=np.log(0.05),
       high=np.log(0.20),
   )

Redshift can be sampled by switching the observation mode and optionally
supplying a tabulated redshift prior:

.. code-block:: python

   cfg.observation.redshift_mode = "fit"
   cfg.prior_config.redshift.z_grid = [0.08, 0.09, 0.10, 0.11, 0.12]
   cfg.prior_config.redshift.pdf = [0.1, 0.4, 1.0, 0.4, 0.1]

Results and plotting
--------------------

The fit result keeps posterior samples, prediction helpers, and save/plot
methods:

.. code-block:: python

   result = fitter.fit()
   samples = result.samples
   summary = result.summary
   prediction = result.predict()

   result.plot_corner()
   fitter.plot_sed(output_path="sed_fit.png")

When ``save_result=True`` or :meth:`jaxsedfit.FitResult.save` is used,
``jaxsedfit`` writes an HDF5 posterior bundle named
``<object_id>_samples.h5``. The bundle contains the fit config, posterior
samples, cached predictive outputs, and summary metadata.

Changing jaxqsofit broad-line components
----------------------------------------

When spectroscopy uses the ``jaxqsofit`` backend, the number of broad
Gaussian components is set by the line-table ``ngauss`` field. The default
``jaxqsofit`` line table uses names such as ``Ha_br``, ``Hb_br``, ``MgII_br``,
``CIV_br``, and ``Lya_br`` for broad components. Copy the table, adjust
``ngauss`` for the rows you want, and assign it before constructing
:class:`jaxsedfit.JAXSEDFit`.

.. code-block:: python

   from copy import deepcopy

   from jaxqsofit.defaults import DEFAULT_LINE_PRIOR_ROWS

   line_table = deepcopy(DEFAULT_LINE_PRIOR_ROWS)

   for row in line_table:
       if row["linename"] in {"Ha_br", "Hb_br", "MgII_br"}:
           row["ngauss"] = 3
       if row["linename"] in {"CIV_br", "Lya_br"}:
           row["ngauss"] = 2

   cfg.spectroscopy_config.backend = "jaxqsofit"
   cfg.spectroscopy_config.jaxqsofit.line_table = line_table
