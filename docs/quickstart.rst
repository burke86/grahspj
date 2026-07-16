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

Aperture-aware likelihood
-------------------------

When ``cfg.likelihood.use_host_capture_model = True``, ``jaxsedfit`` modifies
the model prediction before evaluating the photometric and spectroscopic
likelihoods. The intrinsic SED is built first, then each observation gets its
own captured version of the extended components. Compact AGN continuum and
broad-line-like components are treated as unresolved; stellar host light, host
dust, nebular emission, and narrow-line-like components are treated as
extended.

For each photometric band :math:`b`, the effective angular scale is

.. math::

   \theta_b =
   \begin{cases}
   d_{{\rm ap},b}, & d_{{\rm ap},b}\ {\rm supplied}, \\
   {\rm FWHM}_{{\rm PSF},b}, & {\rm otherwise\ if\ supplied}, \\
   \infty, & {\rm otherwise}.
   \end{cases}

The captured extended-light fraction is a smooth logistic function,

.. math::

   \eta_b =
   \left[
   1 + \exp\left(
   -\alpha_{\rm cap}
   \left[\ln \theta_b - \ln \theta_{\rm cap}\right]
   \right)
   \right]^{-1},

where :math:`\theta_{\rm cap}` and :math:`\alpha_{\rm cap}` are inferred from
the priors ``log_host_capture_scale_arcsec`` and ``host_capture_slope``. Bands
without aperture or PSF metadata are treated as full-capture measurements,
:math:`\eta_b = 1`.

The model flux density compared to photometric band :math:`b` is therefore

.. math::

   F_b^{\rm model}
   =
   F_b^{\rm compact}
   + \eta_b F_b^{\rm extended}.

``jaxsedfit`` then applies the configured broadband likelihood to
:math:`F_b^{\rm model}`. For the default Gaussian family, detections are
approximately

.. math::

   F_b^{\rm obs}
   \sim
   \mathcal{N}
   \left(
   F_b^{\rm model},
   \sigma_{{\rm eff},b}
   \right),

with :math:`\sigma_{{\rm eff},b}` including the catalog flux-density error and
the configured systematic/model-error terms. If
``cfg.likelihood.likelihood_family = "student_t"``, the same captured model
flux is used inside the Student-t likelihood. Upper limits use the configured
one-sided photometric likelihood.

For joint spectrum+SED fitting, the spectrum gets its own capture fraction
from :class:`jaxsedfit.SpectroscopyData`:

.. code-block:: python

   cfg.spectroscopy_list[0].aperture_diameter_arcsec = 3.0  # SDSS fiber
   cfg.spectroscopy_config.fit_scale = True
   cfg.likelihood.use_host_capture_model = True

At each spectral pixel,

.. math::

   f_{\lambda}^{\rm spec}
   =
   s_{\rm spec}
   \left[
   f_{\lambda}^{\rm compact}
   + \eta_{\rm spec} f_{\lambda}^{\rm extended}
   \right],

where :math:`\eta_{\rm spec}` is computed from the fiber/slit aperture or PSF
metadata and :math:`s_{\rm spec}` is the optional gray spectral scale inferred
when ``cfg.spectroscopy_config.fit_scale = True``. This lets the broadband
photometry and the spectrum be different aperture views of the same intrinsic
source, rather than forcing PSF photometry, large-beam photometry, and fiber
spectroscopy to contain the same host fraction.

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
       ssp_imf="chabrier_2003",
       ssp_metallicity_coordinate="absolute_log10_z",
       ssp_solar_metallicity=0.019,
       rest_wave_min=100.0,
       rest_wave_max=3.0e6,
       n_wave=1024,
   )

``ssp_imf`` and ``ssp_metallicity_coordinate`` declare the provenance of the
loaded SSP library. They do not regenerate it. The IMF declaration also selects
the matching DSPS surviving-mass calibration, so it must describe the SSP file
accurately. Supported IMFs are ``chabrier_2003``, ``salpeter_1955``,
``kroupa_2001``, and ``van_dokkum_2008``. Supported metallicity coordinates are
``absolute_log10_z`` and ``log10_z_over_zsun``. Custom IMFs require an SSP format
that supplies its own surviving-mass fractions and are not silently approximated.

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
       zgas=0.019,
       ne=100.0,
       f_esc=0.0,
       f_dust=0.0,
       lines_width=300.0,
   )

By default, ``logU``, ``zgas``, and ``ne`` are fixed to the GRAHSP-like values
``-2.0``, ``0.019``, and ``100 cm^-3``, respectively. Set ``zgas=None`` to tie
the gas metallicity to the host metallicity proxy. Only add priors for
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

Useful options:

``likelihood_family``
   Use ``"gaussian"`` for the default normal likelihood or ``"student_t"`` for
   a heavier-tailed likelihood with ``student_t_df`` degrees of freedom.

``systematics_width``
   Fractional model-error floor. Fix it for simpler geometry, or sample it
   with ``fit_systematics_width=True`` when the data warrant it.

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

Photometric apertures, PSFs, and spectra
----------------------------------------

SED photometry often mixes measurements with different effective apertures:
SDSS PSF fluxes, 2MASS point-source photometry, AllWISE profile-fit
photometry, fixed-aperture measurements, catalog ``AUTO``/total-like fluxes,
and sometimes a spectrum from a fiber or slit. ``jaxsedfit`` handles this with
an empirical extended-light capture model. The model is disabled by default;
enable it when the photometry or spectroscopy does not all measure the same
aperture.

.. code-block:: python

   cfg.likelihood.use_host_capture_model = True

The capture model is driven by explicit angular-size metadata, not by the text
label in ``photometry_method``. For each photometric point, ``jaxsedfit`` uses
``aperture_diameter_arcsec`` when supplied; otherwise it uses
``psf_fwhm_arcsec``. If neither is supplied for a band, that band is treated as
total/full-capture photometry.

Use ``photometry_method`` to record what kind of catalog measurement was used:
``psf`` means point-source/PSF-like photometry, ``profile`` means profile-fit
photometry such as AllWISE ``W?mag``, ``aperture`` means an explicit fixed
aperture, ``auto`` means Kron/AUTO-like photometry, ``model``/``cmodel``/
``petrosian`` mean extended-source model measurements, and ``catalog`` is a
fallback for catalog fluxes whose aperture semantics are not known.

.. code-block:: python

   cfg.photometry = PhotometryData(
       filter_names=[
           "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss",
           "J_2mass", "H_2mass", "Ks_2mass", "W1", "W2",
       ],
       fluxes=[...],
       errors=[...],
       # Metadata labels are useful for provenance and plotting, but they do
       # not by themselves change the aperture model.
       photometry_method=[
           "psf", "psf", "psf", "psf", "psf",
           "psf", "psf", "psf", "profile", "profile",
       ],
       # SDSS and 2MASS point-source measurements use the PSF scale. AllWISE
       # profile-fit measurements use the WISE beam/profile scale.
       psf_fwhm_arcsec=[
           1.4, 1.4, 1.4, 1.4, 1.4,
           2.5, 2.5, 2.5, 6.08, 6.84,
       ],
       aperture_diameter_arcsec=[None] * 10,
   )

Internally, the model builds intrinsic total source components first. It then
applies an aperture-dependent capture fraction only to extended components
such as the stellar host, host dust, nebular emission, and narrow-line-like
emission. Compact AGN continuum and broad-line-like components are not reduced
by the host capture fraction. This is the intended behavior for SED-only fits:
small-PSF optical points can see less host light than large-beam infrared
points while still sharing one intrinsic physical source model.

For joint spectrum+SED fitting, provide the same kind of aperture metadata for
the spectrum. For example, an SDSS spectrum should usually use the 3 arcsec
fiber diameter:

.. code-block:: python

   from jaxsedfit import SpectroscopyData

   cfg.spectroscopy_list = [
       SpectroscopyData(
           wave_obs=wave_obs,
           fluxes=spec_flux,
           errors=spec_err,
           instrument="sdss",
           aperture_diameter_arcsec=3.0,
       )
   ]
   cfg.spectroscopy_config.enabled = True
   cfg.spectroscopy_config.fit_scale = True
   cfg.likelihood.use_host_capture_model = True

With this setup the photometry and spectrum are treated as different views of
the same intrinsic source:

* SDSS PSF photometry measures compact AGN light plus a PSF-captured fraction
  of the extended host.
* The SDSS spectrum measures compact AGN light plus a fiber-captured fraction
  of the extended host.
* 2MASS and AllWISE catalog photometry are finite-resolution measurements; they
  use their PSF/profile scales rather than being automatically treated as total
  host measurements.
* Larger-aperture or total-like catalog photometry can approach the full host
  contribution when supplied with large aperture metadata, or when no spatial
  scale is supplied.

``cfg.spectroscopy_config.fit_scale`` adds a gray spectral calibration/fiber
scale parameter on top of the component capture model. This is useful because
real spectra can have additional absolute calibration or slit-loss offsets
relative to broadband photometry.

When ``save_result=True`` or :meth:`jaxsedfit.FitResult.save` is used,
``jaxsedfit`` writes an HDF5 posterior bundle named
``<object_id>_samples.h5``. The bundle contains the fit config, posterior
samples, cached predictive outputs, and summary metadata.

Photometric apertures, PSFs, and spectra
----------------------------------------

SED photometry often mixes measurements with different effective apertures:
SDSS PSF fluxes, 2MASS point-source photometry, AllWISE profile-fit
photometry, fixed-aperture measurements, catalog ``AUTO``/total-like fluxes,
and sometimes a spectrum from a fiber or slit. ``jaxsedfit`` handles this with
an empirical extended-light capture model. The model is disabled by default;
enable it when the photometry or spectroscopy does not all measure the same
aperture.

.. code-block:: python

   cfg.likelihood.use_host_capture_model = True

The capture model is driven by explicit angular-size metadata, not by the text
label in ``photometry_method``. For each photometric point, ``jaxsedfit`` uses
``aperture_diameter_arcsec`` when supplied; otherwise it uses
``psf_fwhm_arcsec``. If neither is supplied for a band, that band is treated as
total/full-capture photometry.

Use ``photometry_method`` to record what kind of catalog measurement was used:
``psf`` means point-source/PSF-like photometry, ``profile`` means profile-fit
photometry such as AllWISE ``W?mag``, ``aperture`` means an explicit fixed
aperture, ``auto`` means Kron/AUTO-like photometry, ``model``/``cmodel``/
``petrosian`` mean extended-source model measurements, and ``catalog`` is a
fallback for catalog fluxes whose aperture semantics are not known.

.. code-block:: python

   cfg.photometry = PhotometryData(
       filter_names=[
           "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss",
           "J_2mass", "H_2mass", "Ks_2mass", "W1", "W2",
       ],
       fluxes=[...],
       errors=[...],
       # Metadata labels are useful for provenance and plotting, but they do
       # not by themselves change the aperture model.
       photometry_method=[
           "psf", "psf", "psf", "psf", "psf",
           "psf", "psf", "psf", "profile", "profile",
       ],
       # SDSS and 2MASS point-source measurements use the PSF scale. AllWISE
       # profile-fit measurements use the WISE beam/profile scale.
       psf_fwhm_arcsec=[
           1.4, 1.4, 1.4, 1.4, 1.4,
           2.5, 2.5, 2.5, 6.08, 6.84,
       ],
       aperture_diameter_arcsec=[None] * 10,
   )

Internally, the model builds intrinsic total source components first. It then
applies an aperture-dependent capture fraction only to extended components
such as the stellar host, host dust, nebular emission, and narrow-line-like
emission. Compact AGN continuum and broad-line-like components are not reduced
by the host capture fraction. This is the intended behavior for SED-only fits:
small-PSF optical points can see less host light than large-beam infrared
points while still sharing one intrinsic physical source model.

For joint spectrum+SED fitting, provide the same kind of aperture metadata for
the spectrum. For example, an SDSS spectrum should usually use the 3 arcsec
fiber diameter:

.. code-block:: python

   from jaxsedfit import SpectroscopyData

   cfg.spectroscopy_list = [
       SpectroscopyData(
           wave_obs=wave_obs,
           fluxes=spec_flux,
           errors=spec_err,
           instrument="sdss",
           aperture_diameter_arcsec=3.0,
       )
   ]
   cfg.spectroscopy_config.enabled = True
   cfg.spectroscopy_config.fit_scale = True
   cfg.likelihood.use_host_capture_model = True

With this setup the photometry and spectrum are treated as different views of
the same intrinsic source:

* SDSS PSF photometry measures compact AGN light plus a PSF-captured fraction
  of the extended host.
* The SDSS spectrum measures compact AGN light plus a fiber-captured fraction
  of the extended host.
* 2MASS and AllWISE catalog photometry are finite-resolution measurements; they
  use their PSF/profile scales rather than being automatically treated as total
  host measurements.
* Larger-aperture or total-like catalog photometry can approach the full host
  contribution when supplied with large aperture metadata, or when no spatial
  scale is supplied.

``cfg.spectroscopy_config.fit_scale`` adds a gray spectral calibration/fiber
scale parameter on top of the component capture model. This is useful because
real spectra can have additional absolute calibration or slit-loss offsets
relative to broadband photometry.

AGN type and SED line branches
------------------------------

The native SED-scale AGN component is configured through
:class:`jaxsedfit.AGNConfig`. The ``agn_type`` field selects which empirical
AGN emission-line branch is used for broadband line corrections:

.. code-block:: python

   from jaxsedfit import AGNConfig

   cfg.agn = AGNConfig(agn_type=1)

``agn_type=1`` is the broad-line AGN branch. It includes the BLAGN broad-line
template, the Seyfert-2-like narrow-line template, and allows native SED-scale
Fe II and Balmer-continuum components when those switches are enabled.

``agn_type=2`` is the narrow-line/Seyfert-2 branch. It excludes broad AGN
lines and uses the Seyfert-2-like narrow-line template. Native Fe II and
Balmer-continuum components are not used for this branch.

``agn_type=3`` is the LINER branch. It excludes broad AGN lines and uses the
LINER narrow-line template. This is a line-ratio/template choice for
low-ionization narrow-line AGN; it is not a general ``type 3 quasar`` category.

The ``agn_type`` setting affects the native jaxsedfit SED-scale AGN line
corrections. When spectroscopy uses the ``jaxqsofit`` backend, the detailed
spectral Fe II, Balmer continuum, and emission-line model is controlled by the
``jaxqsofit`` configuration and line table. In joint fits, jaxqsofit should own
lines covered by the spectrum, while jaxsedfit's SED-scale line corrections are
most useful for broadband filters outside the spectral coverage.

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
