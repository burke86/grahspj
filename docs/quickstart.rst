Quickstart
==========

``jaxsedfit`` is configured through a single :class:`jaxsedfit.FitConfig`
object. The top-level config groups the observation metadata, photometry and
optional spectroscopy arrays, filter definitions, galaxy and AGN model options,
likelihood settings, inference settings, and optional prior overrides. Build
the config first, pass it to :class:`jaxsedfit.JAXSEDFit`, and then call
:meth:`jaxsedfit.JAXSEDFit.fit`.

.. code-block:: python

   from jaxsedfit import FitConfig, JAXSEDFit, Observation, PhotometryData

   cfg = FitConfig(
       observation=Observation(object_id="demo", redshift=0.1),
       photometry=PhotometryData(
           filter_names=["u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss"],
           fluxes=[0.22, 0.48, 0.73, 0.86, 0.91],
           errors=[0.03, 0.04, 0.05, 0.06, 0.07],
       ),
   )

   cfg.inference.method = "optax+nuts"
   cfg.inference.map_steps = 600
   cfg.inference.learning_rate = 1e-2
   cfg.inference.num_warmup = 50
   cfg.inference.num_samples = 50
   cfg.inference.num_chains = 1
   cfg.output.plot_fig = False
   cfg.output.save_fig = True
   cfg.output.save_result = True
   cfg.output.output_dir = "fit_outputs"

   fitter = JAXSEDFit(cfg)
   result = fitter.fit()
   result.summary
   result.plot_corner()

When ``save_result=True`` or :meth:`jaxsedfit.FitResult.save` is used,
``jaxsedfit`` writes an HDF5 posterior bundle named
``<object_id>_samples.h5``. The bundle contains the fit config, posterior
samples, cached predictive outputs, and summary metadata.

Nested sampling is available through NumPyro's ``jaxns`` wrapper:

.. code-block:: python

   cfg.inference.method = "ns"
   cfg.inference.ns_num_live_points = 200
   cfg.inference.ns_dlogz = 0.1

   fitter = JAXSEDFit(cfg)
   result = fitter.fit()

The component SED plot can also be generated directly:

.. code-block:: python

   prediction = result.predict()
   fitter.plot_sed(output_path="sed_fit.png")
