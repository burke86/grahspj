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
   fitter = JAXSEDFit(cfg)
   fitter.fit(
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

Nested sampling is available through NumPyro's ``jaxns`` wrapper:

.. code-block:: python

   cfg.inference.method = "ns"
   fitter.fit(
       ns_live_points=200,
       ns_dlogz=0.1,
   )

The component SED plot can also be generated directly:

.. code-block:: python

   from jaxsedfit.plotting import plot_fit_sed

   plot_fit_sed(fitter, output_path="sed_fit.png")
