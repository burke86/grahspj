Quickstart
==========

The high-level interface is :class:`jaxsedfit.core.JAXSEDFit`.

.. code-block:: python

   from jaxsedfit.core import JAXSEDFit

   fitter = JAXSEDFit(cfg)
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

Nested sampling is available through NumPyro's ``jaxns`` wrapper:

.. code-block:: python

   fitter.fit(
       fit_method="ns",
       ns_live_points=200,
       ns_dlogz=0.1,
   )

The component SED plot can also be generated directly:

.. code-block:: python

   from jaxsedfit.plotting import plot_fit_sed

   plot_fit_sed(fitter, output_path="sed_fit.png")
