API Reference
=============

.. autosummary::
   :toctree: generated
   :recursive:

   jaxsedfit.JAXSEDFit
   jaxsedfit.FitConfig
   jaxsedfit.Observation
   jaxsedfit.PhotometryData
   jaxsedfit.SpectroscopyData
   jaxsedfit.FilterCurve
   jaxsedfit.FilterSet
   jaxsedfit.GalaxyConfig
   jaxsedfit.AGNConfig
   jaxsedfit.NebularConfig
   jaxsedfit.LikelihoodConfig
   jaxsedfit.InferenceConfig
   jaxsedfit.PriorConfig
   jaxsedfit.RedshiftPriorConfig
   jaxsedfit.StellarMassPriorConfig
   jaxsedfit.MassMetallicityPriorConfig
   jaxsedfit.plot_fit_sed
   jaxsedfit.plot_corner
   jaxsedfit.plot_trace
   jaxsedfit.load_from_samples
   jaxsedfit.style_path

Model Internals
---------------

Most users should interact with :class:`jaxsedfit.JAXSEDFit` and
:class:`jaxsedfit.FitConfig`. The modules below expose the lower-level
NumPyro model, static preload context, and reusable host-galaxy helpers used by
advanced integrations and tests.

Forward model
~~~~~~~~~~~~~

.. currentmodule:: jaxsedfit.model

.. autofunction:: grahsp_photometric_model

.. autofunction:: evaluate_photometric_state

.. autofunction:: photometric_loglike

.. autofunction:: spectroscopic_loglike

Host helpers
~~~~~~~~~~~~

.. currentmodule:: jaxsedfit.host

.. autofunction:: build_host_basis_jax

.. autofunction:: build_host_state

.. autofunction:: host_rest_on_basis

Preload context
~~~~~~~~~~~~~~~

.. currentmodule:: jaxsedfit.preload

.. autoclass:: ModelContext
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LoadedFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LoadedTemplates
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SSPData
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HostBasis
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: build_model_context

.. autofunction:: load_cached_ssp_data
