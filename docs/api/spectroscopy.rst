Spectroscopy Engine
===================

.. currentmodule:: jaxsedfit.spectroscopy

The functions on this page are the supported integration boundary for detailed
quasar spectral modeling. External fitters should use these names instead of
importing implementation helpers from the ``spectral_*`` modules.

.. autoclass:: SpectralComponentConfig
   :members:

.. autofunction:: build_spectral_prior_config

.. autofunction:: build_joint_tied_line_meta

.. autofunction:: evaluate_joint_spectral_components

.. autofunction:: render_joint_feature_state

.. autofunction:: line_complex_dense_mass_blocks

.. autofunction:: normal_lognormal_standardization_reparam
