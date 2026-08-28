Available telescopes and filters
================================

JAXSEDFit ships the filter transmission curves listed below. Pass the
canonical name—or one of the displayed aliases—to
:func:`jaxsedfit.filters.load_filter_curve`, or pass several names to
:func:`jaxsedfit.filters.load_filter_curves`:

.. code-block:: python

   from jaxsedfit.filters import load_filter_curves

   curves = load_filter_curves([
       "galex.FUV",
       "sloan.sdss.g",
       "jwst.nircam.F444W",
   ])

The pivot wavelengths are calculated from the packaged transmission curves.
This reference is generated directly from the filter registry during the
documentation build, so every packaged filter and public alias is included.

Adding a custom filter
----------------------

A custom filter does not need to be added to the package registry. Create a
:class:`jaxsedfit.FilterCurve`, place it in a :class:`jaxsedfit.FilterSet`, and
use exactly the same name in :class:`jaxsedfit.PhotometryData`:

.. code-block:: python

   import numpy as np
   from jaxsedfit import FilterCurve, FilterSet, PhotometryData
   from jaxsedfit.filters import load_filter_curves

   wave_angstrom = np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0])
   response = np.array([0.0, 0.35, 1.0, 0.40, 0.0])

   my_filter = FilterCurve(
       name="my_camera.r",
       wave=wave_angstrom,
       transmission=response,
   )

   # Built-in and custom curves can be used together.
   cfg.filters = FilterSet(
       curves=[
           *load_filter_curves(["galex.NUV", "sloan.sdss.g"]),
           my_filter,
       ]
   )
   cfg.photometry = PhotometryData(
       filter_names=["galex.NUV", "sloan.sdss.g", "my_camera.r"],
       fluxes=[0.12, 0.31, 0.44],       # mJy
       errors=[0.02, 0.03, 0.04],      # mJy
   )

Custom wavelengths must be observed-frame Angstrom values. The wavelength and
response arrays must be one-dimensional, have the same length, and contain at
least three finite, unique wavelength samples. JAXSEDFit sorts the samples,
clips negative responses to zero, sets the first and last response values to
zero, and calculates the pivot wavelength automatically. You can provide
``effective_wavelength`` explicitly, but normally it should be left as
``None`` so it is calculated from the normalized curve.

Loading a local response file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a whitespace-delimited file whose first two columns are wavelength in
Angstrom and response:

.. code-block:: python

   table = np.loadtxt("my_camera_r.dat")
   my_filter = FilterCurve(
       name="my_camera.r",
       wave=table[:, 0],
       transmission=table[:, 1],
   )
   cfg.filters = FilterSet(curves=[my_filter])

The inline ``transmission`` array is interpreted as the response used for
synthetic photometry. If the source file contains a photon-counting throughput,
convert it to the package's energy-response convention first:

.. code-block:: python

   photon_throughput = table[:, 1]
   energy_response = photon_throughput * table[:, 0]
   my_filter = FilterCurve(
       name="my_camera.r",
       wave=table[:, 0],
       transmission=energy_response,
   )

The custom curve name must match its entry in ``photometry.filter_names``.
Custom curves are fit-local; adding one this way does not modify the packaged
registry or make the name globally available to
:func:`jaxsedfit.filters.load_filter_curve`.

Packaged filter registry
------------------------

.. filter-registry::
