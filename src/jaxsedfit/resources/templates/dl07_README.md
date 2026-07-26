# Draine & Li (2007) template provenance

`dl07_templates.h5` contains the Milky-Way `R_V=3.1` Draine & Li (2007)
dust-emission grid distributed by B. T. Draine:

https://www.astro.princeton.edu/~draine/dust/irem4/DL07spec.tgz

The grid contains single-`U_min` spectra and `U^-2` spectra extending from
`U_min` to the fixed `U_max=10^6`, for the published `q_PAH` values. Each
spectrum is stored on the native wavelength grid and normalized to unit
wavelength integral; jaxsedfit restores the relative luminosity of the
power-law component before enforcing host-galaxy energy balance.

Reference: Draine, B. T. & Li, A. 2007, ApJ, 657, 810.
