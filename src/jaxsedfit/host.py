from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from .model import _build_host_state, _host_rest_on_basis
from .preload import (
    HostBasisJax,
    _build_host_basis,
    _build_host_basis_jax,
    load_cached_ssp_data,
)


def build_host_basis_jax(
    rest_wave: np.ndarray,
    *,
    dsps_ssp_fn: str = "tempdata.h5",
    ssp_imf: str = "kroupa_2001",
    t_obs_gyr: float,
    sfh_n_steps: int = 64,
    sfh_t_min_gyr: float = 0.01,
) -> HostBasisJax:
    """Build the JAXSEDFit physical SSP host basis on a rest-frame grid.

    The returned basis preserves the DSPS physical SSP luminosity and mass
    normalization used by JAXSEDFit. It is intended for reuse by spectral
    fitting code that needs host-galaxy SFH parity with JAXSEDFit.

    Parameters
    ----------
    rest_wave : object
        rest_wave value.
    dsps_ssp_fn : object
        dsps_ssp_fn value.
    ssp_imf : str
        IMF provenance of the SSP library and surviving-mass calibration.
    t_obs_gyr : object
        t_obs_gyr value.
    sfh_n_steps : object
        sfh_n_steps value.
    sfh_t_min_gyr : object
        sfh_t_min_gyr value.
    """
    rest_wave = np.asarray(rest_wave, dtype=float)
    if rest_wave.ndim != 1 or rest_wave.size < 2:
        raise ValueError("rest_wave must be a one-dimensional wavelength grid.")
    ssp_data = load_cached_ssp_data(dsps_ssp_fn)
    gal_t_table = np.geomspace(
        max(float(sfh_t_min_gyr), 1e-3),
        max(float(t_obs_gyr), float(sfh_t_min_gyr) * 1.01),
        int(sfh_n_steps),
    )
    host_basis = _build_host_basis(rest_wave, ssp_data, ssp_imf)
    return _build_host_basis_jax(ssp_data, host_basis, gal_t_table)


def build_host_state(
    host_basis_jax: HostBasisJax,
    prior_config: dict[str, Any],
    *,
    host_sfh_model: str = "delayed",
    t_obs_gyr: float,
    redshift: float = 0.0,
    sfh_t_min_gyr: float = 0.01,
    tau_host_prior_scale: float = 0.5,
    ssp_metallicity_coordinate: str = "absolute_log10_z",
    ssp_solar_metallicity: float = 0.019,
    stellar_metallicity: float = 0.019,
    stellar_metallicity_scatter: float = 0.0,
    full_output: bool = True,
) -> dict[str, Any]:
    """Sample/build the JAXSEDFit physical host state from a host basis.

    This is a lightweight public wrapper around the same delayed-tau and
    Diffstar host implementation used internally by JAXSEDFit. The returned
    state contains the physical SSP weights, host rest-frame luminosity SED,
    stellar mass, surviving mass fraction, SFH diagnostics, and metallicity
    diagnostics produced by the underlying JAXSEDFit host model.

    Parameters
    ----------
    host_basis_jax : object
        host_basis_jax value.
    prior_config : object
        prior_config value.
    host_sfh_model : object
        host_sfh_model value.
    t_obs_gyr : object
        t_obs_gyr value.
    redshift : object
        redshift value.
    sfh_t_min_gyr : object
        sfh_t_min_gyr value.
    tau_host_prior_scale : object
        tau_host_prior_scale value.
    ssp_metallicity_coordinate : str
        Coordinate used by ``host_basis_jax.ssp_lgmet``.
    ssp_solar_metallicity : float
        Absolute solar metallicity used for coordinate conversion.
    stellar_metallicity : float
        Fixed absolute stellar metallicity used when no metallicity prior is supplied.
    stellar_metallicity_scatter : float
        Fixed stellar metallicity dispersion in dex when no scatter prior is supplied.
    full_output : object
        full_output value.
    """
    galaxy = SimpleNamespace(
        host_sfh_model=str(host_sfh_model),
        sfh_t_min_gyr=float(sfh_t_min_gyr),
        tau_host_prior_scale=float(tau_host_prior_scale),
        ssp_metallicity_coordinate=str(ssp_metallicity_coordinate),
        ssp_solar_metallicity=float(ssp_solar_metallicity),
        stellar_metallicity=float(stellar_metallicity),
        stellar_metallicity_scatter=float(stellar_metallicity_scatter),
    )
    observation = SimpleNamespace(redshift=float(redshift))
    context = SimpleNamespace(
        host_basis_jax=host_basis_jax,
        t_obs_gyr=float(t_obs_gyr),
        fit_config=SimpleNamespace(galaxy=galaxy, observation=observation),
    )
    return _build_host_state(context, prior_config, full_output=full_output)


def host_rest_on_basis(host_state: dict[str, Any], host_basis_jax: HostBasisJax):
    """Evaluate a sampled JAXSEDFit host state on a compatible host basis.

    Parameters
    ----------
    host_state : object
        host_state value.
    host_basis_jax : object
        host_basis_jax value.
    """
    return _host_rest_on_basis(host_state, host_basis_jax)


__all__ = [
    "HostBasisJax",
    "build_host_basis_jax",
    "build_host_state",
    "host_rest_on_basis",
]
