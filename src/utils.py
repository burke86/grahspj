from __future__ import annotations

from io import BytesIO
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import quote_plus

import numpy as np
from astropy.table import Table

# PSF / image-resolution references used below:
# - GALEX mission overview: FUV/NUV spatial resolutions 4.3"/5.3" FWHM
#   https://galex.stsci.edu/doc/CTM/wiki/Public_documentation/Chapter_2.html
# - SDSS DR17 imaging data quality: median seeing by band in arcsec
#   https://www.sdss4.org/dr17/imaging/other_info/
# - 2MASS beam characterization: Atlas/coadd images have FWHM typically 2.5" in good seeing
#   https://irsa.ipac.caltech.edu/data/2MASS/docs/supplementary/xsc/jarrett_PSF.html
# - WISE PSF major-axis FWHM by band
#   https://irsa.ipac.caltech.edu/data/WISE/docs/release/Prelim/expsup/sec4_3c.html
#
# These are survey-level nominal values, not per-source PSFs from the VizieR SED service.
FILTER_MAP = {
    "GALEX:FUV": {"grahsp_filter": "FUV_galex", "speclite_name": "galex-fuv", "psf_fwhm_arcsec": 4.3},
    "GALEX:NUV": {"grahsp_filter": "NUV_galex", "speclite_name": "galex-nuv", "psf_fwhm_arcsec": 5.3},
    "Johnson:B": {"grahsp_filter": "B_johnson", "speclite_name": "bessell-B", "psf_fwhm_arcsec": None},
    "Johnson:V": {"grahsp_filter": "V_johnson", "speclite_name": "bessell-V", "psf_fwhm_arcsec": None},
    "SDSS:u": {"grahsp_filter": "u_sdss", "speclite_name": "sdss2010-u", "psf_fwhm_arcsec": 1.53},
    "SDSS:g": {"grahsp_filter": "g_sdss", "speclite_name": "sdss2010-g", "psf_fwhm_arcsec": 1.44},
    "SDSS:r": {"grahsp_filter": "r_sdss", "speclite_name": "sdss2010-r", "psf_fwhm_arcsec": 1.32},
    "SDSS:i": {"grahsp_filter": "i_sdss", "speclite_name": "sdss2010-i", "psf_fwhm_arcsec": 1.26},
    "SDSS:z": {"grahsp_filter": "z_sdss", "speclite_name": "sdss2010-z", "psf_fwhm_arcsec": 1.29},
    "2MASS:J": {"grahsp_filter": "J_2mass", "speclite_name": "twomass-J", "psf_fwhm_arcsec": 2.5},
    "2MASS:H": {"grahsp_filter": "H_2mass", "speclite_name": "twomass-H", "psf_fwhm_arcsec": 2.5},
    "2MASS:Ks": {"grahsp_filter": "Ks_2mass", "speclite_name": "twomass-Ks", "psf_fwhm_arcsec": 2.5},
    "2MASS:K": {"grahsp_filter": "Ks_2mass", "speclite_name": "twomass-Ks", "psf_fwhm_arcsec": 2.5},
    "WISE:W1": {"grahsp_filter": "W1", "speclite_name": "wise2010-W1", "psf_fwhm_arcsec": 6.08},
    "WISE:W2": {"grahsp_filter": "W2", "speclite_name": "wise2010-W2", "psf_fwhm_arcsec": 6.84},
    "WISE:W3": {"grahsp_filter": "W3", "speclite_name": "wise2010-W3", "psf_fwhm_arcsec": 7.36},
    "WISE:W4": {"grahsp_filter": "W4", "speclite_name": "wise2010-W4", "psf_fwhm_arcsec": 11.99},
}


def _as_float(value):
    if np.ma.is_masked(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def query_vizier_sed(
    target: str,
    radius: float = 3.0,
    host: tuple[str, int, bool] = ("vizier.cfa.harvard.edu", 443, True),
    fallback_hosts: list[tuple[str, int, bool]] | tuple[tuple[str, int, bool], ...] | None = None,
    timeout: float = 120.0,
    verbose: bool = True,
) -> tuple[list[dict[str, float | str | None]], Table, str]:
    """Query the VizieR SED service and return cleaned rows, the raw table, and the source URL.

    Parameters
    ----------
    target
        Target name or coordinate string accepted by the VizieR SED service.
    radius
        Search radius in arcseconds.
    host
        Primary host triple of ``(hostname, port, use_https)``.
    fallback_hosts
        Optional additional host triples tried in order after ``host`` fails.
    timeout
        Socket timeout in seconds for each host attempt.
    verbose
        If True, print per-host failure messages.
    """
    hosts = [host]
    if fallback_hosts:
        hosts.extend(list(fallback_hosts))

    path = "/viz-bin/sed?-c={target:s}&-c.rs={radius:f}".format(
        target=quote_plus(target),
        radius=radius,
    )

    last_error = None
    for hostname, port, use_https in hosts:
        connection = None
        connection_cls = HTTPSConnection if use_https else HTTPConnection
        scheme = "https" if use_https else "http"
        source_url = f"{scheme}://{hostname}{path}"
        try:
            connection = connection_cls(hostname, port, timeout=timeout)
            connection.request("GET", path)
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} {response.reason}")
            sed_table = Table.read(BytesIO(payload), format="votable")

            selected = {}
            for row in sed_table:
                sed_filter = str(row["sed_filter"]).strip()
                if sed_filter not in FILTER_MAP:
                    continue
                flux_jy = _as_float(row["sed_flux"])
                err_jy = _as_float(row["sed_eflux"])
                freq_ghz = _as_float(row["sed_freq"])
                if not np.isfinite(flux_jy) or flux_jy <= 0.0:
                    continue
                if not np.isfinite(err_jy) or err_jy <= 0.0:
                    continue
                filter_info = FILTER_MAP[sed_filter]
                grahsp_name = str(filter_info["grahsp_filter"])
                speclite_name = str(filter_info["speclite_name"])
                frac_err = err_jy / flux_jy
                candidate = {
                    "vizier_filter": sed_filter,
                    "grahsp_filter": grahsp_name,
                    "speclite_name": speclite_name,
                    "psf_fwhm_arcsec": filter_info["psf_fwhm_arcsec"],
                    "freq_ghz": freq_ghz,
                    "flux_mjy": 1.0e3 * flux_jy,
                    "err_mjy": 1.0e3 * err_jy,
                    "frac_err": frac_err,
                    "catalog": str(row["_tabname"]).strip() if "_tabname" in row.colnames else "",
                }
                current = selected.get(grahsp_name)
                if current is None or candidate["frac_err"] < current["frac_err"]:
                    selected[grahsp_name] = candidate

            phot_rows = sorted(selected.values(), key=lambda r: r["freq_ghz"])
            if len(phot_rows) < 4:
                raise RuntimeError(f"Need at least 4 supported filters to fit; got {len(phot_rows)}")

            return phot_rows, sed_table, source_url
        except Exception as exc:
            last_error = exc
            if verbose:
                print(f"Query failed for {source_url}: {type(exc).__name__}: {exc}")
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    raise RuntimeError("All VizieR mirrors failed for the SED query.") from last_error
