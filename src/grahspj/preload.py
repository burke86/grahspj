from __future__ import annotations

# This module loads vendored resources and model assets used by grahspj.
# Some of those bundled resources originate from GRAHSP/pcigale template and
# filter data distributed under the CeCILL v2 license.
# See LICENSES/CeCILL-v2.txt, THIRD_PARTY_NOTICES.md, and the README files in
# src/grahspj/resources/ for provenance details.

import importlib.util
from importlib import resources
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from dustmaps.sfd import SFDQuery
import extinction
from speclite import filters as speclite_filters

from .config import EmissionLineTemplate, FeIITemplate, FilterCurve, FitConfig


@dataclass
class LoadedFilter:
    """One prepared filter response on the model evaluation grid."""
    name: str
    wave: np.ndarray
    transmission: np.ndarray
    effective_wavelength: float
    interp_indices: np.ndarray
    interp_weight: np.ndarray
    work_wave: np.ndarray


@dataclass
class LoadedTemplates:
    """Template arrays required by the supported AGN and host-dust components."""
    feii_wave: np.ndarray
    feii_lumin: np.ndarray
    line_wave: np.ndarray
    line_blagn: np.ndarray
    line_sy2: np.ndarray
    line_liner: np.ndarray
    dust_alpha_grid: np.ndarray
    dust_wave: np.ndarray
    dust_lumin: np.ndarray


@dataclass
class SSPData:
    """Raw DSPS SSP grids cached for repeated host-model construction."""
    ssp_lgmet: np.ndarray
    ssp_lg_age_gyr: np.ndarray
    ssp_wave: np.ndarray
    ssp_flux: np.ndarray


@dataclass
class ModelContext:
    """Static arrays and metadata required by one grahspj model evaluation."""
    fit_config: FitConfig
    rest_wave: np.ndarray
    obs_wave: np.ndarray
    ssp_data: SSPData
    t_obs_gyr: float
    luminosity_distance_m: float
    gal_t_table: np.ndarray
    filters: list[LoadedFilter]
    templates: LoadedTemplates
    fluxes: np.ndarray
    errors: np.ndarray
    upper_limits: np.ndarray
    data_mask: np.ndarray
    positive_detected_mask: np.ndarray
    mw_ebv: float


_SFD_QUERY_CACHE: dict[str, Any] = {}
_SSP_DATA_CACHE: dict[str, SSPData] = {}
_FILTER_RESPONSE_CACHE: dict[tuple[Any, ...], list[Any]] = {}
_TEMPLATE_CACHE: dict[tuple[Any, ...], LoadedTemplates] = {}
_DALE2014_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_DEFAULT_SPECLITE_NAME_MAP = {
    "u_sdss": "sdss2010-u",
    "g_sdss": "sdss2010-g",
    "r_sdss": "sdss2010-r",
    "i_sdss": "sdss2010-i",
    "z_sdss": "sdss2010-z",
    "J_2mass": "twomass-J",
    "H_2mass": "twomass-H",
    "Ks_2mass": "twomass-Ks",
    "W1": "wise2010-W1",
    "W2": "wise2010-W2",
}
_VENDORED_FILTER_FILES = {
    "IRAC1": "resources/filters/IRAC1.dat",
    "IRAC2": "resources/filters/IRAC2.dat",
}


def _load_ssp_templates(dsps_ssp_fn: str):
    """Load DSPS SSP templates from the configured HDF5 file."""
    from dsps import load_ssp_templates

    return load_ssp_templates(fn=dsps_ssp_fn)


def _get_sfd_query():
    """Return a cached dustmaps SFDQuery instance."""
    cache_key = "default"
    if cache_key not in _SFD_QUERY_CACHE:
        _SFD_QUERY_CACHE[cache_key] = SFDQuery()
    return _SFD_QUERY_CACHE[cache_key]


def _sanitize_speclite_token(value: str, prefix: str) -> str:
    """Normalize a token for safe use in generated speclite filter names."""
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    if not token:
        token = prefix
    if token[0].isdigit():
        token = f"{prefix}_{token}"
    return token


def _as_angstrom_values(values) -> np.ndarray:
    """Convert wavelength-like values to a float Angstrom array."""
    if hasattr(values, "to_value"):
        return np.asarray(values.to_value(u.AA), dtype=float)
    return np.asarray(values, dtype=float)


def _scalar_angstrom_value(value) -> float:
    """Convert one wavelength-like scalar to Angstrom units."""
    if hasattr(value, "to_value"):
        return float(value.to_value(u.AA))
    return float(value)


def _package_resource_path(relpath: str) -> Path:
    """Return an absolute path to a packaged grahspj resource."""
    return Path(str(resources.files("grahspj").joinpath(relpath)))


def _mw_band_attenuation_factor(wave_obs, filt_trans, ebv, r_v=3.1):
    """Compute the Milky Way attenuation factor integrated through one bandpass."""
    wave_obs = np.asarray(wave_obs, dtype=float)
    filt_trans = np.clip(np.asarray(filt_trans, dtype=float), 0.0, None)
    if (not np.isfinite(ebv)) or ebv == 0.0:
        return 1.0
    a_lambda = extinction.fitzpatrick99(wave_obs, a_v=float(r_v) * float(ebv), r_v=float(r_v))
    attenuation = 10.0 ** (-0.4 * np.asarray(a_lambda, dtype=float))
    inv_wave = 1.0 / np.clip(wave_obs, 1e-8, None)
    denom = float(np.trapezoid(filt_trans * inv_wave, wave_obs))
    numer = float(np.trapezoid(filt_trans * attenuation * inv_wave, wave_obs))
    if denom <= 0.0 or numer <= 0.0:
        return 1.0
    return numer / denom


def _map_logzsol_to_dsps_lgmet(logzsol_grid: Sequence[float], ssp_lgmet: np.ndarray) -> np.ndarray:
    """Map log(Z/Zsun) fitting values onto the DSPS metallicity convention."""
    logzsol_grid = np.asarray(logzsol_grid, dtype=float)
    ssp_lgmet = np.asarray(ssp_lgmet, dtype=float)
    cand_direct = logzsol_grid
    cand_shifted = logzsol_grid + np.log10(0.019)

    def mismatch(cand):
        return np.mean([np.min(np.abs(ssp_lgmet - val)) for val in cand])

    return cand_direct if mismatch(cand_direct) <= mismatch(cand_shifted) else cand_shifted


def load_cached_ssp_data(dsps_ssp_fn: str) -> SSPData:
    """Load DSPS SSP data once and cache it by input file path."""
    cache_key = str(Path(dsps_ssp_fn).expanduser().resolve())
    cached = _SSP_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ssp_data = _load_ssp_templates(dsps_ssp_fn)
    loaded = SSPData(
        ssp_lgmet=np.asarray(ssp_data.ssp_lgmet, dtype=float),
        ssp_lg_age_gyr=np.asarray(ssp_data.ssp_lg_age_gyr, dtype=float),
        ssp_wave=np.asarray(ssp_data.ssp_wave, dtype=float),
        ssp_flux=np.asarray(ssp_data.ssp_flux, dtype=float),
    )
    _SSP_DATA_CACHE[cache_key] = loaded
    return loaded


def _locate_grahsp_repo() -> Path | None:
    """Best-effort lookup for a local GRAHSP checkout used by optional fallbacks."""
    candidates = []
    env = Path(str(Path.cwd()))
    candidates.append(env / "GRAHSP")
    candidates.append(Path(__file__).resolve().parents[3] / "GRAHSP")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_pcigale_database_cls():
    """Import and return the optional pcigale Database class if available."""
    try:
        from pcigale.data import Database  # type: ignore
        return Database
    except Exception:
        repo = _locate_grahsp_repo()
        if repo is None:
            return None
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from pcigale.data import Database  # type: ignore
            return Database
        except Exception:
            return None


def _curve_to_speclite_filter(curve: FilterCurve, group_name: str) -> speclite_filters.FilterResponse:
    """Wrap an inline filter curve into a speclite FilterResponse."""
    wave = np.asarray(curve.wave, dtype=float)
    trans = np.clip(np.asarray(curve.transmission, dtype=float), 0.0, None)
    if wave.ndim != 1 or trans.ndim != 1 or wave.size != trans.size:
        raise ValueError(f"Filter curve {curve.name!r} must have 1D wave/transmission arrays of equal length.")
    if wave.size < 3:
        raise ValueError(f"Filter curve {curve.name!r} must have at least 3 wavelength samples.")
    if trans[0] != 0.0 or trans[-1] != 0.0:
        wave = wave.copy()
        trans = trans.copy()
        trans[0] = 0.0
        trans[-1] = 0.0
    meta = {
        "group_name": _sanitize_speclite_token(group_name, "filter"),
        "band_name": _sanitize_speclite_token(curve.name, "band"),
    }
    return speclite_filters.FilterResponse(
        wavelength=wave * u.AA,
        response=trans,
        meta=meta,
    )


def _fetch_database_filter_curve(filter_name: str) -> FilterCurve:
    """Load one filter curve from the optional pcigale database backend."""
    Database = _get_pcigale_database_cls()
    if Database is None:
        raise RuntimeError("Could not import pcigale Database to load filter curves.")
    with Database() as db:
        filt = db.get_filter(filter_name)
    return FilterCurve(
        name=filter_name,
        wave=np.asarray(filt.trans_table[0], dtype=float),
        transmission=np.asarray(filt.trans_table[1], dtype=float),
        effective_wavelength=float(filt.effective_wavelength),
    )


def _load_vendored_filter_curve(filter_name: str) -> FilterCurve | None:
    """Load one vendored filter curve if grahspj ships it locally."""
    relpath = _VENDORED_FILTER_FILES.get(filter_name)
    if relpath is None:
        return None
    path = _package_resource_path(relpath)
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Vendored filter file {path} does not contain two-column transmission data.")
    wave = np.asarray(data[:, 0], dtype=float)
    trans = np.asarray(data[:, 1], dtype=float)
    if trans[0] != 0.0 or trans[-1] != 0.0:
        trans = trans.copy()
        trans[0] = 0.0
        trans[-1] = 0.0
    return FilterCurve(name=filter_name, wave=wave, transmission=trans)


def _load_named_speclite_filter(filter_name: str, cfg: FitConfig):
    """Resolve a configured filter name to a built-in speclite response."""
    speclite_name = cfg.filters.speclite_names.get(filter_name, _DEFAULT_SPECLITE_NAME_MAP.get(filter_name, filter_name))
    try:
        loaded = speclite_filters.load_filters(speclite_name)
    except Exception:
        return None
    if len(loaded) != 1:
        raise ValueError(f"Expected a single speclite filter for {filter_name!r}, got {len(loaded)} from {speclite_name!r}.")
    return loaded[0]


def _filter_response_cache_key(cfg: FitConfig) -> tuple[Any, ...]:
    """Build a stable cache key for resolved filter responses."""
    return (
        tuple(str(name) for name in cfg.photometry.filter_names),
        tuple((curve.name, id(curve.wave), id(curve.transmission), curve.effective_wavelength) for curve in cfg.filters.curves),
        tuple(sorted((str(k), str(v)) for k, v in cfg.filters.speclite_names.items())),
        bool(cfg.filters.use_grahsp_database),
    )


def _load_filter_responses(cfg: FitConfig):
    """Resolve configured filters to speclite responses, using caching."""
    cache_key = _filter_response_cache_key(cfg)
    cached = _FILTER_RESPONSE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    inline_curves = {curve.name: curve for curve in cfg.filters.curves}
    responses = []
    for filter_name in cfg.photometry.filter_names:
        if filter_name in inline_curves:
            responses.append(_curve_to_speclite_filter(inline_curves[filter_name], group_name="inline"))
            continue
        builtin = _load_named_speclite_filter(filter_name, cfg)
        if builtin is not None:
            responses.append(builtin)
            continue
        vendored = _load_vendored_filter_curve(filter_name)
        if vendored is not None:
            responses.append(_curve_to_speclite_filter(vendored, group_name="grahspj"))
            continue
        if not cfg.filters.use_grahsp_database:
            raise ValueError(
                f"Filter {filter_name!r} was not provided inline and could not be loaded by speclite; "
                "GRAHSP database fallback is disabled."
            )
        responses.append(_curve_to_speclite_filter(_fetch_database_filter_curve(filter_name), group_name="grahsp"))
    _FILTER_RESPONSE_CACHE[cache_key] = responses
    return responses


def _load_templates(cfg: FitConfig) -> LoadedTemplates:
    """Load AGN and host-dust template arrays required by the current config."""
    feii = cfg.agn.feii_template
    em = cfg.agn.emission_line_template
    cache_key = (
        feii.name,
        id(feii.wave),
        id(feii.lumin),
        id(em.wave),
        id(em.lumin_blagn),
        id(em.lumin_sy2),
        id(em.lumin_liner),
    )
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dust_alpha_grid, dust_wave, dust_lumin = _load_vendored_dale2014_templates()
    if feii.wave is not None and em.wave is not None:
        loaded = LoadedTemplates(
            feii_wave=np.asarray(feii.wave, dtype=float),
            feii_lumin=np.asarray(feii.lumin, dtype=float),
            line_wave=np.asarray(em.wave, dtype=float),
            line_blagn=np.asarray(em.lumin_blagn, dtype=float),
            line_sy2=np.asarray(em.lumin_sy2, dtype=float),
            line_liner=np.asarray(em.lumin_liner, dtype=float),
            dust_alpha_grid=dust_alpha_grid,
            dust_wave=dust_wave,
            dust_lumin=dust_lumin,
        )
        _TEMPLATE_CACHE[cache_key] = loaded
        return loaded
    if feii.name == "BruhweilerVerner08" and em.wave is None:
        feii_path = _package_resource_path("resources/templates/Fe_d11-m20-20.5.txt")
        feii_data = np.loadtxt(feii_path)
        wave_observed = np.asarray(feii_data[:, 0], dtype=float)
        lnu = np.asarray(feii_data[:, 1], dtype=float)
        z_shift = 4593.4 / 4575.0 - 1.0
        wave_rest = wave_observed / (1.0 + z_shift)
        llam = lnu * 2.99792458e18 / np.clip(wave_observed * wave_observed, 1e-30, None)
        norm = float(llam[int(np.argmin(np.abs(wave_rest - 4575.0)))])
        line_path = _package_resource_path("resources/templates/emission_line_table.formatted")
        line_data = np.loadtxt(
            line_path,
            comments="#",
            dtype=[
                ("name", "U32"),
                ("wave", "f8"),
                ("broad", "f8"),
                ("S2", "f8"),
                ("LINER", "f8"),
            ],
        )
        loaded = LoadedTemplates(
            feii_wave=np.asarray(wave_rest * 0.1, dtype=float),
            feii_lumin=np.asarray(llam / max(norm, 1e-30), dtype=float),
            line_wave=np.asarray(line_data["wave"] * 0.1, dtype=float),
            line_blagn=np.asarray(line_data["broad"], dtype=float),
            line_sy2=np.asarray(line_data["S2"], dtype=float),
            line_liner=np.asarray(line_data["LINER"], dtype=float),
            dust_alpha_grid=dust_alpha_grid,
            dust_wave=dust_wave,
            dust_lumin=dust_lumin,
        )
        _TEMPLATE_CACHE[cache_key] = loaded
        return loaded
    Database = _get_pcigale_database_cls()
    if Database is None:
        raise RuntimeError("Could not import pcigale Database to load AGN templates.")
    with Database() as db:
        feii_db = db.get_ActivateFeII(feii.name)
        em_db = db.get_ActivateMorNetzerEmLines()
    loaded = LoadedTemplates(
        feii_wave=np.asarray(feii_db.wave, dtype=float),
        feii_lumin=np.asarray(feii_db.lumin, dtype=float),
        line_wave=np.asarray(em_db.wave, dtype=float),
        line_blagn=np.asarray(em_db.lumin_BLAGN, dtype=float),
        line_sy2=np.asarray(em_db.lumin_Sy2, dtype=float),
        line_liner=np.asarray(em_db.lumin_LINER, dtype=float),
        dust_alpha_grid=dust_alpha_grid,
        dust_wave=dust_wave,
        dust_lumin=dust_lumin,
    )
    _TEMPLATE_CACHE[cache_key] = loaded
    return loaded


def _load_vendored_dale2014_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalize the vendored Dale 2014 host-dust template grid."""
    cache_key = "dale2014-host"
    cached = _DALE2014_CACHE.get(cache_key)
    if cached is not None:
        return cached
    base = _package_resource_path("resources/templates/dale2014")
    alpha_grid = np.asarray(np.genfromtxt(base / "dhcal.dat")[:, 1], dtype=float)
    raw_templates = np.asarray(np.genfromtxt(base / "spectra.0.00AGN.dat"), dtype=float)
    wave_nm = np.asarray(raw_templates[:, 0] * 1.0e3, dtype=float)
    stell = np.asarray(np.genfromtxt(base / "stellar_SED_age13Gyr_tau10Gyr.spec"), dtype=float)
    wave_stell_nm = np.asarray(stell[:, 0] * 0.1, dtype=float)
    stell_emission_nm = np.asarray(stell[:, 1] * 10.0, dtype=float)
    stell_interp_nm = np.interp(wave_nm, wave_stell_nm, stell_emission_nm)
    dust_lumin_nm = []
    for idx in range(alpha_grid.size):
        lumin_with_stell_nm = np.power(10.0, raw_templates[:, idx + 1]) / np.clip(wave_nm, 1.0e-30, None)
        constant = lumin_with_stell_nm[7] / max(stell_interp_nm[7], 1.0e-30)
        lumin_nm = lumin_with_stell_nm - stell_interp_nm * constant
        lumin_nm = np.clip(lumin_nm, 0.0, None)
        lumin_nm[wave_nm < 2.0e3] = 0.0
        norm = float(np.trapezoid(lumin_nm, x=wave_nm))
        dust_lumin_nm.append(lumin_nm / max(norm, 1.0e-30))
    dust_lumin_nm = np.asarray(dust_lumin_nm, dtype=float)
    wave_ang = wave_nm * 10.0
    dust_lumin_ang = dust_lumin_nm / 10.0
    loaded = (alpha_grid, wave_ang, dust_lumin_ang)
    _DALE2014_CACHE[cache_key] = loaded
    return loaded


def _prepare_loaded_filter(obs_wave: np.ndarray, response: speclite_filters.FilterResponse) -> LoadedFilter:
    """Precompute interpolation metadata for one filter on the model grid."""
    filt_wave = _as_angstrom_values(response.wavelength)
    trans = np.clip(np.asarray(response.response, dtype=float), 0.0, None)
    effective = _scalar_angstrom_value(response.effective_wavelength)
    mask = (obs_wave >= filt_wave[0]) & (obs_wave <= filt_wave[-1])
    work_wave = obs_wave[mask]
    if work_wave.size < 2:
        work_wave = np.linspace(filt_wave[0], filt_wave[-1], min(max(obs_wave.size // 4, 16), 512))
    trans_r = np.interp(work_wave, filt_wave, trans, left=0.0, right=0.0)
    interp_indices = np.searchsorted(obs_wave, work_wave) - 1
    interp_indices = np.clip(interp_indices, 0, obs_wave.size - 2)
    denom = obs_wave[interp_indices + 1] - obs_wave[interp_indices]
    interp_weight = (work_wave - obs_wave[interp_indices]) / np.clip(denom, 1e-12, None)
    return LoadedFilter(
        name=response.name,
        wave=filt_wave,
        transmission=trans_r,
        effective_wavelength=effective,
        interp_indices=interp_indices.astype(int),
        interp_weight=interp_weight.astype(float),
        work_wave=work_wave.astype(float),
    )


def build_model_context(cfg: FitConfig) -> ModelContext:
    """Construct the static context consumed by the grahspj NumPyro model."""
    cfg.validate()
    raw_fluxes = np.asarray(cfg.photometry.fluxes, dtype=float)
    raw_errors = np.asarray(cfg.photometry.errors, dtype=float)
    fluxes = np.asarray(raw_fluxes, dtype=float)
    errors = np.asarray(raw_errors, dtype=float)
    upper_limits = np.asarray(cfg.photometry.is_upper_limit if cfg.photometry.is_upper_limit is not None else np.zeros_like(fluxes, dtype=bool), dtype=bool)
    data_mask = (~upper_limits) & np.isfinite(raw_fluxes) & np.isfinite(raw_errors) & (raw_errors > 0.0)
    positive_detected_mask = (~upper_limits) & np.isfinite(raw_fluxes) & (raw_fluxes > 0.0)
    fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    errors = np.nan_to_num(errors, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)
    errors = np.clip(np.abs(errors), 1.0e-30, 1.0e30)

    rest_wave = np.geomspace(cfg.galaxy.rest_wave_min, cfg.galaxy.rest_wave_max, cfg.galaxy.n_wave).astype(float)
    obs_wave = rest_wave * (1.0 + max(cfg.observation.redshift, 0.0))
    ssp_data = load_cached_ssp_data(cfg.galaxy.dsps_ssp_fn)
    cosmology = FlatLambdaCDM(H0=cfg.galaxy.cosmology_h0, Om0=cfg.galaxy.cosmology_om0)
    t_obs_gyr = float(cosmology.age(max(cfg.observation.redshift, 0.0)).value)
    luminosity_distance_m = float(cosmology.luminosity_distance(max(cfg.observation.redshift, 0.0)).to_value(u.m))
    gal_t_table = np.geomspace(
        max(cfg.galaxy.sfh_t_min_gyr, 1e-3),
        max(t_obs_gyr, cfg.galaxy.sfh_t_min_gyr * 1.01),
        int(cfg.galaxy.sfh_n_steps),
    ).astype(float)

    filter_responses = _load_filter_responses(cfg)
    loaded_filters = [_prepare_loaded_filter(obs_wave, response) for response in filter_responses]
    templates = _load_templates(cfg)

    mw_ebv = 0.0
    if cfg.observation.apply_mw_deredden and cfg.observation.ra is not None and cfg.observation.dec is not None:
        coord = SkyCoord(cfg.observation.ra * u.deg, cfg.observation.dec * u.deg)
        mw_ebv = float(_get_sfd_query()(coord))
        factors = np.array(
            [_mw_band_attenuation_factor(f.work_wave, f.transmission, mw_ebv) for f in loaded_filters],
            dtype=float,
        )
        fluxes = fluxes / np.clip(factors, 1e-12, None)
        errors = errors / np.clip(factors, 1e-12, None)

    return ModelContext(
        fit_config=cfg,
        rest_wave=rest_wave,
        obs_wave=obs_wave,
        ssp_data=ssp_data,
        t_obs_gyr=t_obs_gyr,
        luminosity_distance_m=luminosity_distance_m,
        gal_t_table=gal_t_table,
        filters=loaded_filters,
        templates=templates,
        fluxes=fluxes,
        errors=errors,
        upper_limits=upper_limits,
        data_mask=data_mask,
        positive_detected_mask=positive_detected_mask,
        mw_ebv=mw_ebv,
    )
