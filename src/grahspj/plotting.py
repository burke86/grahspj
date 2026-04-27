from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .mplstyle import use_style


_COMPONENT_STYLE = [
    ("host_obs_sed", "Host stellar", "#2b6cb0", 1.6),
    ("nebular_obs_sed", "Nebular emission", "#319795", 1.1),
    ("dust_obs_sed", "Host dust", "#b7791f", 1.5),
    ("disk_obs_sed", "AGN disk", "#c05621", 1.2),
    ("torus_obs_sed", "Torus", "#805ad5", 1.2),
    ("feii_obs_sed", "Fe II", "#2f855a", 1.0),
    ("line_bl_obs_sed", "AGN lines", "#d53f8c", 1.0),
    ("line_nl_obs_sed", "AGN lines", "#b83280", 1.0),
    ("line_liner_obs_sed", "AGN lines", "#97266d", 1.0),
    ("balmer_obs_sed", "Balmer cont.", "#dd6b20", 1.0),
    ("agn_obs_sed", "AGN total", "#718096", 1.4),
    ("total_obs_sed", "Model total", "#000000", 2.0),
]


def _median_site(pred: dict[str, Any], key: str) -> np.ndarray:
    """Return the median draw for one predictive site."""
    arr = np.asarray(pred[key], dtype=float)
    return np.median(arr, axis=0) if arr.ndim > 1 else arr


def _percentile_site(pred: dict[str, Any], key: str, q: float) -> np.ndarray:
    """Return one percentile across predictive draws for a site."""
    arr = np.asarray(pred[key], dtype=float)
    return np.percentile(arr, q, axis=0) if arr.ndim > 1 else arr


def _to_display_flux_density(obs_wave: np.ndarray, sed: np.ndarray) -> np.ndarray:
    """Convert internal model spectra into displayed mJy values."""
    obs_wave = np.asarray(obs_wave, dtype=float)
    sed = np.asarray(sed, dtype=float)
    return 1.0e-10 / 299792458.0 * 1.0e29 * obs_wave * obs_wave * sed


def _median_effective_variance(fitter, pred: dict[str, Any]) -> np.ndarray:
    """Rebuild the model's effective variance from predictive median sites."""
    pred_fluxes = np.asarray(_median_site(pred, "pred_fluxes"), dtype=float)
    agn_fluxes = np.asarray(_median_site(pred, "agn_fluxes"), dtype=float) if "agn_fluxes" in pred else np.zeros_like(pred_fluxes)
    intrinsic_scatter = float(np.median(np.asarray(pred["intrinsic_scatter_fit"], dtype=float))) if "intrinsic_scatter_fit" in pred else 0.0
    agn_variability_nev = float(np.median(np.asarray(pred["agn_variability_nev"], dtype=float))) if "agn_variability_nev" in pred else 0.0
    transmitted_fraction = (
        np.asarray(_median_site(pred, "transmitted_fraction_fluxes"), dtype=float)
        if "transmitted_fraction_fluxes" in pred
        else np.ones_like(pred_fluxes)
    )
    filter_wavelength = np.asarray([flt.effective_wavelength for flt in fitter.context.filters], dtype=float)
    redshift = float(np.median(np.asarray(pred["redshift_fit"], dtype=float))) if "redshift_fit" in pred else 0.0
    obs_errors = np.asarray(getattr(fitter.context, "errors", fitter.config.photometry.errors), dtype=float)
    cfg = getattr(fitter.config, "likelihood", None)
    if cfg is None:
        class _FallbackLikelihood:
            systematics_width = 0.0
            variability_uncertainty = False
            attenuation_model_uncertainty = False
            lyman_break_uncertainty = False

        cfg = _FallbackLikelihood()

    obs_variance = obs_errors**2 + np.maximum(intrinsic_scatter, 0.0) ** 2
    sys_variance = (float(cfg.systematics_width) * pred_fluxes) ** 2
    var_variance = np.where(bool(cfg.variability_uncertainty), agn_variability_nev * agn_fluxes**2, 0.0)
    if cfg.attenuation_model_uncertainty:
        tf = np.clip(transmitted_fraction, 1e-4, 1.0)
        neg_log = -np.log10(tf + 1e-4)
        log_unc_frac = np.minimum(-4.0 + 2.0 * neg_log, -1.0)
        att_unc = 10 ** log_unc_frac / tf
        sys_variance = sys_variance + (att_unc * pred_fluxes) ** 2
    if cfg.lyman_break_uncertainty:
        ly_unc = np.where(filter_wavelength / (1.0 + redshift) < 150.0, 1.0e8, 0.0)
        sys_variance = sys_variance + (ly_unc * pred_fluxes) ** 2
    return np.nan_to_num(obs_variance + sys_variance + var_variance, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)


def plot_fit_sed(
    fitter,
    output_path: str | Path | None = None,
    posterior: str = "latest",
    show: bool = False,
    annotate_band_names: bool = True,
):
    """Render a component SED plot for a fitted grahspj object."""
    pred = fitter.predict(posterior=posterior)
    obs_wave = _median_site(pred, "obs_wave")
    x_min = min(1.0e2, float(np.nanmin(obs_wave)))
    x_max = max(1.0e6, float(np.nanmax(obs_wave)))
    model_flux = _median_site(pred, "pred_fluxes")
    phot_wave = np.asarray([flt.effective_wavelength for flt in fitter.context.filters], dtype=float)
    obs_flux = np.asarray(fitter.config.photometry.fluxes, dtype=float)
    obs_err = np.asarray(fitter.config.photometry.errors, dtype=float)
    labels = list(fitter.config.photometry.filter_names)
    plotted_components: list[np.ndarray] = []
    legend_labels_seen: set[str] = set()

    with use_style():
        fig, (ax_sed, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.05},
        )

        component_sums = {}
        for key, label, color, lw in _COMPONENT_STYLE:
            if key not in pred:
                continue
            component = _to_display_flux_density(obs_wave, _median_site(pred, key))
            comp_lo = _to_display_flux_density(obs_wave, _percentile_site(pred, key, 16.0))
            comp_hi = _to_display_flux_density(obs_wave, _percentile_site(pred, key, 84.0))
            finite_component = np.asarray(component, dtype=float)
            if not np.any(np.isfinite(finite_component) & (np.abs(finite_component) > 0.0)):
                continue
            plotted_components.append(component)
            plotted_components.append(comp_lo)
            plotted_components.append(comp_hi)
            component_sums[label] = float(np.nansum(np.clip(component, 0.0, None)))
            lo = np.minimum(comp_lo, comp_hi)
            hi = np.maximum(comp_lo, comp_hi)
            finite_band = np.isfinite(lo) & np.isfinite(hi) & (hi > 0.0)
            if np.any(finite_band):
                ax_sed.fill_between(
                    obs_wave,
                    np.where(finite_band, np.clip(lo, 1e-300, None), np.nan),
                    np.where(finite_band, np.clip(hi, 1e-300, None), np.nan),
                    color=color,
                    alpha=0.12,
                    linewidth=0.0,
                    zorder=0,
                )
            plot_label = label if label not in legend_labels_seen else "_nolegend_"
            if plot_label != "_nolegend_":
                legend_labels_seen.add(label)
            if key == "total_obs_sed":
                ax_sed.plot(obs_wave, component, color=color, lw=max(lw - 0.2, 1.4), alpha=0.65, label=plot_label, zorder=1)
            elif key == "host_obs_sed":
                ax_sed.plot(obs_wave, component, color=color, lw=max(lw, 2.3), ls="--", alpha=0.95, label=plot_label, zorder=4)
            elif key == "dust_obs_sed":
                ax_sed.plot(obs_wave, component, color=color, lw=max(lw, 2.1), ls=(0, (4, 2)), alpha=0.95, label=plot_label, zorder=4)
            elif key == "agn_obs_sed":
                ax_sed.plot(obs_wave, component, color=color, lw=max(lw, 2.2), ls="-.", alpha=0.95, label=plot_label, zorder=4)
            else:
                ax_sed.plot(obs_wave, component, color=color, lw=max(lw, 2.0), ls=":", alpha=0.95, label=plot_label, zorder=3)

        ax_sed.errorbar(
            phot_wave,
            obs_flux,
            yerr=obs_err,
            fmt="o",
            color="#c53030",
            ms=5,
            capsize=2,
            label="Observed photometry",
            zorder=5,
        )
        ax_sed.scatter(phot_wave, model_flux, color="#111111", marker="s", s=28, label="Model photometry", zorder=6)
        if annotate_band_names:
            for x, y, label in zip(phot_wave, obs_flux, labels):
                ax_sed.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)

        resid = obs_flux - model_flux
        eff_variance = _median_effective_variance(fitter, pred)
        eff_sigma = np.sqrt(np.clip(eff_variance, 1e-30, 1.0e60))
        ax_resid.errorbar(phot_wave, resid, yerr=eff_sigma, fmt="o", color="black", ms=4, capsize=2)
        ax_resid.axhline(0.0, color="black", lw=1.0, ls="--")
        upper_limits = np.asarray(getattr(fitter.context, "upper_limits", np.zeros_like(obs_flux, dtype=bool)), dtype=bool)
        finite_chi2 = np.isfinite(obs_flux) & np.isfinite(model_flux) & np.isfinite(eff_sigma) & (eff_sigma > 0.0) & (~upper_limits)
        if np.any(finite_chi2):
            chi2 = float(np.sum(((obs_flux[finite_chi2] - model_flux[finite_chi2]) / eff_sigma[finite_chi2]) ** 2))
            # For this Bayesian SED fit, counting sampled variables as "free parameters"
            # makes the usual dof estimate meaningless and often collapses the denominator
            # to 1. Use chi2 per valid band as a stable visual diagnostic instead.
            reduced_chi2 = chi2 / max(1, int(np.sum(finite_chi2)))
            ax_resid.text(
                0.98,
                0.05,
                rf"$\chi^2_\nu = {reduced_chi2:.2f}$",
                transform=ax_resid.transAxes,
                va="bottom",
                ha="right",
                color="#c53030",
                fontsize=10,
            )

        ax_sed.set_xscale("log")
        ax_sed.set_yscale("log")
        ax_sed.set_ylabel("Flux density (mJy)")
        ax_resid.set_ylabel("Obs - Model (mJy)")
        ax_resid.set_xlabel("Observed-frame wavelength (Å)")
        ax_sed.legend(loc="lower right", fontsize=9, ncol=2)

        finite_flux_parts = [np.asarray(obs_flux, dtype=float), np.asarray(model_flux, dtype=float)]
        finite_flux_parts.extend(np.asarray(comp, dtype=float) for comp in plotted_components)
        finite_flux = np.concatenate([arr.ravel() for arr in finite_flux_parts])
        finite_flux = finite_flux[np.isfinite(finite_flux) & (finite_flux > 0.0)]
        if finite_flux.size:
            ymax = float(np.nanmax(finite_flux))
            scale_floor = ymax * 1.0e-6
            visible_flux = finite_flux[finite_flux >= scale_floor]
            if visible_flux.size == 0:
                visible_flux = finite_flux
            ymin = float(np.nanmin(visible_flux))
            ax_sed.set_ylim(ymin * 0.7, ymax * 1.8)
        ax_resid.set_xscale("log")
        ax_sed.set_xlim(x_min, x_max)
        ax_resid.set_xlim(x_min, x_max)

        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
