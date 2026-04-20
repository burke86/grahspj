from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta

from .config import FitConfig, serialize_config
from .model import grahsp_photometric_model
from .preload import ModelContext, build_model_context


def _get_nested_sampler_cls():
    """Resolve NumPyro's optional nested sampler lazily."""
    from numpyro.contrib.nested_sampling import NestedSampler

    return NestedSampler


class GRAHSPJ:
    """High-level single-object fitting interface for grahspj."""
    def __init__(self, config: FitConfig):
        """Initialize the fitter and build its static model context."""
        self.config = config
        self.context: ModelContext = build_model_context(config)
        self.map_result: dict[str, Any] | None = None
        self.nuts_result: dict[str, Any] | None = None
        self.ns_result: dict[str, Any] | None = None
        self.samples: dict[str, Any] | None = None
        self.predictive: dict[str, Any] | None = None
        self._plot_cache: dict[str, Any] | None = None

    def _reset_fit_state(self) -> None:
        """Clear cached inference and predictive state."""
        self.map_result = None
        self.nuts_result = None
        self.ns_result = None
        self.samples = None
        self.predictive = None
        self._plot_cache = None

    def _apply_runtime_overrides(
        self,
        prior_config: dict[str, Any] | None = None,
        dsps_ssp_fn: str | None = None,
    ) -> None:
        """Apply one-off fit-time overrides and rebuild context if required."""
        rebuild_context = False
        if prior_config is not None:
            self.config.prior_config = dict(prior_config)
            self._reset_fit_state()
        if dsps_ssp_fn is not None and str(dsps_ssp_fn) != str(self.config.galaxy.dsps_ssp_fn):
            self.config.galaxy.dsps_ssp_fn = str(dsps_ssp_fn)
            rebuild_context = True
        if rebuild_context:
            self.context = build_model_context(self.config)
            self._reset_fit_state()

    def _model(self):
        """Return the bound NumPyro model for the current context."""
        return grahsp_photometric_model(self.context, include_components=False)

    def _continuum_init_model(self):
        """Return the MAP warm-start model with detailed AGN features disabled."""
        return grahsp_photometric_model(
            self.context,
            include_components=False,
            include_sed_agn_features=False,
            include_spectral_features=False,
        )

    def _predictive_model(self):
        """Return the bound NumPyro model used for posterior predictive products."""
        return grahsp_photometric_model(self.context, include_components=True)

    def _compute_predictive(self) -> dict[str, Any]:
        """Generate and cache predictive outputs from posterior samples."""
        if self.samples is None:
            raise RuntimeError("No fitted posterior available. Run fit_map(), fit_nuts(), or fit_ns() first.")
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 17)
        pred = Predictive(
            self._predictive_model,
            posterior_samples=self.samples,
            return_sites=[
                "pred_fluxes",
                "pred_spectrum_fluxes",
                "spec_wave_obs",
                "spec_spectrum_index",
                "spectrum_scale_fit",
                "log_spectrum_scale_fit",
                "spectrum_host_capture_fraction",
                "spectroscopy_loglike",
                "jqf_continuum_model",
                "jqf_line_model",
                "jqf_line_model_broad",
                "jqf_line_model_narrow",
                "jqf_line_amp_per_component",
                "jqf_line_mu_per_component",
                "jqf_line_sig_per_component",
                "jqf_feii_model",
                "jqf_balmer_model",
                "jqf_total_model",
                "agn_fluxes",
                "host_fluxes",
                "dust_fluxes",
                "disk_fluxes",
                "torus_fluxes",
                "feii_fluxes",
                "line_fluxes",
                "line_bl_fluxes",
                "line_nl_fluxes",
                "line_liner_fluxes",
                "balmer_fluxes",
                "host_age_weights",
                "host_lgmet_weights",
                "host_ssp_weights",
                "gal_sfr_table",
                "gal_smh_table",
                "obs_wave",
                "redshift_fit",
                "total_rest_sed",
                "agn_rest_sed",
                "host_rest_sed",
                "host_absorbed_rest_sed",
                "dust_rest_sed",
                "disk_rest_sed",
                "torus_rest_sed",
                "feii_rest_sed",
                "line_rest_sed",
                "line_bl_rest_sed",
                "line_nl_rest_sed",
                "line_liner_rest_sed",
                "balmer_rest_sed",
                "total_obs_sed",
                "agn_obs_sed",
                "host_obs_sed",
                "dust_obs_sed",
                "disk_obs_sed",
                "torus_obs_sed",
                "feii_obs_sed",
                "line_obs_sed",
                "line_bl_obs_sed",
                "line_nl_obs_sed",
                "line_liner_obs_sed",
                "balmer_obs_sed",
                "log_dust_luminosity_fit",
                "dust_alpha_fit",
                "intrinsic_scatter_fit",
                "fracAGN_5100_fit",
                "log_agn_bol_luminosity_fit",
                "log_disk_luminosity_fit",
                "agn_variability_nev",
                "host_total_fluxes",
                "host_capture_fraction_fluxes",
                "log_host_capture_scale_arcsec_fit",
                "host_capture_slope_fit",
                "transmitted_fraction_fluxes",
                "absolute_flux_scale_logprior",
            ],
        )(rng_key)
        self.predictive = {k: np.asarray(v) for k, v in pred.items()}
        return self.predictive

    def fit(
        self,
        fit_method: str = "optax+nuts",
        progress_bar: bool = True,
        prior_config: dict[str, Any] | None = None,
        dsps_ssp_fn: str | None = None,
        optax_steps: int | None = None,
        optax_lr: float | None = None,
        nuts_warmup: int | None = None,
        nuts_samples: int | None = None,
        nuts_chains: int | None = None,
        ns_live_points: int | None = None,
        ns_max_samples: int | None = None,
        ns_dlogz: float | None = None,
        plot_fig: bool = False,
        save_fig: bool = False,
        save_result: bool = False,
        output_dir: str | Path = ".",
        fig_path: str | Path | None = None,
        result_path: str | Path | None = None,
        use_map_init: bool = True,
        target_accept_prob: float | None = None,
        staged_map: bool = True,
        staged_steps: int | None = None,
        **kwargs,
    ):
        """Run the requested inference path and optional plotting/saving helpers."""
        if "steps" in kwargs and optax_steps is None:
            optax_steps = kwargs.pop("steps")
        if "learning_rate" in kwargs and optax_lr is None:
            optax_lr = kwargs.pop("learning_rate")
        if "num_warmup" in kwargs and nuts_warmup is None:
            nuts_warmup = kwargs.pop("num_warmup")
        if "num_samples" in kwargs and nuts_samples is None:
            nuts_samples = kwargs.pop("num_samples")
        if "num_chains" in kwargs and nuts_chains is None:
            nuts_chains = kwargs.pop("num_chains")
        if "ns_live_points" in kwargs and ns_live_points is None:
            ns_live_points = kwargs.pop("ns_live_points")
        if "ns_max_samples" in kwargs and ns_max_samples is None:
            ns_max_samples = kwargs.pop("ns_max_samples")
        if "ns_dlogz" in kwargs and ns_dlogz is None:
            ns_dlogz = kwargs.pop("ns_dlogz")
        if "target_accept_prob" in kwargs and target_accept_prob is None:
            target_accept_prob = kwargs.pop("target_accept_prob")
        use_map_init_explicit = "use_map_init" in kwargs
        if use_map_init_explicit:
            use_map_init = kwargs.pop("use_map_init")
        if hasattr(self, "_apply_runtime_overrides"):
            self._apply_runtime_overrides(prior_config=prior_config, dsps_ssp_fn=dsps_ssp_fn)
        method = str(fit_method).lower()
        output_dir = Path(output_dir)
        if method == "optax":
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(f"Unknown fit() keyword arguments: {unknown}")
            map_kwargs: dict[str, Any] = {"progress_bar": progress_bar}
            if optax_steps is not None:
                map_kwargs["steps"] = optax_steps
            if optax_lr is not None:
                map_kwargs["learning_rate"] = optax_lr
            map_kwargs["staged"] = staged_map
            if staged_steps is not None:
                map_kwargs["staged_steps"] = staged_steps
            fit_output: dict[str, Any] | Any = self.fit_map(
                **map_kwargs,
            )
        elif method == "nuts":
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(f"Unknown fit() keyword arguments: {unknown}")
            nuts_kwargs: dict[str, Any] = {"progress_bar": progress_bar}
            if nuts_warmup is not None:
                nuts_kwargs["num_warmup"] = nuts_warmup
            if nuts_samples is not None:
                nuts_kwargs["num_samples"] = nuts_samples
            if nuts_chains is not None:
                nuts_kwargs["num_chains"] = nuts_chains
            if target_accept_prob is not None:
                nuts_kwargs["target_accept_prob"] = target_accept_prob
            if use_map_init_explicit or use_map_init is not True:
                nuts_kwargs["use_map_init"] = use_map_init
            fit_output = self.fit_nuts(
                **nuts_kwargs,
            )
        elif method == "optax+nuts":
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(f"Unknown fit() keyword arguments: {unknown}")
            map_kwargs = {"progress_bar": progress_bar}
            if optax_steps is not None:
                map_kwargs["steps"] = optax_steps
            if optax_lr is not None:
                map_kwargs["learning_rate"] = optax_lr
            map_kwargs["staged"] = staged_map
            if staged_steps is not None:
                map_kwargs["staged_steps"] = staged_steps
            map_result = self.fit_map(
                **map_kwargs,
            )
            nuts_kwargs = {"progress_bar": progress_bar}
            if nuts_warmup is not None:
                nuts_kwargs["num_warmup"] = nuts_warmup
            if nuts_samples is not None:
                nuts_kwargs["num_samples"] = nuts_samples
            if nuts_chains is not None:
                nuts_kwargs["num_chains"] = nuts_chains
            if target_accept_prob is not None:
                nuts_kwargs["target_accept_prob"] = target_accept_prob
            if use_map_init_explicit or use_map_init is not True:
                nuts_kwargs["use_map_init"] = use_map_init
            nuts_result = self.fit_nuts(
                **nuts_kwargs,
            )
            fit_output = {"map": map_result, "nuts": nuts_result}
        elif method == "ns":
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(f"Unknown fit() keyword arguments: {unknown}")
            ns_kwargs: dict[str, Any] = {"progress_bar": progress_bar}
            if ns_live_points is not None:
                ns_kwargs["num_live_points"] = ns_live_points
            if ns_max_samples is not None:
                ns_kwargs["max_samples"] = ns_max_samples
            if ns_dlogz is not None:
                ns_kwargs["dlogz"] = ns_dlogz
            fit_output = self.fit_ns(**ns_kwargs)
        else:
            raise ValueError("fit_method must be one of: 'optax+nuts', 'optax', 'nuts', 'ns'")

        saved_result_path = None
        saved_fig_path = None
        fig = None
        if save_result:
            if result_path is None:
                saved_result_path = self.save(output_dir)
            else:
                result_path = Path(result_path)
                result_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "config": serialize_config(self.config),
                    "summary": self.summary() if self.samples is not None else None,
                    "samples": {k: np.asarray(v) for k, v in (self.samples or {}).items()},
                    "predictive": {k: np.asarray(v) for k, v in self.predict().items()} if self.samples is not None else {},
                    "mw_ebv": self.context.mw_ebv,
                }
                with open(result_path, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                saved_result_path = result_path
        if plot_fig or save_fig:
            if fig_path is None and save_fig:
                fig_path = output_dir / f"{self.config.observation.object_id}_sed.png"
            fig = self.plot_sed(output_path=fig_path if save_fig else None, show=plot_fig)
            if save_fig:
                saved_fig_path = Path(fig_path) if fig_path is not None else None

        samples = getattr(self, "samples", None)
        # Lightweight test doubles may call fit() on a partially constructed object
        # without config/context. Preserve the direct fit payload for that case only.
        if not hasattr(self, "config"):
            return fit_output
        return {
            "fit": fit_output,
            "summary": self.summary() if samples is not None else None,
            "figure": fig,
            "figure_path": saved_fig_path,
            "result_path": saved_result_path,
        }

    def _run_map_svi(
        self,
        model_fn,
        *,
        steps: int,
        learning_rate: float,
        progress_bar: bool,
        rng_seed: int,
        init_values: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run one Optax/NumPyro AutoDelta MAP stage."""
        import optax
        from numpyro.optim import optax_to_numpyro

        if init_values:
            guide = AutoDelta(model_fn, init_loc_fn=init_to_value(values=init_values))
        else:
            guide = AutoDelta(model_fn)
        optimizer = optax_to_numpyro(optax.adam(learning_rate))
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
        rng_key = jax.random.PRNGKey(rng_seed)
        svi_result = svi.run(rng_key, steps, progress_bar=progress_bar)
        median = guide.median(svi_result.params)
        return svi_result, median

    def fit_map(
        self,
        steps: int | None = None,
        learning_rate: float | None = None,
        progress_bar: bool = True,
        staged: bool = True,
        staged_steps: int | None = None,
    ):
        """Run the Optax/NumPyro MAP optimization path."""
        steps = int(self.config.inference.map_steps if steps is None else steps)
        learning_rate = float(self.config.inference.learning_rate if learning_rate is None else learning_rate)
        stage1_result = None
        stage1_median = None
        init_values = None
        if staged:
            continuum_steps = int(max(1, steps // 3) if staged_steps is None else staged_steps)
            stage1_result, stage1_median = self._run_map_svi(
                self._continuum_init_model,
                steps=continuum_steps,
                learning_rate=learning_rate,
                progress_bar=progress_bar,
                rng_seed=self.config.inference.seed,
            )
            init_values = {k: np.asarray(v) for k, v in stage1_median.items()}

        svi_result, median = self._run_map_svi(
            self._model,
            steps=steps,
            learning_rate=learning_rate,
            progress_bar=progress_bar,
            rng_seed=self.config.inference.seed + (1 if staged else 0),
            init_values=init_values,
        )
        self.map_result = {
            "params": svi_result.params,
            "median": median,
            "losses": np.asarray(getattr(svi_result, "losses", [])),
            "staged": bool(staged),
        }
        if stage1_result is not None and stage1_median is not None:
            self.map_result["stage1"] = {
                "params": stage1_result.params,
                "median": stage1_median,
                "losses": np.asarray(getattr(stage1_result, "losses", [])),
            }
        self.samples = {k: np.asarray(v)[None, ...] for k, v in median.items()}
        self.predictive = None
        return self.map_result

    def fit_nuts(
        self,
        num_warmup: int | None = None,
        num_samples: int | None = None,
        num_chains: int | None = None,
        target_accept_prob: float | None = None,
        use_map_init: bool = True,
        progress_bar: bool = True,
    ):
        """Run NUTS sampling, optionally initializing from the MAP solution."""
        if use_map_init and self.map_result is None:
            self.fit_map(progress_bar=progress_bar)
        num_warmup = int(self.config.inference.num_warmup if num_warmup is None else num_warmup)
        num_samples = int(self.config.inference.num_samples if num_samples is None else num_samples)
        num_chains = int(self.config.inference.num_chains if num_chains is None else num_chains)
        target_accept_prob = float(self.config.inference.target_accept_prob if target_accept_prob is None else target_accept_prob)
        init_values = None
        if self.map_result is not None:
            init_values = {k: np.asarray(v) for k, v in self.map_result["median"].items() if np.ndim(v) != 0 or np.isfinite(v)}
        kernel = NUTS(self._model, init_strategy=init_to_value(values=init_values) if init_values else None, target_accept_prob=target_accept_prob, dense_mass=False, max_tree_depth=8)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar, jit_model_args=False)
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 1)
        mcmc.run(rng_key)
        samples = mcmc.get_samples()
        self.nuts_result = {"mcmc": mcmc}
        self.samples = {k: np.asarray(v) for k, v in samples.items()}
        self.predictive = None
        return self.nuts_result

    def fit_ns(
        self,
        num_live_points: int | None = None,
        max_samples: int | None = None,
        dlogz: float | None = None,
        progress_bar: bool = True,
    ):
        """Run full-model nested sampling and resample equal-weight posterior draws."""
        NestedSampler = _get_nested_sampler_cls()

        constructor_kwargs: dict[str, Any] = {"verbose": bool(progress_bar)}
        if num_live_points is not None:
            constructor_kwargs["num_live_points"] = int(num_live_points)
        if max_samples is not None:
            constructor_kwargs["max_samples"] = int(max_samples)
        termination_kwargs: dict[str, Any] = {}
        if dlogz is not None:
            termination_kwargs["dlogZ"] = float(dlogz)

        sampler = NestedSampler(
            self._model,
            constructor_kwargs=constructor_kwargs,
            termination_kwargs=termination_kwargs,
        )
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 2)
        sampler.run(rng_key)
        posterior_rng_key = jax.random.PRNGKey(self.config.inference.seed + 3)
        samples = sampler.get_samples(
            posterior_rng_key,
            num_samples=int(self.config.inference.num_samples),
            group_by_chain=False,
        )
        self.ns_result = {
            "sampler": sampler,
            "results": getattr(sampler, "_results", None),
            "constructor_kwargs": dict(getattr(sampler, "constructor_kwargs", constructor_kwargs)),
            "termination_kwargs": dict(getattr(sampler, "termination_kwargs", termination_kwargs)),
        }
        self.samples = {k: np.asarray(v) for k, v in samples.items()}
        self.predictive = None
        return self.ns_result

    def predict(self, posterior: str = "latest") -> dict[str, Any]:
        """Return cached predictive outputs or generate them on demand."""
        if self.predictive is None:
            return self._compute_predictive()
        return self.predictive

    def recovered_log_stellar_mass(self) -> float:
        """Return the median recovered stellar mass from the fitted posterior."""
        if self.samples is not None and "log_stellar_mass" in self.samples:
            return float(np.median(np.asarray(self.samples["log_stellar_mass"], dtype=float)))
        if self.map_result is not None and "median" in self.map_result and "log_stellar_mass" in self.map_result["median"]:
            return float(np.asarray(self.map_result["median"]["log_stellar_mass"], dtype=float))
        raise RuntimeError("No recovered stellar mass available. Run fit_map(), fit_nuts(), or fit_ns() first.")

    def summary(self) -> dict[str, Any]:
        """Summarize posterior medians and selected derived quantities."""
        if self.samples is None:
            raise RuntimeError("No fitted posterior available.")
        out: dict[str, Any] = {}
        for key, value in self.samples.items():
            arr = np.asarray(value)
            out[f"{key}_median"] = np.median(arr, axis=0).tolist() if arr.ndim > 1 else float(np.median(arr))
        if "host_age_weights" in self.samples:
            ages = np.power(10.0, np.asarray(self.context.ssp_data.ssp_lg_age_gyr, dtype=float))
            age_weights = np.median(np.asarray(self.samples["host_age_weights"]), axis=0)
            age_weight_sum = np.sum(age_weights)
            out["host_age_weighted_gyr"] = float(np.sum(age_weights * ages) / age_weight_sum) if age_weight_sum > 0 else -1.0
        if "host_lgmet_weights" in self.samples:
            mets = np.asarray(self.context.ssp_data.ssp_lgmet, dtype=float)
            lgmet_weights = np.median(np.asarray(self.samples["host_lgmet_weights"]), axis=0)
            lgmet_weight_sum = np.sum(lgmet_weights)
            out["host_lgmet_weighted"] = float(np.sum(lgmet_weights * mets) / lgmet_weight_sum) if lgmet_weight_sum > 0 else -99.0
        if "gal_lgmet" in self.samples:
            out["gal_lgmet_fit"] = float(np.median(np.asarray(self.samples["gal_lgmet"], dtype=float)))
        if "gal_lgmet_scatter" in self.samples:
            out["gal_lgmet_scatter_fit"] = float(np.median(np.asarray(self.samples["gal_lgmet_scatter"], dtype=float)))
        if "log_stellar_mass" in self.samples:
            out["log_stellar_mass_fit"] = self.recovered_log_stellar_mass()
        if "dust_alpha" in self.samples:
            out["dust_alpha_fit"] = float(np.median(np.asarray(self.samples["dust_alpha"], dtype=float)))
        if self.predictive is not None:
            out["pred_fluxes_median"] = np.median(np.asarray(self.predictive["pred_fluxes"]), axis=0).tolist()
            if "log_dust_luminosity_fit" in self.predictive:
                out["log_dust_luminosity_fit"] = float(np.median(np.asarray(self.predictive["log_dust_luminosity_fit"], dtype=float)))
            if "log_agn_bol_luminosity_fit" in self.predictive:
                out["log_agn_bol_luminosity_fit"] = float(np.median(np.asarray(self.predictive["log_agn_bol_luminosity_fit"], dtype=float)))
            if "log_disk_luminosity_fit" in self.predictive:
                out["log_disk_luminosity_fit"] = float(np.median(np.asarray(self.predictive["log_disk_luminosity_fit"], dtype=float)))
            if "intrinsic_scatter_fit" in self.predictive:
                out["intrinsic_scatter_fit"] = float(np.median(np.asarray(self.predictive["intrinsic_scatter_fit"], dtype=float)))
            if "absolute_flux_scale_logprior" in self.predictive:
                out["absolute_flux_scale_logprior"] = float(np.median(np.asarray(self.predictive["absolute_flux_scale_logprior"], dtype=float)))
        return out

    def save(self, output_dir: str | Path) -> Path:
        """Serialize config, posterior samples, and predictive outputs to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": serialize_config(self.config),
            "summary": self.summary() if self.samples is not None else None,
            "samples": {k: np.asarray(v) for k, v in (self.samples or {}).items()},
            "predictive": {k: np.asarray(v) for k, v in self.predict().items()} if self.samples is not None else {},
            "mw_ebv": self.context.mw_ebv,
        }
        out = output_dir / f"{self.config.observation.object_id}_posterior.pkl"
        with open(out, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    def plot_sed(
        self,
        output_path: str | Path | None = None,
        posterior: str = "latest",
        show: bool = False,
        annotate_band_names: bool = True,
    ):
        """Plot the fitted SED using the package plotting helper."""
        from .plotting import plot_fit_sed

        return plot_fit_sed(
            self,
            output_path=output_path,
            posterior=posterior,
            show=show,
            annotate_band_names=annotate_band_names,
        )

    @staticmethod
    def _posterior_median_array(value: Any) -> np.ndarray:
        """Return a median predictive array over the leading sample axis."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0 or arr.size == 0:
            return arr
        return np.nanmedian(arr, axis=0)

    @staticmethod
    def _mjy_to_rest_flambda_1e17(wave_obs: np.ndarray, flux_mjy: np.ndarray, redshift: float) -> np.ndarray:
        """Convert observed-frame mJy to jaxqsofit rest-frame f_lambda units."""
        wave_obs = np.asarray(wave_obs, dtype=float)
        flux_mjy = np.asarray(flux_mjy, dtype=float)
        c_ang_s = 2.99792458e18
        flam_obs_cgs = flux_mjy * 1.0e-26 * c_ang_s / np.clip(wave_obs**2, 1.0e-30, None)
        return flam_obs_cgs * (1.0 + float(redshift)) / 1.0e-17

    @staticmethod
    def _obs_flambda_to_rest_flambda_1e17(flux_lambda_obs: np.ndarray, redshift: float) -> np.ndarray:
        """Convert observed-frame W/m^2/Angstrom to jaxqsofit rest-frame units."""
        flux_lambda_obs = np.asarray(flux_lambda_obs, dtype=float)
        return flux_lambda_obs * 1.0e3 * (1.0 + float(redshift)) / 1.0e-17

    def plot_jaxqsofit_spectrum(
        self,
        spectrum_index: int = 0,
        posterior: str = "latest",
        show_plot: bool = True,
        plot_residual: bool = True,
        plot_legend: bool = True,
        ylims: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot the joint spectral fit with jaxqsofit's spectrum plotter."""
        if str(self.config.spectroscopy_config.backend).lower() != "jaxqsofit":
            raise RuntimeError("plot_jaxqsofit_spectrum requires SpectroscopyConfig.backend='jaxqsofit'.")
        if self.context.spec_wave_obs.size == 0:
            raise RuntimeError("No spectroscopy data are available to plot.")
        try:
            from jaxqsofit import QSOFit
        except Exception as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError("plot_jaxqsofit_spectrum requires jaxqsofit on PYTHONPATH.") from exc

        pred = self.predict(posterior=posterior)
        index = np.asarray(self.context.spec_spectrum_index, dtype=int)
        selected = (index == int(spectrum_index)) & np.asarray(self.context.spec_mask, dtype=bool)
        if not np.any(selected):
            raise ValueError(f"No valid spectral pixels found for spectrum_index={spectrum_index}.")

        z = float(self.config.observation.redshift)
        wave_obs = np.asarray(self.context.spec_wave_obs, dtype=float)[selected]
        wave_rest = wave_obs / (1.0 + z)
        flux_rest = self._mjy_to_rest_flambda_1e17(
            wave_obs,
            np.asarray(self.context.spec_fluxes, dtype=float)[selected],
            z,
        )
        err_rest = self._mjy_to_rest_flambda_1e17(
            wave_obs,
            np.asarray(self.context.spec_errors, dtype=float)[selected],
            z,
        )
        spectrum_scale = self._posterior_median_array(pred.get("spectrum_scale_fit", 1.0))
        if np.ndim(spectrum_scale) > 0 and np.size(spectrum_scale) > 1:
            scale_factor = float(np.asarray(spectrum_scale, dtype=float)[int(spectrum_index)])
        else:
            scale_factor = float(np.asarray(spectrum_scale, dtype=float))
        capture_fraction = self._posterior_median_array(pred.get("spectrum_host_capture_fraction", 1.0))
        if np.ndim(capture_fraction) > 0 and np.size(capture_fraction) > 1:
            host_capture = float(np.asarray(capture_fraction, dtype=float)[int(spectrum_index)])
        else:
            host_capture = float(np.asarray(capture_fraction, dtype=float))

        def component(name: str, apply_scale: bool = True) -> np.ndarray:
            if name not in pred:
                return np.zeros_like(wave_rest)
            comp_mjy = self._posterior_median_array(pred[name])[selected]
            if apply_scale:
                comp_mjy = scale_factor * comp_mjy
            return self._mjy_to_rest_flambda_1e17(wave_obs, comp_mjy, z)

        def obs_sed_component(name: str, multiplier: float = 1.0) -> np.ndarray:
            if name not in pred or "obs_wave" not in pred:
                return np.zeros_like(wave_rest)
            source_wave = np.asarray(self._posterior_median_array(pred["obs_wave"]), dtype=float)
            source_flux = np.asarray(self._posterior_median_array(pred[name]), dtype=float)
            if source_wave.size == 0 or source_flux.size != source_wave.size:
                return np.zeros_like(wave_rest)
            flux_lambda = np.interp(wave_obs, source_wave, source_flux, left=0.0, right=0.0)
            return self._obs_flambda_to_rest_flambda_1e17(scale_factor * multiplier * flux_lambda, z)

        def keep_component(arr: np.ndarray) -> bool:
            finite = np.asarray(arr, dtype=float)
            finite = finite[np.isfinite(finite)]
            return finite.size > 0 and float(np.nanmax(np.abs(finite))) > 0.0

        plotter = QSOFit.__new__(QSOFit)
        plotter.z = z
        plotter.wave = wave_rest
        plotter.flux = flux_rest
        plotter.err = err_rest
        plotter.wave_prereduced = wave_rest
        plotter.flux_prereduced = flux_rest
        plotter.err_prereduced = err_rest
        plotter.model_total = component("pred_spectrum_fluxes", apply_scale=False)
        plotter.host = obs_sed_component("host_obs_sed", multiplier=host_capture)
        disk_component = obs_sed_component("disk_obs_sed")
        plotter.f_pl_model = disk_component if keep_component(disk_component) else component("jqf_continuum_model")
        plotter.f_pl_model_intrinsic = plotter.f_pl_model
        plotter.f_fe_mgii_model = component("jqf_feii_model")
        plotter.f_fe_balmer_model = np.zeros_like(wave_rest)
        plotter.f_bc_model = component("jqf_balmer_model")
        plotter.f_line_model = component("jqf_line_model")
        custom_components = {
            "grahspj_torus": obs_sed_component("torus_obs_sed"),
            "grahspj_host_dust": obs_sed_component("dust_obs_sed"),
            "grahspj_sed_feii": obs_sed_component("feii_obs_sed"),
            "grahspj_sed_balmer": obs_sed_component("balmer_obs_sed"),
            "grahspj_sed_lines": obs_sed_component("line_obs_sed"),
        }
        plotter.custom_components = {
            name: model for name, model in custom_components.items() if keep_component(model)
        }
        plotter.qso = (
            plotter.f_pl_model
            + sum(plotter.custom_components.values(), np.zeros_like(wave_rest))
            + plotter.f_fe_mgii_model
            + plotter.f_bc_model
            + plotter.f_line_model
        )
        plotter.f_poly_model = np.ones_like(wave_rest)
        plotter.custom_line_components = {}
        plotter.use_psf_phot = False
        plotter.psf_model = np.array([])
        plotter.host_psf = np.array([])
        plotter.scale_psf = 1.0
        plotter.eta_psf = 1.0
        plotter.save_fig = False
        plotter.output_path = "."
        plotter.filename = str(self.config.observation.object_id)
        plotter.verbose = False
        plotter.line_component_amp_median = self._posterior_median_array(pred.get("jqf_line_amp_per_component", []))
        plotter.line_component_mu_median = self._posterior_median_array(pred.get("jqf_line_mu_per_component", []))
        plotter.line_component_sig_median = self._posterior_median_array(pred.get("jqf_line_sig_per_component", []))
        plotter.tied_line_meta = {"names": [""] * len(np.atleast_1d(plotter.line_component_amp_median))}

        plotter.plot_fig(
            plot_legend=plot_legend,
            ylims=ylims,
            plot_residual=plot_residual,
            show_plot=show_plot,
            **kwargs,
        )
        return getattr(plotter, "fig", None)
