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


class GRAHSPJ:
    """High-level single-object fitting interface for grahspj."""
    def __init__(self, config: FitConfig):
        """Initialize the fitter and build its static model context."""
        self.config = config
        self.context: ModelContext = build_model_context(config)
        self.map_result: dict[str, Any] | None = None
        self.nuts_result: dict[str, Any] | None = None
        self.samples: dict[str, Any] | None = None
        self.predictive: dict[str, Any] | None = None
        self._plot_cache: dict[str, Any] | None = None

    def _reset_fit_state(self) -> None:
        """Clear cached inference and predictive state."""
        self.map_result = None
        self.nuts_result = None
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

    def _predictive_model(self):
        """Return the bound NumPyro model used for posterior predictive products."""
        return grahsp_photometric_model(self.context, include_components=True)

    def _compute_predictive(self) -> dict[str, Any]:
        """Generate and cache predictive outputs from posterior samples."""
        if self.samples is None:
            raise RuntimeError("No fitted posterior available. Run fit_map() or fit_nuts() first.")
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 17)
        pred = Predictive(
            self._predictive_model,
            posterior_samples=self.samples,
            return_sites=[
                "pred_fluxes",
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
                "dust_luminosity",
                "dust_alpha_fit",
                "intrinsic_scatter_fit",
                "agn_bol_luminosity",
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
        plot_fig: bool = False,
        save_fig: bool = False,
        save_result: bool = False,
        output_dir: str | Path = ".",
        fig_path: str | Path | None = None,
        result_path: str | Path | None = None,
        use_map_init: bool = True,
        target_accept_prob: float | None = None,
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
        else:
            raise ValueError("fit_method must be one of: 'optax+nuts', 'optax', 'nuts'")

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

    def fit_map(self, steps: int | None = None, learning_rate: float | None = None, progress_bar: bool = True):
        """Run the Optax/NumPyro MAP optimization path."""
        import optax
        from numpyro.optim import optax_to_numpyro

        steps = int(self.config.inference.map_steps if steps is None else steps)
        learning_rate = float(self.config.inference.learning_rate if learning_rate is None else learning_rate)
        guide = AutoDelta(self._model)
        optimizer = optax_to_numpyro(optax.adam(learning_rate))
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        rng_key = jax.random.PRNGKey(self.config.inference.seed)
        svi_result = svi.run(rng_key, steps, progress_bar=progress_bar)
        params = svi_result.params
        median = guide.median(params)
        self.map_result = {"params": params, "median": median, "losses": np.asarray(getattr(svi_result, "losses", []))}
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
        raise RuntimeError("No recovered stellar mass available. Run fit_map() or fit_nuts() first.")

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
            if "dust_luminosity" in self.predictive:
                out["dust_luminosity_fit"] = float(np.median(np.asarray(self.predictive["dust_luminosity"], dtype=float)))
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
