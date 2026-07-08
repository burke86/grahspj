from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta

from .config import FitConfig, _coerce_prior_config, fit_config_from_mapping, serialize_config
from .model import grahsp_photometric_model
from .preload import ModelContext, build_model_context
from .results import FitResult, _FitState, median_mapping


def _get_nested_sampler_cls():
    """Resolve NumPyro's optional nested sampler lazily."""
    from numpyro.contrib.nested_sampling import NestedSampler

    return NestedSampler


class JAXSEDFit:
    """High-level single-object fitting interface for jaxsedfit."""
    _POSTERIOR_BUNDLE_SUFFIX = ".h5"

    def __init__(self, config: FitConfig):
        """Initialize the fitter and build its static model context.

        Parameters
        ----------
        config : object
            config value.
        """
        self.config = config
        self.context: ModelContext = build_model_context(config)
        self._fit_state = _FitState()

    def _ensure_fit_state(self) -> _FitState:
        """Return the internal fit state, creating it for legacy/test objects."""
        state = self.__dict__.get("_fit_state")
        if state is None:
            state = _FitState()
            self.__dict__["_fit_state"] = state
        return state

    @property
    def map_result(self) -> dict[str, Any] | None:
        """Latest MAP inference payload mirrored from the internal fit state."""
        return self._ensure_fit_state().map_result

    @map_result.setter
    def map_result(self, value: dict[str, Any] | None) -> None:
        """map_result helper.

        Parameters
        ----------
        value : object
            value value.
        """
        state = self._ensure_fit_state()
        state.map_result = value
        if value is not None:
            state.method = "map"

    @property
    def nuts_result(self) -> dict[str, Any] | None:
        """Latest NUTS inference payload mirrored from the internal fit state."""
        return self._ensure_fit_state().nuts_result

    @nuts_result.setter
    def nuts_result(self, value: dict[str, Any] | None) -> None:
        """nuts_result helper.

        Parameters
        ----------
        value : object
            value value.
        """
        state = self._ensure_fit_state()
        state.nuts_result = value
        if value is not None:
            state.method = "nuts"

    @property
    def ns_result(self) -> dict[str, Any] | None:
        """Latest nested-sampling payload mirrored from the internal fit state."""
        return self._ensure_fit_state().ns_result

    @ns_result.setter
    def ns_result(self, value: dict[str, Any] | None) -> None:
        """ns_result helper.

        Parameters
        ----------
        value : object
            value value.
        """
        state = self._ensure_fit_state()
        state.ns_result = value
        if value is not None:
            state.method = "ns"

    @property
    def samples(self) -> dict[str, Any] | None:
        """Posterior samples mirrored from the internal fit state."""
        return self._ensure_fit_state().samples

    @samples.setter
    def samples(self, value: dict[str, Any] | None) -> None:
        """samples helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_fit_state().samples = value

    @property
    def predictive(self) -> dict[str, Any] | None:
        """Posterior predictive outputs mirrored from the internal fit state."""
        return self._ensure_fit_state().predictive

    @predictive.setter
    def predictive(self, value: dict[str, Any] | None) -> None:
        """predictive helper.

        Parameters
        ----------
        value : object
            value value.
        """
        state = self._ensure_fit_state()
        state.predictive = value
        state.predictive_cache = None if value is None else {"plot:all": value}

    @property
    def _plot_cache(self) -> dict[str, Any] | None:
        """Plot cache mirrored from the internal fit state."""
        return self._ensure_fit_state().plot_cache

    @_plot_cache.setter
    def _plot_cache(self, value: dict[str, Any] | None) -> None:
        """_plot_cache helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_fit_state().plot_cache = value

    @property
    def _saved_summary(self) -> Mapping[str, Any] | None:
        """Summary restored from a saved posterior bundle."""
        return self._ensure_fit_state().summary

    @_saved_summary.setter
    def _saved_summary(self, value: Mapping[str, Any] | None) -> None:
        """_saved_summary helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_fit_state().summary = value

    @property
    def _loaded_posterior_path(self) -> Path | None:
        """Path restored from a saved posterior bundle."""
        return self._ensure_fit_state().path

    @_loaded_posterior_path.setter
    def _loaded_posterior_path(self, value: str | Path | None) -> None:
        """_loaded_posterior_path helper.

        Parameters
        ----------
        value : object
            value value.
        """
        self._ensure_fit_state().path = None if value is None else Path(value)

    def _reset_fit_state(self) -> None:
        """Clear cached inference and predictive state."""
        self._fit_state = _FitState()

    def _apply_runtime_overrides(
        self,
        prior_config: dict[str, Any] | None = None,
        dsps_ssp_fn: str | None = None,
    ) -> None:
        """Apply one-off fit-time overrides and rebuild context if required.

        Parameters
        ----------
        prior_config : object
            prior_config value.
        dsps_ssp_fn : object
            dsps_ssp_fn value.
        """
        rebuild_context = False
        if prior_config is not None:
            self.config.prior_config = _coerce_prior_config(prior_config)
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

    @staticmethod
    def _prediction_kind(kind: str) -> str:
        """Normalize the prediction product set name.

        Parameters
        ----------
        kind : object
            kind value.
        """
        normalized = str(kind).lower()
        aliases = {"full": "plot", "all": "plot", "sed": "plot", "photo": "photometry"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"plot", "photometry"}:
            raise ValueError("predict(kind=...) must be either 'plot' or 'photometry'.")
        return normalized

    @staticmethod
    def _subset_prediction_samples(samples: Mapping[str, Any], max_draws: int | None) -> dict[str, Any]:
        """Return posterior samples optionally limited along the leading draw axis.

        Parameters
        ----------
        samples : object
            samples value.
        max_draws : object
            max_draws value.
        """
        out: dict[str, Any] = {}
        n = None if max_draws is None else max(int(max_draws), 1)
        for key, value in samples.items():
            arr = np.asarray(value)
            out[key] = arr if n is None or arr.ndim == 0 else arr[:n]
        return out

    @staticmethod
    def _median_prediction_samples(samples: Mapping[str, Any]) -> dict[str, Any]:
        """Return one posterior draw built from per-site medians.

        Parameters
        ----------
        samples : object
            samples value.
        """
        return {key: np.expand_dims(np.asarray(value), axis=0) for key, value in median_mapping(samples).items()}

    @staticmethod
    def _predictive_return_sites(kind: str) -> list[str]:
        """Return deterministic sites needed for a prediction product set.

        Parameters
        ----------
        kind : object
            kind value.
        """
        photometry_sites = [
            "pred_fluxes",
            "pred_spectrum_fluxes",
            "spec_continuum_model_fluxes",
            "spec_host_model_fluxes",
            "spec_disk_model_fluxes",
            "spec_torus_model_fluxes",
            "spec_wave_obs",
            "spec_spectrum_index",
            "spectrum_scale_fit",
            "log_spectrum_scale_fit",
            "spectrum_host_capture_fraction",
            "spectroscopy_loglike",
            "spectroscopy_likelihood_weight",
            "jqf_continuum_model",
            "jqf_line_model",
            "jqf_line_model_aperture",
            "jqf_line_model_broad",
            "jqf_line_model_narrow",
            "jqf_line_model_narrow_aperture",
            "jqf_line_amp_per_component",
            "jqf_line_mu_per_component",
            "jqf_line_sig_per_component",
            "jqf_line_narrow_fwhm_kms",
            "jqf_line_narrow_amp_scale",
            "jqf_feii_model",
            "jqf_balmer_model",
            "jqf_total_model",
            "rest_wave",
            "obs_wave",
            "redshift_fit",
            "nebular_line_scale_fit",
            "log_dust_luminosity_fit",
            "dust_alpha_fit",
            "nebular_logU_fit",
            "nebular_zgas_fit",
            "nebular_ne_fit",
            "nebular_f_esc_fit",
            "nebular_f_dust_fit",
            "nebular_f_dust_fraction_fit",
            "nebular_lines_width_fit",
            "nebular_corr_fit",
            "nebular_n_ly_young_fit",
            "nebular_n_ly_old_fit",
            "fracAGN_5100_fit",
            "log_agn_bol_luminosity_fit",
            "log_disk_luminosity_fit",
            "agn_variability_nev",
            "host_total_fluxes",
            "host_capture_source_fluxes",
            "agn_narrow_line_fluxes_total",
            "captured_agn_narrow_line_fluxes",
            "extended_capture_source_fluxes",
            "captured_extended_source_fluxes",
            "host_capture_fraction_fluxes",
            "log_host_capture_scale_arcsec_fit",
            "host_capture_slope_fit",
            "transmitted_fraction_fluxes",
            "absolute_flux_scale_logprior",
        ]
        if kind == "photometry":
            return photometry_sites
        return photometry_sites + [
            "agn_fluxes",
            "host_fluxes",
            "dust_fluxes",
            "nebular_fluxes",
            "nebular_lines_fluxes",
            "nebular_continuum_fluxes",
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
            "total_rest_sed",
            "agn_rest_sed",
            "host_rest_sed",
            "host_total_rest_sed",
            "host_absorbed_rest_sed",
            "dust_rest_sed",
            "nebular_rest_sed",
            "nebular_lines_rest_sed",
            "nebular_continuum_rest_sed",
            "nebular_absorption_rest_sed",
            "disk_rest_sed",
            "torus_rest_sed",
            "feii_rest_sed",
            "line_rest_sed",
            "line_bl_rest_sed",
            "line_nl_rest_sed",
            "line_liner_rest_sed",
            "balmer_rest_sed",
            "total_obs_sed",
            "total_local_lines_obs_wave",
            "total_local_lines_obs_sed",
            "agn_obs_sed",
            "host_obs_sed",
            "host_total_obs_sed",
            "dust_obs_sed",
            "nebular_obs_sed",
            "nebular_lines_obs_sed",
            "nebular_lines_local_obs_wave",
            "nebular_lines_local_obs_sed",
            "nebular_continuum_obs_sed",
            "disk_obs_sed",
            "torus_obs_sed",
            "feii_obs_sed",
            "line_obs_sed",
            "line_bl_obs_sed",
            "line_nl_obs_sed",
            "line_liner_obs_sed",
            "balmer_obs_sed",
        ]

    def _make_result(
        self,
        *,
        method: str,
        path: str | Path | None = None,
        figure: Any = None,
        summary: dict[str, Any] | None = None,
    ) -> FitResult:
        """Build a public result object from the current mirrored fit state.

        Parameters
        ----------
        method : object
            method value.
        path : object
            path value.
        figure : object
            figure value.
        summary : object
            summary value.
        """
        state = self._ensure_fit_state()
        state.method = method
        if path is not None:
            state.path = Path(path)
        if figure is not None:
            state.figure = figure
        samples = state.samples
        map_result = state.map_result
        if method == "map" and map_result is not None and "median" in map_result:
            median = dict(map_result["median"])
        else:
            median = median_mapping(samples)
        if summary is None and samples is not None:
            try:
                summary = self.summary()
            except AttributeError:
                summary = None
        if summary is not None:
            state.summary = summary
        return FitResult(
            fitter=self,
            samples=samples,
            median=median,
            method=method,
            summary=summary,
            path=state.path,
            figure=state.figure,
            _state=state,
        )

    def _compute_predictive(
        self,
        *,
        _state: _FitState | None = None,
        kind: str = "plot",
        max_draws: int | None = None,
        posterior_samples: Mapping[str, Any] | None = None,
        cache: bool = True,
    ) -> dict[str, Any]:
        """Generate and cache predictive outputs from posterior samples.

        Parameters
        ----------
        _state : object
            _state value.
        kind : object
            kind value.
        max_draws : object
            max_draws value.
        posterior_samples : object
            posterior_samples value.
        cache : object
            cache value.
        """
        state = self._ensure_fit_state() if _state is None else _state
        kind = self._prediction_kind(kind)
        samples = state.samples if posterior_samples is None else posterior_samples
        if samples is None:
            raise RuntimeError("No fitted posterior available. Run fit_map(), fit_nuts(), or fit_ns() first.")
        cache_key = f"{kind}:{'all' if max_draws is None else int(max_draws)}"
        if cache and posterior_samples is None and state.predictive_cache is not None and cache_key in state.predictive_cache:
            return dict(state.predictive_cache[cache_key])
        include_components = kind == "plot"
        draw_samples = self._subset_prediction_samples(samples, max_draws)
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 17)
        pred = Predictive(
            lambda: grahsp_photometric_model(self.context, include_components=include_components),
            posterior_samples=draw_samples,
            return_sites=self._predictive_return_sites(kind),
        )(rng_key)
        predictive = {k: np.asarray(v) for k, v in pred.items()}
        if cache and posterior_samples is None:
            if state.predictive_cache is None:
                state.predictive_cache = {}
            state.predictive_cache[cache_key] = predictive
            if kind == "plot" and max_draws is None:
                state.predictive = predictive
        return predictive

    def fit(
        self,
        progress_bar: bool = True,
    ):
        """Run the requested inference path and optional plotting/saving helpers.

        Parameters
        ----------
        progress_bar : object
            progress_bar value.
        """
        inference = self.config.inference
        output = self.config.output
        method = str(inference.method).lower()
        output_dir = Path(output.output_dir)
        if method == "optax":
            map_kwargs: dict[str, Any] = {
                "progress_bar": progress_bar,
                "steps": inference.map_steps,
                "learning_rate": inference.learning_rate,
                "staged": inference.staged_map,
            }
            if inference.staged_steps is not None:
                map_kwargs["staged_steps"] = inference.staged_steps
            fit_output: dict[str, Any] | Any = self.fit_map(
                **map_kwargs,
            )
        elif method == "nuts":
            nuts_kwargs: dict[str, Any] = {
                "progress_bar": progress_bar,
                "num_warmup": inference.num_warmup,
                "num_samples": inference.num_samples,
                "num_chains": inference.num_chains,
                "target_accept_prob": inference.target_accept_prob,
                "dense_mass": inference.dense_mass,
                "max_tree_depth": inference.max_tree_depth,
                "use_map_init": inference.use_map_init,
            }
            fit_output = self.fit_nuts(
                **nuts_kwargs,
            )
        elif method == "optax+nuts":
            map_kwargs = {
                "progress_bar": progress_bar,
                "steps": inference.map_steps,
                "learning_rate": inference.learning_rate,
                "staged": inference.staged_map,
            }
            if inference.staged_steps is not None:
                map_kwargs["staged_steps"] = inference.staged_steps
            map_result = self.fit_map(
                **map_kwargs,
            )
            nuts_kwargs = {
                "progress_bar": progress_bar,
                "num_warmup": inference.num_warmup,
                "num_samples": inference.num_samples,
                "num_chains": inference.num_chains,
                "target_accept_prob": inference.target_accept_prob,
                "dense_mass": inference.dense_mass,
                "max_tree_depth": inference.max_tree_depth,
                "use_map_init": inference.use_map_init,
            }
            nuts_result = self.fit_nuts(
                **nuts_kwargs,
            )
            fit_output = {"map": map_result, "nuts": nuts_result}
        elif method == "ns":
            ns_kwargs: dict[str, Any] = {"progress_bar": progress_bar}
            if inference.ns_num_live_points is not None:
                ns_kwargs["num_live_points"] = inference.ns_num_live_points
            if inference.ns_max_samples is not None:
                ns_kwargs["max_samples"] = inference.ns_max_samples
            if inference.ns_dlogz is not None:
                ns_kwargs["dlogz"] = inference.ns_dlogz
            if inference.ns_resamples is not None:
                ns_kwargs["num_resamples"] = inference.ns_resamples
            ns_kwargs["difficult_model"] = bool(inference.ns_difficult_model)
            ns_kwargs["parameter_estimation"] = bool(inference.ns_parameter_estimation)
            if inference.ns_num_parallel_workers is not None:
                ns_kwargs["num_parallel_workers"] = inference.ns_num_parallel_workers
            if inference.ns_init_efficiency_threshold is not None:
                ns_kwargs["init_efficiency_threshold"] = inference.ns_init_efficiency_threshold
            if inference.ns_max_likelihood_evals is not None:
                ns_kwargs["max_likelihood_evals"] = inference.ns_max_likelihood_evals
            if inference.ns_efficiency_threshold is not None:
                ns_kwargs["efficiency_threshold"] = inference.ns_efficiency_threshold
            fit_output = self.fit_ns(**ns_kwargs)
        else:
            raise ValueError("method must be one of: 'optax+nuts', 'optax', 'nuts', 'ns'")

        saved_result_path = None
        saved_fig_path = None
        fig = None
        if output.save_result:
            if output.result_path is None:
                saved_result_path = self.save(output_dir)
            else:
                saved_result_path = self.save(output.result_path)
        if output.plot_fig or output.save_fig:
            fig_path = Path(output.fig_path) if output.fig_path is not None else None
            if fig_path is None and output.save_fig:
                fig_path = output_dir / f"{self.config.observation.object_id}_sed.png"
            fig = self.plot_sed(output_path=fig_path if output.save_fig else None, show=output.plot_fig or output.show_plot)
            if output.save_fig:
                saved_fig_path = Path(fig_path) if fig_path is not None else None

        # Lightweight test doubles may call fit() on a partially constructed object
        # without config/context. Preserve the direct fit payload for that case only.
        if not hasattr(self, "config"):
            return fit_output
        return self._make_result(
            method=method,
            path=saved_result_path,
            figure=fig,
            summary=self.summary() if getattr(self, "samples", None) is not None else None,
        )

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
        """Run one Optax/NumPyro AutoDelta MAP stage.

        Parameters
        ----------
        model_fn : object
            model_fn value.
        steps : object
            steps value.
        learning_rate : object
            learning_rate value.
        progress_bar : object
            progress_bar value.
        rng_seed : object
            rng_seed value.
        init_values : object
            init_values value.
        """
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
        staged: bool | None = None,
        staged_steps: int | None = None,
    ):
        """Run the Optax/NumPyro MAP optimization path.

        Parameters
        ----------
        steps : object
            steps value.
        learning_rate : object
            learning_rate value.
        progress_bar : object
            progress_bar value.
        staged : object
            staged value.
        staged_steps : object
            staged_steps value.
        """
        self._reset_fit_state()
        steps = int(self.config.inference.map_steps if steps is None else steps)
        learning_rate = float(self.config.inference.learning_rate if learning_rate is None else learning_rate)
        staged = bool(self.config.inference.staged_map if staged is None else staged)
        if staged_steps is None:
            staged_steps = self.config.inference.staged_steps
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
        return self._make_result(method="map")

    def fit_nuts(
        self,
        num_warmup: int | None = None,
        num_samples: int | None = None,
        num_chains: int | None = None,
        target_accept_prob: float | None = None,
        dense_mass: bool | None = None,
        max_tree_depth: int | None = None,
        use_map_init: bool = True,
        progress_bar: bool = True,
    ):
        """Run NUTS sampling, optionally initializing from the MAP solution.

        Parameters
        ----------
        num_warmup : object
            num_warmup value.
        num_samples : object
            num_samples value.
        num_chains : object
            num_chains value.
        target_accept_prob : object
            target_accept_prob value.
        dense_mass : object
            dense_mass value.
        max_tree_depth : object
            max_tree_depth value.
        use_map_init : object
            use_map_init value.
        progress_bar : object
            progress_bar value.
        """
        if use_map_init and self.map_result is None:
            self.fit_map(progress_bar=progress_bar)
        map_result = self.map_result
        self._fit_state = _FitState(map_result=map_result, method="nuts")
        num_warmup = int(self.config.inference.num_warmup if num_warmup is None else num_warmup)
        num_samples = int(self.config.inference.num_samples if num_samples is None else num_samples)
        num_chains = int(self.config.inference.num_chains if num_chains is None else num_chains)
        target_accept_prob = float(self.config.inference.target_accept_prob if target_accept_prob is None else target_accept_prob)
        dense_mass = bool(self.config.inference.dense_mass if dense_mass is None else dense_mass)
        max_tree_depth = int(self.config.inference.max_tree_depth if max_tree_depth is None else max_tree_depth)
        init_values = None
        if self.map_result is not None:
            init_values = {k: np.asarray(v) for k, v in self.map_result["median"].items() if np.ndim(v) != 0 or np.isfinite(v)}
        kernel = NUTS(
            self._model,
            init_strategy=init_to_value(values=init_values) if init_values else None,
            target_accept_prob=target_accept_prob,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar, jit_model_args=False)
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 1)
        mcmc.run(rng_key)
        samples = mcmc.get_samples()
        self.nuts_result = {"mcmc": mcmc}
        self.samples = {k: np.asarray(v) for k, v in samples.items()}
        self.predictive = None
        return self._make_result(method="nuts")

    def fit_ns(
        self,
        num_live_points: int | None = None,
        max_samples: int | None = None,
        dlogz: float | None = None,
        num_resamples: int | None = None,
        difficult_model: bool | None = None,
        parameter_estimation: bool | None = None,
        num_parallel_workers: int | None = None,
        init_efficiency_threshold: float | None = None,
        max_likelihood_evals: int | None = None,
        efficiency_threshold: float | None = None,
        ns_difficult_model: bool | None = None,
        ns_parameter_estimation: bool | None = None,
        ns_num_parallel_workers: int | None = None,
        ns_init_efficiency_threshold: float | None = None,
        ns_max_likelihood_evals: int | None = None,
        ns_efficiency_threshold: float | None = None,
        ns_resamples: int | None = None,
        progress_bar: bool = True,
    ):
        """Run full-model nested sampling and resample equal-weight posterior draws.

        Parameters
        ----------
        num_live_points : object
            num_live_points value.
        max_samples : object
            max_samples value.
        dlogz : object
            dlogz value.
        num_resamples : object
            num_resamples value.
        difficult_model : object
            difficult_model value.
        parameter_estimation : object
            parameter_estimation value.
        num_parallel_workers : object
            num_parallel_workers value.
        init_efficiency_threshold : object
            init_efficiency_threshold value.
        max_likelihood_evals : object
            max_likelihood_evals value.
        efficiency_threshold : object
            efficiency_threshold value.
        ns_difficult_model : object
            ns_difficult_model value.
        ns_parameter_estimation : object
            ns_parameter_estimation value.
        ns_num_parallel_workers : object
            ns_num_parallel_workers value.
        ns_init_efficiency_threshold : object
            ns_init_efficiency_threshold value.
        ns_max_likelihood_evals : object
            ns_max_likelihood_evals value.
        ns_efficiency_threshold : object
            ns_efficiency_threshold value.
        ns_resamples : object
            ns_resamples value.
        progress_bar : object
            progress_bar value.
        """
        self._reset_fit_state()
        NestedSampler = _get_nested_sampler_cls()

        if ns_difficult_model is not None:
            difficult_model = ns_difficult_model
        if ns_parameter_estimation is not None:
            parameter_estimation = ns_parameter_estimation
        if ns_num_parallel_workers is not None and num_parallel_workers is None:
            num_parallel_workers = ns_num_parallel_workers
        if ns_init_efficiency_threshold is not None and init_efficiency_threshold is None:
            init_efficiency_threshold = ns_init_efficiency_threshold
        if ns_max_likelihood_evals is not None and max_likelihood_evals is None:
            max_likelihood_evals = ns_max_likelihood_evals
        if ns_efficiency_threshold is not None and efficiency_threshold is None:
            efficiency_threshold = ns_efficiency_threshold
        if ns_resamples is not None and num_resamples is None:
            num_resamples = ns_resamples
        inference = self.config.inference
        if num_live_points is None:
            num_live_points = inference.ns_num_live_points
        if max_samples is None:
            max_samples = inference.ns_max_samples
        if dlogz is None:
            dlogz = inference.ns_dlogz
        if num_resamples is None:
            num_resamples = inference.ns_resamples
        if difficult_model is None:
            difficult_model = inference.ns_difficult_model
        if parameter_estimation is None:
            parameter_estimation = inference.ns_parameter_estimation
        if num_parallel_workers is None:
            num_parallel_workers = inference.ns_num_parallel_workers
        if init_efficiency_threshold is None:
            init_efficiency_threshold = inference.ns_init_efficiency_threshold
        if max_likelihood_evals is None:
            max_likelihood_evals = inference.ns_max_likelihood_evals
        if efficiency_threshold is None:
            efficiency_threshold = inference.ns_efficiency_threshold

        constructor_kwargs: dict[str, Any] = {"verbose": bool(progress_bar)}
        if num_live_points is not None:
            constructor_kwargs["num_live_points"] = int(num_live_points)
        constructor_kwargs["max_samples"] = None if max_samples is None else int(max_samples)
        if difficult_model:
            constructor_kwargs["difficult_model"] = bool(difficult_model)
        if parameter_estimation:
            constructor_kwargs["parameter_estimation"] = bool(parameter_estimation)
        if num_parallel_workers is not None:
            constructor_kwargs["num_parallel_workers"] = int(num_parallel_workers)
        if init_efficiency_threshold is not None:
            constructor_kwargs["init_efficiency_threshold"] = float(init_efficiency_threshold)
        termination_kwargs: dict[str, Any] = {}
        if dlogz is not None:
            termination_kwargs["dlogZ"] = float(dlogz)
        if max_likelihood_evals is not None:
            termination_kwargs["max_num_likelihood_evaluations"] = int(max_likelihood_evals)
        if efficiency_threshold is not None:
            termination_kwargs["efficiency_threshold"] = float(efficiency_threshold)

        sampler = NestedSampler(
            self._model,
            constructor_kwargs=constructor_kwargs,
            termination_kwargs=termination_kwargs,
        )
        rng_key = jax.random.PRNGKey(self.config.inference.seed + 2)
        sampler.run(rng_key)
        posterior_rng_key = jax.random.PRNGKey(self.config.inference.seed + 3)
        posterior_num_samples = int(self.config.inference.num_samples if num_resamples is None else num_resamples)
        samples = sampler.get_samples(
            posterior_rng_key,
            num_samples=posterior_num_samples,
            group_by_chain=False,
        )
        self.ns_result = {
            "sampler": sampler,
            "results": getattr(sampler, "_results", None),
            "constructor_kwargs": dict(getattr(sampler, "constructor_kwargs", constructor_kwargs)),
            "termination_kwargs": dict(getattr(sampler, "termination_kwargs", termination_kwargs)),
            "num_resamples": posterior_num_samples,
        }
        self.samples = {k: np.asarray(v) for k, v in samples.items()}
        self.predictive = None
        return self._make_result(method="ns")

    def predict(
        self,
        posterior: str = "latest",
        *,
        kind: str = "plot",
        max_draws: int | None = None,
        _state: _FitState | None = None,
    ) -> dict[str, Any]:
        """Return cached predictive outputs or generate them on demand.

        ``kind="photometry"`` returns lightweight photometry/spectrum products.
        ``kind="plot"`` returns the full component SED products used by plotting.

        Parameters
        ----------
        posterior : object
            posterior value.
        kind : object
            kind value.
        max_draws : object
            max_draws value.
        _state : object
            _state value.
        """
        state = self._ensure_fit_state() if _state is None else _state
        kind = self._prediction_kind(kind)
        if kind == "plot" and max_draws is None and state.predictive is not None:
            return dict(state.predictive)
        if kind == "plot" and max_draws is None and state is self._ensure_fit_state():
            return self._compute_predictive()
        return self._compute_predictive(_state=state, kind=kind, max_draws=max_draws)

    def predict_median(
        self,
        posterior: str = "latest",
        *,
        kind: str = "plot",
        _state: _FitState | None = None,
    ) -> dict[str, Any]:
        """Evaluate predictive products once at the posterior median parameters.

        Parameters
        ----------
        posterior : object
            posterior value.
        kind : object
            kind value.
        _state : object
            _state value.
        """
        state = self._ensure_fit_state() if _state is None else _state
        if state.samples is None:
            raise RuntimeError("No fitted posterior available. Run fit_map(), fit_nuts(), or fit_ns() first.")
        median_samples = self._median_prediction_samples(state.samples)
        return self._compute_predictive(
            _state=state,
            kind=kind,
            posterior_samples=median_samples,
            cache=False,
        )

    def recovered_log_stellar_mass(self, *, _state: _FitState | None = None) -> float:
        """Return the median recovered stellar mass from the fitted posterior.

        Parameters
        ----------
        _state : object
            _state value.
        """
        state = self._ensure_fit_state() if _state is None else _state
        if state.samples is not None and "log_stellar_mass" in state.samples:
            return float(np.median(np.asarray(state.samples["log_stellar_mass"], dtype=float)))
        if state.map_result is not None and "median" in state.map_result and "log_stellar_mass" in state.map_result["median"]:
            return float(np.asarray(state.map_result["median"]["log_stellar_mass"], dtype=float))
        raise RuntimeError("No recovered stellar mass available. Run fit_map(), fit_nuts(), or fit_ns() first.")

    def summary(self, *, _state: _FitState | None = None) -> dict[str, Any]:
        """Summarize posterior medians and selected derived quantities.

        Parameters
        ----------
        _state : object
            _state value.
        """
        state = self._ensure_fit_state() if _state is None else _state
        if state.samples is None:
            raise RuntimeError("No fitted posterior available.")
        out: dict[str, Any] = {}
        for key, value in state.samples.items():
            arr = np.asarray(value)
            out[f"{key}_median"] = np.median(arr, axis=0).tolist() if arr.ndim > 1 else float(np.median(arr))
        if "host_age_weights" in state.samples:
            ages = np.power(10.0, np.asarray(self.context.ssp_data.ssp_lg_age_gyr, dtype=float))
            age_weights = np.median(np.asarray(state.samples["host_age_weights"]), axis=0)
            age_weight_sum = np.sum(age_weights)
            out["host_age_weighted_gyr"] = float(np.sum(age_weights * ages) / age_weight_sum) if age_weight_sum > 0 else -1.0
        if "host_lgmet_weights" in state.samples:
            mets = np.asarray(self.context.ssp_data.ssp_lgmet, dtype=float)
            lgmet_weights = np.median(np.asarray(state.samples["host_lgmet_weights"]), axis=0)
            lgmet_weight_sum = np.sum(lgmet_weights)
            out["host_lgmet_weighted"] = float(np.sum(lgmet_weights * mets) / lgmet_weight_sum) if lgmet_weight_sum > 0 else -99.0
        if "gal_lgmet" in state.samples:
            out["gal_lgmet_fit"] = float(np.median(np.asarray(state.samples["gal_lgmet"], dtype=float)))
        if "gal_lgmet_scatter" in state.samples:
            out["gal_lgmet_scatter_fit"] = float(np.median(np.asarray(state.samples["gal_lgmet_scatter"], dtype=float)))
        if "log_stellar_mass" in state.samples:
            out["log_stellar_mass_fit"] = self.recovered_log_stellar_mass(_state=state)
        if "dust_alpha" in state.samples:
            out["dust_alpha_fit"] = float(np.median(np.asarray(state.samples["dust_alpha"], dtype=float)))
        if state.predictive is not None:
            out["pred_fluxes_median"] = np.median(np.asarray(state.predictive["pred_fluxes"]), axis=0).tolist()
            if "log_dust_luminosity_fit" in state.predictive:
                out["log_dust_luminosity_fit"] = float(np.median(np.asarray(state.predictive["log_dust_luminosity_fit"], dtype=float)))
            if "log_agn_bol_luminosity_fit" in state.predictive:
                out["log_agn_bol_luminosity_fit"] = float(np.median(np.asarray(state.predictive["log_agn_bol_luminosity_fit"], dtype=float)))
            if "log_disk_luminosity_fit" in state.predictive:
                out["log_disk_luminosity_fit"] = float(np.median(np.asarray(state.predictive["log_disk_luminosity_fit"], dtype=float)))
            if "absolute_flux_scale_logprior" in state.predictive:
                out["absolute_flux_scale_logprior"] = float(np.median(np.asarray(state.predictive["absolute_flux_scale_logprior"], dtype=float)))
        state.summary = out
        return out

    @staticmethod
    def _hdf5_scalar_string_dtype():
        """Return the UTF-8 scalar string dtype used in HDF5 bundles."""
        return h5py.string_dtype(encoding="utf-8")

    @classmethod
    def _write_hdf5_node(cls, parent, name, value):
        """Write one recursively serialized Python value into an HDF5 group.

        Parameters
        ----------
        parent : object
            parent value.
        name : object
            name value.
        value : object
            value value.
        """
        if value is None:
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "none"
            return

        if isinstance(value, dict):
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "dict"
            for idx, (key, item) in enumerate(value.items()):
                item_grp = grp.create_group(f"item_{idx:08d}")
                cls._write_hdf5_node(item_grp, "key", str(key))
                cls._write_hdf5_node(item_grp, "value", item)
            return

        if isinstance(value, list):
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "list"
            for idx, item in enumerate(value):
                cls._write_hdf5_node(grp, f"item_{idx:08d}", item)
            return

        if isinstance(value, tuple):
            grp = parent.create_group(name)
            grp.attrs["node_type"] = "tuple"
            for idx, item in enumerate(value):
                cls._write_hdf5_node(grp, f"item_{idx:08d}", item)
            return

        if isinstance(value, (np.ndarray, np.generic)):
            arr = np.asarray(value)
            ds_kwargs = {}
            if arr.ndim > 0:
                ds_kwargs["compression"] = "gzip"
                ds_kwargs["shuffle"] = True
            ds = parent.create_dataset(name, data=arr, **ds_kwargs)
            ds.attrs["node_type"] = "ndarray"
            return

        if isinstance(value, bool):
            ds = parent.create_dataset(name, data=np.bool_(value))
            ds.attrs["node_type"] = "scalar_bool"
            return

        if isinstance(value, int):
            ds = parent.create_dataset(name, data=np.int64(value))
            ds.attrs["node_type"] = "scalar_int"
            return

        if isinstance(value, float):
            ds = parent.create_dataset(name, data=np.float64(value))
            ds.attrs["node_type"] = "scalar_float"
            return

        if isinstance(value, str):
            ds = parent.create_dataset(name, data=np.array(value, dtype=cls._hdf5_scalar_string_dtype()))
            ds.attrs["node_type"] = "scalar_str"
            return

        raise TypeError(f"Unsupported value type in posterior bundle: {type(value)!r}")

    @classmethod
    def _read_hdf5_node(cls, parent, name):
        """Read one recursively serialized Python value from an HDF5 group.

        Parameters
        ----------
        parent : object
            parent value.
        name : object
            name value.
        """
        node = parent[name]
        if isinstance(node, h5py.Dataset):
            node_type = node.attrs.get("node_type", "ndarray")
            if isinstance(node_type, bytes):
                node_type = node_type.decode("utf-8")
            if node_type == "scalar_str":
                return node.asstr()[()]
            value = node[()]
            if node_type == "scalar_bool":
                return bool(value)
            if node_type == "scalar_int":
                return int(value)
            if node_type == "scalar_float":
                return float(value)
            return np.asarray(value)

        node_type = node.attrs.get("node_type", "")
        if isinstance(node_type, bytes):
            node_type = node_type.decode("utf-8")
        if node_type == "none":
            return None
        if node_type == "dict":
            out = {}
            for item_name in sorted(node.keys()):
                item_grp = node[item_name]
                key = cls._read_hdf5_node(item_grp, "key")
                out[str(key)] = cls._read_hdf5_node(item_grp, "value")
            return out
        if node_type == "list":
            return [cls._read_hdf5_node(node, item_name) for item_name in sorted(node.keys())]
        if node_type == "tuple":
            return tuple(cls._read_hdf5_node(node, item_name) for item_name in sorted(node.keys()))
        raise TypeError(f"Unsupported HDF5 node type in posterior bundle: {node_type!r}")

    @staticmethod
    def _write_array_group(parent, values: dict[str, Any]) -> None:
        """Write an array mapping as compressed HDF5 datasets.

        Parameters
        ----------
        parent : object
            parent value.
        values : object
            values value.
        """
        for name, value in values.items():
            arr = np.asarray(value)
            ds_kwargs = {}
            if arr.ndim > 0:
                ds_kwargs["compression"] = "gzip"
                ds_kwargs["shuffle"] = True
            parent.create_dataset(str(name), data=arr, **ds_kwargs)

    @classmethod
    def _posterior_bundle_path(cls, path: str | Path | None, object_id: str) -> Path:
        """Resolve an output directory or explicit HDF5 path.

        Parameters
        ----------
        path : object
            path value.
        object_id : object
            object_id value.
        """
        resolved = Path("." if path is None else path)
        if resolved.suffix == cls._POSTERIOR_BUNDLE_SUFFIX:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            return resolved
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved / f"{object_id}_samples{cls._POSTERIOR_BUNDLE_SUFFIX}"

    def save(self, output_dir: str | Path | None = None, *, _state: _FitState | None = None) -> Path:
        """Serialize config, posterior samples, and predictive outputs to HDF5.

        Parameters
        ----------
        output_dir : object
            output_dir value.
        _state : object
            _state value.
        """
        state = self._ensure_fit_state() if _state is None else _state
        out = self._posterior_bundle_path(output_dir, self.config.observation.object_id)
        samples = {k: np.asarray(v) for k, v in (state.samples or {}).items()}
        predictive = {k: np.asarray(v) for k, v in self.predict(_state=state).items()} if state.samples is not None else {}

        with h5py.File(out, "w") as h5f:
            h5f.attrs["posterior_bundle_format"] = "jaxsedfit_samples_meta_v1"
            self._write_hdf5_node(h5f, "config", serialize_config(self.config))
            self._write_hdf5_node(h5f, "summary", self.summary(_state=state) if state.samples is not None else None)
            self._write_hdf5_node(h5f, "mw_ebv", self.context.mw_ebv)
            samples_grp = h5f.create_group("samples")
            self._write_array_group(samples_grp, samples)
            predictive_grp = h5f.create_group("predictive")
            self._write_array_group(predictive_grp, predictive)
        state.path = out
        return out

    @staticmethod
    def _resolve_posterior_path(path: str | Path | None = None) -> Path:
        """Resolve a saved HDF5 posterior path or unique posterior in a directory.


        Parameters
        ----------
        path : object
            path value.
        """
        if path is None:
            path = "."
        resolved = Path(path)
        if resolved.is_dir():
            matches = sorted(resolved.glob("*_samples.h5"))
            if not matches:
                raise FileNotFoundError(f"No *_samples.h5 file found under: {resolved}")
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple *_samples.h5 files found under: {resolved}. "
                    "Pass an explicit posterior file path."
                )
            resolved = matches[0]
        if not resolved.exists():
            raise FileNotFoundError(f"Posterior bundle not found: {resolved}")
        return resolved

    @classmethod
    def load(cls, path: str | Path | None = None) -> "JAXSEDFit":
        """Load a posterior bundle written by :meth:`save`.

        Parameters
        ----------
        path
            Path to a ``*_samples.h5`` file, or a directory containing exactly
            one such file. Defaults to the current directory.

        Returns
        -------
        JAXSEDFit
            A configured fitter with posterior samples and cached predictive
            outputs restored from disk.
        """
        posterior_path = cls._resolve_posterior_path(path)
        with h5py.File(posterior_path, "r") as h5f:
            if "samples" not in h5f or "config" not in h5f:
                raise ValueError(f"Unsupported posterior bundle schema: {posterior_path}")
            payload = {
                "config": cls._read_hdf5_node(h5f, "config"),
                "summary": cls._read_hdf5_node(h5f, "summary") if "summary" in h5f else None,
                "mw_ebv": cls._read_hdf5_node(h5f, "mw_ebv") if "mw_ebv" in h5f else None,
                "samples": {k: np.asarray(h5f["samples"][k][()]) for k in h5f["samples"].keys()},
                "predictive": {k: np.asarray(h5f["predictive"][k][()]) for k in h5f.get("predictive", {}).keys()},
            }
        if not isinstance(payload, dict) or "config" not in payload:
            raise ValueError(f"Unsupported posterior bundle schema: {posterior_path}")

        config = payload["config"]
        if not isinstance(config, FitConfig):
            config = fit_config_from_mapping(config)
        fitter = cls(config)
        samples = payload.get("samples")
        fitter.samples = None if samples is None else {k: np.asarray(v) for k, v in samples.items()}
        predictive = payload.get("predictive")
        fitter.predictive = None if predictive is None else {k: np.asarray(v) for k, v in predictive.items()}
        fitter._saved_summary = payload.get("summary")
        fitter._loaded_posterior_path = posterior_path
        return fitter

    load_from_samples = load

    @classmethod
    def load_result(cls, path: str | Path | None = None) -> FitResult:
        """Load a posterior bundle and wrap it in a :class:`FitResult`.

        Parameters
        ----------
        path : object
            path value.
        """
        fitter = cls.load(path)
        return fitter._make_result(
            method="loaded",
            path=getattr(fitter, "_loaded_posterior_path", None),
            summary=getattr(fitter, "_saved_summary", None),
        )

    def plot_sed(
        self,
        output_path: str | Path | None = None,
        posterior: str = "latest",
        show: bool = False,
        annotate_band_names: bool = True,
    ):
        """Plot the fitted SED using the package plotting helper.

        Parameters
        ----------
        output_path : object
            output_path value.
        posterior : object
            posterior value.
        show : object
            show value.
        annotate_band_names : object
            annotate_band_names value.
        """
        from .plotting import plot_fit_sed

        return plot_fit_sed(
            self,
            output_path=output_path,
            posterior=posterior,
            show=show,
            annotate_band_names=annotate_band_names,
        )

    def plot_corner(
        self,
        output_path: str | Path | None = None,
        params: list[str] | tuple[str, ...] | None = None,
        max_params: int | None = 12,
        labels: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        truths: dict[str, float | None] | list[float | None] | tuple[float | None, ...] | None = None,
        show: bool = False,
        **corner_kwargs,
    ):
        """Plot scalar posterior samples with the corner package.

        Parameters
        ----------
        output_path : object
            output_path value.
        params : object
            params value.
        max_params : object
            max_params value.
        labels : object
            labels value.
        truths : object
            truths value.
        show : object
            show value.
        **corner_kwargs : dict
            Additional keyword arguments.
        """
        from .plotting import plot_corner

        return plot_corner(
            self,
            output_path=output_path,
            params=params,
            max_params=max_params,
            labels=labels,
            truths=truths,
            show=show,
            **corner_kwargs,
        )

    def plot_trace(
        self,
        output_path: str | Path | None = None,
        params: list[str] | tuple[str, ...] | None = None,
        max_params: int | None = 12,
        show: bool = False,
    ):
        """Plot scalar posterior sample traces, preserving chains when available.

        Parameters
        ----------
        output_path : object
            output_path value.
        params : object
            params value.
        max_params : object
            max_params value.
        show : object
            show value.
        """
        from .plotting import plot_trace

        return plot_trace(
            self,
            output_path=output_path,
            params=params,
            max_params=max_params,
            show=show,
        )

    @staticmethod
    def _posterior_median_array(value: Any) -> np.ndarray:
        """Return a median predictive array over the leading sample axis.

        Parameters
        ----------
        value : object
            value value.
        """
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0 or arr.size == 0:
            return arr
        return np.nanmedian(arr, axis=0)

    @staticmethod
    def _mjy_to_rest_flambda_1e17(wave_obs: np.ndarray, flux_mjy: np.ndarray, redshift: float) -> np.ndarray:
        """Convert observed-frame mJy to jaxqsofit rest-frame f_lambda units.

        Parameters
        ----------
        wave_obs : object
            wave_obs value.
        flux_mjy : object
            flux_mjy value.
        redshift : object
            redshift value.
        """
        wave_obs = np.asarray(wave_obs, dtype=float)
        flux_mjy = np.asarray(flux_mjy, dtype=float)
        c_ang_s = 2.99792458e18
        flam_obs_cgs = flux_mjy * 1.0e-26 * c_ang_s / np.clip(wave_obs**2, 1.0e-30, None)
        return flam_obs_cgs * (1.0 + float(redshift)) / 1.0e-17

    @staticmethod
    def _obs_flambda_to_rest_flambda_1e17(flux_lambda_obs: np.ndarray, redshift: float) -> np.ndarray:
        """Convert observed-frame W/m^2/Angstrom to jaxqsofit rest-frame units.

        Parameters
        ----------
        flux_lambda_obs : object
            flux_lambda_obs value.
        redshift : object
            redshift value.
        """
        flux_lambda_obs = np.asarray(flux_lambda_obs, dtype=float)
        return flux_lambda_obs * 1.0e3 * (1.0 + float(redshift)) / 1.0e-17

    def plot_jaxqsofit_spectrum(
        self,
        spectrum_index: int = 0,
        posterior: str = "latest",
        show_nebular_lines: bool = False,
        show_plot: bool = True,
        plot_residual: bool = True,
        plot_legend: bool = True,
        ylims: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot the joint spectral fit with jaxqsofit's spectrum plotter.

        Parameters
        ----------
        spectrum_index : object
            spectrum_index value.
        posterior : object
            posterior value.
        show_nebular_lines : object
            show_nebular_lines value.
        show_plot : object
            show_plot value.
        plot_residual : object
            plot_residual value.
        plot_legend : object
            plot_legend value.
        ylims : object
            ylims value.
        **kwargs : dict
            Additional keyword arguments.
        """
        if str(self.config.spectroscopy_config.backend).lower() != "jaxqsofit":
            raise RuntimeError("plot_jaxqsofit_spectrum requires SpectroscopyConfig.backend='jaxqsofit'.")
        if self.context.spec_wave_obs.size == 0:
            raise RuntimeError("No spectroscopy data are available to plot.")
        try:
            from jaxqsofit import JAXQSOFit
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
            """Return one posterior-median spectral component on the rest grid.

            Parameters
            ----------
            name : object
                name value.
            apply_scale : object
                apply_scale value.
            """
            if name not in pred:
                return np.zeros_like(wave_rest)
            comp_mjy = self._posterior_median_array(pred[name])[selected]
            if apply_scale:
                comp_mjy = scale_factor * comp_mjy
            return self._mjy_to_rest_flambda_1e17(wave_obs, comp_mjy, z)

        def obs_sed_component(name: str, multiplier: float = 1.0) -> np.ndarray:
            """Return one posterior-median observed SED component on the spectrum grid.

            Parameters
            ----------
            name : object
                name value.
            multiplier : object
                multiplier value.
            """
            if name not in pred or "obs_wave" not in pred:
                return np.zeros_like(wave_rest)
            source_wave = np.asarray(self._posterior_median_array(pred["obs_wave"]), dtype=float)
            source_flux = np.asarray(self._posterior_median_array(pred[name]), dtype=float)
            if source_wave.size == 0 or source_flux.size != source_wave.size:
                return np.zeros_like(wave_rest)
            flux_lambda = np.interp(wave_obs, source_wave, source_flux, left=0.0, right=0.0)
            return self._obs_flambda_to_rest_flambda_1e17(scale_factor * multiplier * flux_lambda, z)

        def keep_component(arr: np.ndarray) -> bool:
            """Return True for finite, nonzero component arrays worth plotting.


            Parameters
            ----------
            arr : object
                arr value.
            """
            finite = np.asarray(arr, dtype=float)
            finite = finite[np.isfinite(finite)]
            return finite.size > 0 and float(np.nanmax(np.abs(finite))) > 0.0

        def draw_scale(n_draws: int) -> np.ndarray:
            """Return one spectrum-scale factor per posterior draw.

            Parameters
            ----------
            n_draws : object
                n_draws value.
            """
            raw = np.asarray(pred.get("spectrum_scale_fit", scale_factor), dtype=float)
            if raw.ndim == 0:
                return np.full(n_draws, float(raw), dtype=float)
            if raw.shape[0] == n_draws:
                if raw.ndim > 1 and raw.shape[-1] > 1:
                    return np.asarray(raw[:, int(spectrum_index)], dtype=float)
                return np.asarray(raw.reshape(n_draws, -1)[:, 0], dtype=float)
            return np.full(n_draws, scale_factor, dtype=float)

        def band_from_draws(draws: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
            """Return 16-84% bands for posterior draws on the plot wavelength grid.


            Parameters
            ----------
            draws : object
                draws value.
            """
            if draws is None:
                return None
            arr = np.asarray(draws, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != wave_rest.size or arr.size == 0:
                return None
            return tuple(np.nanpercentile(arr, [16.0, 84.0], axis=0))

        def spectrum_draws(name: str, apply_scale: bool = True) -> np.ndarray | None:
            """Return spectral-component posterior draws in qsofit rest-frame units.

            Parameters
            ----------
            name : object
                name value.
            apply_scale : object
                apply_scale value.
            """
            if name not in pred:
                return None
            arr = np.asarray(pred[name], dtype=float)
            if arr.ndim == 1:
                if arr.shape[0] == selected.shape[0]:
                    draws = arr[None, selected]
                elif arr.shape[0] == wave_rest.size:
                    draws = arr[None, :]
                else:
                    return None
            elif arr.ndim >= 2 and arr.shape[-1] == selected.shape[0]:
                draws = arr.reshape((-1, arr.shape[-1]))[:, selected]
            elif arr.ndim >= 2 and arr.shape[-1] == wave_rest.size:
                draws = arr.reshape((-1, arr.shape[-1]))
            else:
                return None
            if apply_scale:
                draws = draw_scale(draws.shape[0])[:, None] * draws
            return self._mjy_to_rest_flambda_1e17(wave_obs[None, :], draws, z)

        def obs_sed_draws(name: str, multiplier: float = 1.0) -> np.ndarray | None:
            """Return observed-SED posterior draws interpolated to the spectrum grid.

            Parameters
            ----------
            name : object
                name value.
            multiplier : object
                multiplier value.
            """
            if name not in pred or "obs_wave" not in pred:
                return None
            source_wave = np.asarray(self._posterior_median_array(pred["obs_wave"]), dtype=float)
            source_flux = np.asarray(pred[name], dtype=float)
            if source_wave.ndim != 1 or source_wave.size == 0:
                return None
            if source_flux.ndim == 1:
                flux_draws = source_flux[None, :]
            elif source_flux.ndim >= 2:
                flux_draws = source_flux.reshape((-1, source_flux.shape[-1]))
            else:
                return None
            if flux_draws.shape[-1] != source_wave.size:
                return None
            interp_draws = np.vstack([
                np.interp(wave_obs, source_wave, draw, left=0.0, right=0.0)
                for draw in flux_draws
            ])
            scaled = draw_scale(interp_draws.shape[0])[:, None] * float(multiplier) * interp_draws
            return self._obs_flambda_to_rest_flambda_1e17(scaled, z)

        plotter = JAXQSOFit.__new__(JAXQSOFit)
        plotter.z = z
        plotter.wave = wave_rest
        plotter.flux = flux_rest
        plotter.err = err_rest
        plotter.wave_prereduced = wave_rest
        plotter.flux_prereduced = flux_rest
        plotter.err_prereduced = err_rest
        plotter.model_total = component("pred_spectrum_fluxes", apply_scale=False)
        spec_host_component = component("spec_host_model_fluxes")
        plotter.host = (
            spec_host_component
            if keep_component(spec_host_component)
            else obs_sed_component("host_obs_sed", multiplier=host_capture)
        )
        spec_disk_component = component("spec_disk_model_fluxes")
        disk_component = spec_disk_component if keep_component(spec_disk_component) else obs_sed_component("disk_obs_sed")
        plotter.f_pl_model = disk_component if keep_component(disk_component) else component("jqf_continuum_model")
        plotter.f_pl_model_intrinsic = plotter.f_pl_model
        plotter.f_fe_mgii_model = component("jqf_feii_model")
        plotter.f_fe_balmer_model = np.zeros_like(wave_rest)
        plotter.f_bc_model = component("jqf_balmer_model")
        jqf_line_site = "jqf_line_model_aperture" if "jqf_line_model_aperture" in pred else "jqf_line_model"
        plotter.f_line_model = component(jqf_line_site)
        spec_torus_component = component("spec_torus_model_fluxes")
        custom_components = {
            "jaxsedfit_torus": spec_torus_component if keep_component(spec_torus_component) else obs_sed_component("torus_obs_sed"),
            "jaxsedfit_host_dust": obs_sed_component("dust_obs_sed", multiplier=host_capture),
            "jaxsedfit_sed_balmer": obs_sed_component("balmer_obs_sed"),
        }
        if show_nebular_lines:
            custom_components["jaxsedfit_nebular_lines"] = obs_sed_component("nebular_lines_obs_sed")
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
        pred_bands = {}
        total_band = band_from_draws(spectrum_draws("pred_spectrum_fluxes", apply_scale=False))
        if total_band is not None:
            pred_bands["total_model"] = total_band
        host_draw_values = (
            spectrum_draws("spec_host_model_fluxes")
            if keep_component(spec_host_component)
            else obs_sed_draws("host_obs_sed", multiplier=host_capture)
        )
        host_band = band_from_draws(host_draw_values)
        if host_band is not None:
            pred_bands["host"] = host_band
        if keep_component(disk_component):
            disk_draw_values = spectrum_draws("spec_disk_model_fluxes") if keep_component(spec_disk_component) else obs_sed_draws("disk_obs_sed")
        else:
            disk_draw_values = spectrum_draws("jqf_continuum_model")
        disk_band = band_from_draws(disk_draw_values)
        if disk_band is not None:
            pred_bands["PL"] = disk_band
            pred_bands["PL_intrinsic"] = disk_band
        for key, site in (
            ("FeII", "jqf_feii_model"),
            ("Balmer_cont", "jqf_balmer_model"),
            ("lines", jqf_line_site),
        ):
            band = band_from_draws(spectrum_draws(site))
            if band is not None:
                pred_bands[key] = band
        torus_draw_values = spectrum_draws("spec_torus_model_fluxes") if keep_component(spec_torus_component) else obs_sed_draws("torus_obs_sed")
        custom_draws = {
            "jaxsedfit_torus": torus_draw_values,
            "jaxsedfit_host_dust": obs_sed_draws("dust_obs_sed", multiplier=host_capture),
            "jaxsedfit_sed_balmer": obs_sed_draws("balmer_obs_sed"),
        }
        if show_nebular_lines:
            custom_draws["jaxsedfit_nebular_lines"] = obs_sed_draws("nebular_lines_obs_sed")
        for name, draws in custom_draws.items():
            if name in plotter.custom_components:
                band = band_from_draws(draws)
                if band is not None:
                    pred_bands[name] = band
        plotter.f_poly_model = np.ones_like(wave_rest)
        plotter.custom_line_components = {}
        plotter.pred_bands = pred_bands
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
