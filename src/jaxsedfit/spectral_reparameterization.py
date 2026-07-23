"""Exact NumPyro coordinate transformations shared by spectral fitters."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import Reparam


NORMAL_LOGNORMAL_STANDARDIZATION = (
    "jaxqsofit_normal_lognormal_standardization"
)


def standardized_prior_site(site_name: str) -> str:
    """Return the auxiliary standard-Normal site for one physical prior."""
    return f"{str(site_name)}_std"


def _scoped_auxiliary_names(
    site_name: str,
    auxiliary_name: str,
) -> tuple[str, str]:
    """Return public and local auxiliary names under a NumPyro scope handler."""
    site_name = str(site_name)
    auxiliary_name = str(auxiliary_name)
    if "/" not in site_name:
        return auxiliary_name, auxiliary_name
    scope_prefix = site_name.rsplit("/", 1)[0]
    local_name = auxiliary_name.rsplit("/", 1)[-1]
    public_name = (
        auxiliary_name
        if "/" in auxiliary_name
        else f"{scope_prefix}/{auxiliary_name}"
    )
    return public_name, local_name


class NormalLogNormalStandardizeReparam(Reparam):
    """Sample a scalar Normal or LogNormal prior on a standard-Normal axis.

    The original physical sample site is retained as a deterministic value.
    This is an exact coordinate transformation: it changes neither the prior
    nor any downstream likelihood or prediction.
    """

    def __init__(
        self,
        auxiliary_name: str,
        sampling_name: str | None = None,
    ):
        self.auxiliary_name = str(auxiliary_name)
        self.sampling_name = str(
            self.auxiliary_name if sampling_name is None else sampling_name
        )

    @staticmethod
    def _validate(fn):
        if fn.batch_shape or fn.event_shape:
            raise ValueError(
                "Normal/LogNormal standardization currently supports only "
                "scalar sites."
            )
        if not isinstance(fn, (dist.Normal, dist.LogNormal)):
            raise ValueError(
                "Normal/LogNormal standardization requires a scalar Normal "
                f"or LogNormal prior, got {type(fn).__name__}."
            )

    def __call__(self, name, fn, obs):
        if obs is not None:
            raise ValueError(
                "Normal/LogNormal standardization requires a latent site."
            )
        self._validate(fn)
        dtype = jnp.result_type(fn.loc, fn.scale)
        standardized = numpyro.sample(
            self.sampling_name,
            dist.Normal(
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
            ),
        )
        unconstrained = fn.loc + fn.scale * standardized
        value = (
            jnp.exp(unconstrained)
            if isinstance(fn, dist.LogNormal)
            else unconstrained
        )
        return None, value

    def transform_initial_value(self, fn, value):
        """Map a physical initial value to its standard-Normal coordinate."""
        self._validate(fn)
        value = jnp.asarray(value)
        unconstrained = (
            jnp.log(value) if isinstance(fn, dist.LogNormal) else value
        )
        return (unconstrained - fn.loc) / fn.scale


def normal_lognormal_standardization_reparam(site):
    """Resolve a model-provided scalar prior standardization."""
    metadata = (site.get("infer") or {}).get(
        NORMAL_LOGNORMAL_STANDARDIZATION
    )
    if metadata is None:
        return None
    auxiliary_name, sampling_name = _scoped_auxiliary_names(
        site["name"],
        metadata["auxiliary_name"],
    )
    return NormalLogNormalStandardizeReparam(
        auxiliary_name,
        sampling_name,
    )
