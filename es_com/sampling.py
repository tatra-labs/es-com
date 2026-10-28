"""Where the population comes from, and why antithetic sampling moves the reference point.

The centroid of the population is `theta + sigma * eps_bar`. Mirroring each perturbation makes
`eps_bar` exactly zero, so the centroid is exactly `theta` — not approximately, not in
expectation, but to the last bit. Without mirroring it only approaches `theta` as the population
grows, at the usual `1/sqrt(N)` rate, and `experiments/identity.py` measures that.
"""

import numpy as np

__all__ = ["antithetic", "plain", "sample"]


def plain(rng, population, parameters):
    """`population` independent standard normal perturbations."""
    return rng.standard_normal((population, parameters))


def antithetic(rng, population, parameters):
    """Half as many perturbations, each used twice with opposite signs.

    `population` must be even; the mean of the result is exactly zero.
    """
    if population % 2:
        raise ValueError("antithetic sampling needs an even population, got %d" % population)
    half = rng.standard_normal((population // 2, parameters))
    return np.concatenate([half, -half], axis=0)


SAMPLERS = {"plain": plain, "antithetic": antithetic}


def sample(kind, rng, population, parameters):
    if kind not in SAMPLERS:
        raise ValueError("unknown sampler: %r" % kind)
    return SAMPLERS[kind](rng, population, parameters)
