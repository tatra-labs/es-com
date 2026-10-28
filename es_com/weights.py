"""Ways of turning rewards into recombination weights, and whether each one is a centre of mass.

A weighting that sums to zero is a displacement between the population's centroid and *some*
balance point; `es_com.as_masses` says which. A weighting that does not sum to zero is a
displacement measured from `theta` instead. That is the only distinction that matters here, and
`sums_to_zero` is how each one below declares which side it is on.
"""

import numpy as np

__all__ = ["WEIGHTINGS", "centred_rank", "mean_subtracted", "raw", "standardised", "sums_to_zero"]


def raw(rewards):
    """The rewards themselves. Does not sum to zero, so the reference point stays at theta."""
    return np.asarray(rewards, dtype=np.float64)


def mean_subtracted(rewards):
    """The usual choice, and the one the centre-of-mass reading is written for."""
    rewards = np.asarray(rewards, dtype=np.float64)
    return rewards - rewards.mean()


def centred_rank(rewards):
    """Ranks mapped to [-0.5, 0.5]. Throws away reward scale, keeps the order."""
    rewards = np.asarray(rewards, dtype=np.float64)
    order = rewards.argsort()
    ranks = np.empty(rewards.shape, dtype=np.float64)
    ranks[order] = np.arange(rewards.size, dtype=np.float64)
    if rewards.size > 1:
        ranks /= rewards.size - 1
    return ranks - 0.5


def standardised(rewards, eps=1e-8):
    """Mean-subtracted and divided by the spread, which is what most implementations do."""
    rewards = np.asarray(rewards, dtype=np.float64)
    centred = rewards - rewards.mean()
    return centred / (rewards.std() + eps)


WEIGHTINGS = {
    "raw": raw,
    "mean_subtracted": mean_subtracted,
    "centred_rank": centred_rank,
    "standardised": standardised,
}


def sums_to_zero(name):
    """Whether this weighting leaves the centroid as the reference point."""
    return name != "raw"
