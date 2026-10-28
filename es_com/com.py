"""An evolution-strategies step, read as a displacement between two balance points.

An ES step perturbs the parameters into a population, scores it, and recombines:

    p_i = theta + sigma * eps_i            Delta theta = (alpha / N) * sum_i w_i eps_i

Treat each perturbed model as a particle and its reward as that particle's mass. Before the rewards
arrive the population balances at its plain centroid, `p_bar`. After they arrive it balances at the
reward-weighted centre of mass, `p_com`. The claim this module exists to check is that for
mean-subtracted weights the update *is* the displacement between those two points:

    Delta theta = (alpha * r_bar / sigma) * (p_com - p_bar)

exactly, for any finite population, not in expectation. `identity_residual` measures how exactly.

Three things follow, and each is a test:

* adding a constant to every reward leaves `Delta theta` untouched, though it moves `p_com` and
  `r_bar` separately — so the physical picture can always be arranged to have non-negative masses;
* with antithetic sampling `p_bar` is exactly `theta`, so the step is a displacement measured from
  the model itself, and without it that only holds as the population grows;
* using raw rewards instead of mean-subtracted ones measures the same centre of mass from `theta`
  rather than from `p_bar`. Mean subtraction moves the reference point, nothing else.
"""

import numpy as np

__all__ = [
    "as_masses",
    "center_of_mass",
    "centroid",
    "displacement_update",
    "es_update",
    "identity_residual",
    "perturb",
]


def perturb(theta, noise, sigma):
    """The population `p_i = theta + sigma * eps_i`, shaped (population, parameters)."""
    return np.asarray(theta, dtype=np.float64) + sigma * np.asarray(noise, dtype=np.float64)


def centroid(population):
    """`p_bar`: where the population balances before any reward is known."""
    return np.asarray(population, dtype=np.float64).mean(axis=0)


def center_of_mass(population, masses):
    """`p_com`: where it balances once each particle carries its reward as mass."""
    population = np.asarray(population, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    total = masses.sum()
    if np.isclose(total, 0.0):
        raise ValueError("the masses sum to zero, so there is no centre of mass to speak of")
    return (masses[:, None] * population).sum(axis=0) / total


def es_update(noise, weights, alpha, sigma=None):
    """The ordinary ES step, `(alpha / N) * sum_i w_i eps_i`.

    `sigma` is accepted and ignored: the step is written in terms of the noise, which is where the
    scale already lives. It is in the signature so that callers can pass the same arguments here
    and to `displacement_update` without rearranging them.
    """
    noise = np.asarray(noise, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return alpha * (weights[:, None] * noise).mean(axis=0)


def as_masses(weights, floor=1.0):
    """Shift sum-to-zero weights up until they are all positive, so they can act as masses.

    The paper notes that rewards can always be made non-negative by adding a constant, because the
    mean-subtracted update does not notice. The same move works for *any* weighting that sums to
    zero — centred ranks, standardised rewards — which is what makes the centre-of-mass reading
    general rather than a special case of raw rewards.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return weights + (floor - weights.min())


def displacement_update(population, masses, alpha, sigma):
    """The same step written geometrically: `(alpha * m_bar / sigma) * (p_com - p_bar)`."""
    population = np.asarray(population, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    gap = center_of_mass(population, masses) - centroid(population)
    return alpha * masses.mean() / sigma * gap


def identity_residual(noise, masses, alpha, sigma, theta=None, relative=True):
    """How far apart the two ways of writing the step are. Should be rounding error and nothing more.

    `masses` are used as they are: centring them gives the mean-subtracted update, so pass the raw
    rewards and both sides will agree on those.
    """
    noise = np.asarray(noise, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    theta = np.zeros(noise.shape[1]) if theta is None else np.asarray(theta, dtype=np.float64)

    population = perturb(theta, noise, sigma)
    direct = es_update(noise, masses - masses.mean(), alpha)
    geometric = displacement_update(population, masses, alpha, sigma)

    difference = np.abs(direct - geometric).max()
    if not relative:
        return float(difference)
    scale = max(np.abs(direct).max(), 1e-300)
    return float(difference / scale)


def reference_point(population, masses, alpha, sigma, theta, centred=True):
    """Which point the step is a displacement *from*.

    With centred weights it is the population's own centroid; with raw rewards it is `theta`. The
    destination, `p_com`, is the same either way.
    """
    return centroid(population) if centred else np.asarray(theta, dtype=np.float64)
