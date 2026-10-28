"""Tests for the centre-of-mass reading of an evolution-strategies step.

The headline is an identity, so most of these assert equality to rounding error rather than to a
tolerance someone chose. The exceptions are the two places where the geometry is exact but the
arithmetic is not, and those are tested for the behaviour they actually have.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import es_com as es  # noqa: E402


def setup(population=32, parameters=8, sigma=0.2, seed=0, sampler="plain"):
    rng = np.random.default_rng(seed)
    noise = es.sample(sampler, rng, population, parameters)
    theta = rng.normal(size=parameters)
    rewards = rng.normal(4.0, 1.5, size=population)
    return noise, theta, rewards


# --- the identity ----------------------------------------------------------------------------


@pytest.mark.parametrize("population", [2, 7, 64, 256])
@pytest.mark.parametrize("parameters", [1, 5, 128])
def test_the_step_is_the_displacement_between_the_two_balance_points(population, parameters):
    noise, theta, rewards = setup(population, parameters, seed=population + parameters)
    assert es.identity_residual(noise, rewards, 0.37, 0.2, theta) < 1e-9


def test_it_holds_for_a_population_of_two():
    """'Any finite population' includes the smallest one that has a centroid at all."""
    noise, theta, rewards = setup(population=2, parameters=3, seed=5)
    assert es.identity_residual(noise, rewards, 1.0, 0.5, theta) < 1e-12


def test_adding_a_constant_to_every_reward_changes_nothing():
    noise, theta, rewards = setup(seed=1)
    population = es.perturb(theta, noise, 0.2)
    base = es.es_update(noise, rewards - rewards.mean(), 0.5)
    for shift in (1.0, 100.0, 10000.0):
        moved = rewards + shift
        assert np.allclose(base, es.es_update(noise, moved - moved.mean(), 0.5), atol=1e-12)
        # The centre of mass itself does move; the scale factor moves to compensate.
        assert not np.allclose(
            es.center_of_mass(population, rewards),
            es.center_of_mass(population, moved),
            atol=1e-6,
        )
        assert np.allclose(
            base, es.displacement_update(population, moved, 0.5, 0.2), rtol=1e-6
        )


def test_negative_rewards_still_work_once_shifted_into_masses():
    noise, theta, _ = setup(seed=2)
    rewards = np.random.default_rng(9).normal(-50.0, 3.0, size=noise.shape[0])
    population = es.perturb(theta, noise, 0.2)
    direct = es.es_update(noise, rewards - rewards.mean(), 0.5)
    masses = es.as_masses(rewards - rewards.mean())
    assert masses.min() > 0
    assert np.allclose(direct, es.displacement_update(population, masses, 0.5, 0.2), rtol=1e-9)


# --- what the reference point is -------------------------------------------------------------


def test_antithetic_sampling_puts_the_centroid_exactly_on_theta():
    noise, theta, _ = setup(population=64, sampler="antithetic", seed=3)
    population = es.perturb(theta, noise, 0.7)
    assert np.abs(es.centroid(population) - theta).max() < 1e-15


def test_without_mirroring_the_centroid_is_merely_close():
    _, theta, _ = setup(population=64, seed=3)
    rng = np.random.default_rng(3)
    gaps = []
    for population in (16, 1024):
        noise = es.sample("plain", rng, population, 32)
        points = es.perturb(np.zeros(32), noise, 1.0)
        gaps.append(float(np.linalg.norm(es.centroid(points))))
    assert gaps[0] > gaps[1] > 0.0


def test_raw_rewards_measure_the_same_point_from_theta():
    noise, theta, rewards = setup(seed=4)
    population = es.perturb(theta, noise, 0.2)
    raw_step = es.es_update(noise, es.raw(rewards), 0.5)
    from_theta = 0.5 * rewards.mean() / 0.2 * (es.center_of_mass(population, rewards) - theta)
    assert np.allclose(raw_step, from_theta, rtol=1e-9)


def test_which_weightings_leave_the_centroid_as_the_reference():
    rewards = np.random.default_rng(0).normal(size=16)
    for name, weighting in es.WEIGHTINGS.items():
        total = float(weighting(rewards).sum())
        if es.sums_to_zero(name):
            assert abs(total) < 1e-9, name
        else:
            assert abs(total) > 1e-6, name


# --- the generalisation beyond mean-subtracted rewards ----------------------------------------


@pytest.mark.parametrize("name", ["mean_subtracted", "centred_rank", "standardised"])
def test_any_sum_to_zero_weighting_is_a_centre_of_mass_displacement(name):
    """The paper writes the picture for mean-subtracted rewards; it is not limited to them."""
    noise, theta, rewards = setup(seed=6)
    population = es.perturb(theta, noise, 0.2)
    weights = es.WEIGHTINGS[name](rewards)
    direct = es.es_update(noise, weights, 0.5)
    geometric = es.displacement_update(population, es.as_masses(weights), 0.5, 0.2)
    assert np.allclose(direct, geometric, rtol=1e-8)


@pytest.mark.parametrize("floor", [1e-2, 1.0, 50.0])
def test_the_constant_that_makes_the_masses_positive_is_arbitrary(floor):
    noise, theta, rewards = setup(seed=7)
    population = es.perturb(theta, noise, 0.2)
    weights = es.centred_rank(rewards)
    direct = es.es_update(noise, weights, 0.5)
    geometric = es.displacement_update(population, es.as_masses(weights, floor=floor), 0.5, 0.2)
    assert np.allclose(direct, geometric, rtol=1e-7)


def test_centred_ranks_are_centred_and_ordered():
    rewards = np.array([3.0, -1.0, 7.0, 0.0])
    ranks = es.centred_rank(rewards)
    assert float(ranks.sum()) == pytest.approx(0.0)
    assert ranks.argmax() == 2 and ranks.argmin() == 1


# --- where the geometry is exact but the arithmetic is not ------------------------------------


def test_the_geometric_form_loses_precision_as_sigma_shrinks():
    """A real caveat: it rebuilds a small displacement by subtracting two nearby points."""
    noise, theta, rewards = setup(population=128, parameters=64, seed=8)
    residuals = [
        es.identity_residual(noise, rewards, 0.37, sigma, theta)
        for sigma in (10.0, 1.0, 1e-2, 1e-4)
    ]
    assert residuals == sorted(residuals)
    assert residuals[0] < 1e-13
    assert residuals[-1] > residuals[0]


# --- the small print -------------------------------------------------------------------------


def test_a_massless_population_has_no_centre_of_mass():
    population = np.zeros((4, 2))
    with pytest.raises(ValueError):
        es.center_of_mass(population, np.array([1.0, -1.0, 2.0, -2.0]))


def test_antithetic_sampling_needs_an_even_population():
    with pytest.raises(ValueError):
        es.sample("antithetic", np.random.default_rng(0), 7, 3)


def test_unknown_names_are_refused():
    with pytest.raises(ValueError):
        es.sample("quasirandom", np.random.default_rng(0), 8, 3)


def test_as_masses_lifts_everything_above_zero():
    weights = np.array([-3.0, 0.0, 5.0])
    masses = es.as_masses(weights, floor=0.25)
    assert masses.min() == pytest.approx(0.25)
    assert np.allclose(masses - masses.mean(), weights - weights.mean())
