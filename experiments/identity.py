"""Check the centre-of-mass reading of an ES step, exactly, across every setting that matters.

The claim is an identity, not an approximation, so the only honest test is whether the two sides
agree to rounding error for arbitrary populations, dimensions, noise scales and weightings.

    python experiments/identity.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import es_com as es  # noqa: E402

DOUBLE_EPS = np.finfo(np.float64).eps


def residual_table(args):
    """The identity across populations, dimensions and noise scales."""
    rows = []
    rng = np.random.default_rng(args.seed)
    for population in (2, 8, 64, 512):
        for parameters in (1, 16, 1024):
            for sigma in (1e-3, 0.1, 10.0):
                noise = es.sample("plain", rng, population, parameters)
                theta = rng.normal(size=parameters)
                rewards = rng.normal(5.0, 2.0, size=population)
                rows.append({
                    "population": population,
                    "parameters": parameters,
                    "sigma": sigma,
                    "residual": es.identity_residual(noise, rewards, 0.37, sigma, theta),
                })
    return rows


def weighting_table(args):
    """Every sum-to-zero weighting is the same displacement, once its weights are read as masses."""
    rng = np.random.default_rng(args.seed + 1)
    population, parameters, sigma, alpha = 32, 12, 0.2, 0.5
    noise = es.sample("plain", rng, population, parameters)
    theta = rng.normal(size=parameters)
    rewards = rng.normal(3.0, 1.5, size=population)
    pop = es.perturb(theta, noise, sigma)

    rows = []
    for name, weighting in es.WEIGHTINGS.items():
        weights = weighting(rewards)
        direct = es.es_update(noise, weights, alpha)
        if es.sums_to_zero(name):
            masses = es.as_masses(weights)
            geometric = es.displacement_update(pop, masses, alpha, sigma)
            reference = "centroid"
        else:
            # Raw rewards do not sum to zero; the displacement is measured from theta.
            geometric = alpha * weights.mean() / sigma * (
                es.center_of_mass(pop, weights) - theta
            )
            reference = "theta"
        scale = max(float(np.abs(direct).max()), 1e-300)
        rows.append({
            "weighting": name,
            "reference": reference,
            "weight_sum": float(weights.sum()),
            "residual": float(np.abs(direct - geometric).max() / scale),
        })
    return rows


def floor_table(args):
    """The arbitrary constant that makes the masses positive does not change the step."""
    rng = np.random.default_rng(args.seed + 2)
    population, parameters, sigma, alpha = 24, 6, 0.3, 0.4
    noise = es.sample("plain", rng, population, parameters)
    theta = rng.normal(size=parameters)
    weights = es.centred_rank(rng.normal(size=population))
    pop = es.perturb(theta, noise, sigma)
    direct = es.es_update(noise, weights, alpha)

    rows = []
    for floor in (1e-3, 1.0, 1e3, 1e6):
        geometric = es.displacement_update(pop, es.as_masses(weights, floor=floor), alpha, sigma)
        scale = max(float(np.abs(direct).max()), 1e-300)
        rows.append({
            "floor": floor,
            "residual": float(np.abs(direct - geometric).max() / scale),
        })
    return rows


def centroid_gap(args):
    """How far the population's centroid sits from theta, with and without mirroring."""
    rows = []
    for population in (2, 8, 32, 128, 512, 2048):
        for kind in ("plain", "antithetic"):
            gaps = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + population)
                noise = es.sample(kind, rng, population, args.parameters)
                theta = np.zeros(args.parameters)
                pop = es.perturb(theta, noise, 1.0)
                gaps.append(float(np.linalg.norm(es.centroid(pop) - theta)))
            rows.append({
                "population": population,
                "sampler": kind,
                "gap": float(np.mean(gaps)),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parameters", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/identity.json")
    parser.add_argument("--figure", default="figures/centroid.png")
    args = parser.parse_args()

    print("Delta theta  ==  (alpha * r_bar / sigma) * (p_com - p_bar)")
    print("relative residual, float64 eps = %.2e\n" % DOUBLE_EPS)
    residuals = residual_table(args)
    print("%12s %12s %10s %14s" % ("population", "parameters", "sigma", "residual"))
    for row in residuals:
        print("%12d %12d %10.0e %14.2e"
              % (row["population"], row["parameters"], row["sigma"], row["residual"]))
    worst = max(row["residual"] for row in residuals)
    print("\nworst case %.2e, which is %.0f times float64 eps" % (worst, worst / DOUBLE_EPS))

    print("\n\nEvery sum-to-zero weighting is the same displacement:")
    print("%18s %12s %14s %14s" % ("weighting", "reference", "sum of weights", "residual"))
    weightings = weighting_table(args)
    for row in weightings:
        print("%18s %12s %14.2e %14.2e"
              % (row["weighting"], row["reference"], row["weight_sum"], row["residual"]))

    print("\n\nThe constant that makes the masses positive is arbitrary:")
    floors = floor_table(args)
    for row in floors:
        print("  floor %10.0e -> residual %.2e" % (row["floor"], row["residual"]))

    print("\n\nDistance from the population's centroid to theta (sigma = 1):")
    print("%12s %14s %14s" % ("population", "plain", "antithetic"))
    gaps = centroid_gap(args)
    for population in sorted({row["population"] for row in gaps}):
        chosen = {row["sampler"]: row["gap"] for row in gaps if row["population"] == population}
        print("%12d %14.4f %14.2e" % (population, chosen["plain"], chosen["antithetic"]))
    print("\nMirroring makes the centroid exactly theta. Without it the gap falls as 1/sqrt(N),")
    print("which is how large a population has to be before the two readings coincide.")

    payload = {
        "residuals": residuals,
        "weightings": weightings,
        "floors": floors,
        "centroid_gap": gaps,
        "float64_eps": DOUBLE_EPS,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % args.out)
    if args.figure:
        plot(gaps, args.figure)


def plot(gaps, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    populations = sorted({row["population"] for row in gaps})
    for kind, colour in (("plain", "#c2492f"), ("antithetic", "#2f6fc2")):
        values = [
            max(row["gap"], 1e-17)
            for population in populations
            for row in gaps
            if row["population"] == population and row["sampler"] == kind
        ]
        axes.loglog(populations, values, marker="o", ms=4, lw=1.6, color=colour, label=kind)
    reference = [gaps[0]["gap"] * (populations[0] / n) ** 0.5 for n in populations]
    axes.loglog(populations, reference, ls=":", lw=1.0, color="#999999", label="1/sqrt(N)")
    axes.set_xlabel("population size")
    axes.set_ylabel("distance from centroid to theta")
    axes.set_title("Where the step is measured from", fontsize=11)
    axes.legend(frameon=False, fontsize=8)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(alpha=0.25, linewidth=0.5, which="both")
    figure.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
