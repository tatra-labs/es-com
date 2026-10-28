"""Draw the step: the population, where it balances, and where the rewards move the balance to.

Four frames of an ES run on a two-dimensional objective. In each one the particles are the
perturbed models, drawn with an area proportional to the mass they carry; the hollow circle is the
plain centroid, the filled one the reward-weighted centre of mass, and the arrow is the step the
optimiser actually takes. The arrow is the displacement between the two circles, scaled — that is
the whole claim, drawn.

    python experiments/visualise.py
    python experiments/visualise.py --sampler antithetic
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import es_com as es  # noqa: E402


def objective(points):
    """A tilted bowl with a ridge, so the centre of mass has somewhere interesting to sit."""
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    x, y = points[:, 0], points[:, 1]
    return -(0.5 * x**2 + 1.6 * y**2) + 1.4 * np.sin(1.7 * x) * np.cos(1.2 * y)


def step(theta, args, rng):
    noise = es.sample(args.sampler, rng, args.population, 2)
    population = es.perturb(theta, noise, args.sigma)
    rewards = objective(population)
    weights = es.mean_subtracted(rewards)
    masses = es.as_masses(weights)
    update = es.es_update(noise, weights, args.alpha)
    return {
        "population": population,
        "masses": masses,
        "centroid": es.centroid(population),
        "com": es.center_of_mass(population, masses),
        "update": update,
        "theta": theta.copy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.55)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--sampler", default="antithetic", choices=["plain", "antithetic"])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--start", type=float, nargs=2, default=[-2.3, 1.9])
    parser.add_argument("--figure", default="figures/geometry.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    theta = np.array(args.start, dtype=np.float64)
    frames = []
    for _ in range(args.frames):
        frame = step(theta, args, rng)
        frames.append(frame)
        theta = theta + frame["update"]

    for index, frame in enumerate(frames):
        gap = frame["com"] - frame["centroid"]
        update = frame["update"]
        scale = args.alpha * frame["masses"].mean() / args.sigma
        rebuilt = scale * gap
        cosine = float(gap @ update / (np.linalg.norm(gap) * np.linalg.norm(update)))
        frame["scale"] = scale
        print(
            "step %d  cos(gap, update) = %.12f   |gap| %.4f  x %.2f  =  |update| %.4f"
            "   (difference %.1e)"
            % (index + 1, cosine, np.linalg.norm(gap), scale, np.linalg.norm(update),
               float(np.abs(update - rebuilt).max()))
        )

    plot(frames, args)


def plot(frames, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    everything = np.concatenate([frame["population"] for frame in frames])
    low = everything.min(axis=0) - 0.6
    high = everything.max(axis=0) + 0.6
    grid_x, grid_y = np.meshgrid(
        np.linspace(low[0], high[0], 220), np.linspace(low[1], high[1], 220)
    )
    field = objective(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).reshape(grid_x.shape)

    figure, axes = plt.subplots(1, len(frames), figsize=(3.3 * len(frames), 3.5), dpi=160)
    axes = np.atleast_1d(axes)
    for panel, (frame, index) in zip(axes, [(f, i) for i, f in enumerate(frames)]):
        panel.contourf(grid_x, grid_y, field, levels=24, cmap="bone", alpha=0.55)
        masses = frame["masses"]
        sizes = 4.0 + 150.0 * (masses - masses.min()) / (np.ptp(masses) + 1e-12)
        panel.scatter(
            frame["population"][:, 0], frame["population"][:, 1],
            s=sizes, c="#2f6fc2", alpha=0.55, edgecolors="none", zorder=2,
        )
        start = frame["centroid"]
        # The raw displacement, and the step it becomes once scaled. Drawn from the same point so
        # that the one being a multiple of the other is visible rather than asserted.
        panel.annotate(
            "", xy=start + frame["update"], xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.8), zorder=6,
        )
        panel.annotate(
            "", xy=frame["com"], xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#c2492f", lw=3.0), zorder=7,
        )
        panel.scatter(*start, s=90, facecolors="none", edgecolors="#111111",
                      linewidths=1.6, zorder=8, label="centroid (= theta)")
        panel.scatter(*frame["com"], s=70, c="#c2492f", zorder=9, label="centre of mass")
        panel.set_title("step %d   (x %.1f)" % (index + 1, frame["scale"]), fontsize=10)
        panel.set_xticks([])
        panel.set_yticks([])
        panel.set_xlim(low[0], high[0])
        panel.set_ylim(low[1], high[1])
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    for panel in axes:
        panel.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Red: centroid to centre of mass.  Black: the step the optimiser takes, which is the "
        "red arrow scaled by alpha * m_bar / sigma.  (%s sampling, N = %d)"
        % (args.sampler, args.population),
        fontsize=9,
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
    figure.savefig(args.figure)
    print("wrote %s" % args.figure)


if __name__ == "__main__":
    main()
