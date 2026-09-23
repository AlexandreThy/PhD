"""
Large amplitude movements (longmovement_1 and longmovement_2), parameters of
the "Large amplitude" cell of CurrentParts/PaperPlot.ipynb: T = 0.6 s, N = 60,
WP = 2e4, WV = 1, WR = 0.1, FL r = 1e-4, delay 60 ms, motor noise 5e-4.

For each movement and controller: single-trial hand paths, angular velocity
profiles (mean +- std; shoulder solid, elbow dashed) and the paper movement cost.

    python FL_with-iLQG/large_amplitude.py [--trials 100] [--jobs 8]
"""

import argparse
import sys

from paper_controllers import (
    COLORS, CONTROLLERS, HERE, LABELS, cost_function, delete_axis, plan_muscle, run_trials,
)

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from Helpers.Helpers import longmovement_1, longmovement_2

MOVEMENT_TIME = 0.6
NUM_ITER = 60
WR = 0.1
MAX_SHOWN = 15  # single trials drawn per hand-path panel


def simulate_movement(start, target, trials, jobs):
    plan = plan_muscle(MOVEMENT_TIME, NUM_ITER, start, target, WR)
    tasks = [
        dict(controller=c, duration=MOVEMENT_TIME, num_iter=NUM_ITER, start=start, target=target,
             wr=WR, noise=True, seed=trial, plan=plan)
        for trial in range(trials) for c in CONTROLLERS
    ]
    results = run_trials(tasks, jobs)
    n_c = len(CONTROLLERS)
    # (controller, trial, ...)
    paths = np.array([[r[0], r[1]] for r in results]).reshape(trials, n_c, 2, NUM_ITER + 1).swapaxes(0, 1)
    velocities = np.array([r[2][:, 2:4].T for r in results]).reshape(trials, n_c, 2, NUM_ITER + 1).swapaxes(0, 1)
    costs = np.array([cost_function(r[2], r[3], target, WR) for r in results]).reshape(trials, n_c).T
    return paths, velocities, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    movements = [longmovement_1(), longmovement_2()]
    n_c = len(CONTROLLERS)
    time_ms = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)

    fig = plt.figure(figsize=(4.5 * n_c, 22))
    gs = gridspec.GridSpec(5, n_c, height_ratios=[1.2, 1, 1.2, 1, 1.2])

    print(f"Paper cost (WR={WR}), mean +- std over {args.trials} noisy trials")
    for m, (start, target) in enumerate(movements):
        print(f"\nMovement {m + 1}: {np.round(start, 1)} -> {np.round(target, 1)}")
        paths, velocities, costs = simulate_movement(start, target, args.trials, args.jobs)

        for c, name in enumerate(CONTROLLERS):
            print(f"  {LABELS[name]:28s} {costs[c].mean():8.4f} +- {costs[c].std():.4f}")

            ax = fig.add_subplot(gs[2 * m, c])
            for trial in range(min(MAX_SHOWN, args.trials)):
                ax.plot(*paths[c, trial], color=COLORS[name], linewidth=0.6)
            ax.add_patch(plt.Circle(start, 1.5, color="grey"))
            ax.add_patch(plt.Rectangle((target[0] - 1.5, target[1] - 1.5), 3, 3, color="grey"))
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            delete_axis(ax, ("top", "right", "bottom", "left"))
            ax.set_title(f"Movement {m + 1}\n{LABELS[name]}", fontsize=11)

            ax = fig.add_subplot(gs[2 * m + 1, c])
            ax.axhline(0, color="black", linewidth=0.8)
            for joint, style in enumerate(["-", "--"]):
                vel = velocities[c, :, joint] * 180 / np.pi
                mean, std = vel.mean(axis=0), vel.std(axis=0)
                ax.plot(time_ms, mean, color=COLORS[name], linestyle=style,
                        label=["shoulder", "elbow"][joint])
                ax.fill_between(time_ms, mean - std, mean + std, color=COLORS[name], alpha=0.3)
            ax.set_ylim(-270, 320)
            ax.set_xlabel("Time [ms]")
            if c == 0:
                ax.set_ylabel("Angular velocity [deg/s]")
                ax.legend(fontsize=8)
            delete_axis(ax)

        ax = fig.add_subplot(gs[4, 2 * m: 2 * m + 2] if n_c >= 4 else gs[4, m])
        box = ax.boxplot(costs.T, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black"))
        for patch, name in zip(box["boxes"], CONTROLLERS):
            patch.set_facecolor(COLORS[name])
        ax.set_yscale("log")
        ax.set_xticks(range(1, n_c + 1), [LABELS[n].replace(" (", "\n(") for n in CONTROLLERS], fontsize=8)
        ax.set_ylabel("Movement cost")
        ax.set_title(f"Movement {m + 1}")
        delete_axis(ax)

    fig.tight_layout()
    out = HERE / "large_amplitude.svg"
    fig.savefig(out)
    print(f"\nFigure saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
