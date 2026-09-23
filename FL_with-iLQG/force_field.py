"""
Force field task, parameters of the "Force Field" cell of
CurrentParts/PaperPlot.ipynb: start [0, 45], target [0, 60], T = 0.6 s,
N = 60, WP = 2e4, WV = 1, WR = 0.05, FL r = 1e-4, ff_power = -4e-4,
delay 60 ms, motor noise 5e-4. Every controller is run with and without the
force field (the controllers do not know about the field).

For each controller: single-trial hand paths (force field in colour, null
field in grey), angular velocity profiles (mean +- std in the force field,
null-field mean in grey; shoulder solid, elbow dashed) and the paper movement
cost with and without the field.

    python FL_with-iLQG/force_field.py [--trials 50] [--jobs 8]
"""

import argparse
import sys

from paper_controllers import (
    COLORS, CONTROLLERS, HERE, LABELS, cost_function, delete_axis, plan_muscle, run_trials,
)

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

START = [0, 45]
TARGET = [0, 60]
MOVEMENT_TIME = 0.6
NUM_ITER = 60
WR = 0.05
FF_POWER = -4e-4
MAX_SHOWN = 15  # single trials drawn per hand-path panel


def simulate(ff, trials, jobs, plan):
    tasks = [
        dict(controller=c, duration=MOVEMENT_TIME, num_iter=NUM_ITER, start=START, target=TARGET,
             wr=WR, noise=True, ff=ff, ff_power=FF_POWER if ff else 0.0, seed=trial, plan=plan)
        for trial in range(trials) for c in CONTROLLERS
    ]
    results = run_trials(tasks, jobs)
    n_c = len(CONTROLLERS)
    paths = np.array([[r[0], r[1]] for r in results]).reshape(trials, n_c, 2, NUM_ITER + 1).swapaxes(0, 1)
    velocities = np.array([r[2][:, 2:4].T for r in results]).reshape(trials, n_c, 2, NUM_ITER + 1).swapaxes(0, 1)
    costs = np.array([cost_function(r[2], r[3], TARGET, WR) for r in results]).reshape(trials, n_c).T
    return paths, velocities, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    # The FL-iLQG plan does not depend on the (unknown) force field.
    plan = plan_muscle(MOVEMENT_TIME, NUM_ITER, START, TARGET, WR)
    ff = simulate(True, args.trials, args.jobs, plan)
    null = simulate(False, args.trials, args.jobs, plan)

    n_c = len(CONTROLLERS)
    time_ms = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)
    fig = plt.figure(figsize=(4 * n_c, 13))
    gs = gridspec.GridSpec(3, n_c, height_ratios=[1.4, 1, 1])

    print(f"Paper cost (WR={WR}), mean +- std over {args.trials} noisy trials")
    print(f"  {'':28s} {'null field':>18s} {'force field':>18s}")
    for c, name in enumerate(CONTROLLERS):
        print(f"  {LABELS[name]:28s} {null[2][c].mean():8.4f} +- {null[2][c].std():.4f}"
              f" {ff[2][c].mean():8.4f} +- {ff[2][c].std():.4f}")

        ax = fig.add_subplot(gs[0, c])
        for trial in range(min(MAX_SHOWN, args.trials)):
            ax.plot(*null[0][c, trial], color="grey", linewidth=0.6)
            ax.plot(*ff[0][c, trial], color=COLORS[name], linewidth=0.6)
        ax.add_patch(plt.Circle(START, 0.7, color="grey"))
        ax.add_patch(plt.Rectangle((TARGET[0] - 0.7, TARGET[1] - 0.7), 1.4, 1.4, color="grey"))
        ax.set_aspect("equal")
        ax.set_xlim(-6, 6)
        ax.set_xticks([])
        ax.set_yticks([])
        delete_axis(ax, ("top", "right", "bottom", "left"))
        ax.set_title(LABELS[name], fontsize=11)

        ax = fig.add_subplot(gs[1, c])
        ax.axhline(0, color="black", linewidth=0.8)
        for joint, style in enumerate(["-", "--"]):
            vel = ff[1][c, :, joint] * 180 / np.pi
            mean, std = vel.mean(axis=0), vel.std(axis=0)
            ax.plot(time_ms, null[1][c, :, joint].mean(axis=0) * 180 / np.pi, color="grey", linestyle=style)
            ax.plot(time_ms, mean, color=COLORS[name], linestyle=style, label=["shoulder", "elbow"][joint])
            ax.fill_between(time_ms, mean - std, mean + std, color=COLORS[name], alpha=0.3)
        ax.set_xlabel("Time [ms]")
        if c == 0:
            ax.set_ylabel("Angular velocity [deg/s]")
            ax.legend(fontsize=8)
        delete_axis(ax)

    ax = fig.add_subplot(gs[2, :])
    positions_null = np.arange(n_c) * 3 + 1
    for costs, positions, alpha in [(null[2], positions_null, 0.35), (ff[2], positions_null + 1, 1.0)]:
        box = ax.boxplot(costs.T, positions=positions, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black"))
        for patch, name in zip(box["boxes"], CONTROLLERS):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(alpha)
    ax.set_yscale("log")
    ax.set_xticks(positions_null + 0.5, [LABELS[n] for n in CONTROLLERS], fontsize=9)
    ax.set_ylabel("Movement cost")
    ax.set_title("Movement cost: null field (light) / force field (dark)")
    delete_axis(ax)

    fig.tight_layout()
    out = HERE / "force_field.svg"
    fig.savefig(out)
    print(f"\nFigure saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
