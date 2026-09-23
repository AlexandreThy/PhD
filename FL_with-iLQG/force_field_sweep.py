"""
Robustness to an (unknown) force field: ILQG, FL, and their combination.

Force field task of PaperPlot.ipynb (start [0, 45], target [0, 60], T = 0.6 s,
N = 60, WR = 0.05), with the four controllers of paper_controllers.py:
  ILQG            : linear approximation of the dynamics around a nominal trajectory
  FL              : Controllers/FL.py, LQR on the virtual acceleration (r ||v||^2)
  FL-iLQG (FL split / min-norm split) : iLQG on the muscle cost in the FL coordinates

  Top left    : paper cost vs force field strength, 60 ms delay (noise-free, and
                mean of NUM_TRIALS noisy trials).
  Bottom left : cost increase caused by the field, J(ff) - J(0), noise-free.
  Right       : noise-free paper cost vs sensory delay, one panel per field strength.

    python FL_with-iLQG/force_field_sweep.py [--trials 20] [--jobs 2]
"""

import argparse
import sys

import paper_controllers
from paper_controllers import (
    COLORS, CONTROLLERS, HERE, LABELS, cost_function, delete_axis, plan_muscle, run, run_trials,
)

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

START = [0, 45]
TARGET = [0, 60]
MOVEMENT_TIME = 0.6
NUM_ITER = 60
WR = 0.05
FF_POWERS = [-2e-3, -1e-3, -4e-4, 0.0, 4e-4, 1e-3, 2e-3, 4e-3]
DELAYS = [0.0, 0.02, 0.04, 0.06, 0.08]
FF_FOR_DELAY_SWEEP = [-1e-3, 4e-4, 1e-3, 2e-3]


def noise_free_cost(controller, ff_power, plan):
    """Paper cost of one noise-free movement (NaN if the movement diverges)."""
    try:
        X, Y, x, u = run(controller, MOVEMENT_TIME, NUM_ITER, START, TARGET, WR, noise=False,
                         ff=ff_power != 0, ff_power=ff_power, plan=plan)
    except np.linalg.LinAlgError:
        return np.nan
    return cost_function(x, u, TARGET, WR)


def fmt(value, width=9):
    return f"{'diverged':>{width}s}" if not np.isfinite(value) else f"{value:{width}.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    plan = plan_muscle(MOVEMENT_TIME, NUM_ITER, START, TARGET, WR)
    short = {"ILQG": "ILQG", "FL": "FL", "FL_muscle": "FLiLQG", "FL_muscle_mn": "FLiLQG mn"}

    # Cost vs field strength, 60 ms delay
    free = {c: np.array([noise_free_cost(c, f, plan) for f in FF_POWERS]) for c in CONTROLLERS}
    tasks = [
        dict(controller=c, duration=MOVEMENT_TIME, num_iter=NUM_ITER, start=START, target=TARGET,
             wr=WR, noise=True, ff=f != 0, ff_power=f, seed=trial, plan=plan)
        for f in FF_POWERS for trial in range(args.trials) for c in CONTROLLERS
    ]
    results = run_trials(tasks, args.jobs)
    costs = np.array([cost_function(r[2], r[3], TARGET, WR) for r in results])
    noisy = np.nanmean(costs.reshape(len(FF_POWERS), args.trials, len(CONTROLLERS)), axis=1)

    i0 = FF_POWERS.index(0.0)
    print(f"Paper cost (WR={WR}), 60 ms delay. FLiLQG = FL split, FLiLQG mn = min-norm split")
    for title, table in [("noise-free", np.array([free[c] for c in CONTROLLERS]).T),
                         (f"mean of {args.trials} noisy trials", noisy),
                         ("increase due to the field, noise-free: J(ff) - J(0)",
                          np.array([free[c] - free[c][i0] for c in CONTROLLERS]).T)]:
        print(f"\n{title}")
        print(f"{'ff_power':>9s} " + "".join(f"{short[c]:>10s}" for c in CONTROLLERS))
        for i, f in enumerate(FF_POWERS):
            print(f"{f:9.0e} " + "".join(" " + fmt(v) for v in table[i]))

    # Cost vs delay, noise-free
    by_delay = {}
    for d in DELAYS:
        paper_controllers.DELAY = d
        for f in FF_FOR_DELAY_SWEEP:
            for c in CONTROLLERS:
                by_delay[(d, f, c)] = noise_free_cost(c, f, plan)
    paper_controllers.DELAY = 0.06

    print("\nNoise-free paper cost vs delay")
    for f in FF_FOR_DELAY_SWEEP:
        print(f"ff_power = {f:g}")
        print(f"  {'delay':>6s} " + "".join(f"{short[c]:>10s}" for c in CONTROLLERS))
        for d in DELAYS:
            print(f"  {int(d * 1000):4d}ms " + "".join(" " + fmt(by_delay[(d, f, c)]) for c in CONTROLLERS))

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.6, 1, 1])
    ax_ff = fig.add_subplot(gs[0, 0])
    ax_inc = fig.add_subplot(gs[1, 0])
    for k, c in enumerate(CONTROLLERS):
        ax_ff.plot(FF_POWERS, free[c], "-o", color=COLORS[c], label=f"{LABELS[c]}, noise-free")
        ax_ff.plot(FF_POWERS, noisy[:, k], "--s", color=COLORS[c], alpha=0.7, label=f"{LABELS[c]}, noisy mean")
        increase = free[c] - free[c][i0]
        ax_inc.plot([f for f in FF_POWERS if f != 0], [v for f, v in zip(FF_POWERS, increase) if f != 0],
                    "-o", color=COLORS[c], label=LABELS[c])
    for ax, ylabel, title in [(ax_ff, "Movement cost", "Cost vs force field, 60 ms delay (missing = diverged)"),
                              (ax_inc, "J(ff) - J(0)", "Cost increase caused by the field (noise-free)")]:
        ax.set_yscale("log")
        ax.set_xlabel("Force field strength (ff_power)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        delete_axis(ax)
    ax_ff.legend(fontsize=7, ncol=2)
    ax_inc.legend(fontsize=8)

    delays_ms = np.array(DELAYS) * 1000
    for i, f in enumerate(FF_FOR_DELAY_SWEEP):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
        for c in CONTROLLERS:
            ax.plot(delays_ms, [by_delay[(d, f, c)] for d in DELAYS], "-o", color=COLORS[c], label=LABELS[c])
        ax.set_yscale("log")
        ax.set_xlabel("Sensory delay [ms]")
        ax.set_ylabel("Movement cost (noise-free)")
        ax.set_title(f"ff_power = {f:g}")
        delete_axis(ax)
    fig.tight_layout()
    out = HERE / "force_field_sweep.svg"
    fig.savefig(out)
    print(f"\nFigure saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
