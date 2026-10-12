"""
Robustness to an (unknown) force field: ILQG, FL, and their combination,
with noisy simulations (mean +- std over NUM_TRIALS trials).

Force field task of PaperPlot.ipynb (start [0, 45], target [0, 60], T = 0.6 s,
N = 60, WR = 0.05, motor noise 5e-4), with the four controllers of
paper_controllers.py:
  ILQG            : linear approximation of the dynamics around a nominal trajectory
  FL              : Controllers/FL.py, LQR on the virtual acceleration (r ||v||^2)
  FL-iLQG (FL split / min-norm split) : iLQG on the muscle cost in the FL coordinates

  Top left    : paper cost vs force field strength, 60 ms delay.
  Bottom left : cost increase caused by the field, mean J(ff) - mean J(0).
  Right       : paper cost vs sensory delay, one panel per field strength.

Every controller sees the same noise sequence in a given trial (same seed).
Diverged trials (NaN) are left out of the mean and std, and counted.

    python FL_with-iLQG/force_field_sweep.py [--trials 50] [--jobs 4]
"""

import argparse
import sys

from paper_controllers import (
    COLORS, CONTROLLERS, DELAY, HERE, LABELS, cost_function, delete_axis, plan_muscle, run_trials,
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
SHORT = {"ILQG": "ILQG", "FL": "FL", "FL_muscle": "FLiLQG", "FL_muscle_mn": "FLiLQG mn"}


def simulate(conditions, trials, jobs, plan):
    """
    Noisy costs for every (ff_power, delay) condition.

    Returns {(ff_power, delay): array (trials, controllers)}, NaN for diverged trials.
    """
    tasks = [
        dict(controller=c, duration=MOVEMENT_TIME, num_iter=NUM_ITER, start=START, target=TARGET,
             wr=WR, noise=True, ff=f != 0, ff_power=f, seed=trial, plan=plan, delay=d)
        for (f, d) in conditions for trial in range(trials) for c in CONTROLLERS
    ]
    results = run_trials(tasks, jobs)
    costs = np.array([cost_function(r[2], r[3], TARGET, WR) for r in results])
    costs = costs.reshape(len(conditions), trials, len(CONTROLLERS))
    return {cond: costs[i] for i, cond in enumerate(conditions)}


def fmt(mean, std, n_nan, trials):
    if n_nan == trials:
        return f"{'diverged':>21s}"
    cell = f"{mean:9.2f} +-{std:8.2f}"
    return cell + ("*" if n_nan else " ")


def band(ax, x, mean, std, color, label, marker="o"):
    """Mean line with a +-std band; the lower edge is clipped to stay positive on the log axis."""
    ax.plot(x, mean, "-" + marker, color=color, label=label, markersize=4)
    ax.fill_between(x, np.maximum(mean - std, mean * 1e-2), mean + std, color=color, alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    plan = plan_muscle(MOVEMENT_TIME, NUM_ITER, START, TARGET, WR)
    conditions = [(f, DELAY) for f in FF_POWERS]
    conditions += [(f, d) for d in DELAYS if d != DELAY for f in FF_FOR_DELAY_SWEEP]
    print(f"Simulating {len(conditions) * args.trials * len(CONTROLLERS)} noisy trials...")
    costs = simulate(conditions, args.trials, args.jobs, plan)

    def stats(f, d):
        c = costs[(f, d)]
        return np.nanmean(c, axis=0), np.nanstd(c, axis=0), np.isnan(c).sum(axis=0)

    print(f"\nPaper cost (WR={WR}), mean +- std over {args.trials} noisy trials "
          f"(* = some trials diverged and are excluded)")
    print(f"FLiLQG = FL-iLQG with the FL split, FLiLQG mn = with the min-norm split")
    print(f"\n60 ms delay\n{'ff_power':>9s} " + "".join(f"{SHORT[c]:>22s}" for c in CONTROLLERS))
    for f in FF_POWERS:
        mean, std, n_nan = stats(f, DELAY)
        print(f"{f:9.0e} " + "".join(fmt(mean[k], std[k], n_nan[k], args.trials) for k in range(len(CONTROLLERS))))
    for f in FF_FOR_DELAY_SWEEP:
        print(f"\nff_power = {f:g}\n{'delay':>9s} " + "".join(f"{SHORT[c]:>22s}" for c in CONTROLLERS))
        for d in DELAYS:
            mean, std, n_nan = stats(f, d)
            print(f"{int(d * 1000):7d}ms " + "".join(fmt(mean[k], std[k], n_nan[k], args.trials)
                                                    for k in range(len(CONTROLLERS))))

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.6, 1, 1])
    ax_ff = fig.add_subplot(gs[0, 0])
    ax_inc = fig.add_subplot(gs[1, 0])
    by_ff = [stats(f, DELAY) for f in FF_POWERS]
    means = np.array([m for m, _, _ in by_ff])
    stds = np.array([s for _, s, _ in by_ff])
    i0 = FF_POWERS.index(0.0)
    nonzero = [i for i, f in enumerate(FF_POWERS) if f != 0]
    ff_nonzero = np.array(FF_POWERS)[nonzero]
    for k, c in enumerate(CONTROLLERS):
        band(ax_ff, FF_POWERS, means[:, k], stds[:, k], COLORS[c], LABELS[c])
        band(ax_inc, ff_nonzero, means[nonzero, k] - means[i0, k], stds[nonzero, k], COLORS[c], LABELS[c])
    for ax, ylabel, title in [
        (ax_ff, "Movement cost", f"Cost vs force field, 60 ms delay ({args.trials} noisy trials)"),
        (ax_inc, "mean J(ff) - mean J(0)", "Cost increase caused by the field"),
    ]:
        ax.set_yscale("log")
        ax.set_xlabel("Force field strength (ff_power)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        delete_axis(ax)
    ax_ff.legend(fontsize=8)

    delays_ms = np.array(DELAYS) * 1000
    for i, f in enumerate(FF_FOR_DELAY_SWEEP):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
        by_delay = [stats(f, d) for d in DELAYS]
        for k, c in enumerate(CONTROLLERS):
            band(ax, delays_ms, np.array([m[k] for m, _, _ in by_delay]),
                 np.array([s[k] for _, s, _ in by_delay]), COLORS[c], LABELS[c])
        ax.set_yscale("log")
        ax.set_xlabel("Sensory delay [ms]")
        ax.set_ylabel("Movement cost")
        ax.set_title(f"ff_power = {f:g}", fontsize=11)
        delete_axis(ax)

    fig.suptitle("Mean +- std over noisy trials (band lower edge clipped on the log axis)", fontsize=10)
    fig.tight_layout()
    out = HERE / "force_field_sweep.svg"
    fig.savefig(out)
    print(f"\nFigure saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
