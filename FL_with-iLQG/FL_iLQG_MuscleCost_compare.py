"""
FL-iLQG on the muscle-command cost versus FL and ILQG, scored with the
Cost_function of CurrentParts/PaperPlot.ipynb, on the 15 cm center-out task
(T = 0.4 s, N = 40, WR = 0.1, as in the notebook cost cells).

  1. Optimality of the FL-iLQG plans (both muscle splits): random perturbations
     of the planned virtual command vbar must all increase its (noise-free) cost.
  2. Noise-free cost of the four controllers of paper_controllers.py, plus FL
     at its best motor weight r (grid search) for each target.
  3. Noisy cost averaged over NUM_TRIALS trials per target.

ILQG (Controllers/ILQG.py, r1 = WR) minimises the paper cost over all muscle
commands, so it is the optimal reference the other controllers are compared to.

    python FL_with-iLQG/FL_iLQG_MuscleCost_compare.py [--trials 50] [--jobs 8]
"""

import argparse
import sys

from paper_controllers import (
    COLORS, CONTROLLERS, HERE, LABELS, WP, WV, cost_function, plan_muscle, run_trials,
)

import numpy as np
from matplotlib import pyplot as plt

from Controllers.FL import simulate_FL
from FL_iLQG_MuscleCost import MuscleCommandCost, initial_state, linear_dynamics, rollout, total_cost

MOVEMENT_TIME = 0.4
NUM_ITER = 40
START = [0, 40]
AMPLITUDE = 15
WR = 0.1
R_FL_GRID = np.logspace(-6, -2, 17)
NUM_PERTURBATIONS = 20
SHORT = {"ILQG": "ILQG", "FL": "FL",
         "FL_muscle": "FLiLQG", "FL_muscle_mn": "FLiLQG mn"}


def perturbation_check(tg, plans, rng):
    """Smallest relative cost increase over random perturbations of vbar, for both splits (must be > 0)."""
    return min(_perturbation_check(tg, plans[mn], mn, rng) for mn in (False, True))


def _perturbation_check(tg, plan, min_norm, rng):
    zbar, vbar, _, _ = plan
    A, B = linear_dynamics(MOVEMENT_TIME / NUM_ITER)
    z0 = initial_state(START, tg)
    cost = MuscleCommandCost(WP, WV, WR, min_norm)
    J0 = total_cost(zbar, vbar, cost)
    increases = []
    for _ in range(NUM_PERTURBATIONS):
        dv = rng.normal(size=vbar.shape) * 1e-2 * np.max(np.abs(vbar))
        increases.append(total_cost(rollout(z0, vbar + dv, A, B), vbar + dv, cost) / J0 - 1)
    return min(increases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=50, help="noisy trials per target")
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    targets = [[START[0] + AMPLITUDE * np.cos(a), START[1] + AMPLITUDE * np.sin(a)] for a in angles]
    common = dict(duration=MOVEMENT_TIME, num_iter=NUM_ITER, start=START, wr=WR)

    plans = [plan_muscle(MOVEMENT_TIME, NUM_ITER, START, tg, WR) for tg in targets]

    # Noise-free runs, with FL also at its best r for each target
    print("Noise-free runs...")
    tasks = [dict(controller=c, target=tg, noise=False, plan=plans[t], **common)
             for t, tg in enumerate(targets) for c in CONTROLLERS]
    results = run_trials(tasks, args.jobs)
    J_free = np.array([cost_function(r[2], r[3], task["target"], WR) for r, task in zip(results, tasks)])
    J_free = J_free.reshape(len(targets), len(CONTROLLERS))

    # Noisy runs: same seed for every controller within a (trial, target)
    print(f"Noisy runs ({args.trials} trials per target)...")
    tasks = [dict(controller=c, target=tg, noise=True, seed=1000 * t + trial, plan=plans[t], **common)
             for t, tg in enumerate(targets) for trial in range(args.trials) for c in CONTROLLERS]
    results = run_trials(tasks, args.jobs)
    J_noisy = np.array([cost_function(r[2], r[3], task["target"], WR) for r, task in zip(results, tasks)])
    J_noisy = J_noisy.reshape(len(targets), args.trials, len(CONTROLLERS)).mean(axis=1)
    paths = np.array([[r[0], r[1]] for r in results]).transpose(0, 2, 1)
    mean_paths = paths.reshape(len(targets), args.trials, len(CONTROLLERS), NUM_ITER + 1, 2).mean(axis=1)

    print(f"\nPaper cost, WP={WP}, WV={WV}, WR={WR}\n")
    names = [SHORT[c] for c in CONTROLLERS]
    header = (f"{'tg':>2s} {'dJ/J pert':>9s} | noise-free: "
              + " ".join(f"{n:>9s}" for n in names) + f" {'FL best r':>9s} | noisy: "
              + " ".join(f"{n:>9s}" for n in names))
    print("(FLiLQG = FL-iLQG muscle with the FL split, FLiLQG mn = with the min-norm split)")
    print(header)
    print("-" * len(header))
    ok = True
    for t, tg in enumerate(targets):
        min_increase = perturbation_check(tg, plans[t], rng)
        fl_best = min(
            cost_function(*_run_fl_r(tg, r)[2:], tg, WR) for r in R_FL_GRID
        )
        print(f"{t:2d} {min_increase:9.2e} | noise-free: "
              + " ".join(f"{v:9.4f}" for v in J_free[t]) + f" {fl_best:9.4f} | noisy: "
              + " ".join(f"{v:9.4f}" for v in J_noisy[t]))
        mus = CONTROLLERS.index("FL_muscle")
        ok &= all(p[3]["converged"] for p in plans[t].values()) and min_increase > 0 and J_free[t, mus] < fl_best

    print("\nMean over targets:")
    for c, name in enumerate(CONTROLLERS):
        print(f"  {LABELS[name]:32s} noise-free {J_free[:, c].mean():.4f}   noisy {J_noisy[:, c].mean():.4f}")
    opt = CONTROLLERS.index("ILQG")
    for name in ("FL_muscle", "FL_muscle_mn"):
        c = CONTROLLERS.index(name)
        print(f"{LABELS[name]} / optimal ILQG (noise-free): {J_free[:, c].mean() / J_free[:, opt].mean():.3f}")
    print("PASSED: FL-iLQG plan is a local optimum and beats FL at its best r on every target."
          if ok else "FAILED: see table")

    fig = plt.figure(figsize=(13, 6))
    ax_traj = fig.add_subplot(1, 2, 1)
    for c, name in enumerate(CONTROLLERS):
        for t in range(len(targets)):
            ax_traj.plot(*mean_paths[t, c].T, color=COLORS[name], linewidth=2,
                         label=LABELS[name] if t == 0 else None)
    ax_traj.scatter(*np.array(targets).T, color="black", s=15)
    ax_traj.axis("equal")
    ax_traj.set_xlabel("X [cm]")
    ax_traj.set_ylabel("Y [cm]")
    ax_traj.set_title(f"Mean of {args.trials} noisy trials")
    ax_traj.legend(fontsize=8)

    ax_cost = fig.add_subplot(1, 2, 2, projection="polar")
    closed = np.append(angles, angles[0])
    for c, name in enumerate(CONTROLLERS):
        ax_cost.plot(closed, np.append(J_noisy[:, c], J_noisy[0, c]), "--o", color=COLORS[name], label=LABELS[name])
    ax_cost.set_title(f"Paper cost (mean of {args.trials} noisy trials)")
    fig.tight_layout()
    out = HERE / "FL_iLQG_MuscleCost_compare.svg"
    fig.savefig(out)
    print(f"Figure saved to {out}")
    return 0 if ok else 1


def _run_fl_r(tg, r):
    """Noise-free FL with motor weight r (for the grid search on r)."""
    return simulate_FL(Duration=MOVEMENT_TIME, w1=WP, w2=WP, w3=WV, w4=WV, r=r, targets=tg,
                       starting_point=START, Num_iter=NUM_ITER)


if __name__ == "__main__":
    sys.exit(main())
