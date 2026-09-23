"""
Check that FL_iLQG_Combined reproduces classical FL on the center-out task.

On the feedback-linearised system the dynamics are linear and the FL cost is
quadratic, so iLQG must:
  1. converge after a single update,
  2. return feedback gains L_k = -L^FL_k and a nominal plan vbar_k = -L^FL_k zbar_k,
  3. hence produce the same trajectories and muscle commands as simulate_FL,
     with and without noise (same random seed).

The figure shows the noisy case, averaged over NUM_TRIALS trials per target.

Parameters are those of the paper figures (Final_Code/common.py).
The figure is saved next to this script, as FL_iLQG_verif.svg.

    python FL_with-iLQG/FL_iLQG_verif.py
    python FL_with-iLQG/FL_iLQG_verif.py --no-plot
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from matplotlib import pyplot as plt

from Controllers.FL import compute_linear_control_gains, compute_path, simulate_FL
from FL_iLQG_Combined import plan_FL_iLQG, simulate_FL_iLQG

# Paper parameters, copied from Final_Code/common.py (not imported: common.py
# pulls in every controller, so any breakage there would break this check).
WP = 20000
WV = 1
WR_FL = 6e-5
MOTOR_NOISE = 5e-4
DELAY = 0.06
START = [0, 40]
WC = 0.1
TAU_PATH = 0.15

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
PERCENT = 0.75  # simulate_FL default
NUM_TRIALS = 50  # noisy trials per target averaged in the figure

# Relative tolerances. Both controllers do the same floating point work up to
# the order of the Riccati operations, so differences are at round-off level.
GAIN_RTOL = 1e-8
TRAJ_ATOL_CM = 1e-6
COMMAND_RTOL = 1e-6

CASES = [
    # name, path cost weight, noise, force field
    ("no path cost, noise-free", 0, False, False),
    ("path cost, noise-free", WC, False, False),
    ("path cost, noisy", WC, True, False),
]
NOISY_CASE = CASES[-1]


def common_kwargs(target, wc):
    return dict(
        Duration=MOVEMENT_TIME, w1=WP, w2=WP, w3=WV, w4=WV, r=WR_FL,
        targets=target, starting_point=START, Num_iter=NUM_ITER,
        wp=wc, taupath=TAU_PATH, percent=PERCENT,
    )


def check_plan(target, wc):
    """Compare the iLQG plan with the FL LQR gains. Returns (gain error, plan error, iterations)."""
    kwargs = common_kwargs(target, wc)
    z0, zbar, vbar, L, info = plan_FL_iLQG(**kwargs)

    Qk = compute_path(np.array(START), np.array(target), wc, PERCENT)
    L_FL = compute_linear_control_gains(
        NUM_ITER, MOVEMENT_TIME, Qk, TAU_PATH, motor_cost=WR_FL,
        cost_weights=[WP, WP, WV, WV],
    )
    gain_err = np.max(np.abs(L + L_FL)) / np.max(np.abs(L_FL))
    v_FL = -np.einsum("kij,kj->ki", L_FL, zbar[:-1])
    plan_err = np.max(np.abs(vbar - v_FL)) / np.max(np.abs(v_FL))
    return gain_err, plan_err, info["iterations"], info["converged"]


def run_pair(target, wc, noise, ff, seed):
    """Simulate FL and FL-iLQG from the same random seed."""
    kwargs = common_kwargs(target, wc)
    kwargs.update(Delay=DELAY, Activate_Noise=noise, FF=ff,
                  motornoise_variance=MOTOR_NOISE)
    np.random.seed(seed)
    X1, Y1, x1, u1 = simulate_FL(**kwargs)
    np.random.seed(seed)
    X2, Y2, x2, u2 = simulate_FL_iLQG(**kwargs)
    traj_err = max(np.max(np.abs(X1 - X2)), np.max(np.abs(Y1 - Y2)))
    cmd_err = np.max(np.abs(u1 - u2)) / np.max(np.abs(u1))
    return traj_err, cmd_err, (X1, Y1), (X2, Y2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    targets = [[START[0] + AMPLITUDE * np.cos(a), START[1] + AMPLITUDE * np.sin(a)] for a in angles]
    failures = []
    trajectories = {}

    print(f"Center-out, {len(targets)} targets, amplitude {AMPLITUDE} cm, "
          f"T = {MOVEMENT_TIME} s, N = {NUM_ITER}\n")
    header = f"{'case':32s} {'target':>6s} {'iLQG it':>7s} {'gain err':>10s} {'plan err':>10s} {'traj err [cm]':>13s} {'cmd err':>10s}"
    print(header)
    print("-" * len(header))
    for name, wc, noise, ff in CASES:
        for t, target in enumerate(targets):
            gain_err, plan_err, iterations, converged = check_plan(target, wc)
            traj_err, cmd_err, fl, comb = run_pair(target, wc, noise, ff, args.seed + t)
            trajectories[(name, t)] = (fl, comb)
            print(f"{name:32s} {t:6d} {iterations:7d} {gain_err:10.2e} {plan_err:10.2e} {traj_err:13.2e} {cmd_err:10.2e}")

            ok = (
                converged
                and iterations == 1
                and gain_err < GAIN_RTOL
                and plan_err < GAIN_RTOL
                and traj_err < TRAJ_ATOL_CM
                and cmd_err < COMMAND_RTOL
            )
            if not ok:
                failures.append((name, t))

    print()
    if failures:
        print(f"FAILED for {len(failures)} (case, target) pairs: {failures}")
    else:
        print("PASSED: FL-iLQG reproduces classical FL on every target and case "
              "(single iLQG update, identical gains, trajectories and commands).")

    if not args.no_plot:
        name, wc, noise, ff = NOISY_CASE
        print(f"\nAveraging {NUM_TRIALS} noisy trials per target for the figure...")
        # (controller, trial, target, timestep, xy)
        paths = np.zeros((2, NUM_TRIALS, len(targets), NUM_ITER + 1, 2))
        worst = 0.0
        for trial in range(NUM_TRIALS):
            for t, target in enumerate(targets):
                traj_err, _, fl, comb = run_pair(target, wc, noise, ff, args.seed + 1000 * trial + t)
                paths[0, trial, t] = np.array(fl).T
                paths[1, trial, t] = np.array(comb).T
                worst = max(worst, traj_err)
        print(f"Largest FL / FL-iLQG trajectory difference over the {NUM_TRIALS} trials: {worst:.2e} cm")
        if worst > TRAJ_ATOL_CM:
            failures.append(("figure trials", worst))
            print("FAILED: the noisy trials of the figure differ")

        mean = paths.mean(axis=1)
        fig, ax = plt.subplots(figsize=(6, 6))
        for t in range(len(targets)):
            ax.plot(*mean[0, t].T, color="#0072B2", linewidth=3, label="FL" if t == 0 else None)
            ax.plot(*mean[1, t].T, "--", color="#E69F00", linewidth=1.5, label="FL-iLQG" if t == 0 else None)
        ax.scatter(*np.array(targets).T, color="black", s=15)
        ax.set_title(f"{name}, mean of {NUM_TRIALS} trials")
        ax.axis("equal")
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        ax.legend()
        fig.tight_layout()
        out = Path(__file__).resolve().parent / "FL_iLQG_verif.svg"
        fig.savefig(out)
        print(f"Figure saved to {out}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
