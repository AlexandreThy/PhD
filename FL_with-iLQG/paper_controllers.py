"""
Controllers and cost of CurrentParts/PaperPlot.ipynb, shared by the figure
scripts of this folder.

Four controllers, all run on the same arm / muscle model with delay and noise:
  ILQG      : simulate_ILQG of Controllers/ILQG.py with r1 = WR. Its cost is half
              Cost_function (no dt on the effort), so it is the optimal reference.
  FL        : simulate_FL with r = 1e-4, as in the notebook
  FL_muscle : FL-iLQG on the muscle commands (FL_iLQG_MuscleCost.py), wr = WR,
              with the muscle split of Controllers/FL.py
  FL_muscle_mn : same, with the minimum-norm muscle split (min_norm=True)
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from Controllers.FL import compute_angles_from_cartesian, simulate_FL
from Controllers.ILQG import simulate_ILQG
from FL_iLQG_MuscleCost import plan_FL_iLQG_muscle, simulate_FL_iLQG_muscle

DELAY = 0.06
MOTOR_NOISE = 5e-4
WP = 20000
WV = 1
R_FL = 1e-4  # r of simulate_FL in the notebook

CONTROLLERS = ["ILQG", "FL", "FL_muscle", "FL_muscle_mn"]
LABELS = {
    "ILQG": "ILQG (optimal)",
    "FL": f"FL (r={R_FL})",
    "FL_muscle": "FL-iLQG muscle (FL split)",
    "FL_muscle_mn": "FL-iLQG muscle (min-norm split)",
}
COLORS = {"ILQG": "#009E73", "FL": "#0072B2",
          "FL_muscle": "#CC79A7", "FL_muscle_mn": "#56B4E9"}


def cost_function(x, u, tg, wr, wp=WP, wv=WV):
    """Cost_function of PaperPlot.ipynb."""
    target1, target2 = compute_angles_from_cartesian(tg[0], tg[1])
    thetas, thetae, omegas, omegae = x[-1, :4]
    return (
        wp * (thetas - target1) ** 2
        + wp * (thetae - target2) ** 2
        + wv * (omegas**2 + omegae**2)
        + np.sum(u * u) * wr
    )


def plan_muscle(duration, num_iter, start, target, wr):
    """FL-iLQG muscle plans {min_norm: plan}, computed once and reused over trials."""
    return {mn: plan_FL_iLQG_muscle(duration, WP, WV, wr, target, start, num_iter, mn)
            for mn in (False, True)}


def run(controller, duration, num_iter, start, target, wr, noise=True,
        ff=False, ff_power=0.0, seed=None, plan=None):
    """One trial. Returns (X, Y, joint state (num_iter+1, 4), muscle commands (num_iter, 6))."""
    if seed is not None:
        np.random.seed(seed)
    if controller == "ILQG":
        X, Y, x, u = simulate_ILQG(
            duration, WP, WV, wr, target, start, num_iter, delay=DELAY, Noise=noise,
            print_iterations=False, FF=ff, ff_power=ff_power, motornoise_variance=MOTOR_NOISE,
        )
    elif controller == "FL":
        X, Y, x, u = simulate_FL(
            Duration=duration, w1=WP, w2=WP, w3=WV, w4=WV, r=R_FL, Num_iter=num_iter,
            starting_point=start, targets=target, Delay=DELAY, Activate_Noise=noise,
            FF=ff, ff_power=ff_power, motornoise_variance=MOTOR_NOISE,
        )
    elif controller in ("FL_muscle", "FL_muscle_mn"):
        min_norm = controller == "FL_muscle_mn"
        X, Y, x, u = simulate_FL_iLQG_muscle(
            Duration=duration, wr=wr, targets=target, starting_point=start,
            Num_iter=num_iter, Delay=DELAY, Activate_Noise=noise, FF=ff,
            ff_power=ff_power, motornoise_variance=MOTOR_NOISE,
            plan=None if plan is None else plan[min_norm], min_norm=min_norm,
        )
    else:
        raise ValueError(controller)
    return X, Y, x[:, :4], u


def _run_task(kwargs):
    # Rejected line-search trial steps of simulate_ILQG can overflow; the
    # results are checked for NaN by run_trials instead.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        return run(**kwargs)
    except np.linalg.LinAlgError:
        # A diverged trial (NaN state) makes the linear algebra fail: record it as NaN.
        n = kwargs["num_iter"]
        return np.full(n + 1, np.nan), np.full(n + 1, np.nan), np.full((n + 1, 4), np.nan), np.full((n, 6), np.nan)


def run_trials(tasks, jobs=None):
    """Run a list of `run` keyword dicts in parallel, in order."""
    # Each worker imports numpy / matplotlib (~100 MB): more than a few workers
    # can exhaust memory (MemoryError at start-up), hence the cap of 4.
    jobs = jobs or max(1, min(4, (os.cpu_count() or 2) - 1))
    if jobs == 1:
        results = [_run_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_run_task, tasks, chunksize=4))
    bad = sum(np.isnan(r[2]).any() for r in results)
    if bad:
        print(f"WARNING: {bad} of {len(results)} trials diverged (NaN)")
    return results


def delete_axis(ax, sides=("top", "right")):
    for side in sides:
        ax.spines[side].set_visible(False)
