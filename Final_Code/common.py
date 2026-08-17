"""
Shared setup for the paper figure scripts.

Every script in this folder imports from here so that the simulation parameters,
the cost function and the plotting style are defined in exactly one place.

Run any script directly, e.g.:

    python Final_Code/centerout_trajectories.py
    python Final_Code/centerout_cost_polar.py --amplitude 15 --duration 0.4
    python Final_Code/centerout_cost_polar.py --num-sim 5 --jobs 1   # quick check

Figures are written to Final_Code/figures/ unless --outdir says otherwise.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from math import cos, pi, sin
from matplotlib import pyplot as plt

from Controllers.FL import ToCartesian, compute_path, simulate_FL
from Controllers.ILQG import FULL_PLANT, Plant, simulate_ILQG
from Controllers.LQGControllers import DLQG_6Muscles
from Helpers.Helpers import (
    compute_angles_from_cartesian,
    delete_axis,
    delete_ticks,
    get_colors_from_colormap,
    longmovement_1,
    longmovement_2,
)

FIGURE_DIR = Path(__file__).resolve().parent / "figures"

# ----------------------------------------------------------------------------
# Simulation parameters shared by the figures
# ----------------------------------------------------------------------------
WP = 20000  # position (target) cost weight
WV = 1  # terminal velocity cost weight
WR = 0.02  # motor cost of ILQG and DLQG
WR_FL = 6e-5  # motor cost of FL
MOTOR_NOISE = 5e-4  # motor noise variance
DELAY = 0.06  # sensory feedback delay [s]
START = [0, 40]  # center-out starting position [cm]

# Straight-path cost parameters (FL only), from a grid search over both, redone
# for WR_FL = 6e-5. At TAU_PATH = 0.15 the peak lateral deviation of the long
# movements falls from 13.6 to 0.8 cm and from 11.9 to 0.6 cm between WC = 0 and
# WC = 0.1 -- a straight hand path -- while the endpoint error stays under
# 0.01 cm. Both are saturated past WC ~ 0.1. See Final_Code/README.md.
WC = 0.1  # weight of the straight-path cost
TAU_PATH = 0.15  # time constant of the straight-path cost decay

COLORS = ["#009E73", "#0072B2", "#E69F00"]
LEGEND = ["ILQG", "FL", "DLQG"]
NUM_CONTROLLERS = 3


# ----------------------------------------------------------------------------
# Cost functions
# ----------------------------------------------------------------------------
def Cost_function(x, u, w1=WP, w2=WV, r=WR, tg=(0, 0)):
    """Total movement cost: target accuracy, terminal velocity and motor cost."""
    target1, target2 = compute_angles_from_cartesian(tg[0], tg[1])
    thetas, thetae, omegas, omegae = x[-1, :4]

    return (
        w1 * (thetas - target1) ** 2
        + w1 * (thetae - target2) ** 2
        + w2 * (omegas**2 + omegae**2)
        + np.sum(u * u) * r
    )


def Cost_r(x, u, w1=WP, w2=WV, r=WR, tg=(0, 0)):
    """Motor cost component alone."""
    return np.sum(u * u) * r


def cost_components(x, u, w1=WP, w2=WV, r=WR, tg=(0, 0)):
    """
    The three terms of Cost_function, kept apart.

    Every controller is scored with the shared r, including FL, which is
    *optimised* with the much smaller WR_FL. Scoring each one under the weights
    it optimised would compare three different quantities.
    """
    target1, target2 = compute_angles_from_cartesian(tg[0], tg[1])
    thetas, thetae, omegas, omegae = x[-1, :4]

    return (
        w1 * ((thetas - target1) ** 2 + (thetae - target2) ** 2),
        w2 * (omegas**2 + omegae**2),
        np.sum(u * u) * r,
    )


def Compute_Cartesian_Speed(x, L1=30, L2=33):
    """Hand velocity in cartesian coordinates from the joint state x[:4]."""
    ts, te, os_, oe = x[0], x[1], x[2], x[3]
    Velx = -L1 * np.sin(ts) * os_ - L2 * np.sin(ts + te) * (os_ + oe)
    Vely = L1 * np.cos(ts) * os_ + L2 * np.cos(ts + te) * (os_ + oe)
    return Velx, Vely, np.sqrt(Velx**2 + Vely**2)


# ----------------------------------------------------------------------------
# Controller wrappers
# ----------------------------------------------------------------------------
def run_ilqg(duration, num_iter, start, target, noise=True, ff=False, ff_power=0.0,
             wp=WP, wv=WV, wr=WR, motor_noise=MOTOR_NOISE, plant=FULL_PLANT):
    """Run ILQG. Returns (X, Y, state, command).

    `plant` selects which arm nonlinearities are kept; the default keeps all
    three, so every caller but nonlinearity_ablation.py can ignore it.
    """
    X, Y, x, u = simulate_ILQG(
        duration, wp, wv, wr, target, start, num_iter,
        delay=DELAY, Noise=noise, print_iterations=False,
        FF=ff, ff_power=ff_power, motornoise_variance=motor_noise, plant=plant,
    )
    return X, Y, x, u


def run_fl(duration, num_iter, start, target, noise=True, ff=False, ff_power=0.0,
           wp=WP, wv=WV, wr=WR_FL, motor_noise=MOTOR_NOISE,
           wc=0, taupath=TAU_PATH):
    """Run the feedback linearisation controller. Returns (X, Y, state, command)."""
    X, Y, x, u = simulate_FL(
        Duration=duration, w1=wp, w2=wp, w3=wv, w4=wv, r=wr,
        Num_iter=num_iter, starting_point=start, targets=target,
        Delay=DELAY, Activate_Noise=noise, FF=ff, ff_power=ff_power,
        motornoise_variance=motor_noise, wp=wc, taupath=taupath,
    )
    return X, Y, x, u


def run_dlqg(duration, num_iter, start, target, noise=True, ff=False, ff_power=0.0,
             wp=WP, wv=WV, wr=WR, motor_noise=MOTOR_NOISE):
    """Run DLQG with six muscles. Returns (X, Y, state, command).

    The state is transposed to (time, variable) so that it matches the layout
    returned by the other two controllers.
    """
    X, Y, u, z = DLQG_6Muscles(
        w1=wp, w2=wp, w3=wv, w4=wv, Duration=duration, r1=wr,
        Num_iter=num_iter, starting_point=start, targets=target, plot=False,
        Delay=DELAY, Activate_Noise=noise, FF=ff, ff_power=ff_power,
        motornoise_variance=motor_noise,
    )
    return X, Y, z.T, u


class ControllerDiverged(RuntimeError):
    """A controller produced a non-finite state for the requested movement."""


def guarded(runner, label, *args, **kwargs):
    """
    Call a controller and turn a divergence into a message that says what broke.

    Without this, a NaN state surfaces as a bare LinAlgError from inside a
    worker process, with no indication of which condition caused it.
    """
    try:
        return runner(*args, **kwargs)
    except np.linalg.LinAlgError as exc:
        raise ControllerDiverged(
            f"{label} diverged ({exc}). The normalised muscle length goes "
            "negative along this movement, so l**1.55 in the force-length "
            "curve returns NaN. See Final_Code/README.md."
        ) from exc


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def centerout_targets(start=START, amplitude=15, num_targets=8):
    """The `num_targets` center-out targets at `amplitude` cm from `start`."""
    angles = np.linspace(0, 2 * pi, num_targets + 1)[:-1]
    return [[start[0] + cos(a) * amplitude, start[1] + sin(a) * amplitude] for a in angles]


def style_polar_axis(ax, radial_ticks, num_targets=8, rmax=None,
                     radial_fontsize=22, degree_fontsize=17):
    """
    Shared look of the center-out polar figures.

    Each direction is named at the outer end of its own radial line, and the
    radial ticks are the only rings: the rim is dropped so that it cannot be
    read as one more.
    """
    angles = np.linspace(0, 2 * pi, num_targets + 1)[:-1]
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{d:.0f}°" for d in np.degrees(angles)],
                       fontsize=degree_fontsize)

    ax.spines["polar"].set_visible(False)
    ax.set_yticks(radial_ticks, labels=[f"{t:g}" for t in radial_ticks],
                  fontsize=radial_fontsize)
    ax.set_rlabel_position(22.5)  # between two spokes, so the labels stay clear
    if rmax is not None:
        ax.set_ylim(0, rmax)


# ----------------------------------------------------------------------------
# Parallel execution
# ----------------------------------------------------------------------------
def default_jobs():
    return max(1, (os.cpu_count() or 1) - 1)


def run_tasks(worker, tasks, jobs=None, desc=None):
    """
    Map `worker` over `tasks`, in worker processes when jobs > 1.

    Results come back in task order, so the figures are identical to a serial
    run apart from the noise realisations (which are random and unseeded in
    either case). `worker` must be a module-level function so it can be pickled.
    """
    tasks = list(tasks)
    jobs = default_jobs() if jobs is None else jobs
    jobs = max(1, min(jobs, len(tasks))) if tasks else 1

    if desc:
        print(f"{desc}: {len(tasks)} simulations on {jobs} process(es)", flush=True)

    if jobs == 1:
        return [worker(t) for t in tasks]

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(worker, tasks, chunksize=1))


# ----------------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------------
def build_parser(description, num_sim_default=100):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num-sim", type=int, default=num_sim_default,
                        help="number of noisy repetitions per condition")
    parser.add_argument("--jobs", type=int, default=None,
                        help="worker processes (default: cpu_count - 1, 1 disables)")
    parser.add_argument("--outdir", type=Path, default=FIGURE_DIR,
                        help="directory for the generated figures")
    parser.add_argument("--no-show", action="store_true",
                        help="save the figures without opening a window")
    return parser


def save_figure(fig, outdir, name, dpi=300):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"wrote {path}", flush=True)
    return path


def finish(show):
    if show:
        plt.show()
    else:
        plt.close("all")
