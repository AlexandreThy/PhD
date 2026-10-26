"""
Pairwise RMS distance between hand trajectories that start from different postures.

Script version of Simulations.ipynb, for a different question: does the same
reach (same direction, same amplitude) give the same hand trajectory wherever
it starts? Eight starting points are spread over the part of the workspace where
every target of both amplitudes is reachable, and for each reach direction the
noiseless FL trajectories from those points are compared two by two.

Trajectories from different starts sit at different places, so each one is taken
as its displacement from its own start, d(t) = p(t) - p(0), and

    RMS(a, b) = sqrt( mean_t || d_a(t) - d_b(t) ||^2 )   [cm]

over the time steps. It is zero when the two reaches follow the same path with
the same timing, whatever the posture. Each amplitude has its own movement time
(DURATIONS), from the time-cost estimate: the effort of FL scales as D^2.2 / T^2.9,
so the duration minimising effort + c * T grows as about D^0.56.

    python Experiment_1/trajectory_rms.py              # 10 cm, then 20 cm
    python Experiment_1/trajectory_rms.py --amplitudes 10

Writes, in Experiment_1/figures/:
    TrajectoryRMS_starts.svg        the eight starting postures
    TrajectoryRMS_<amp>cm.svg       one 8 x 8 matrix per reach direction
    TrajectoryRMS_paths_<amp>cm.svg the eight displacements per direction
    TrajectoryRMS.npz               the matrices and the trajectories
"""

import argparse
import sys
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from FL import L1_CM, L2_CM, compute_angles_from_cartesian, simulate_FL  # noqa: E402

DURATIONS = {10: 0.4, 20: 0.6, 25: 0.7}  # movement time [s] of each amplitude [cm]
DT = 0.01  # simulation step [s]
WP, WV, WRFL = 2e4, 10, 6e-5
AMPLITUDES = (10, 20)
NUM_DIRECTIONS = 8
L1, L2 = L1_CM, L2_CM  # segment lengths [cm], from FL.kinematic_params
# Eight starting points, evenly spread over the region where every target of
# both amplitudes is reachable: the hand spans |L2 - L1| to L1 + L2 from the
# shoulder, so a start reaches all eight targets of the largest amplitude only
# between R_MIN and R_MAX. Two distances (at a quarter and three quarters of
# that range) by four angles around the shoulder.
R_MIN = abs(L2 - L1) + max(AMPLITUDES)
R_MAX = L1 + L2 - max(AMPLITUDES)
START_RADII = tuple(round(R_MIN + f * (R_MAX - R_MIN), 1) for f in (0.25, 0.75))  # [cm]
START_ANGLES = (70, 90, 110, 130)  # [deg], 90 = straight ahead
OUTDIR = HERE / "figures"
COLORS = plt.cm.tab10(np.arange(8))


def centerout_reachingtask(start=[0, 30], L=10, num_targ=8):
    tg = np.zeros((num_targ, 2))
    for i, angles in enumerate(np.linspace(0, 2 * pi, num_targ + 1)[:-1]):
        tg[i] = np.array([start[0] + cos(angles) * L, start[1] + sin(angles) * L])
    return tg


def starting_points():
    """The eight starts [cm], ordered by distance then angle, and their postures [deg]."""
    points = np.array([[r * np.cos(np.radians(phi)), r * np.sin(np.radians(phi))]
                       for r in START_RADII for phi in START_ANGLES])
    postures = np.degrees([compute_angles_from_cartesian(*p) for p in points])
    return points, postures


def num_steps(amplitude):
    return int(round(DURATIONS[amplitude] / DT))


def displacement(start, target, amplitude):
    """
    Noiseless FL hand path from start to target in the movement time of
    `amplitude`, minus its start: (num_steps + 1, 2) [cm].
    """
    X, Y, _, _ = simulate_FL(Duration=DURATIONS[amplitude], Num_iter=num_steps(amplitude),
                             starting_point=start, targets=target, Activate_Noise=False,
                             wp=WP, wv=WV, wr=WRFL)
    path = np.stack([X, Y], axis=1)
    return path - path[0]


def pairwise_rms(paths):
    """RMS over time of the distance between every pair of displacements."""
    diff = paths[:, None] - paths[None, :]  # (start, start, time, xy)
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=-1))


def simulate(amplitude, points):
    """Displacements (direction, start, time, xy) and RMS matrices (direction, start, start)."""
    paths = np.zeros((NUM_DIRECTIONS, len(points), num_steps(amplitude) + 1, 2))
    for s, start in enumerate(points):
        targets = centerout_reachingtask(start=start, L=amplitude, num_targ=NUM_DIRECTIONS)
        for d in range(NUM_DIRECTIONS):
            paths[d, s] = displacement(start, targets[d], amplitude)
    rms = np.array([pairwise_rms(paths[d]) for d in range(NUM_DIRECTIONS)])
    return paths, rms


def plot_starts(points, postures):
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    for s, ((x, y), (ts, te)) in enumerate(zip(points, postures)):
        elbow = L1 * np.array([np.cos(np.radians(ts)), np.sin(np.radians(ts))])
        ax.plot([0, elbow[0], x], [0, elbow[1], y], color=COLORS[s], linewidth=1.5,
                alpha=0.6)
        ax.plot(x, y, "o", color=COLORS[s], markersize=10, zorder=3)
        ax.annotate(f"{s + 1}", (x, y), textcoords="offset points", xytext=(7, 5),
                    fontsize=12, color=COLORS[s], fontweight="bold")
    angles = np.linspace(0, pi, 200)
    for r in (R_MIN, R_MAX):
        ax.plot(r * np.cos(angles), r * np.sin(angles), ":", color="grey", linewidth=1)
    ax.plot(0, 0, "o", color="grey", markersize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("Hand x [cm]")
    ax.set_ylabel("Hand y [cm]")
    ax.set_title(f"Starting postures (dotted: region where every 10 and 20 cm\n"
                 f"target is reachable, {R_MIN:.1f} to {R_MAX:.1f} cm from the shoulder)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTDIR / "TrajectoryRMS_starts.svg")
    return fig


def plot_rms(rms, amplitude):
    """One 8 x 8 matrix per direction, on one colour scale for the amplitude."""
    vmax = rms.max()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
    labels = [str(s + 1) for s in range(rms.shape[1])]
    for d, ax in enumerate(axes.ravel()):
        im = ax.imshow(rms[d], cmap="viridis", vmin=0, vmax=vmax)
        for a in range(rms.shape[1]):
            for b in range(rms.shape[2]):
                ax.text(b, a, f"{rms[d, a, b]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if rms[d, a, b] < 0.6 * vmax else "black")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Starting point")
        ax.set_ylabel("Starting point")
        off = rms[d][~np.eye(rms.shape[1], dtype=bool)]
        ax.set_title(f"Direction {d * 45} deg  (mean {off.mean():.2f}, max {off.max():.2f} cm)",
                     fontsize=12)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label("RMS distance between displacements [cm]")
    fig.suptitle(f"Pairwise RMS distance between noiseless FL hand trajectories, "
                 f"{amplitude} cm reaches in {DURATIONS[amplitude] * 1000:.0f} ms",
                 fontsize=15)
    fig.savefig(OUTDIR / f"TrajectoryRMS_{amplitude}cm.svg")
    return fig


def plot_paths(paths, amplitude):
    """The eight displacements of each direction, from a common origin."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
    for d, ax in enumerate(axes.ravel()):
        for s in range(paths.shape[1]):
            ax.plot(paths[d, s, :, 0], paths[d, s, :, 1], color=COLORS[s], linewidth=2,
                    label=f"{s + 1}")
        a = np.radians(d * 45)
        end = amplitude * np.array([np.cos(a), np.sin(a)])
        ax.plot([0, end[0]], [0, end[1]], ":", color="grey")
        ax.plot(0, 0, "o", color="grey")
        # The same square window around every reach, so the panels share one scale.
        centre, half = end / 2, 0.65 * amplitude
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal")
        ax.set_title(f"Direction {d * 45} deg", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(title="start", fontsize=8, ncol=2, frameon=False)
    fig.suptitle(f"Hand displacement from each starting point, {amplitude} cm reaches in "
                 f"{DURATIONS[amplitude] * 1000:.0f} ms (dotted: straight line)", fontsize=15)
    fig.savefig(OUTDIR / f"TrajectoryRMS_paths_{amplitude}cm.svg")
    return fig


def print_matrices(rms, amplitude):
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    for d in range(NUM_DIRECTIONS):
        print(f"\n{amplitude} cm, direction {d * 45} deg: RMS [cm]")
        print(rms[d])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=AMPLITUDES)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    points, postures = starting_points()
    print("Starting points (x, y [cm]; shoulder, elbow [deg]):")
    for s, (p, q) in enumerate(zip(points, postures)):
        print(f"  {s + 1}: ({p[0]:6.1f}, {p[1]:5.1f})   {q[0]:6.1f}, {q[1]:6.1f}")
    plot_starts(points, postures)

    results = {"points": points, "postures": postures}
    for amplitude in args.amplitudes:
        amplitude = int(amplitude) if float(amplitude).is_integer() else amplitude
        paths, rms = simulate(amplitude, points)
        print_matrices(rms, amplitude)
        plot_rms(rms, amplitude)
        plot_paths(paths, amplitude)
        results[f"rms_{amplitude}cm"] = rms
        results[f"paths_{amplitude}cm"] = paths
    np.savez(OUTDIR / "TrajectoryRMS.npz", **results)
    print(f"\nwrote figures and TrajectoryRMS.npz to {OUTDIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
