"""
Mean center-out hand trajectories of the three controllers for several motor
costs r1, one panel per r1.

r1 is the motor cost of ILQG and LQG, as in `centerout_cost_polar.py --r1`.
FL keeps its own WR_FL, so its trajectories do not depend on r1: it is
simulated once and drawn on every panel as the reference.

    python Final_Code/centerout_trajectories_r1.py
    python Final_Code/centerout_trajectories_r1.py --r1 0.02 0.2 0.5
    python Final_Code/centerout_trajectories_r1.py --num-sim 5 --jobs 1
    python Final_Code/centerout_trajectories_r1.py --replot
"""

from matplotlib.lines import Line2D

from common import (
    FIGURE_SUBDIRS,
    COLORS, LEGEND, START, WR, build_parser, centerout_targets, finish, np,
    plt, run_fl, run_ilqg, run_lqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
NUM_TARGETS = 8
R1_VALUES = (WR, 0.2, 0.5)
FIGURE_NAME = "Centerout_mean_r1"


def _worker(task):
    """Run one (controller, r1, target) simulation. Must stay top-level."""
    controller, r1, target, start = task
    if controller == 0:
        X, Y, _, _ = run_ilqg(MOVEMENT_TIME, NUM_ITER, start, target, wr=r1)
    elif controller == 1:
        X, Y, _, _ = run_fl(MOVEMENT_TIME, NUM_ITER, start, target)
    else:
        X, Y, _, _ = run_lqg(MOVEMENT_TIME, NUM_ITER, start, target, wr=r1)
    return np.array([X, Y]).T


def simulate(r1_values, num_sim, jobs, start, amplitude):
    """Mean trajectories, shape (r1, controller, target, timestep, xy)."""
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    # FL does not use r1: one run set, shared by every panel.
    keys = [(0, r1) for r1 in r1_values] + [(1, None)] + [(2, r1) for r1 in r1_values]
    tasks = [(c, r1, target, start)
             for c, r1 in keys for _ in range(num_sim) for target in targets]
    flat = run_tasks(_worker, tasks, jobs, desc="center-out mean vs r1")
    runs = np.array(flat).reshape(len(keys), num_sim, NUM_TARGETS, NUM_ITER + 1, 2)
    means = dict(zip(keys, runs.mean(axis=1)))

    return np.array([
        [means[(0, r1)], means[(1, None)], means[(2, r1)]] for r1 in r1_values
    ]), np.array(targets)


def _draw_markers(ax, start, targets):
    for target in targets:
        ax.plot([target[0]], [target[1]], marker="s", markersize=14,
                markeredgecolor="grey", markerfacecolor="white", zorder=0,
                markeredgewidth=3)
    ax.plot([start[0]], [start[1]], marker="o", markersize=14,
            markeredgecolor="grey", markerfacecolor="white", zorder=0,
            markeredgewidth=3)
    ax.set_aspect("equal")
    ax.axis("off")


def plot(r1_values, mean_traj, targets, start, outdir):
    fig, axes = plt.subplots(1, len(r1_values), figsize=(6 * len(r1_values), 6.5))
    for ax, r1, per_controller in zip(np.atleast_1d(axes), r1_values, mean_traj):
        for c, traj in enumerate(per_controller):
            for t in range(len(targets)):
                ax.plot(traj[t, :, 0], traj[t, :, 1], color=COLORS[c], linewidth=2.5)
        _draw_markers(ax, start, targets)
        ax.set_title(f"$r_1$ = {r1:g}", fontsize=22)

    handles = [Line2D([], [], color=COLORS[c], linewidth=3) for c in range(len(LEGEND))]
    fig.legend(handles, LEGEND, loc="lower center", ncol=len(LEGEND),
               fontsize=18, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, outdir, f"{FIGURE_NAME}.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=30,
                          subdir=FIGURE_SUBDIRS["trajectories"])
    parser.add_argument("--r1", type=float, nargs="+", default=list(R1_VALUES),
                        help="motor costs of ILQG and LQG, one panel each")
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE,
                        help="reach amplitude in cm")
    parser.add_argument("--replot", action="store_true",
                        help=f"redraw from the saved {FIGURE_NAME}.npz, no simulation")
    args = parser.parse_args()

    data_file = args.outdir / f"{FIGURE_NAME}.npz"
    if args.replot:
        saved = np.load(data_file)
        r1_values, mean_traj, targets = saved["r1"], saved["mean"], saved["targets"]
    else:
        r1_values = args.r1
        mean_traj, targets = simulate(r1_values, args.num_sim, args.jobs,
                                      START, args.amplitude)
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez(data_file, r1=r1_values, mean=mean_traj, targets=targets)
        print(f"wrote {data_file}", flush=True)

    plot(r1_values, mean_traj, targets, START, args.outdir)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
