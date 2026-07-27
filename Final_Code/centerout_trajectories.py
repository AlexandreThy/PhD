"""
Center-out reaching trajectories of the three controllers.

For each controller, plots the individual noisy trajectories towards the eight
targets and, in a second figure, the trajectory averaged over repetitions.
Replaces the three near-identical notebook cells with one parameterised pass.

    python Final_Code/centerout_trajectories.py
    python Final_Code/centerout_trajectories.py --num-sim 5 --jobs 1
"""

from common import (
    COLORS, LEGEND, START, build_parser, centerout_targets, finish, np,
    plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
NUM_TARGETS = 8
MAX_TRAJECTORIES_SHOWN = 15

RUNNERS = (run_ilqg, run_fl, run_dlqg)


def _worker(task):
    """Run one (controller, repetition, target) simulation. Must stay top-level."""
    controller, _, target, duration, num_iter, start = task
    X, Y, _, _ = RUNNERS[controller](duration, num_iter, start, target)
    return np.array([X, Y]).T


def simulate(controller, num_sim, jobs, start, amplitude):
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [
        (controller, rep, target, MOVEMENT_TIME, NUM_ITER, start)
        for rep in range(num_sim)
        for target in targets
    ]
    flat = run_tasks(_worker, tasks, jobs, desc=f"{LEGEND[controller]} center-out")
    # (repetition, target, timestep, xy)
    trajectories = np.array(flat).reshape(num_sim, len(targets), NUM_ITER + 1, 2)
    return targets, trajectories


def _draw_markers(ax, start, targets):
    for target in targets:
        ax.plot([target[0]], [target[1]], marker="s", markersize=25,
                markeredgecolor="grey", markerfacecolor="white", zorder=0,
                markeredgewidth=5)
    ax.plot([start[0]], [start[1]], marker="o", markersize=25,
            markeredgecolor="grey", markerfacecolor="white", zorder=0,
            markeredgewidth=5)
    ax.axis("equal")
    ax.axis("off")


def plot_controller(controller, targets, trajectories, start, outdir):
    color = COLORS[controller]
    name = LEGEND[controller]

    fig, ax = plt.subplots(figsize=(8, 8))
    for rep in range(min(trajectories.shape[0], MAX_TRAJECTORIES_SHOWN)):
        for t in range(len(targets)):
            ax.plot(trajectories[rep, t, :, 0], trajectories[rep, t, :, 1],
                    color=color, linewidth=1, label=name)
    _draw_markers(ax, start, targets)
    save_figure(fig, outdir, f"{name}_Centerout.svg")

    fig_mean, ax_mean = plt.subplots(figsize=(8, 8))
    for t in range(len(targets)):
        ax_mean.plot(np.mean(trajectories[:, t, :, 0], axis=0),
                     np.mean(trajectories[:, t, :, 1], axis=0),
                     color=color, linewidth=3, label=name)
    _draw_markers(ax_mean, start, targets)
    save_figure(fig_mean, outdir, f"{name}_Centerout_mean.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=15)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE,
                        help="reach amplitude in cm")
    args = parser.parse_args()

    for controller in range(len(RUNNERS)):
        targets, trajectories = simulate(controller, args.num_sim, args.jobs,
                                         START, args.amplitude)
        plot_controller(controller, targets, trajectories, START, args.outdir)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
