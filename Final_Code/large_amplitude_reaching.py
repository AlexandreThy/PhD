"""
Large amplitude (55 cm) reaching for the two long movements.

For each of the two long movements, plots the trajectories of the three
controllers on one pair of axes, their joint angular velocity profiles and the
distribution of the movement cost, on a log and then on a linear scale.

    python Final_Code/large_amplitude_reaching.py
    python Final_Code/large_amplitude_reaching.py --num-sim 5 --jobs 1
"""

from matplotlib import gridspec
from matplotlib.lines import Line2D

from common import (
    COLORS, Cost_function, LEGEND, NUM_CONTROLLERS, build_parser,
    compute_angles_from_cartesian, cost_components, delete_axis, finish,
    guarded, longmovement_1, longmovement_2, np, pi, plt, run_dlqg, run_fl,
    run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
MAX_TRAJECTORIES_SHOWN = 10
# The notebook drew movement 2 on the top row and movement 1 below it
MOVEMENTS = (longmovement_2, longmovement_1)
MOVEMENT_BY_NUMBER = {1: longmovement_1, 2: longmovement_2}
COST_TERMS = ("target\naccuracy", "terminal\nvelocity", "motor")
JOINTS = ("shoulder", "elbow")


def _worker(task):
    """Run the three controllers for one repetition. Stays top-level."""
    start, target, duration, num_iter, movement_name = task
    runs = (
        guarded(run_ilqg, f"ILQG on {movement_name}", duration, num_iter, start, target),
        guarded(run_fl, f"FL on {movement_name}", duration, num_iter, start, target),
        guarded(run_dlqg, f"DLQG on {movement_name}", duration, num_iter, start, target),
    )
    target_angles = np.array(compute_angles_from_cartesian(target[0], target[1]))
    return {
        "cost": np.array([Cost_function(x, u, tg=target) for _, _, x, u in runs]),
        "components": np.array([cost_components(x, u, tg=target)
                                for _, _, x, u in runs]),
        "velocity": np.array([x[:, 2:4].T for _, _, x, _ in runs]),
        # Signed, so that overshooting and falling short stay distinguishable.
        "terminal_error": np.array([x[-1, :2] - target_angles
                                    for _, _, x, _ in runs]),
        "terminal_velocity": np.array([x[-1, 2:4] for _, _, x, _ in runs]),
        "trajectory": np.array([[X, Y] for X, Y, _, _ in runs]),
    }


def simulate(movement, num_sim, jobs):
    start, target = movement()
    tasks = [(start, target, MOVEMENT_TIME, NUM_ITER, movement.__name__)
             for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc=f"{movement.__name__}")
    data = {key: np.array([r[key] for r in results]) for key in results[0]}
    return start, target, data


def _mark_endpoints(ax, start, target):
    ax.add_patch(plt.Circle((target[0], target[1]), 1.5, edgecolor="grey",
                            facecolor="grey", linewidth=3))
    ax.add_patch(plt.Rectangle((start[0] - 1.5, start[1] - 1.5), 3, 3,
                               edgecolor="grey", facecolor="grey", linewidth=3))
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xticks([])
    delete_axis(ax)


def plot_trajectories(ax, start, target, trajectory):
    """All three controllers on one pair of axes, colour telling them apart."""
    shown = min(trajectory.shape[0], MAX_TRAJECTORIES_SHOWN)
    for i in range(NUM_CONTROLLERS):
        for rep in range(shown):
            ax.plot(trajectory[rep, i, 0], trajectory[rep, i, 1],
                    color=COLORS[i], linewidth=0.5)
    _mark_endpoints(ax, start, target)
    # "best" rather than a fixed corner: the two movements bow in opposite
    # directions, so no single corner is free on both panels.
    ax.legend(handles=[Line2D([], [], color=c, lw=3, label=n)
                       for c, n in zip(COLORS, LEGEND)],
              fontsize=18, loc="best", frameon=False)


def plot_velocities(vel_axes, velocity, time, label_y):
    for i in range(NUM_CONTROLLERS):
        ax = vel_axes[i]
        ax.plot(time, np.zeros(len(time)), color="black")
        for joint, linestyle in enumerate(["-", "--"]):
            trace = velocity[:, i, joint] * 180 / pi
            mean_vel, std_vel = np.mean(trace, axis=0), np.std(trace, axis=0)
            ax.plot(time, mean_vel, color=COLORS[i], linestyle=linestyle)
            ax.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                            color=COLORS[i], alpha=0.3)

        delete_axis(ax, sides=["top", "right"])
        ax.set_xlabel("Time [ms]", fontsize=22)
        ax.set_xticks([0, 600])
        ax.tick_params(labelsize=20)
        ax.grid(axis="y")

    vel_axes[0].set_ylabel(label_y, fontsize=22)


def share_velocity_axis(vel_rows):
    """
    One y range over every velocity panel, wide enough for all of them.

    Taken from what was drawn rather than fixed, because the old (-270, 320)
    cut the tops off the fastest traces; autoscale has already allowed for the
    +/- SD bands. Ticks land on the multiples of 90 that fit inside.
    """
    axes = [ax for row in vel_rows for ax in row]
    low = min(ax.get_ylim()[0] for ax in axes)
    high = max(ax.get_ylim()[1] for ax in axes)
    ticks = np.arange(np.ceil(low / 90) * 90, np.floor(high / 90) * 90 + 1, 90)

    for row in vel_rows:
        for i, ax in enumerate(row):
            ax.set_ylim(low, high)
            # Only the leftmost panel of each row carries the tick labels
            ax.set_yticks(ticks, labels=[f"{t:g}" for t in ticks] if i == 0
                          else [""] * len(ticks))


def plot_cost(ax, cost, log=True):
    """The same distribution twice: log first, then linear underneath it."""
    if log:
        ax.set_yscale("log")
    box = ax.boxplot(cost, patch_artist=True, medianprops=dict(color="black"),
                     whiskerprops=dict(color="black"),
                     capprops=dict(color="black"), showfliers=False)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
    ax.set_xticks(np.arange(NUM_CONTROLLERS) + 1, labels=LEGEND, fontsize=20)
    delete_axis(ax, sides=["top", "right", "bottom"])
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel(f"Movement Cost ({'log' if log else 'linear'})", fontsize=22)


def _grouped_boxplot(ax, values, group_labels, ylabel, title, log=False,
                     zero_line=False):
    """
    One box per controller, in a group per label.

    `values` is (repetition, controller, group).
    """
    if log:
        ax.set_yscale("log")
    span = NUM_CONTROLLERS + 1
    for group in range(len(group_labels)):
        box = ax.boxplot(values[:, :, group], patch_artist=True, showfliers=False,
                         positions=np.arange(NUM_CONTROLLERS) + group * span,
                         medianprops=dict(color="black"),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"))
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)

    if zero_line:
        ax.axhline(0, color="grey", linewidth=1, zorder=0)

    centre = (NUM_CONTROLLERS - 1) / 2
    ax.set_xticks([centre + g * span for g in range(len(group_labels))],
                  labels=group_labels, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    delete_axis(ax, sides=["top", "right", "bottom"])


def plot_terminal(per_movement, outdir):
    """Where each controller ends up, and what the cost is made of."""
    fig, axes = plt.subplots(3, len(per_movement), figsize=(14, 15),
                             layout="constrained")

    for column, (number, data) in enumerate(per_movement):
        _grouped_boxplot(axes[0, column], data["components"], COST_TERMS,
                         "Movement cost", f"Cost by term, movement {number}",
                         log=True)
        _grouped_boxplot(axes[1, column],
                         np.degrees(data["terminal_velocity"]), JOINTS,
                         "Terminal velocity [deg/s]",
                         f"Terminal joint velocity, movement {number}",
                         zero_line=True)
        _grouped_boxplot(axes[2, column],
                         np.degrees(data["terminal_error"]), JOINTS,
                         "Terminal error [deg]",
                         f"Terminal joint position, movement {number}",
                         zero_line=True)

    axes[0, 0].legend(handles=[Line2D([], [], color=c, lw=3, label=n)
                               for c, n in zip(COLORS, LEGEND)],
                      fontsize=12, loc="upper left", frameon=False)
    fig.suptitle("Where the long movements end, and what the cost is made of\n"
                 "(joint panels are signed: 0 is the target, reached at rest)",
                 fontsize=16)
    save_figure(fig, outdir, "long_move_terminal.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--movements", type=int, nargs="+", choices=(1, 2),
                        default=[2, 1],
                        help="which long movements to simulate (row order)")
    args = parser.parse_args()

    # One column per movement for the trajectories and the two cost rows; the
    # velocity profiles keep a row of three panels each, one per controller.
    # The trajectory row is drawn to scale and much wider than tall, so it needs
    # far less height than the four time series and boxplot rows.
    fig = plt.figure(figsize=(22, 32))
    gs = gridspec.GridSpec(5, 6, height_ratios=[0.4, 1, 1, 0.8, 0.8])
    traj_axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    vel_rows = [[fig.add_subplot(gs[row, 2 * i:2 * i + 2]) for i in range(3)]
                for row in (1, 2)]
    cost_log_axes = [fig.add_subplot(gs[3, :3]), fig.add_subplot(gs[3, 3:])]
    cost_linear_axes = [fig.add_subplot(gs[4, :3]), fig.add_subplot(gs[4, 3:])]

    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)

    per_movement = []
    for column, number in enumerate(args.movements):
        movement = MOVEMENT_BY_NUMBER[number]
        start, target, data = simulate(movement, args.num_sim, args.jobs)
        plot_trajectories(traj_axes[column], start, target, data["trajectory"])
        plot_velocities(vel_rows[column], data["velocity"], time,
                        "Angular Velocity [deg/sec]")
        plot_cost(cost_log_axes[column], data["cost"], log=True)
        plot_cost(cost_linear_axes[column], data["cost"], log=False)
        per_movement.append((number, data))

    share_velocity_axis(vel_rows)

    save_figure(fig, args.outdir, "LongMove.svg", dpi=200)
    plot_terminal(per_movement, args.outdir)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
