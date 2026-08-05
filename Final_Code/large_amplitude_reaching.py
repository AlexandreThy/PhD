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
    COLORS, Cost_function, LEGEND, NUM_CONTROLLERS, build_parser, delete_axis,
    finish, guarded, longmovement_1, longmovement_2, np, pi, plt, run_dlqg,
    run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
MAX_TRAJECTORIES_SHOWN = 10
# The notebook drew movement 2 on the top row and movement 1 below it
MOVEMENTS = (longmovement_2, longmovement_1)
MOVEMENT_BY_NUMBER = {1: longmovement_1, 2: longmovement_2}


def _worker(task):
    """Run the three controllers for one repetition. Stays top-level."""
    start, target, duration, num_iter, movement_name = task
    runs = (
        guarded(run_ilqg, f"ILQG on {movement_name}", duration, num_iter, start, target),
        guarded(run_fl, f"FL on {movement_name}", duration, num_iter, start, target),
        guarded(run_dlqg, f"DLQG on {movement_name}", duration, num_iter, start, target),
    )
    cost = np.array([Cost_function(x, u, tg=target) for _, _, x, u in runs])
    velocity = np.array([x[:, 2:4].T for _, _, x, _ in runs])
    trajectory = np.array([[X, Y] for X, Y, _, _ in runs])
    return cost, velocity, trajectory


def simulate(movement, num_sim, jobs):
    start, target = movement()
    tasks = [(start, target, MOVEMENT_TIME, NUM_ITER, movement.__name__)
             for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc=f"{movement.__name__}")

    cost = np.array([r[0] for r in results])
    velocity = np.array([r[1] for r in results])
    trajectory = np.array([r[2] for r in results])
    return start, target, cost, velocity, trajectory


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

    for column, number in enumerate(args.movements):
        movement = MOVEMENT_BY_NUMBER[number]
        start, target, cost, velocity, trajectory = simulate(
            movement, args.num_sim, args.jobs)
        plot_trajectories(traj_axes[column], start, target, trajectory)
        plot_velocities(vel_rows[column], velocity, time,
                        "Angular Velocity [deg/sec]")
        plot_cost(cost_log_axes[column], cost, log=True)
        plot_cost(cost_linear_axes[column], cost, log=False)

    share_velocity_axis(vel_rows)

    save_figure(fig, args.outdir, "LongMove.svg", dpi=200)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
