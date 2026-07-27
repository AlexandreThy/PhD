"""
Large amplitude (55 cm) reaching for the two long movements.

For each of the two long movements, plots the trajectories of the three
controllers, their joint angular velocity profiles and the distribution of the
movement cost.

    python Final_Code/large_amplitude_reaching.py
    python Final_Code/large_amplitude_reaching.py --num-sim 5 --jobs 1
"""

from matplotlib import gridspec

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


def plot_row(traj_axes, vel_axes, start, target, velocity, trajectory, time,
             label_y):
    shown = min(trajectory.shape[0], MAX_TRAJECTORIES_SHOWN)
    for i in range(NUM_CONTROLLERS):
        for rep in range(shown):
            traj_axes[i].plot(trajectory[rep, i, 0], trajectory[rep, i, 1],
                              color=COLORS[i], linewidth=0.5)
        _mark_endpoints(traj_axes[i], start, target)

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
        ax.set_ylim(-270, 320)
        ticks = [-270, -180, -90, 0, 90, 180, 270]
        # Only the leftmost panel of each row carries the tick labels
        ax.set_yticks(ticks, labels=ticks if i == 0 else [""] * len(ticks))
        ax.tick_params(labelsize=20)
        ax.grid(axis="y")

    vel_axes[0].set_ylabel(label_y, fontsize=22)


def plot_cost(ax, cost):
    ax.set_yscale("log")
    box = ax.boxplot(cost, patch_artist=True, medianprops=dict(color="black"),
                     whiskerprops=dict(color="black"),
                     capprops=dict(color="black"), showfliers=False)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
    ax.set_xticks([])
    delete_axis(ax)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel("Movement Cost", fontsize=22)


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--movements", type=int, nargs="+", choices=(1, 2),
                        default=[2, 1],
                        help="which long movements to simulate (row order)")
    args = parser.parse_args()

    fig = plt.figure(figsize=(22, 40))
    gs = gridspec.GridSpec(5, 6)
    traj_rows = [[fig.add_subplot(gs[row, 2 * i:2 * i + 2]) for i in range(3)]
                 for row in (0, 1)]
    vel_rows = [[fig.add_subplot(gs[row, 2 * i:2 * i + 2]) for i in range(3)]
                for row in (2, 3)]
    cost_axes = [fig.add_subplot(gs[4, :3]), fig.add_subplot(gs[4, 3:])]

    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)

    for row, number in enumerate(args.movements):
        movement = MOVEMENT_BY_NUMBER[number]
        start, target, cost, velocity, trajectory = simulate(
            movement, args.num_sim, args.jobs)
        plot_row(traj_rows[row], vel_rows[row], start, target, velocity,
                 trajectory, time, "Angular Velocity [deg/sec]")
        plot_cost(cost_axes[row], cost)

    save_figure(fig, args.outdir, "LongMove.svg", dpi=200)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
