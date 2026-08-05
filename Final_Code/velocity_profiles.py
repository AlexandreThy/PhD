"""
Kinematics and cost breakdown for two reach directions.

One row per target direction: the mean +/- SD shoulder (solid) and elbow
(dashed) angular velocity of the three controllers, the cartesian hand speed
sqrt(vx**2 + vy**2), and the three terms of the movement cost as grouped bars.

    python Final_Code/velocity_profiles.py
    python Final_Code/velocity_profiles.py --num-sim 5 --jobs 1
"""

from matplotlib.lines import Line2D

from common import (
    COLORS, Compute_Cartesian_Speed, LEGEND, NUM_CONTROLLERS, START, WP, WR, WV,
    build_parser, compute_angles_from_cartesian, delete_axis, finish, np, pi,
    plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
# The two directions the notebook plotted, in radians
DIRECTIONS = [0, pi/2]
COST_TERMS = ("target\naccuracy", "terminal\nvelocity", "motor")


def cost_components(x, u, target):
    """
    The three terms of common.Cost_function, kept apart.

    Every controller is scored with the shared WR, including FL, which is
    *optimised* with the smaller WR_FL. Scoring each one under its own weights
    would compare three different quantities.
    """
    target1, target2 = compute_angles_from_cartesian(target[0], target[1])
    thetas, thetae, omegas, omegae = x[-1, :4]
    return (
        WP * ((thetas - target1) ** 2 + (thetae - target2) ** 2),
        WV * (omegas**2 + omegae**2),
        np.sum(u * u) * WR,
    )


def _worker(task):
    """Run the three controllers for one repetition. Stays top-level."""
    target, duration, num_iter, start = task
    runs = (
        run_ilqg(duration, num_iter, start, target),
        run_fl(duration, num_iter, start, target),
        run_dlqg(duration, num_iter, start, target),
    )
    return {
        # Shoulder and elbow angular velocity of each controller
        "joint_vel": np.array([x[:, 2:4] for _, _, x, _ in runs]),
        "speed": np.array([Compute_Cartesian_Speed(x.T)[2] for _, _, x, _ in runs]),
        "cost": np.array([cost_components(x, u, target) for _, _, x, u in runs]),
    }


def simulate(angle, num_sim, jobs, start, amplitude):
    target = [start[0] + np.cos(angle) * amplitude,
              start[1] + np.sin(angle) * amplitude]
    tasks = [(target, MOVEMENT_TIME, NUM_ITER, start) for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs,
                        desc=f"velocity profile at {np.degrees(angle):.0f} deg")
    # Each entry is (repetition, controller, ...)
    return {key: np.array([r[key] for r in results]) for key in results[0]}


def _mean_band(ax, time, trace, color, **kwargs):
    """Mean over repetitions with a +/- SD band."""
    mean, std = np.mean(trace, axis=0), np.std(trace, axis=0)
    ax.plot(time, mean, color=color, linewidth=2, **kwargs)
    ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.3)


def plot_joint_panel(ax, velocities, time):
    for joint in range(2):
        linestyle = "-" if joint == 0 else "--"
        for controller in range(NUM_CONTROLLERS):
            _mean_band(ax, time, velocities[:, controller, :, joint] * 180 / pi,
                       COLORS[controller], linestyle=linestyle,
                       label=LEGEND[controller] if joint == 0 else None)

    ax.plot([0, MOVEMENT_TIME * 1000], [0, 0], color="grey", linestyle="--")
    ax.set_yticks([-90, 0, 90])
    ax.set_xticks([0, 200, 400])
    ax.tick_params(labelsize=16)
    delete_axis(ax)


def plot_speed_panel(ax, speeds, time):
    for controller in range(NUM_CONTROLLERS):
        _mean_band(ax, time, speeds[:, controller], COLORS[controller])

    ax.set_xticks([0, 200, 400])
    ax.tick_params(labelsize=16)
    delete_axis(ax)


def plot_cost_panel(ax, costs):
    """One group of bars per cost term, one bar per controller."""
    means = np.mean(costs, axis=0)
    # SEM, not SD: the bar is a mean, so this is the error on the mean. On the
    # log scale an SD wider than the mean would run off the bottom of the axis.
    errors = np.std(costs, axis=0) / np.sqrt(costs.shape[0])

    width = 0.8 / NUM_CONTROLLERS
    positions = np.arange(len(COST_TERMS))
    for controller in range(NUM_CONTROLLERS):
        offset = (controller - (NUM_CONTROLLERS - 1) / 2) * width
        ax.bar(positions + offset, means[controller], width,
               yerr=errors[controller], color=COLORS[controller],
               error_kw=dict(ecolor="black", lw=1))

    ax.set_yscale("log")
    ax.set_xticks(positions, labels=COST_TERMS)
    ax.tick_params(labelsize=13)
    delete_axis(ax, sides=["top", "right"])


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE,
                        help="reach amplitude in cm")
    args = parser.parse_args()

    time = np.linspace(0, MOVEMENT_TIME, NUM_ITER + 1) * 1000
    fig, axes = plt.subplots(len(DIRECTIONS), 3, figsize=(12, 7),
                             layout="constrained")

    for row, angle in enumerate(DIRECTIONS):
        data = simulate(angle, args.num_sim, args.jobs, START, args.amplitude)
        plot_joint_panel(axes[row, 0], data["joint_vel"], time)
        plot_speed_panel(axes[row, 1], data["speed"], time)
        plot_cost_panel(axes[row, 2], data["cost"])

        axes[row, 0].set_ylabel(f"{np.degrees(angle):.0f} deg reach", fontsize=15)

    # The hand speed and the cost share a scale down each column, otherwise the
    # two directions are drawn to different scales and cannot be compared.
    for col in (1, 2):
        limits = [ax.get_ylim() for ax in axes[:, col]]
        for ax in axes[:, col]:
            ax.set_ylim(min(lo for lo, _ in limits), max(hi for _, hi in limits))

    axes[0, 0].set_title("Joint angular velocity [deg/s]", fontsize=14)
    axes[0, 1].set_title("Hand speed [cm/s]", fontsize=14)
    axes[0, 2].set_title("Movement cost by term", fontsize=14)
    for col in range(2):
        axes[-1, col].set_xlabel("Time [ms]", fontsize=14)

    axes[0, 0].legend(handles=[
        *(Line2D([], [], color=c, lw=3, label=n) for c, n in zip(COLORS, LEGEND)),
        Line2D([], [], color="black", ls="-", label="shoulder"),
        Line2D([], [], color="black", ls="--", label="elbow"),
    ], fontsize=11, loc="upper right", frameon=False)

    save_figure(fig, args.outdir, "Kinematiccenterout.svg")
    finish(not args.no_show)


if __name__ == "__main__":
    main()
