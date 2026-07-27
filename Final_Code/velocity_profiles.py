"""
Joint angular velocity profiles for two reach directions.

Each panel shows the mean +/- SD shoulder (solid) and elbow (dashed) angular
velocity of the three controllers for one target direction.

    python Final_Code/velocity_profiles.py
    python Final_Code/velocity_profiles.py --num-sim 5 --jobs 1
"""

from common import (
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, delete_axis, finish,
    np, pi, plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 10
# The two directions the notebook plotted, in radians
DIRECTIONS = [pi, pi * 7 / 4]


def _worker(task):
    """Run the three controllers for one repetition. Stays top-level."""
    target, duration, num_iter, start = task
    _, _, x_ilqg, _ = run_ilqg(duration, num_iter, start, target)
    _, _, x_fl, _ = run_fl(duration, num_iter, start, target)
    _, _, x_dlqg, _ = run_dlqg(duration, num_iter, start, target)
    # Shoulder and elbow angular velocity of each controller
    return np.array([x_ilqg[:, 2:4], x_fl[:, 2:4], x_dlqg[:, 2:4]])


def simulate(angle, num_sim, jobs, start, amplitude):
    target = [start[0] + np.cos(angle) * amplitude,
              start[1] + np.sin(angle) * amplitude]
    tasks = [(target, MOVEMENT_TIME, NUM_ITER, start) for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs,
                        desc=f"velocity profile at {np.degrees(angle):.0f} deg")
    # (repetition, controller, timestep, joint)
    return np.array(results)


def plot_panel(ax, velocities, time):
    for joint in range(2):
        linestyle = "-" if joint == 0 else "--"
        for controller in range(NUM_CONTROLLERS):
            trace = velocities[:, controller, :, joint] * 180 / pi
            mean_vel = np.mean(trace, axis=0)
            std_vel = np.std(trace, axis=0)
            ax.plot(time, mean_vel, color=COLORS[controller], linestyle=linestyle,
                    linewidth=2, label=LEGEND[controller] if joint == 0 else None)
            ax.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                            color=COLORS[controller], alpha=0.3)

    ax.plot([0, MOVEMENT_TIME * 1000], [0, 0], color="grey", linestyle="--")
    ax.set_yticks([-90, 0, 90])
    ax.set_xticks([0, 200, 400])
    ax.tick_params(labelsize=16)
    delete_axis(ax)


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE,
                        help="reach amplitude in cm")
    args = parser.parse_args()

    time = np.linspace(0, MOVEMENT_TIME, NUM_ITER + 1) * 1000
    fig, axes = plt.subplots(len(DIRECTIONS), 1, figsize=(4, 8))

    for ax, angle in zip(np.atleast_1d(axes), DIRECTIONS):
        velocities = simulate(angle, args.num_sim, args.jobs, START, args.amplitude)
        plot_panel(ax, velocities, time)

    save_figure(fig, args.outdir, "Kinematiccenterout.svg")
    finish(not args.no_show)


if __name__ == "__main__":
    main()
