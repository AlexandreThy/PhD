"""
Motor cost (r * sum u**2) of the three controllers over a center-out reach.

Eight directions, 15 cm from [0, 40], 400 ms. The left panel resolves the cost
by direction, the right one averages over directions and shows the mean per
controller. `centerout_cost_polar.py` already computes this quantity for the
same condition and stores it in its .npz, but only ever plots the total cost.

Every controller is scored with the shared WR, including FL, which is
*optimised* with the much smaller WR_FL. Scoring each one under its own weight
would compare three different quantities rather than the same one.

    python Final_Code/centerout_motor_cost.py
    python Final_Code/centerout_motor_cost.py --num-sim 5 --jobs 1
"""

from common import (
    COLORS, Cost_r, LEGEND, NUM_CONTROLLERS, START, WR, build_parser,
    centerout_targets, delete_axis, finish, np, plt, run_dlqg, run_fl,
    run_ilqg, run_tasks, save_figure,
)

NUM_TARGETS = 8
AMPLITUDE = 15
DURATION = 0.4
NUM_ITER = 40


def _worker(task):
    """Motor cost of the three controllers for one (repetition, target)."""
    target, duration, num_iter, start = task
    _, _, x_ilqg, u_ilqg = run_ilqg(duration, num_iter, start, target)
    _, _, x_fl, u_fl = run_fl(duration, num_iter, start, target)
    _, _, x_dlqg, u_dlqg = run_dlqg(duration, num_iter, start, target)

    runs = ((x_ilqg, u_ilqg), (x_fl, u_fl), (x_dlqg, u_dlqg))
    return np.array([Cost_r(x, u, tg=target) for x, u in runs])


def simulate(num_sim, jobs, amplitude, duration, start):
    num_iter = int(round(duration * 100))
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [(target, duration, num_iter, start)
             for _ in range(num_sim)
             for target in targets]
    results = run_tasks(_worker, tasks, jobs,
                        desc=f"motor cost {amplitude:g} cm / {duration*1000:.0f} ms")
    # (repetition, direction, controller)
    return np.array(results).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)


def plot_polar(ax, motor):
    """Mean motor cost against reach direction, closed back to the first one."""
    mean_by_direction = np.mean(motor, axis=0)
    closed = np.vstack([mean_by_direction, mean_by_direction[0]])
    angles = np.linspace(0, 2 * np.pi, NUM_TARGETS + 1)

    for i in range(NUM_CONTROLLERS):
        ax.scatter(angles, closed[:, i], color=COLORS[i], s=20, zorder=10)
        ax.plot(angles, closed[:, i], color=COLORS[i], linewidth=2,
                linestyle="--", label=LEGEND[i])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=11)
    ax.set_title("Motor cost by reach direction", fontsize=14, pad=18)
    ax.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.15, -0.05),
              frameon=False)


def plot_summary(ax, motor):
    """Mean over directions and repetitions, with the SEM over repetitions."""
    per_repetition = np.mean(motor, axis=1)
    means = np.mean(per_repetition, axis=0)
    errors = np.std(per_repetition, axis=0) / np.sqrt(per_repetition.shape[0])

    ax.bar(np.arange(NUM_CONTROLLERS), means, 0.6, yerr=errors, color=COLORS,
           error_kw=dict(ecolor="black", lw=1.5))
    ax.set_xticks(np.arange(NUM_CONTROLLERS), labels=LEGEND, fontsize=13)
    ax.set_ylabel(f"Motor cost   ({WR:g} * sum u^2)", fontsize=14)
    ax.set_title("Averaged over directions", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    delete_axis(ax, sides=["top", "right"])


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE,
                        help="reach amplitude in cm")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help="movement duration in s")
    args = parser.parse_args()

    motor = simulate(args.num_sim, args.jobs, args.amplitude, args.duration, START)

    fig = plt.figure(figsize=(12, 5.5), layout="constrained")
    plot_polar(fig.add_subplot(1, 2, 1, projection="polar"), motor)
    plot_summary(fig.add_subplot(1, 2, 2), motor)
    fig.suptitle(f"Motor cost, center-out {args.amplitude:g} cm from "
                 f"[{START[0]:g}, {START[1]:g}] in {args.duration*1000:.0f} ms "
                 f"({motor.shape[0]} trials)", fontsize=15)

    name = (f"MotorCost_{int(args.amplitude)}cm_"
            f"{int(args.duration * 1000)}ms")
    save_figure(fig, args.outdir, f"{name}.svg")

    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f"{name}.npz"
    np.savez(path, motor=motor, legend=np.array(LEGEND))
    print(f"wrote {path}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
