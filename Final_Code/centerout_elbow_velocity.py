"""
Terminal elbow velocity of the three controllers, by reach direction.

Center-out, 15 cm from [0, 40] in 400 ms. The target is meant to be reached at
rest, so the elbow velocity left at the last timestep is itself the error; it is
plotted as a magnitude, which is also what a radial axis can show.

    python Final_Code/centerout_elbow_velocity.py
    python Final_Code/centerout_elbow_velocity.py --num-sim 5 --jobs 1
"""

from common import (
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, centerout_targets,
    finish, np, pi, plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
    style_polar_axis,
)

NUM_TARGETS = 8
AMPLITUDE = 15
DURATION = 0.4
# deg/s. The terminal elbow velocity spans about 1.6 to 13.5 deg/s over the
# three controllers, so these two rings sit at roughly a third and two thirds
# of the radius.
RADIAL_TICKS = (5, 10)


def _worker(task):
    """Terminal elbow angular velocity of the three controllers, in deg/s."""
    target, duration, num_iter, start = task
    _, _, x_ilqg, _ = run_ilqg(duration, num_iter, start, target)
    _, _, x_fl, _ = run_fl(duration, num_iter, start, target)
    _, _, x_dlqg, _ = run_dlqg(duration, num_iter, start, target)

    # Column 3 of the state is the elbow angular velocity, for all three.
    return np.array([abs(x[-1, 3]) * 180 / pi
                     for x in (x_ilqg, x_fl, x_dlqg)])


def simulate(num_sim, jobs, amplitude, duration, start):
    num_iter = int(round(duration * 100))
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [(target, duration, num_iter, start)
             for _ in range(num_sim)
             for target in targets]
    results = run_tasks(_worker, tasks, jobs,
                        desc=f"elbow velocity {amplitude:g} cm / "
                             f"{duration*1000:.0f} ms")
    # (repetition, direction, controller)
    return np.array(results).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)


def plot(mean_by_direction, outdir):
    # Close the curve by repeating the first direction at 2*pi.
    closed = np.vstack([mean_by_direction, mean_by_direction[0]])
    angles = np.linspace(0, 2 * np.pi, NUM_TARGETS + 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for i in range(NUM_CONTROLLERS):
        ax.plot(angles, closed[:, i], color=COLORS[i], linewidth=2.5,
                label=LEGEND[i])

    style_polar_axis(ax, RADIAL_TICKS, NUM_TARGETS,
                     rmax=max(max(RADIAL_TICKS), closed.max()) * 1.05)
    save_figure(fig, outdir, "elbow_vel_co.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE,
                        help="reach amplitude in cm")
    parser.add_argument("--duration", type=float, default=DURATION,
                        help="movement duration in s")
    args = parser.parse_args()

    velocity = simulate(args.num_sim, args.jobs, args.amplitude, args.duration,
                        START)
    plot(np.mean(velocity, axis=0), args.outdir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / "elbow_vel_co.npz"
    np.savez(path, velocity=velocity, legend=np.array(LEGEND))
    print(f"wrote {path}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
