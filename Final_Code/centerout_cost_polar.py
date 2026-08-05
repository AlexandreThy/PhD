"""
Movement cost of the three controllers as a function of reach direction.

Draws the polar cost plot for a center-out task and writes the averaged costs
next to the figure. One parameterised script replaces the six notebook cells
that differed only in reach amplitude and movement duration.

    python Final_Code/centerout_cost_polar.py                       # all conditions
    python Final_Code/centerout_cost_polar.py --amplitude 15 --duration 0.4
    python Final_Code/centerout_cost_polar.py --num-sim 5 --jobs 1   # quick check
"""

from common import (
    COLORS, Cost_function, Cost_r, LEGEND, NUM_CONTROLLERS, START,
    build_parser, centerout_targets, finish, np, plt, run_dlqg, run_fl,
    run_ilqg, run_tasks, save_figure, style_polar_axis,
)

NUM_TARGETS = 8
# (amplitude [cm], duration [s]) reproduced from the notebook
CONDITIONS = [(15, 0.4), (15, 0.6), (10, 0.4), (10, 0.6), (20, 0.4), (20, 0.6)]
# The only two radial references drawn, on every condition.
RADIAL_TICKS = (2, 4)
YLIM = {(15, 0.6): (0, 5)}


def num_iter_for(duration):
    """The notebook used 40 steps for 400 ms and 60 for 600 ms."""
    return int(round(duration * 100))


def _worker(task):
    """Run the three controllers for one (repetition, target). Stays top-level."""
    target, duration, num_iter, start = task
    _, _, x_ilqg, u_ilqg = run_ilqg(duration, num_iter, start, target)
    _, _, x_fl, u_fl = run_fl(duration, num_iter, start, target)
    _, _, x_dlqg, u_dlqg = run_dlqg(duration, num_iter, start, target)

    runs = ((x_ilqg, u_ilqg), (x_fl, u_fl), (x_dlqg, u_dlqg))
    total = np.array([Cost_function(x, u, tg=target) for x, u in runs])
    motor = np.array([Cost_r(x, u, tg=target) for x, u in runs])
    return total, motor


def simulate(amplitude, duration, num_sim, jobs, start):
    num_iter = num_iter_for(duration)
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [
        (target, duration, num_iter, start)
        for _ in range(num_sim)
        for target in targets
    ]
    results = run_tasks(
        _worker, tasks, jobs,
        desc=f"cost polar {amplitude:g} cm / {duration*1000:.0f} ms",
    )

    total = np.array([r[0] for r in results]).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)
    motor = np.array([r[1] for r in results]).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)
    return np.mean(total, axis=0), np.mean(motor, axis=0)


def plot(amplitude, duration, mean_total, outdir, start):
    # Close the polar curve by repeating the first direction at 2*pi.
    closed = np.vstack([mean_total, mean_total[0]])
    angles = np.linspace(0, 2 * np.pi, NUM_TARGETS + 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for i in range(NUM_CONTROLLERS):
        ax.plot(angles, closed[:, i], color=COLORS[i], linewidth=2.5,
                label=LEGEND[i])

    rmax = YLIM.get((amplitude, duration),
                    (0, max(max(RADIAL_TICKS), closed.max()) * 1.05))[1]
    style_polar_axis(ax, RADIAL_TICKS, NUM_TARGETS, rmax)

    name = f"Cfy{int(start[1])}_{int(amplitude)}cm_{int(duration * 1000)}ms"
    save_figure(fig, outdir, f"{name}.svg")
    return name


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--amplitude", type=float, default=None,
                        help="reach amplitude in cm (default: all conditions)")
    parser.add_argument("--duration", type=float, default=None,
                        help="movement duration in s (default: all conditions)")
    args = parser.parse_args()

    if args.amplitude is None and args.duration is None:
        conditions = CONDITIONS
    elif args.amplitude is not None and args.duration is not None:
        conditions = [(args.amplitude, args.duration)]
    else:
        parser.error("give both --amplitude and --duration, or neither")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for amplitude, duration in conditions:
        mean_total, mean_motor = simulate(amplitude, duration, args.num_sim,
                                          args.jobs, START)
        name = plot(amplitude, duration, mean_total, args.outdir, START)
        np.savez(args.outdir / f"{name}_cost.npz",
                 total=mean_total, motor=mean_motor)
        print(f"wrote {args.outdir / (name + '_cost.npz')}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
