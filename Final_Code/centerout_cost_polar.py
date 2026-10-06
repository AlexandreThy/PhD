"""
Movement cost of the three controllers as a function of reach direction.

Draws the polar cost plot for a center-out task and writes the averaged costs
next to the figure. One parameterised script replaces the six notebook cells
that differed only in reach amplitude and movement duration.

    python Final_Code/centerout_cost_polar.py                       # all conditions
    python Final_Code/centerout_cost_polar.py --amplitude 15 --duration 0.4
    python Final_Code/centerout_cost_polar.py --num-sim 5 --jobs 1   # quick check
    python Final_Code/centerout_cost_polar.py --replot   # redraw from saved costs
    python Final_Code/centerout_cost_polar.py --amplitude 15 --duration 0.4 --r1 1e-4
    python Final_Code/centerout_cost_polar.py --amplitude 10 --duration 0.4 --start-y 35

Each condition gives two figures: all three controllers, and ILQG and FL alone
(*_ILQG_FL.svg) on a scale fitted to those two.
"""

from matplotlib.ticker import MaxNLocator

from common import (
    FIGURE_SUBDIRS,
    COLORS, Cost_function, Cost_r, LEGEND, NUM_CONTROLLERS, START, WR,
    build_parser, centerout_targets, finish, np, plt, run_lqg, run_fl,
    run_ilqg, run_tasks, save_figure, style_polar_axis,
)

NUM_TARGETS = 8
# (amplitude [cm], duration [s]) reproduced from the notebook
CONDITIONS = [(15, 0.4), (15, 0.6), (10, 0.4), (10, 0.6), (20, 0.4), (20, 0.6)]
# The only two radial references drawn, on every condition.
RADIAL_TICKS = (2, 4)
YLIM = {(15, 0.6): (0, 5)}
# Controllers of the second figure of each condition, drawn on its own scale.
WITHOUT_LQG = (0, 1)  # ILQG, FL


def num_iter_for(duration):
    """The notebook used 40 steps for 400 ms and 60 for 600 ms."""
    return int(round(duration * 100))


def _worker(task):
    """Run the three controllers for one (repetition, target). Stays top-level."""
    target, duration, num_iter, start, r1 = task
    _, _, x_ilqg, u_ilqg = run_ilqg(duration, num_iter, start, target, wr=r1)
    _, _, x_fl, u_fl = run_fl(duration, num_iter, start, target)
    _, _, x_lqg, u_lqg = run_lqg(duration, num_iter, start, target, wr=r1)

    runs = ((x_ilqg, u_ilqg), (x_fl, u_fl), (x_lqg, u_lqg))
    total = np.array([Cost_function(x, u, r=r1, tg=target) for x, u in runs])
    motor = np.array([Cost_r(x, u, r=r1, tg=target) for x, u in runs])
    return total, motor


def simulate(amplitude, duration, num_sim, jobs, start, r1=WR):
    num_iter = num_iter_for(duration)
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [
        (target, duration, num_iter, start, r1)
        for _ in range(num_sim)
        for target in targets
    ]
    results = run_tasks(
        _worker, tasks, jobs,
        desc=f"cost polar {amplitude:g} cm / {duration*1000:.0f} ms",
    )

    total = np.array([r[0] for r in results]).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)
    motor = np.array([r[1] for r in results]).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)
    # Mean and standard deviation over repetitions, per (direction, controller)
    return (np.mean(total, axis=0), np.mean(motor, axis=0),
            np.std(total, axis=0), np.std(motor, axis=0))


def condition_name(amplitude, duration, start, r1=WR):
    name = f"Cfy{int(start[1])}_{int(amplitude)}cm_{int(duration * 1000)}ms"
    # The default motor cost keeps the historical file names.
    return name if r1 == WR else f"{name}_r1_{r1:g}"


def auto_radial_ticks(rmax, count=2):
    """`count` round radial ticks inside (0, rmax] for a plot without LQG."""
    ticks = MaxNLocator(nbins=count + 1, steps=[1, 2, 2.5, 5, 10]).tick_values(0, rmax)
    ticks = [t for t in ticks if 0 < t <= rmax]
    return tuple(ticks[-count:])


def plot(amplitude, duration, mean_total, sd_total, outdir, start,
         controllers=tuple(range(NUM_CONTROLLERS)), suffix="", r1=WR):
    """
    Mean cost per direction with a +/- one SD band, for the given controllers.

    With all three controllers the scale and ticks are the fixed ones shared by
    every condition; with a subset they are fitted to the curves drawn, so the
    ILQG/FL comparison is not flattened by the much larger LQG cost.
    """
    # Close the polar curves by repeating the first direction at 2*pi.
    closed = np.vstack([mean_total, mean_total[0]])
    closed_sd = np.vstack([sd_total, sd_total[0]])
    lower = np.clip(closed - closed_sd, 0, None)  # a cost cannot be negative
    upper = closed + closed_sd
    angles = np.linspace(0, 2 * np.pi, NUM_TARGETS + 1)
    controllers = list(controllers)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for i in controllers:
        # Shaded band: mean +/- one standard deviation over repetitions
        ax.fill_between(angles, lower[:, i], upper[:, i], color=COLORS[i],
                        alpha=0.2, linewidth=0)
        ax.plot(angles, closed[:, i], color=COLORS[i], linewidth=2.5,
                label=LEGEND[i])

    if len(controllers) == NUM_CONTROLLERS:
        ticks = RADIAL_TICKS
        rmax = YLIM.get((amplitude, duration),
                        (0, max(max(RADIAL_TICKS), upper.max()) * 1.05))[1]
    else:
        rmax = upper[:, controllers].max() * 1.05
        ticks = auto_radial_ticks(rmax)
    style_polar_axis(ax, ticks, NUM_TARGETS, rmax)

    name = condition_name(amplitude, duration, start, r1)
    save_figure(fig, outdir, f"{name}{suffix}.svg")
    return name


def plot_all(amplitude, duration, mean_total, sd_total, outdir, start, r1=WR):
    """The three-controller figure and the ILQG/FL-only one."""
    name = plot(amplitude, duration, mean_total, sd_total, outdir, start, r1=r1)
    plot(amplitude, duration, mean_total, sd_total, outdir, start,
         controllers=WITHOUT_LQG, suffix="_ILQG_FL", r1=r1)
    return name


def main():
    parser = build_parser(__doc__, num_sim_default=100,
                          subdir=FIGURE_SUBDIRS["cost_polar"])
    parser.add_argument("--amplitude", type=float, default=None,
                        help="reach amplitude in cm (default: all conditions)")
    parser.add_argument("--duration", type=float, default=None,
                        help="movement duration in s (default: all conditions)")
    parser.add_argument("--r1", type=float, default=WR,
                        help="motor cost of ILQG and LQG, also used to score "
                             "all three controllers (FL keeps WR_FL); default WR")
    parser.add_argument("--start-y", type=float, default=START[1],
                        help="starting hand height in cm (default: START); "
                             "names the figures Cfy<start-y>_*")
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved *_cost.npz, no simulation")
    args = parser.parse_args()

    if args.amplitude is None and args.duration is None:
        conditions = CONDITIONS
    elif args.amplitude is not None and args.duration is not None:
        conditions = [(args.amplitude, args.duration)]
    else:
        parser.error("give both --amplitude and --duration, or neither")

    start = [START[0], args.start_y]
    args.outdir.mkdir(parents=True, exist_ok=True)
    for amplitude, duration in conditions:
        if args.replot:
            saved = np.load(args.outdir / f"{condition_name(amplitude, duration, start, args.r1)}_cost.npz")
            plot_all(amplitude, duration, saved["total"], saved["total_sd"],
                     args.outdir, start, args.r1)
            continue
        mean_total, mean_motor, sd_total, sd_motor = simulate(
            amplitude, duration, args.num_sim, args.jobs, start, args.r1)
        name = plot_all(amplitude, duration, mean_total, sd_total, args.outdir,
                        start, args.r1)
        np.savez(args.outdir / f"{name}_cost.npz",
                 total=mean_total, motor=mean_motor,
                 total_sd=sd_total, motor_sd=sd_motor)
        print(f"wrote {args.outdir / (name + '_cost.npz')}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
