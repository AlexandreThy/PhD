"""
Mean movement cost over directions against the speed of the movement.

Two x axes for the same costs (read from centerout_cost_polar.py):
  - the nominal mean speed, amplitude / duration, a task parameter;
  - the peak cartesian hand speed, simulated here: per repetition the maximum
    of sqrt(vx**2 + vy**2) over the movement, averaged over repetitions and
    then over the eight directions. Each controller has its own peak speed,
    so each trace has its own x values.
One trace per (controller, movement duration), as in centerout_cost_amplitude.py.

    python Final_Code/centerout_cost_polar.py            # writes the costs first
    python Final_Code/centerout_cost_velocity.py         # simulates peak speeds
    python Final_Code/centerout_cost_velocity.py --replot   # from saved peak speeds
    python Final_Code/centerout_cost_velocity.py --num-sim 5 --jobs 1   # quick check

Each x axis gives a linear and a logarithmic (*_log.svg) cost figure.
"""

from matplotlib.ticker import FuncFormatter

from centerout_cost_amplitude import (
    AMPLITUDES, COST_DIR, DURATIONS, LINESTYLES, MARKERS, load_costs,
)
from centerout_cost_polar import NUM_TARGETS, condition_name, num_iter_for
from common import (
    FIGURE_SUBDIRS,
    COLORS, Compute_Cartesian_Speed, LEGEND, NUM_CONTROLLERS, START,
    build_parser, centerout_targets, finish, np, plt, run_fl, run_ilqg, run_lqg,
    run_tasks, save_figure,
)

XAXES = {
    "nominal": "Movement amplitude / duration [cm/s]",
    "peak": "Peak cartesian hand speed [cm/s]",
}


def _worker(task):
    """Peak hand speed of the three controllers for one (repetition, target)."""
    target, duration, num_iter, start = task
    runs = (
        run_ilqg(duration, num_iter, start, target),
        run_fl(duration, num_iter, start, target),
        run_lqg(duration, num_iter, start, target),
    )
    return np.array([Compute_Cartesian_Speed(x.T)[2].max() for _, _, x, _ in runs])


def simulate_peak_speed(amplitude, duration, num_sim, jobs, start):
    """Mean and SD over repetitions of the peak speed, per (direction, controller)."""
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [
        (target, duration, num_iter_for(duration), start)
        for _ in range(num_sim)
        for target in targets
    ]
    results = run_tasks(
        _worker, tasks, jobs,
        desc=f"peak speed {amplitude:g} cm / {duration*1000:.0f} ms",
    )
    peak = np.array(results).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS)
    return np.mean(peak, axis=0), np.std(peak, axis=0)


def load_peak_speeds(cost_dir=COST_DIR, start=START):
    """Peak speed per direction, shape (durations, amplitudes, 8, controllers)."""
    return np.array([
        [np.load(cost_dir / f"{condition_name(amplitude, duration, start)}_peakspeed.npz")["peak"]
         for amplitude in AMPLITUDES]
        for duration in DURATIONS
    ])


def plot(costs, peak_speeds, xaxis, outdir, log=False):
    # Mean over the eight directions: (durations, amplitudes, controllers)
    mean_cost = np.mean(costs, axis=2)
    mean_peak = np.mean(peak_speeds, axis=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(NUM_CONTROLLERS):
        for d, duration in enumerate(DURATIONS):
            if xaxis == "nominal":
                speed = np.array(AMPLITUDES) / duration
            else:
                speed = mean_peak[d, :, i]
            ax.plot(speed, mean_cost[d, :, i], color=COLORS[i],
                    linestyle=LINESTYLES[duration], marker=MARKERS[duration],
                    markersize=8, linewidth=2,
                    label=f"{LEGEND[i]}, {duration * 1000:.0f} ms")

    ax.set_xlabel(XAXES[xaxis])
    ax.set_ylabel("Mean cost over directions")
    suffix = ""
    if log:
        ax.set_yscale("log")
        # Plain numbers (1, 10, 100) rather than powers of ten
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        suffix = "_log"
    else:
        ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    save_figure(fig, outdir, f"CostVsSpeed_{xaxis}{suffix}.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100,
                          subdir=FIGURE_SUBDIRS["cost_polar"])
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved *_peakspeed.npz, no simulation")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.replot:
        for duration in DURATIONS:
            for amplitude in AMPLITUDES:
                peak, peak_sd = simulate_peak_speed(amplitude, duration,
                                                    args.num_sim, args.jobs, START)
                path = args.outdir / f"{condition_name(amplitude, duration, START)}_peakspeed.npz"
                np.savez(path, peak=peak, peak_sd=peak_sd)
                print(f"wrote {path}", flush=True)

    costs = load_costs()
    peak_speeds = load_peak_speeds(args.outdir)
    for xaxis in XAXES:
        plot(costs, peak_speeds, xaxis, args.outdir)
        plot(costs, peak_speeds, xaxis, args.outdir, log=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
