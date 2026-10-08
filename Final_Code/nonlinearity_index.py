"""
Peak joint power index against the movement cost, per reach direction.

For each of eight center-out directions the joint power (torque times angular
velocity) is taken at its peak over the movement, then correlated with the LQG
movement cost of the same direction: once with the total cost, once with the
motor cost alone. Ported from CurrentParts/NonlinearityIndex.py.

The costs it correlates against come from centerout_cost_polar.py, so run that
first for the matching conditions (10 cm / 400 ms and 15 cm / 600 ms):

    python Final_Code/centerout_cost_polar.py
    python Final_Code/nonlinearity_index.py                          # both
    python Final_Code/nonlinearity_index.py --amplitude 10 --duration 0.4
"""

from scipy import stats

from centerout_cost_polar import condition_name, num_iter_for
from common import (
    FIGURE_DIR, FIGURE_SUBDIRS,
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, centerout_targets,
    finish, np, pi, plt, run_lqg, run_fl, run_ilqg, run_tasks, save_figure,
)
# The muscle model of the controllers, so the torque below is the torque the
# simulation actually produced. See the note in compute_torque.
from Controllers.ILQG import MOMENT_ARM, muscle_force_scaling

NUM_TARGETS = 8
# (amplitude [cm], duration [s]); the costs are read from the
# <condition>_cost.npz files written by centerout_cost_polar.py
CONDITIONS = [(10, 0.4), (15, 0.6)]
# Column of the cost array the correlations use as the x axis
REFERENCE_CONTROLLER = 2  # LQG


def compute_torque(x, u):
    """
    Joint torque produced by the muscle commands u at state x.

    The force-length and force-velocity gains come from the controllers rather
    than being written out again here, so that this torque matches the dynamics
    that generated the trajectory. The original script inlined an older
    force-length curve, exp(+|(l**1.55-1)/0.81|), which rises as the muscle
    leaves its optimal length instead of falling, and no longer matched the
    model the controllers integrate.
    """
    fl, ff_v = muscle_force_scaling(x)[3:]
    return MOMENT_ARM @ (u * fl * ff_v)


def compute_effort(x, u):
    """Joint power, torque . angular velocity, at each timestep."""
    N = np.zeros(x.shape[0] - 1)
    for i in range(x.shape[0] - 1):
        torque = compute_torque(x[i], u[i])
        N[i] = torque[0] * x[i, 2] + torque[1] * x[i, 3]
    return N


def _worker(task):
    """Joint power over time for the three controllers, one repetition."""
    target, duration, num_iter, start = task
    _, _, x_ilqg, u_ilqg = run_ilqg(duration, num_iter, start, target)
    _, _, x_fl, u_fl = run_fl(duration, num_iter, start, target)
    _, _, x_lqg, u_lqg = run_lqg(duration, num_iter, start, target)
    return np.array([
        compute_effort(x_ilqg, u_ilqg),
        compute_effort(x_fl[:, :4], u_fl),
        compute_effort(x_lqg[:, :4], u_lqg),
    ])


def simulate(num_sim, jobs, start, amplitude, duration):
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [(target, duration, num_iter_for(duration), start)
             for _ in range(num_sim) for target in targets]
    results = run_tasks(_worker, tasks, jobs,
                        desc=f"nonlinearity index {amplitude:g} cm / {duration:g} s")
    # (repetition, direction, controller, timestep)
    effort = np.array(results).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS, -1)
    # peak power within each movement, then averaged over repetitions
    return np.mean(np.max(effort, axis=3), axis=0)  # (direction, controller)


def load_costs(outdir, name, amplitude, duration):
    """Total and motor cost per direction, as written by centerout_cost_polar.py."""
    path = outdir / f"{name}_cost.npz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Generate it first with:\n"
            f"    python Final_Code/centerout_cost_polar.py "
            f"--amplitude {amplitude:g} --duration {duration:g}"
        )
    saved = np.load(path)
    return saved["total"], saved["motor"]


def regress(cost_column, peak):
    result = stats.linregress(cost_column, peak)
    return result.rvalue**2, result.slope, result.intercept


def plot_polar(peak, r2_total, r2_motor, outdir, num_sim, tag):
    """Peak power per direction, closing the curve back to the first direction."""
    angles = np.linspace(0, 2 * pi, NUM_TARGETS + 1)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    for i in range(NUM_CONTROLLERS):
        closed = np.append(peak[:, i], peak[0, i])
        ax.plot(angles, closed, color=COLORS[i], label=LEGEND[i])
        ax.scatter(angles, closed, color=COLORS[i])

    for k, i in enumerate(range(NUM_CONTROLLERS)):
        ax.text(0.5, 1.12 - 0.04 * k,
                f"{LEGEND[i]} : r2 total = {r2_total[i]:.2f}, "
                f"r2 motor = {r2_motor[i]:.2f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.11, 1.1), fontsize=10)
    ax.set_title(f"Peak joint power index by direction, {tag}\n"
                 f"({num_sim} trials, r2 against "
                 f"{LEGEND[REFERENCE_CONTROLLER]} cost)", fontsize=13, pad=70)
    save_figure(fig, outdir, f"PeakPower_polar_{tag}.svg")


def plot_scatter(cost_column, peak, fits, xlabel, title, filename, outdir):
    """
    Peak power against the reference cost, the three controllers on one axes:
    one colour each for the points and their regression line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    order = np.argsort(cost_column)
    for i in range(NUM_CONTROLLERS):
        r2, slope, intercept = fits[i]
        ax.scatter(cost_column, peak[:, i], marker="o", color=COLORS[i], s=40)
        ax.plot(cost_column[order], slope * cost_column[order] + intercept,
                color=COLORS[i], linestyle="--", linewidth=2,
                label=f"{LEGEND[i]}  (r2 = {r2:.2f})")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Peak joint power index", fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=11, loc="best")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, filename)


def main():
    parser = build_parser(__doc__, num_sim_default=100,
                          subdir=FIGURE_SUBDIRS["nonlinearity_index"])
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

    ref = LEGEND[REFERENCE_CONTROLLER]
    for amplitude, duration in conditions:
        name = condition_name(amplitude, duration, START)
        tag = f"{int(amplitude)}cm_{int(duration * 1000)}ms"
        total_cost, motor_cost = load_costs(FIGURE_DIR / FIGURE_SUBDIRS["cost_polar"], name,
                                           amplitude, duration)
        peak = simulate(args.num_sim, args.jobs, START, amplitude, duration)

        total_column = total_cost[:NUM_TARGETS, REFERENCE_CONTROLLER]
        motor_column = motor_cost[:NUM_TARGETS, REFERENCE_CONTROLLER]
        total_fits = [regress(total_column, peak[:, i]) for i in range(NUM_CONTROLLERS)]
        motor_fits = [regress(motor_column, peak[:, i]) for i in range(NUM_CONTROLLERS)]

        plot_polar(peak, [f[0] for f in total_fits], [f[0] for f in motor_fits],
                   args.outdir, args.num_sim, tag)
        plot_scatter(total_column, peak, total_fits,
                     f"Total {ref} movement cost",
                     f"Peak joint power against total movement cost, {tag}",
                     f"PeakPower_vs_total_{tag}.svg", args.outdir)
        plot_scatter(motor_column, peak, motor_fits,
                     f"{ref} motor cost",
                     f"Peak joint power against motor cost, {tag}",
                     f"PeakPower_vs_motor_{tag}.svg", args.outdir)
        np.savez(args.outdir / f"PeakPower_{tag}.npz", peak=peak,
                 total=total_column, motor=motor_column,
                 legend=np.array(LEGEND))

        print(f"\n{tag}: r2 of peak joint power against {ref} cost")
        for i in range(NUM_CONTROLLERS):
            print(f"  {LEGEND[i]:5s}  total r2 = {total_fits[i][0]:.3f} "
                  f"(slope {total_fits[i][1]:+.4g})   "
                  f"motor r2 = {motor_fits[i][0]:.3f} "
                  f"(slope {motor_fits[i][1]:+.4g})")

    finish(not args.no_show)


if __name__ == "__main__":
    main()
