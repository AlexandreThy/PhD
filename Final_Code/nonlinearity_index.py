"""
Peak joint power index against the movement cost, per reach direction.

For each of eight center-out directions the joint power (torque times angular
velocity) is taken at its peak over the movement, then correlated with the total
DLQG movement cost of the same direction. Ported from
CurrentParts/NonlinearityIndex.py.

The cost it correlates against comes from centerout_cost_polar.py, so run that
first for the matching condition (15 cm, 400 ms):

    python Final_Code/centerout_cost_polar.py --amplitude 15 --duration 0.4
    python Final_Code/nonlinearity_index.py
"""

from scipy import stats

from common import (
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, centerout_targets,
    finish, np, pi, plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
)
# The muscle model of the controllers, so the torque below is the torque the
# simulation actually produced. See the note in compute_torque.
from Controllers.ILQG import MOMENT_ARM, muscle_force_scaling

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
NUM_TARGETS = 8
# Written by centerout_cost_polar.py for the 15 cm / 400 ms condition
COST_FILE = "Costr.npz"
# Column of the cost array the correlations use as the x axis
REFERENCE_CONTROLLER = 2  # DLQG


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
    _, _, x_dlqg, u_dlqg = run_dlqg(duration, num_iter, start, target)
    return np.array([
        compute_effort(x_ilqg, u_ilqg),
        compute_effort(x_fl[:, :4], u_fl),
        compute_effort(x_dlqg[:, :4], u_dlqg),
    ])


def simulate(num_sim, jobs, start, amplitude):
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [(target, MOVEMENT_TIME, NUM_ITER, start)
             for _ in range(num_sim) for target in targets]
    results = run_tasks(_worker, tasks, jobs, desc="nonlinearity index")
    # (repetition, direction, controller, timestep)
    effort = np.array(results).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS, -1)
    # peak power within each movement, then averaged over repetitions
    return np.mean(np.max(effort, axis=3), axis=0)  # (direction, controller)


def load_total_cost(outdir):
    """Total movement cost per direction, as written by centerout_cost_polar.py."""
    path = outdir / COST_FILE
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Generate it first with:\n"
            f"    python Final_Code/centerout_cost_polar.py "
            f"--amplitude 15 --duration 0.4"
        )
    return np.load(path)["my_array"]


def regress(cost_column, peak):
    result = stats.linregress(cost_column, peak)
    return result.rvalue**2, result.slope, result.intercept


def plot_polar(peak, r2, outdir, num_sim):
    """Peak power per direction, closing the curve back to the first direction."""
    angles = np.linspace(0, 2 * pi, NUM_TARGETS + 1)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    for i in range(NUM_CONTROLLERS):
        closed = np.append(peak[:, i], peak[0, i])
        ax.plot(angles, closed, color=COLORS[i], label=LEGEND[i])
        ax.scatter(angles, closed, color=COLORS[i])

    for k, i in enumerate(range(NUM_CONTROLLERS)):
        ax.text(0, 1.10 - 0.05 * k, f"{LEGEND[i]} : r2 = {r2[i]:.2f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.11, 1.1), fontsize=10)
    ax.set_title(f"Peak joint power index by direction\n"
                 f"({num_sim} trials, r2 against total "
                 f"{LEGEND[REFERENCE_CONTROLLER]} movement cost)", fontsize=13)
    save_figure(fig, outdir, "Corr_Plots_1DLQG.svg")


def plot_scatter(cost_column, peak, fits, xlabel, title, filename, outdir):
    """One panel per controller: peak power against the reference cost."""
    fig, ax = plt.subplots(NUM_CONTROLLERS, figsize=(8, 8))
    for i in range(NUM_CONTROLLERS):
        r2, slope, intercept = fits[i]
        ax[i].scatter(cost_column, peak[:, i], marker="o", color=COLORS[i],
                      linewidth=2, label=f"{LEGEND[i]} peak power")
        ax[i].plot(cost_column, slope * cost_column + intercept, color=COLORS[i],
                   linestyle="--", label=f"{LEGEND[i]} fit")
        ax[i].text(0.02, 0.90, f"{LEGEND[i]} : r2 = {r2:.2f}",
                   transform=ax[i].transAxes, fontsize=12)
        ax[i].set_xlabel(xlabel, fontsize=12)
        ax[i].set_ylabel("Peak joint power index", fontsize=12)
        ax[i].grid(True)
        ax[i].legend(fontsize=9, loc="lower right")
    ax[0].set_title(title, fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, filename)


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    args = parser.parse_args()

    total_cost = load_total_cost(args.outdir)
    peak = simulate(args.num_sim, args.jobs, START, AMPLITUDE)

    total_column = total_cost[:NUM_TARGETS, REFERENCE_CONTROLLER]
    total_fits = [regress(total_column, peak[:, i]) for i in range(NUM_CONTROLLERS)]

    plot_polar(peak, [f[0] for f in total_fits], args.outdir, args.num_sim)
    plot_scatter(total_column, peak, total_fits,
                 f"Total {LEGEND[REFERENCE_CONTROLLER]} movement cost",
                 "Peak joint power against total movement cost",
                 "Corr_Plots_2DLQG.svg", args.outdir)

    print(f"\nr2 of peak joint power against total "
          f"{LEGEND[REFERENCE_CONTROLLER]} movement cost:")
    for i in range(NUM_CONTROLLERS):
        r2, slope, _ = total_fits[i]
        print(f"  {LEGEND[i]:5s}  r2 = {r2:.3f}   slope = {slope:+.4g}")

    finish(not args.no_show)


if __name__ == "__main__":
    main()
