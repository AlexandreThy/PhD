"""
LQG total movement cost against two kinematic variables, per center-out direction.

    - Joint excursion: norm of the joint displacement from the starting posture
      to the target posture, sqrt(dtheta_s**2 + dtheta_e**2) [deg], and the
      shoulder |dtheta_s| and elbow |dtheta_e| excursions alone. Geometry only,
      the same for every controller.
    - Maximal lateral deviation: largest distance of the LQG hand path from the
      straight start-target line [cm], taken in each noisy trial, then averaged
      over trials.

Two figures, 15 cm / 400 ms from START, WP1's (macaque) muscle parameters:
    - 8 directions: the LQG costs of centerout_cost_polar.py
      (Cfy40_15cm_400ms_cost.npz), i.e. the points of Cfy40_15cm_400ms.svg. The
      lateral deviation of those directions comes from the simulation below
      (0, 45, ... deg are among its directions);
    - 24 directions (every 15 deg), cost and deviation both simulated here, to
      check the relations with three times as many points.

    python Final_Code/cost_vs_kinematics.py --jobs 1
    python Final_Code/cost_vs_kinematics.py --num-sim 10 --jobs 1   # quick check

Use --jobs 1 on a machine with little virtual memory: each worker process loads
its own numpy / scipy, and 15 of them can exhaust the Windows paging file.
"""

from scipy import stats

from common import (
    FIGURE_DIR, FIGURE_SUBDIRS, START, build_parser, centerout_targets,
    compute_angles_from_cartesian, cost_components, finish, np, plt, run_lqg,
    run_tasks, save_figure,
)

AMPLITUDE, DURATION, NUM_ITER = 15, 0.4, 40
NUM_DIRECTIONS = 24  # must be a multiple of 8, so the 8 directions are among them
COLOR = "#E69F00"  # LQG, COLORS[2] in common.py
TAG = f"{AMPLITUDE}cm_{int(DURATION * 1000)}ms"


def joint_displacement(target, start=START):
    """Joint displacement start -> target, (shoulder, elbow) [deg]."""
    th0 = np.array(compute_angles_from_cartesian(*start))
    th1 = np.array(compute_angles_from_cartesian(*target))
    return np.degrees(th1 - th0)


def max_lateral_deviation(X, Y, target, start=START):
    """Largest distance of the hand path from the straight start-target line [cm]."""
    direction = np.subtract(target, start)
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    path = np.stack([np.asarray(X), np.asarray(Y)], axis=-1) - np.asarray(start)
    return np.max(np.abs(path @ normal))


def _worker(target):
    """Total LQG cost and maximal lateral deviation of one noisy trial."""
    X, Y, x, u = run_lqg(DURATION, NUM_ITER, START, target, noise=True)
    return sum(cost_components(x[:, :4], u, tg=target)), max_lateral_deviation(X, Y, target)


def simulate(num_sim, jobs):
    """Mean and SEM over trials of the cost and the deviation, (variable, direction)."""
    targets = centerout_targets(START, AMPLITUDE, NUM_DIRECTIONS)
    tasks = [t for _ in range(num_sim) for t in targets]
    out = np.array(run_tasks(_worker, tasks, jobs, desc=f"LQG, {NUM_DIRECTIONS} directions"))
    out = out.reshape(num_sim, NUM_DIRECTIONS, 2).transpose(2, 0, 1)  # (variable, trial, direction)
    return out.mean(axis=1), out.std(axis=1) / np.sqrt(num_sim)


def stored_8_directions():
    """Mean and SEM of the LQG costs written by centerout_cost_polar.py."""
    saved = np.load(FIGURE_DIR / FIGURE_SUBDIRS["cost_polar"] / f"Cfy40_{TAG}_cost.npz")
    # centerout_cost_polar.py does not store its trial count; it defaults to 100
    return saved["total"][:, 2], saved["total_sd"][:, 2] / np.sqrt(100)


def scatter_panel(ax, x, x_err, cost, cost_err, xlabel, angles):
    """Cost against one variable, with the regression line. Returns (r, p, rho, p_rho)."""
    res = stats.linregress(x, cost)
    rho, p_rho = stats.spearmanr(x, cost)
    ax.errorbar(x, cost, xerr=x_err, yerr=cost_err, fmt="o", color=COLOR, ecolor="0.6",
                capsize=3, markersize=7)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, res.slope * xs + res.intercept, "--", color=COLOR,
            label=f"r$^2$ = {res.rvalue**2:.2f}")
    step = 1 if len(angles) <= 8 else 3  # label every 45 deg when crowded
    for k in range(0, len(angles), step):
        ax.annotate(f"{angles[k]:.0f}°", (x[k], cost[k]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="0.3")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("LQG total movement cost", fontsize=12)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return res.rvalue, res.pvalue, rho, p_rho


def plot(displacement, deviation, deviation_err, cost, cost_err, source, outdir):
    """displacement: (direction, 2) joint displacement [deg], shoulder then elbow."""
    n = len(cost)
    angles = np.arange(n) * 360 / n
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    axes = axes.ravel()
    fits = {
        "joint excursion": scatter_panel(axes[0], np.linalg.norm(displacement, axis=1), None,
                                         cost, cost_err,
                                         "Joint excursion $\\|\\Delta\\theta\\|$ [deg]", angles),
        "max lateral deviation": scatter_panel(axes[1], deviation, deviation_err, cost, cost_err,
                                               "Maximal lateral deviation of the LQG hand path [cm]",
                                               angles),
        "shoulder excursion": scatter_panel(axes[2], np.abs(displacement[:, 0]), None, cost, cost_err,
                                            "Shoulder excursion $|\\Delta\\theta_s|$ [deg]", angles),
        "elbow excursion": scatter_panel(axes[3], np.abs(displacement[:, 1]), None, cost, cost_err,
                                         "Elbow excursion $|\\Delta\\theta_e|$ [deg]", angles),
    }
    fig.suptitle(f"LQG movement cost against kinematics, {AMPLITUDE} cm / "
                 f"{DURATION * 1000:.0f} ms, {n} directions ({source})", fontsize=13)
    save_figure(fig, outdir, f"CostVsKinematics_{TAG}_{n}dir.svg")
    return fits


def main():
    parser = build_parser(__doc__, num_sim_default=200,
                          subdir=FIGURE_SUBDIRS["nonlinearity_index"])
    args = parser.parse_args()

    targets = centerout_targets(START, AMPLITUDE, NUM_DIRECTIONS)
    displacement = np.array([joint_displacement(t) for t in targets])  # (direction, 2)
    (cost, deviation), (cost_err, deviation_err) = simulate(args.num_sim, args.jobs)

    every = NUM_DIRECTIONS // 8  # the 8 directions 0, 45, ... deg
    cost8, cost8_err = stored_8_directions()
    fits8 = plot(displacement[::every], deviation[::every], deviation_err[::every], cost8, cost8_err,
                 "stored costs", args.outdir)
    fitsn = plot(displacement, deviation, deviation_err, cost, cost_err,
                 f"{args.num_sim} noisy trials each", args.outdir)

    np.savez(args.outdir / f"CostVsKinematics_{TAG}.npz", angles=np.arange(NUM_DIRECTIONS) * 360 / NUM_DIRECTIONS,
             displacement=displacement, excursion=np.linalg.norm(displacement, axis=1),
             deviation=deviation, deviation_sem=deviation_err,
             cost=cost, cost_sem=cost_err, cost_8_stored=cost8, cost_8_stored_sem=cost8_err)

    for label, fits in (("8 directions", fits8), (f"{NUM_DIRECTIONS} directions", fitsn)):
        print(f"\n{label}: LQG total cost against")
        for name, (r, p, rho, p_rho) in fits.items():
            print(f"  {name:22s} r = {r:+.2f} (r2 = {r**2:.2f}, p = {p:.2g})   "
                  f"Spearman rho = {rho:+.2f} (p = {p_rho:.2g})")

    finish(not args.no_show)


if __name__ == "__main__":
    main()
