"""
LQG movement cost, and LQG / ILQG cost ratio, against kinematic variables, per
center-out direction.

    - Joint excursion: norm of the joint displacement from the starting posture
      to the target posture, sqrt(dtheta_s**2 + dtheta_e**2) [deg], and the
      shoulder |dtheta_s| and elbow |dtheta_e| excursions alone. Geometry only,
      the same for every controller.
    - Maximal lateral deviation: largest distance of the LQG hand path from the
      straight start-target line [cm], taken in each noisy trial, then averaged
      over trials.

Figures, 15 cm / 400 ms from START, WP1's (macaque) muscle parameters:
    - 8 directions: the costs of centerout_cost_polar.py
      (Cfy40_15cm_400ms_cost.npz), i.e. the points of Cfy40_15cm_400ms.svg. The
      lateral deviation of those directions comes from the simulation below
      (0, 45, ... deg are among its directions);
    - 24 directions (every 15 deg), costs and deviation all simulated here, to
      check the relations with three times as many points.
Each is drawn twice: for the LQG total cost (CostVsKinematics_*) and for the
LQG / ILQG ratio, the mean LQG cost over the mean ILQG cost of the same
direction (CostRatioVsKinematics_*). CostRatio_polar_* shows that ratio per
direction, the 24 simulated directions and the 8 stored ones.

    python Final_Code/cost_vs_kinematics.py --jobs 4
    python Final_Code/cost_vs_kinematics.py --num-sim 10 --jobs 1   # quick check
    python Final_Code/cost_vs_kinematics.py --replot   # redraw from the saved .npz

ILQG takes about 0.45 s per trial, so 24 directions x 200 trials is about 35 min
on one process. Keep --jobs low on a machine with little virtual memory: each
worker process loads its own numpy / scipy, and 15 of them exhausted the Windows
paging file.
"""

import warnings

from scipy import stats

from common import (
    FIGURE_DIR, FIGURE_SUBDIRS, START, build_parser, centerout_targets,
    compute_angles_from_cartesian, cost_components, finish, np, pi, plt, run_ilqg,
    run_lqg, run_tasks, save_figure, style_polar_axis,
)

AMPLITUDE, DURATION, NUM_ITER = 15, 0.4, 40
NUM_DIRECTIONS = 24  # must be a multiple of 8, so the 8 directions are among them
COLOR = "#E69F00"  # LQG, COLORS[2] in common.py
RATIO_COLOR = "#CC3311"
TAG = f"{AMPLITUDE}cm_{int(DURATION * 1000)}ms"
COST_LABEL = "LQG movement cost"
# Font sizes [pt], large enough for the panels to go into a paper figure as is
LABEL_SIZE = 20
TICK_SIZE = 17
ANGLE_SIZE = 16
LEGEND_SIZE = 17
TITLE_SIZE = 20
RATIO_LABEL = "LQG cost / mean ILQG cost"


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
    """LQG cost, LQG maximal lateral deviation and ILQG cost of one noisy trial."""
    X, Y, x, u = run_lqg(DURATION, NUM_ITER, START, target, noise=True)
    lqg_cost = sum(cost_components(x[:, :4], u, tg=target))
    deviation = max_lateral_deviation(X, Y, target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ComplexWarning from the ILQG backward pass
        _, _, x, u = run_ilqg(DURATION, NUM_ITER, START, target, noise=True)
    return lqg_cost, deviation, sum(cost_components(x[:, :4], u, tg=target))


def simulate(num_sim, jobs):
    """Per-trial LQG cost, LQG deviation and ILQG cost, (variable, trial, direction)."""
    targets = centerout_targets(START, AMPLITUDE, NUM_DIRECTIONS)
    tasks = [t for _ in range(num_sim) for t in targets]
    out = np.array(run_tasks(_worker, tasks, jobs,
                             desc=f"LQG + ILQG, {NUM_DIRECTIONS} directions"))
    return out.reshape(num_sim, NUM_DIRECTIONS, 3).transpose(2, 0, 1)


def mean_sem(trials):
    """Mean and SEM over trials (axis 0)."""
    return trials.mean(axis=0), trials.std(axis=0) / np.sqrt(trials.shape[0])


def ratio_sem(num, num_sem, den, den_sem):
    """Ratio of two means and its SEM (first-order error propagation)."""
    ratio = num / den
    return ratio, ratio * np.sqrt((num_sem / num) ** 2 + (den_sem / den) ** 2)


def stored_8_directions():
    """Mean and SEM of the LQG and ILQG costs written by centerout_cost_polar.py."""
    saved = np.load(FIGURE_DIR / FIGURE_SUBDIRS["cost_polar"] / f"Cfy40_{TAG}_cost.npz")
    # centerout_cost_polar.py does not store its trial count; it defaults to 100
    sem = saved["total_sd"] / np.sqrt(100)
    return saved["total"][:, 2], sem[:, 2], saved["total"][:, 0], sem[:, 0]


def scatter_panel(ax, x, x_err, y, y_err, xlabel, ylabel, angles, color):
    """y against one variable, with the regression line. Returns (r, p, rho, p_rho)."""
    res = stats.linregress(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", color=color, ecolor="0.6",
                capsize=4, markersize=10)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, res.slope * xs + res.intercept, "--", color=color, linewidth=2)
    step = 1 if len(angles) <= 8 else 3  # label every 45 deg when crowded
    for k in range(0, len(angles), step):
        ax.annotate(f"{angles[k]:.0f}°", (x[k], y[k]), textcoords="offset points",
                    xytext=(8, 5), fontsize=ANGLE_SIZE, color="0.3")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    # Above the panel rather than as a legend, where it would cover points
    ax.set_title(f"r$^2$ = {res.rvalue**2:.2f}", fontsize=LEGEND_SIZE, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return res.rvalue, res.pvalue, rho, p_rho


def plot(displacement, deviation, deviation_err, y, y_err, source, outdir,
         ylabel=COST_LABEL, name="CostVsKinematics", color=COLOR):
    """displacement: (direction, 2) joint displacement [deg], shoulder then elbow."""
    n = len(y)
    angles = np.arange(n) * 360 / n
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    panels = [
        ("joint excursion", np.linalg.norm(displacement, axis=1), None,
         "Joint excursion $\\|\\Delta\\theta\\|$ [deg]"),
        ("max lateral deviation", deviation, deviation_err,
         "Max. lateral deviation [cm]"),
        ("shoulder excursion", np.abs(displacement[:, 0]), None,
         "Shoulder excursion $|\\Delta\\theta_s|$ [deg]"),
        ("elbow excursion", np.abs(displacement[:, 1]), None,
         "Elbow excursion $|\\Delta\\theta_e|$ [deg]"),
    ]
    fits = {label: scatter_panel(ax, x, x_err, y, y_err, xlabel, ylabel, angles, color)
            for ax, (label, x, x_err, xlabel) in zip(axes.ravel(), panels)}
    fig.suptitle(f"{ylabel} against kinematics\n{AMPLITUDE} cm / "
                 f"{DURATION * 1000:.0f} ms, {n} directions ({source})", fontsize=TITLE_SIZE)
    save_figure(fig, outdir, f"{name}_{TAG}_{n}dir.svg")
    return fits


def plot_ratio_polar(ratio, ratio_err, ratio8, ratio8_err, num_sim, outdir):
    """LQG / ILQG cost ratio per direction: the simulated directions and the 8 stored ones."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    angles = np.linspace(0, 2 * pi, len(ratio), endpoint=False)
    closed = np.append(angles, angles[0])

    def close(v):
        return np.append(v, v[0])

    ax.plot(closed, close(ratio), color=RATIO_COLOR, linewidth=2,
            label=f"{len(ratio)} directions, {num_sim} trials each")
    ax.fill_between(closed, close(ratio - ratio_err), close(ratio + ratio_err),
                    color=RATIO_COLOR, alpha=0.2, linewidth=0)
    angles8 = np.linspace(0, 2 * pi, 8, endpoint=False)
    ax.errorbar(angles8, ratio8, yerr=ratio8_err, fmt="o", color="black", markersize=7,
                capsize=3, label=f"8 directions, stored costs (Cfy40_{TAG})")
    rmax = 1.1 * max(np.max(ratio + ratio_err), np.max(ratio8 + ratio8_err))
    ticks = [t for t in (1, 2, 4, 6, 8, 10, 15, 20) if t < rmax]
    style_polar_axis(ax, ticks, num_targets=8, rmax=rmax, radial_fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)
    ax.set_title(f"{RATIO_LABEL}, {AMPLITUDE} cm / {DURATION * 1000:.0f} ms",
                 fontsize=14, pad=25)
    save_figure(fig, outdir, f"CostRatio_polar_{TAG}.svg")


def plot_ratio_polar_8(ratio8, ratio8_err, outdir):
    """LQG / ILQG cost ratio for the 8 stored directions only."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    angles = np.linspace(0, 2 * pi, 8, endpoint=False)
    closed = np.append(angles, angles[0])
    lo, hi = np.append(ratio8 - ratio8_err, ratio8[0] - ratio8_err[0]), \
        np.append(ratio8 + ratio8_err, ratio8[0] + ratio8_err[0])
    ax.plot(closed, np.append(ratio8, ratio8[0]), "o-", color=RATIO_COLOR, linewidth=2,
            markersize=7)
    ax.fill_between(closed, lo, hi, color=RATIO_COLOR, alpha=0.2, linewidth=0)
    rmax = 1.1 * np.max(ratio8 + ratio8_err)
    ticks = [t for t in (1, 2, 4, 6, 8, 10, 15, 20) if t < rmax]
    style_polar_axis(ax, ticks, num_targets=8, rmax=rmax, radial_fontsize=14)
    ax.set_title(f"{RATIO_LABEL}, {AMPLITUDE} cm / {DURATION * 1000:.0f} ms, 8 directions",
                 fontsize=14, pad=25)
    save_figure(fig, outdir, f"CostRatio_polar_{TAG}_8dir.svg")


def report(title, fits):
    print(f"\n{title}")
    for name, (r, p, rho, p_rho) in fits.items():
        print(f"  {name:22s} r = {r:+.2f} (r2 = {r**2:.2f}, p = {p:.2g})   "
              f"Spearman rho = {rho:+.2f} (p = {p_rho:.2g})")


def main():
    parser = build_parser(__doc__, num_sim_default=200,
                          subdir=FIGURE_SUBDIRS["nonlinearity_index"])
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved .npz instead of simulating")
    args = parser.parse_args()
    data_path = args.outdir / f"CostVsKinematics_{TAG}.npz"

    targets = centerout_targets(START, AMPLITUDE, NUM_DIRECTIONS)
    displacement = np.array([joint_displacement(t) for t in targets])  # (direction, 2)
    trials = np.load(data_path)["trials"] if args.replot else simulate(args.num_sim, args.jobs)
    num_sim = trials.shape[1]
    (cost, cost_err), (deviation, deviation_err), (ilqg, ilqg_err) = map(mean_sem, trials)
    ratio, ratio_err = ratio_sem(cost, cost_err, ilqg, ilqg_err)

    every = NUM_DIRECTIONS // 8  # the 8 directions 0, 45, ... deg
    cost8, cost8_err, ilqg8, ilqg8_err = stored_8_directions()
    ratio8, ratio8_err = ratio_sem(cost8, cost8_err, ilqg8, ilqg8_err)
    disp8, dev8, dev8_err = displacement[::every], deviation[::every], deviation_err[::every]
    source = f"{num_sim} noisy trials each"

    fits = {
        "8 directions: LQG total cost against":
            plot(disp8, dev8, dev8_err, cost8, cost8_err, "stored costs", args.outdir),
        f"{NUM_DIRECTIONS} directions: LQG total cost against":
            plot(displacement, deviation, deviation_err, cost, cost_err, source, args.outdir),
        "8 directions: LQG / ILQG cost ratio against":
            plot(disp8, dev8, dev8_err, ratio8, ratio8_err, "stored costs", args.outdir,
                 RATIO_LABEL, "CostRatioVsKinematics", RATIO_COLOR),
        f"{NUM_DIRECTIONS} directions: LQG / ILQG cost ratio against":
            plot(displacement, deviation, deviation_err, ratio, ratio_err, source, args.outdir,
                 RATIO_LABEL, "CostRatioVsKinematics", RATIO_COLOR),
    }
    plot_ratio_polar(ratio, ratio_err, ratio8, ratio8_err, num_sim, args.outdir)
    plot_ratio_polar_8(ratio8, ratio8_err, args.outdir)

    if not args.replot:
        # trials: (LQG cost, LQG max lateral deviation, ILQG cost) x trial x direction
        np.savez(data_path, angles=np.arange(NUM_DIRECTIONS) * 360 / NUM_DIRECTIONS,
                 displacement=displacement, excursion=np.linalg.norm(displacement, axis=1),
                 trials=trials, deviation=deviation, deviation_sem=deviation_err,
                 cost=cost, cost_sem=cost_err, ilqg=ilqg, ilqg_sem=ilqg_err,
                 ratio=ratio, ratio_sem=ratio_err, cost_8_stored=cost8,
                 cost_8_stored_sem=cost8_err, ilqg_8_stored=ilqg8, ratio_8_stored=ratio8)

    for title, f in fits.items():
        report(title, f)
    angles = np.arange(NUM_DIRECTIONS) * 360 / NUM_DIRECTIONS
    print(f"\nLQG / ILQG ratio per direction ({source}):")
    print("  " + " ".join(f"{a:.0f}:{r:.1f}" for a, r in zip(angles, ratio)))
    print("  stored 8: " + " ".join(f"{a}:{r:.1f}" for a, r in zip(range(0, 360, 45), ratio8)))

    finish(not args.no_show)


if __name__ == "__main__":
    main()
