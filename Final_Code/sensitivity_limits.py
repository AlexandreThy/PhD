"""
How far the motor cost must rise before LQG beats FL, and what that costs in accuracy.

One center-out reach (20 cm in 400 ms from [0, 40], 90 deg by default) over a
sweep of the motor cost wr. ILQG and LQG optimise with wr; FL keeps its own
motor weight WR_FL, as in sensitivity_analysis.py. All three are scored with
the movement cost of common.py at that wr. Also records the distance from the
hand to the target at the end of the movement.

Panels: movement cost against wr (the LQG / FL crossover marked), endpoint
distance against wr, and the mean hand paths at one highlighted wr.

    python Final_Code/sensitivity_limits.py
    python Final_Code/sensitivity_limits.py --direction 0 --highlight-wr 10
    python Final_Code/sensitivity_limits.py --num-sim 20 --jobs 1
"""

from common import (
    FIGURE_SUBDIRS,
    COLORS, Cost_function, LEGEND, NUM_CONTROLLERS, START,
    build_parser, delete_axis, finish, guarded, np, plt, run_fl, run_ilqg,
    run_lqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 20
WR_SWEEP = (0.1, 0.3, 1, 3, 10, 20, 25, 30, 50, 100, 300, 1000)
HIGHLIGHT_WR = 30
LQG, FL = 2, 1  # indices in LEGEND


def target_for(direction_deg, start=START, amplitude=AMPLITUDE):
    angle = np.radians(direction_deg)
    return [start[0] + np.cos(angle) * amplitude, start[1] + np.sin(angle) * amplitude]


def _worker(task):
    """The three controllers at one motor cost. Stays top-level."""
    wr, target = task
    runs = (
        guarded(run_ilqg, "ILQG", MOVEMENT_TIME, NUM_ITER, START, target, wr=wr),
        guarded(run_fl, "FL", MOVEMENT_TIME, NUM_ITER, START, target),
        guarded(run_lqg, "LQG", MOVEMENT_TIME, NUM_ITER, START, target, wr=wr),
    )
    return {
        "cost": np.array([Cost_function(x[:, :4], u, r=wr, tg=target) for _, _, x, u in runs]),
        "end_dist": np.array([np.hypot(X[-1] - target[0], Y[-1] - target[1])
                              for X, Y, _, _ in runs]),
        "path": np.array([np.stack([X, Y]) for X, Y, _, _ in runs]),
    }


def simulate(direction_deg, num_sim, jobs, wr_sweep=WR_SWEEP):
    target = target_for(direction_deg)
    tasks = [(wr, target) for wr in wr_sweep for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc=f"motor cost sweep, {direction_deg:g} deg")
    # (wr, trial, controller, ...)
    return {key: np.array([r[key] for r in results]).reshape(
        len(wr_sweep), num_sim, *results[0][key].shape) for key in results[0]}


def crossover(wr_sweep, mean_cost):
    """First wr at which LQG's mean cost drops below FL's, interpolated in log wr."""
    diff = mean_cost[:, LQG] - mean_cost[:, FL]
    for k in range(1, len(wr_sweep)):
        if diff[k - 1] > 0 >= diff[k]:
            a, b = np.log(wr_sweep[k - 1]), np.log(wr_sweep[k])
            return float(np.exp(a + (b - a) * diff[k - 1] / (diff[k - 1] - diff[k])))
    return None


def plot(data, wr_sweep, direction_deg, highlight_wr, num_sim, outdir):
    wr = np.asarray(wr_sweep, float)
    mean_cost = data["cost"].mean(axis=1)
    sem_cost = data["cost"].std(axis=1) / np.sqrt(num_sim)
    mean_dist = data["end_dist"].mean(axis=1)
    sem_dist = data["end_dist"].std(axis=1) / np.sqrt(num_sim)
    cross = crossover(wr, mean_cost)
    h = int(np.argmin(np.abs(wr - highlight_wr)))

    fig, (ax_cost, ax_dist, ax_path) = plt.subplots(1, 3, figsize=(18, 5.5))
    for c in range(NUM_CONTROLLERS):
        ax_cost.errorbar(wr, mean_cost[:, c], yerr=sem_cost[:, c], fmt="-o", ms=4,
                         color=COLORS[c], capsize=2, label=LEGEND[c])
        ax_dist.errorbar(wr, mean_dist[:, c], yerr=sem_dist[:, c], fmt="-o", ms=4,
                         color=COLORS[c], capsize=2, label=LEGEND[c])
    for ax in (ax_cost, ax_dist):
        ax.set_xscale("log")
        ax.axvline(wr[h], color="grey", linestyle=":", linewidth=1)
        if cross is not None:
            ax.axvline(cross, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Motor cost $w_r$ (ILQG and LQG optimise with it; all scored with it)",
                      fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        delete_axis(ax, sides=["top", "right"])
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("Movement cost (mean +/- SEM)", fontsize=12)
    ax_cost.set_title("Movement cost" + (f"   (LQG < FL from $w_r$ = {cross:.0f})"
                                         if cross is not None else ""), fontsize=13)
    ax_dist.set_ylabel("Distance to target at the end [cm]", fontsize=12)
    ax_dist.set_title("Endpoint distance (mean +/- SEM)", fontsize=13)

    target = target_for(direction_deg)
    for c in range(NUM_CONTROLLERS):
        paths = data["path"][h, :, c]
        mean = paths.mean(axis=0)
        for p in paths[:10]:
            ax_path.plot(p[0], p[1], color=COLORS[c], alpha=0.25, linewidth=0.8)
        ax_path.plot(mean[0], mean[1], color=COLORS[c], linewidth=3,
                     label=f"{LEGEND[c]}: {mean_dist[h, c]:.2f} cm")
    ax_path.plot(*START, "o", color="grey", ms=10)
    ax_path.add_patch(plt.Circle(target, 0.5, color="grey", fill=False, linewidth=2))
    ax_path.set_aspect("equal")
    ax_path.set_title(f"Hand paths at $w_r$ = {wr[h]:g} (mean, 10 trials)", fontsize=13)
    ax_path.legend(title="end distance", fontsize=10, frameon=False, loc="best")
    ax_path.set_xlabel("hand x [cm]", fontsize=11)
    ax_path.set_ylabel("hand y [cm]", fontsize=11)
    delete_axis(ax_path, sides=["top", "right"])

    fig.suptitle(f"{AMPLITUDE} cm reach at {direction_deg:g} deg in "
                 f"{MOVEMENT_TIME * 1000:.0f} ms from {list(START)}, {num_sim} trials per "
                 f"$w_r$", fontsize=14)
    fig.tight_layout()
    save_figure(fig, outdir, f"SensitivityLimits_{direction_deg:g}deg.svg")
    return cross, h


def main():
    parser = build_parser(__doc__, num_sim_default=200,
                          subdir=FIGURE_SUBDIRS["sensitivity"])
    parser.add_argument("--direction", type=float, default=90,
                        help="reach direction in degrees")
    parser.add_argument("--highlight-wr", type=float, default=HIGHLIGHT_WR,
                        help="motor cost of the hand path panel")
    args = parser.parse_args()

    data = simulate(args.direction, args.num_sim, args.jobs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f"SensitivityLimits_{args.direction:g}deg.npz"
    np.savez(path, wr=np.array(WR_SWEEP), cost=data["cost"], end_dist=data["end_dist"],
             legend=np.array(LEGEND))
    print(f"wrote {path}", flush=True)
    cross, h = plot(data, WR_SWEEP, args.direction, args.highlight_wr, args.num_sim,
                    args.outdir)

    print(f"\n{'wr':>7}  " + "  ".join(f"{n + ' cost':>12}" for n in LEGEND) + "  "
          + "  ".join(f"{n + ' dist cm':>12}" for n in LEGEND))
    for k, wr in enumerate(WR_SWEEP):
        print(f"{wr:7g}  " + "  ".join(f"{v:12.2f}" for v in data["cost"][k].mean(0)) + "  "
              + "  ".join(f"{v:12.2f}" for v in data["end_dist"][k].mean(0)))
    if cross is not None:
        print(f"\nLQG cheaper than FL from wr = {cross:.1f}")
    finish(not args.no_show)


if __name__ == "__main__":
    main()
