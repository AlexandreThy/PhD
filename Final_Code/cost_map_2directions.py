"""
DLQG movement cost as a function of the starting arm configuration.

Every reach is identical -- 15 cm, in one of two directions -- and only the
posture it starts from changes, sampled on a grid of shoulder and elbow angles.
Any variation in the cost is therefore due to the configuration alone.

Ported from CurrentParts/2Dir.py, which sampled starting positions on a
cartesian grid of hand positions. Over that grid the hand height and the
distance from the shoulder are collinear (a cubic fit of the cost on either
gives R^2 = 0.93), so it could not separate the starting posture from the plain
hand position. Sampling the joint angles directly varies the two independently.

    python Final_Code/cost_map_2directions.py
    python Final_Code/cost_map_2directions.py --num-sim 2 --jobs 1
"""

from matplotlib.colors import LogNorm

from common import (
    Cost_function, build_parser, delete_axis, finish, np, plt, run_dlqg,
    run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
L1, L2 = 30, 33  # upper arm and forearm, as in the controllers
# Starting postures. The ranges keep the start and the target of both reaches
# well inside the workspace, which spans |L2 - L1| to L1 + L2 from the shoulder.
SHOULDER_ANGLES = np.linspace(10, 55, 12)  # degrees
ELBOW_ANGLES = np.linspace(85, 125, 9)  # degrees
DIRECTIONS = [90, 315]

CHEAP_COLOR = "#009E73"
COSTLY_COLOR = "#B90072"


def hand_position(shoulder_deg, elbow_deg):
    """Hand position [cm] for a starting posture given in degrees."""
    ts, te = np.radians(shoulder_deg), np.radians(elbow_deg)
    return np.array([np.cos(ts + te) * L2 + np.cos(ts) * L1,
                     np.sin(ts + te) * L2 + np.sin(ts) * L1])


def reach_target(start, direction_deg):
    angle = np.radians(direction_deg)
    return start + np.array([np.cos(angle), np.sin(angle)]) * AMPLITUDE


def _worker(task):
    """Mean DLQG cost over repetitions for one starting posture and direction."""
    shoulder_deg, elbow_deg, direction_deg, num_sim = task
    start = hand_position(shoulder_deg, elbow_deg)
    target = reach_target(start, direction_deg)
    if not (L2 - L1) + 0.5 < np.hypot(*target) < (L1 + L2) - 0.5:
        return np.nan  # target outside the reachable annulus

    costs = np.zeros(num_sim)
    for sim in range(num_sim):
        _, _, x, u = run_dlqg(MOVEMENT_TIME, NUM_ITER, list(start), list(target))
        costs[sim] = Cost_function(x, u, tg=target)
    return float(np.mean(costs))


def simulate(num_sim, jobs):
    tasks = [(shoulder, elbow, direction, num_sim)
             for direction in DIRECTIONS
             for shoulder in SHOULDER_ANGLES
             for elbow in ELBOW_ANGLES]
    results = run_tasks(_worker, tasks, jobs, desc="DLQG cost map")

    flat = np.array(results).reshape(len(DIRECTIONS), len(SHOULDER_ANGLES),
                                     len(ELBOW_ANGLES))
    return {direction: flat[i] for i, direction in enumerate(DIRECTIONS)}


def _extreme_postures(cost):
    """Grid indices of the cheapest and the costliest starting posture."""
    cheapest = np.unravel_index(np.nanargmin(cost), cost.shape)
    costliest = np.unravel_index(np.nanargmax(cost), cost.shape)
    return cheapest, costliest


def _plot_joint_map(ax, cost, direction_deg, norm):
    """Cost over the two starting joint angles."""
    im = ax.imshow(cost.T, origin="lower", cmap="plasma", norm=norm,
                   aspect="auto", interpolation="bicubic",
                   extent=[SHOULDER_ANGLES.min(), SHOULDER_ANGLES.max(),
                           ELBOW_ANGLES.min(), ELBOW_ANGLES.max()])
    for idx, color in zip(_extreme_postures(cost), (CHEAP_COLOR, COSTLY_COLOR)):
        ax.plot(SHOULDER_ANGLES[idx[0]], ELBOW_ANGLES[idx[1]], marker="o",
                markersize=11, markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=2.5)
    ax.set_xlabel("Starting shoulder angle [deg]", fontsize=13)
    ax.set_ylabel("Starting elbow angle [deg]", fontsize=13)
    ax.set_title(f"Cost over starting posture, {direction_deg} deg reach",
                 fontsize=13)
    ax.tick_params(labelsize=11)
    return im


def _plot_elbow_collapse(ax, cost, direction_deg):
    """
    Cost against the elbow angle alone.

    One grey line per shoulder angle. They fall close together, so the cost is
    essentially a function of how extended the arm is, not of where the shoulder
    points -- and the slope reverses between the two reach directions.
    """
    for row in cost:
        ax.plot(ELBOW_ANGLES, row, color="grey", alpha=0.35, linewidth=0.8)
    ax.fill_between(ELBOW_ANGLES, np.nanmin(cost, axis=0), np.nanmax(cost, axis=0),
                    color="#0072B2", alpha=0.15)
    ax.plot(ELBOW_ANGLES, np.nanmean(cost, axis=0), color="#0072B2", linewidth=2.5)

    ax.set_yscale("log")
    ax.set_xlabel("Starting elbow angle [deg]", fontsize=13)
    ax.set_ylabel("Movement cost", fontsize=13)
    # Quoted off the mean profile rather than off the extreme cells: the largest
    # and smallest single estimates are the noisiest, and would overstate this.
    profile = np.nanmean(cost, axis=0)
    ax.set_title(f"{profile.max() / profile.min():.0f}x across the elbow range",
                 fontsize=13)
    delete_axis(ax, sides=["top", "right"])
    ax.tick_params(labelsize=11)


def _draw_arm(ax, shoulder_deg, elbow_deg, direction_deg, color, label):
    """The two links, and the reach that starts from that posture."""
    ts, te = np.radians(shoulder_deg), np.radians(elbow_deg)
    elbow = np.array([np.cos(ts) * L1, np.sin(ts) * L1])
    hand = elbow + np.array([np.cos(ts + te) * L2, np.sin(ts + te) * L2])

    ax.plot([0, elbow[0], hand[0]], [0, elbow[1], hand[1]], color=color,
            linewidth=2.5, marker="o", markersize=5, zorder=3, label=label)
    ax.annotate("", xy=reach_target(hand, direction_deg), xytext=hand, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.5,
                                mutation_scale=18))


def _plot_workspace(ax, cost, direction_deg, norm):
    """Where the sampled postures put the hand, and the two extreme postures."""
    starts = np.array([[hand_position(s, e) for e in ELBOW_ANGLES]
                       for s in SHOULDER_ANGLES])
    ax.scatter(starts[..., 0], starts[..., 1], c=cost, cmap="plasma", norm=norm,
               s=16, zorder=2)

    cheapest, costliest = _extreme_postures(cost)
    for idx, color, name in ((cheapest, CHEAP_COLOR, "cheapest"),
                             (costliest, COSTLY_COLOR, "costliest")):
        _draw_arm(ax, SHOULDER_ANGLES[idx[0]], ELBOW_ANGLES[idx[1]], direction_deg,
                  color, f"{name} ({cost[idx]:.1f})")

    ax.plot(0, 0, marker="o", markersize=9, color="grey", zorder=5)  # shoulder
    ax.set_aspect("equal")
    ax.set_xlabel("Hand x [cm]", fontsize=13)
    ax.set_ylabel("Hand y [cm]", fontsize=13)
    ax.set_title("Same reach, two postures", fontsize=13)
    ax.legend(fontsize=10, loc="lower left", frameon=False)
    delete_axis(ax, sides=["top", "right"])
    ax.tick_params(labelsize=11)


def plot(cost_maps, outdir, num_sim):
    values = np.concatenate([cost_maps[d].ravel() for d in DIRECTIONS])
    values = values[np.isfinite(values)]
    norm = LogNorm(vmin=values.min(), vmax=values.max())

    # Constrained layout, because tight_layout cannot place a colourbar that is
    # shared across a grid of axes without walking it over the middle column.
    fig, axes = plt.subplots(len(DIRECTIONS), 3, figsize=(15, 9),
                             layout="constrained")
    for row, direction in enumerate(DIRECTIONS):
        cost = cost_maps[direction]
        im = _plot_joint_map(axes[row, 0], cost, direction, norm)
        _plot_elbow_collapse(axes[row, 1], cost, direction)
        _plot_workspace(axes[row, 2], cost, direction, norm)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="right",
                        shrink=0.6, pad=0.01)
    cbar.set_label(f"Mean movement cost over {num_sim} trials", fontsize=12)

    fig.suptitle(f"Identical {AMPLITUDE} cm reaches: the cost is set by the "
                 f"posture they start from", fontsize=16)
    save_figure(fig, outdir, "DLQG_CostMap_90_315.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=10)
    args = parser.parse_args()

    cost_maps = simulate(args.num_sim, args.jobs)
    plot(cost_maps, args.outdir, args.num_sim)

    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / "DLQG_CostMap_90_315.npz"
    np.savez(path, shoulder_angles=SHOULDER_ANGLES, elbow_angles=ELBOW_ANGLES,
             **{f"deg_{d}": cost_maps[d] for d in DIRECTIONS})
    print(f"wrote {path}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
