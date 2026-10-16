"""
LQG movement cost as a function of the starting arm configuration.

Every reach is identical -- 15 cm, in one of two directions -- and only the
posture it starts from changes, sampled on a grid of shoulder and elbow angles.
Any variation in the cost is therefore due to the configuration alone.

Ported from CurrentParts/2Dir.py, which sampled starting positions on a
cartesian grid of hand positions. Over that grid the hand height and the
distance from the shoulder are collinear (a cubic fit of the cost on either
gives R^2 = 0.93), so it could not separate the starting posture from the plain
hand position. Sampling the joint angles directly varies the two independently.

ILQG is run on the same grid, and the last column shows the relative cost of
LQG, (mean LQG cost - mean ILQG cost) / mean ILQG cost, over the starting
posture.

    python Final_Code/cost_map_2directions.py
    python Final_Code/cost_map_2directions.py --num-sim 2 --jobs 1
    python Final_Code/cost_map_2directions.py --replot   # from the saved .npz

--directions picks the reach directions; the files are named after them
(LQG_CostMap_<d1>_<d2>..., default 90 and 315):

    python Final_Code/cost_map_2directions.py --directions 0 --num-sim 200

LQG_ILQG_CostRatio_90_315.svg is the workspace view of the fourth column,
coloured by the cost ratio LQG / ILQG instead of the LQG cost.
"""

from matplotlib.colors import LogNorm, SymLogNorm

from common import (
    FIGURE_SUBDIRS, START,
    Cost_function, build_parser, delete_axis, finish, np, plt, run_ilqg,
    run_lqg, run_tasks, save_figure,
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


def _tag():
    """File name suffix of the current directions, e.g. 90_315."""
    return "_".join(f"{d:g}" for d in DIRECTIONS)

CHEAP_COLOR = "#009E73"
COSTLY_COLOR = "#B90072"
REL_LINTHRESH = 10  # [%] linear range of the relative-cost colour scale


def hand_position(shoulder_deg, elbow_deg):
    """Hand position [cm] for a starting posture given in degrees."""
    ts, te = np.radians(shoulder_deg), np.radians(elbow_deg)
    return np.array([np.cos(ts + te) * L2 + np.cos(ts) * L1,
                     np.sin(ts + te) * L2 + np.sin(ts) * L1])


def reach_target(start, direction_deg):
    angle = np.radians(direction_deg)
    return start + np.array([np.cos(angle), np.sin(angle)]) * AMPLITUDE


def _worker(task):
    """Mean LQG and ILQG costs over repetitions for one posture and direction."""
    shoulder_deg, elbow_deg, direction_deg, num_sim = task
    start = hand_position(shoulder_deg, elbow_deg)
    target = reach_target(start, direction_deg)
    if not (L2 - L1) + 0.5 < np.hypot(*target) < (L1 + L2) - 0.5:
        return np.nan, np.nan  # target outside the reachable annulus

    costs = np.zeros((num_sim, 2))
    for sim in range(num_sim):
        for j, runner in enumerate((run_lqg, run_ilqg)):
            _, _, x, u = runner(MOVEMENT_TIME, NUM_ITER, list(start), list(target))
            costs[sim, j] = Cost_function(x[:, :4], u, tg=target)
    return tuple(np.mean(costs, axis=0))


def simulate(num_sim, jobs):
    """Mean LQG and ILQG cost maps, each {direction: (shoulder, elbow) array}."""
    tasks = [(shoulder, elbow, direction, num_sim)
             for direction in DIRECTIONS
             for shoulder in SHOULDER_ANGLES
             for elbow in ELBOW_ANGLES]
    results = run_tasks(_worker, tasks, jobs, desc="LQG / ILQG cost map")

    flat = np.array(results).reshape(len(DIRECTIONS), len(SHOULDER_ANGLES),
                                     len(ELBOW_ANGLES), 2)
    lqg = {direction: flat[i, ..., 0] for i, direction in enumerate(DIRECTIONS)}
    ilqg = {direction: flat[i, ..., 1] for i, direction in enumerate(DIRECTIONS)}
    return lqg, ilqg


def relative_cost(lqg, ilqg):
    """(LQG - ILQG) / ILQG, from the mean costs of each posture."""
    return (lqg - ilqg) / ilqg


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


def _plot_relative_map(ax, relative, direction_deg, norm):
    """Relative cost of LQG against ILQG, over the two starting joint angles."""
    im = ax.imshow(relative.T * 100, origin="lower", cmap="RdBu_r", norm=norm,
                   aspect="auto", interpolation="bicubic",
                   extent=[SHOULDER_ANGLES.min(), SHOULDER_ANGLES.max(),
                           ELBOW_ANGLES.min(), ELBOW_ANGLES.max()])
    ax.set_xlabel("Starting shoulder angle [deg]", fontsize=13)
    ax.set_ylabel("Starting elbow angle [deg]", fontsize=13)
    ax.set_title(f"(LQG - ILQG) / ILQG, {direction_deg} deg reach", fontsize=13)
    ax.tick_params(labelsize=11)
    return im


def _start_positions():
    """Hand position of every sampled posture, shaped like the cost grid."""
    return np.array([[hand_position(s, e) for e in ELBOW_ANGLES]
                     for s in SHOULDER_ANGLES])


def _plot_cartesian_map(ax, cost, direction_deg, norm):
    """
    The same cost, over where the posture puts the hand.

    The grid is regular in joint angles, so in hand coordinates it is a curved
    quadrilateral rather than a rectangle. A triangulation fills exactly the
    region that was sampled; imshow would have to invent a bounding box and
    colour in corners no simulation ever visited.
    """
    starts = _start_positions()
    x, y, values = starts[..., 0].ravel(), starts[..., 1].ravel(), cost.ravel()
    finite = np.isfinite(values)
    im = ax.tripcolor(x[finite], y[finite], values[finite], shading="gouraud",
                      cmap="plasma", norm=norm)

    
    ax.plot([0], [40], marker="o", markersize=11,
            markerfacecolor="none", markeredgecolor="green", markeredgewidth=2.5)

    ax.set_aspect("equal")
    ax.set_xlabel("Initial hand x [cm]", fontsize=13)
    ax.set_ylabel("Initial hand y [cm]", fontsize=13)
    ax.set_title(f"Cost over starting hand position,\n{direction_deg} deg reach",
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


def _plot_workspace(ax, cost, direction_deg, norm, names=("cheapest", "costliest"),
                    title="Same reach, two postures", fmt=".1f"):
    """Where the sampled postures put the hand, and the two extreme postures."""
    starts = _start_positions()
    ax.scatter(starts[..., 0], starts[..., 1], c=cost, cmap="plasma", norm=norm,
               s=16, zorder=2)

    cheapest, costliest = _extreme_postures(cost)
    for idx, color, name in ((cheapest, CHEAP_COLOR, names[0]),
                             (costliest, COSTLY_COLOR, names[1])):
        _draw_arm(ax, SHOULDER_ANGLES[idx[0]], ELBOW_ANGLES[idx[1]], direction_deg,
                  color, f"{name} ({cost[idx]:{fmt}})")

    ax.plot(0, 0, marker="o", markersize=9, color="grey", zorder=5)  # shoulder
    ax.set_aspect("equal")
    ax.set_xlabel("Hand x [cm]", fontsize=13)
    ax.set_ylabel("Hand y [cm]", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="lower left", frameon=False)
    delete_axis(ax, sides=["top", "right"])
    ax.tick_params(labelsize=11)


def plot(cost_maps, ilqg_maps, outdir, num_sim):
    values = np.concatenate([cost_maps[d].ravel() for d in DIRECTIONS])
    values = values[np.isfinite(values)]
    norm = LogNorm(vmin=values.min(), vmax=values.max())

    # Diverging scale centred on 0: blue where LQG is cheaper than ILQG, red
    # where it costs more. Shared by both directions. The relative cost spans a
    # few percent to several hundred times, so the scale is logarithmic beyond
    # +-REL_LINTHRESH percent and linear inside it.
    relative = {d: relative_cost(cost_maps[d], ilqg_maps[d]) for d in DIRECTIONS}
    rel_values = np.concatenate([relative[d].ravel() for d in DIRECTIONS]) * 100
    rel_bound = np.abs(rel_values[np.isfinite(rel_values)]).max()
    rel_norm = SymLogNorm(linthresh=REL_LINTHRESH, vmin=-rel_bound, vmax=rel_bound,
                          base=10)

    # Constrained layout, because tight_layout cannot place a colourbar that is
    # shared across a grid of axes without walking it over the middle column.
    # The cartesian map sits next to the joint one: same cost, same colour scale,
    # only the coordinates the starting posture is named by change.
    fig, axes = plt.subplots(len(DIRECTIONS), 5, figsize=(25, 4.5 * len(DIRECTIONS)),
                             layout="constrained", squeeze=False)
    for row, direction in enumerate(DIRECTIONS):
        cost = cost_maps[direction]
        im = _plot_joint_map(axes[row, 0], cost, direction, norm)
        _plot_cartesian_map(axes[row, 1], cost, direction, norm)
        _plot_elbow_collapse(axes[row, 2], cost, direction)
        _plot_workspace(axes[row, 3], cost, direction, norm)
        im_rel = _plot_relative_map(axes[row, 4], relative[direction], direction,
                                    rel_norm)

    cbar = fig.colorbar(im, ax=axes[:, :4].ravel().tolist(), location="right",
                        shrink=0.6, pad=0.01)
    cbar.set_label(f"Mean LQG movement cost over {num_sim} trials", fontsize=12)
    cbar_rel = fig.colorbar(im_rel, ax=axes[:, 4].ravel().tolist(),
                            location="right", shrink=0.6, pad=0.01)
    cbar_rel.set_label("Relative cost of LQG vs ILQG [%]", fontsize=12)

    fig.suptitle(f"Identical {AMPLITUDE} cm reaches: the cost is set by the "
                 f"posture they start from", fontsize=16)
    save_figure(fig, outdir, f"LQG_CostMap_{_tag()}.svg")


def plot_ratio_workspace(cost_maps, ilqg_maps, outdir, num_sim):
    """
    The fourth column of LQG_CostMap_90_315.svg coloured by LQG cost / ILQG
    cost. The two directions span very different ratios (about 4 to 110 at
    90 deg, 1 to 8 at 315 deg), so each gets its own logarithmic scale and
    colourbar rather than one shared scale that would flatten the 315 deg panel.
    """
    fig, axes = plt.subplots(1, len(DIRECTIONS), figsize=(7 * len(DIRECTIONS), 6),
                             layout="constrained", squeeze=False)
    for ax, direction in zip(axes[0], DIRECTIONS):
        ratio = cost_maps[direction] / ilqg_maps[direction]
        finite = ratio[np.isfinite(ratio)]
        norm = LogNorm(vmin=finite.min(), vmax=finite.max())
        _plot_workspace(ax, ratio, direction, norm,
                        names=("lowest ratio", "highest ratio"),
                        title=f"LQG / ILQG cost, {direction} deg reach", fmt=".2f")
        # The starting point of the center-out task, for reference.
        ax.plot(START[0], START[1], marker="*", markersize=20, color="white",
                markeredgecolor="black", markeredgewidth=1.5, linestyle="none",
                zorder=6, label=f"center-out start {list(START)}")
        ax.legend(fontsize=10, loc="lower left", frameon=False)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax,
                            location="right", shrink=0.8, pad=0.01)
        cbar.set_label(f"Mean LQG cost / mean ILQG cost ({num_sim} trials each)",
                       fontsize=11)
    save_figure(fig, outdir, f"LQG_ILQG_CostRatio_{_tag()}.svg")


def load_maps(path):
    """The LQG and ILQG cost maps saved by main()."""
    saved = np.load(path)
    return ({d: saved[f"deg_{d}"] for d in DIRECTIONS},
            {d: saved[f"ilqg_deg_{d}"] for d in DIRECTIONS})


def main():
    global DIRECTIONS
    parser = build_parser(__doc__, num_sim_default=10,
                          subdir=FIGURE_SUBDIRS["two_directions"])
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved .npz, no simulation; "
                             "--num-sim then only labels the figures")
    parser.add_argument("--directions", type=float, nargs="+", default=DIRECTIONS,
                        help="reach directions in degrees (default 90 315)")
    args = parser.parse_args()
    DIRECTIONS = [int(d) if float(d).is_integer() else d for d in args.directions]

    path = args.outdir / f"LQG_CostMap_{_tag()}.npz"
    if args.replot:
        cost_maps, ilqg_maps = load_maps(path)
        plot(cost_maps, ilqg_maps, args.outdir, args.num_sim)
        plot_ratio_workspace(cost_maps, ilqg_maps, args.outdir, args.num_sim)
        finish(not args.no_show)
        return

    cost_maps, ilqg_maps = simulate(args.num_sim, args.jobs)
    plot(cost_maps, ilqg_maps, args.outdir, args.num_sim)
    plot_ratio_workspace(cost_maps, ilqg_maps, args.outdir, args.num_sim)

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(path, shoulder_angles=SHOULDER_ANGLES, elbow_angles=ELBOW_ANGLES,
             **{f"deg_{d}": cost_maps[d] for d in DIRECTIONS},
             **{f"ilqg_deg_{d}": ilqg_maps[d] for d in DIRECTIONS},
             **{f"relative_deg_{d}": relative_cost(cost_maps[d], ilqg_maps[d])
                for d in DIRECTIONS})
    print(f"wrote {path}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
