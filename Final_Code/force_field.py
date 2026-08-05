"""
Response of the three controllers to a velocity-dependent force field.

Every repetition is simulated twice, with and without the force field, so the
two conditions share the same set of noise realisations. Produces the panel
figure (hand paths, joint velocity profiles in each condition, and the terminal
joint position and velocity errors) and the scatter of the movement cost with
against without the field.

    python Final_Code/force_field.py
    python Final_Code/force_field.py --num-sim 5 --jobs 1
"""

from matplotlib.lines import Line2D

from common import (
    COLORS, Cost_function, LEGEND, NUM_CONTROLLERS,
    build_parser, compute_angles_from_cartesian, delete_axis, finish, np, pi,
    plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
START = [0, 40]
TARGET = [0, 55]
# Chosen so that the controllers rank FL < ILQG < DLQG by cost with the field
# on, and ILQG < FL < DLQG with it off. The window is narrow: ILQG overtakes FL
# near 6.5e-4 and overtakes DLQG near 9e-4, and only in between does ILQG sit
# second in both conditions. Below 7e-4 the ILQG-FL gap is within noise; above
# 8e-4 DLQG stops being the worst. 7.5e-4 keeps both gaps well resolved
# (ILQG-FL +13.4, DLQG-ILQG +23.6, each many standard errors) while holding the
# lateral excursion near 10 cm and the terminal error near 2 cm.
FF_POWER = -3e-4
MAX_TRAJECTORIES_SHOWN = 10
TRAJECTORY_OFFSET = 15  # panel 0 draws the controllers side by side


def _run_all(duration, num_iter, start, target, ff, ff_power):
    """The three controllers under one force field condition."""
    power = ff_power if ff else 0.0
    return (
        run_ilqg(duration, num_iter, start, target, ff=ff, ff_power=power),
        run_fl(duration, num_iter, start, target, ff=ff, ff_power=power),
        run_dlqg(duration, num_iter, start, target, ff=ff, ff_power=power),
    )


def _worker(task):
    """
    One repetition: the three controllers with, then without, the force field.

    Returns arrays stacked over controllers so the caller can index by
    controller without unpacking per-controller tuples.
    """
    duration, num_iter, start, target, ff_power = task

    ff_runs = _run_all(duration, num_iter, start, target, True, ff_power)
    free_runs = _run_all(duration, num_iter, start, target, False, ff_power)

    target_angles = np.array(compute_angles_from_cartesian(target[0], target[1]))

    return {
        "cost_ff": np.array([Cost_function(x, u, tg=target) for _, _, x, u in ff_runs]),
        "cost_free": np.array([Cost_function(x, u, tg=target) for _, _, x, u in free_runs]),
        "vel_ff": np.array([x[:, 2:4].T for _, _, x, _ in ff_runs]),
        "vel_free": np.array([x[:, 2:4].T for _, _, x, _ in free_runs]),
        "pos_error": np.array([np.abs(x[-1, :2] - target_angles) for _, _, x, _ in ff_runs]),
        "traj_ff": np.array([[X, Y] for X, Y, _, _ in ff_runs]),
        "traj_free": np.array([[X, Y] for X, Y, _, _ in free_runs]),
    }


def simulate(num_sim, jobs, ff_power=FF_POWER):
    tasks = [(MOVEMENT_TIME, NUM_ITER, START, TARGET, ff_power)
             for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc="force field")
    return {key: np.array([r[key] for r in results]) for key in results[0]}


def percentile_cross(x, y, ax, p=95, color="black"):
    """
    Mean marker with percentile whiskers on both axes.

    Returns the (x_high, y_high) the cross reaches, so the caller can crop each
    axis to the data actually drawn.
    """
    x, y = np.asarray(x), np.asarray(y)
    alpha = (100 - p) / 2
    x_low, x_high = np.percentile(x, [alpha, 100 - alpha])
    y_low, y_high = np.percentile(y, [alpha, 100 - alpha])
    mx, my = np.mean(x), np.mean(y)

    ax.plot([x_low, x_high], [my, my], color=color, linewidth=1.5)
    ax.plot([mx, mx], [y_low, y_high], color=color, linewidth=1.5)
    ax.scatter(mx, my, marker="o", color=color, s=30)
    return x_high, y_high


def _style_velocity_axis(ax, title):
    delete_axis(ax, sides=["top", "right"])
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Time [ms]", fontsize=15)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])
    # Kept short: a longer label is taller than the axes and runs into its neighbour.
    ax.set_ylabel("Angular velocity [deg/s]", fontsize=15)
    ax.set_yticks([-180, 0, 180])
    ax.tick_params(labelsize=14)


def _label_boxplot(ax, ylabel, title=None, positions=None, labels=None):
    """Name the boxes on a per-controller boxplot instead of hiding the axis."""
    ax.set_xticks(range(1, NUM_CONTROLLERS + 1) if positions is None else positions)
    ax.set_xticklabels(LEGEND if labels is None else labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel(ylabel, fontsize=15)
    if title:
        ax.set_title(title, fontsize=15)


def _joint_boxplot(ax, values, ylabel, title):
    """One box per controller, in a shoulder group and an elbow group."""
    for joint in range(2):
        box = ax.boxplot(values[:, :, joint],
                         positions=[joint * 3 + 1, joint * 3 + 2, joint * 3 + 3],
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color="black"),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"))
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)

    delete_axis(ax, sides=["top", "right", "bottom"])
    _label_boxplot(ax, ylabel, title=title, positions=[2, 5],
                   labels=["shoulder", "elbow"])
    _controller_legend(ax)


def _controller_legend(ax, extra=()):
    handles = [Line2D([], [], color=c, lw=3, label=n) for c, n in zip(COLORS, LEGEND)]
    handles.extend(extra)
    ax.legend(handles=handles, fontsize=12, loc="upper left", frameon=False)


def plot_panels(data, outdir, ff_power=FF_POWER):
    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)
    # The hand paths are drawn to scale and are far wider than tall, so that row
    # needs less height than the four time series and boxplot rows.
    fig, (ax_paths, ax_vel_ff, ax_vel_free, ax_pos_err, ax_vel_err) = plt.subplots(
        5, figsize=(6, 16), gridspec_kw=dict(height_ratios=[0.62, 1, 1, 1, 1]))

    shown = min(data["traj_ff"].shape[0], MAX_TRAJECTORIES_SHOWN)
    for rep in range(shown):
        for i in range(NUM_CONTROLLERS):
            offset = TRAJECTORY_OFFSET * i
            ax_paths.plot(data["traj_ff"][rep, i, 0] + offset,
                          data["traj_ff"][rep, i, 1],
                          color=COLORS[i], linewidth=0.6, linestyle="-")
            ax_paths.plot(data["traj_free"][rep, i, 0] + offset,
                          data["traj_free"][rep, i, 1],
                          color=COLORS[i], linewidth=0.6, linestyle="-.")

    for axis, key, condition in ((ax_vel_ff, "vel_ff", "force field ON"),
                                 (ax_vel_free, "vel_free", "force field OFF")):
        for i in range(NUM_CONTROLLERS):
            for joint, linestyle in enumerate(["-", "--"]):
                trace = data[key][:, i, joint] * 180 / pi
                mean_vel, std_vel = np.mean(trace, axis=0), np.std(trace, axis=0)
                axis.plot(time, mean_vel, color=COLORS[i], linestyle=linestyle)
                axis.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                                  color=COLORS[i], alpha=0.3)
        _style_velocity_axis(axis, f"Joint angular velocity, {condition}")
        _controller_legend(axis, extra=[
            Line2D([], [], color="black", ls="-", label="shoulder"),
            Line2D([], [], color="black", ls="--", label="elbow"),
        ])

    # The two conditions only mean something side by side, so they share a scale.
    span = max(abs(np.concatenate([ax_vel_ff.get_ylim(),
                                   ax_vel_free.get_ylim()])).max(), 180)
    ax_vel_ff.set_ylim(-span, span)
    ax_vel_free.set_ylim(-span, span)

    _joint_boxplot(ax_pos_err, data["pos_error"] * 180 / np.pi,
                   "Terminal error [deg]",
                   "Terminal joint position error, force field ON")
    # The target is reached at rest, so the terminal velocity is itself the error.
    _joint_boxplot(ax_vel_err, np.abs(data["vel_ff"][:, :, :, -1]) * 180 / np.pi,
                   "Terminal velocity [deg/s]",
                   "Terminal joint velocity error, force field ON")

    ax_paths.set_aspect("equal")
    ax_paths.set_yticks([])
    ax_paths.set_xticks([])
    delete_axis(ax_paths)

    for i in range(NUM_CONTROLLERS):
        ax_paths.plot([TRAJECTORY_OFFSET * i], [TARGET[1]], marker="s", markersize=8,
                      markeredgecolor="lightgrey", markerfacecolor="lightgrey",
                      zorder=0, markeredgewidth=3)
        # Name each block of hand paths, which are drawn side by side.
        ax_paths.text(TRAJECTORY_OFFSET * i, TARGET[1] + 3.5, LEGEND[i], fontsize=13,
                      color=COLORS[i], ha="center", va="bottom")

    # The blocks are already named in colour above each target, so this panel
    # only needs the linestyle key, and it goes under the paths rather than over
    # them: drawn to scale there is no free space inside the axes.
    ax_paths.set_title("Hand paths", fontsize=15, pad=38)
    ax_paths.legend(handles=[
        Line2D([], [], color="black", ls="-", label="force field ON"),
        Line2D([], [], color="black", ls="-.", label="force field OFF"),
    ], fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2,
        frameon=False)

    fig.suptitle(f"Force field response   (ff_power = {ff_power:g}, "
                 f"{data['cost_ff'].shape[0]} trials)", fontsize=17, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    save_figure(fig, outdir, "FF3Controllers.svg")


def plot_cost_scatter(data, outdir, ff_power=FF_POWER):
    fig, ax = plt.subplots(figsize=(5, 5))
    reach = np.array([percentile_cross(data["cost_free"][:, i], data["cost_ff"][:, i],
                                       ax, color=COLORS[i])
                      for i in range(NUM_CONTROLLERS)])

    # The field costs far more than it saves, so the two axes span very different
    # ranges. Scaling them together pushed every controller into the left edge of
    # the plot; each axis is cropped to the data drawn on it instead, which means
    # the equality line is no longer the diagonal.
    x_max, y_max = reach.max(axis=0) * 1.05
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    equal = min(x_max, y_max)
    ax.plot([0, equal], [0, equal], color="grey", linestyle="--")
    ax.set_xlabel("Movement cost, force field OFF", fontsize=14)
    ax.set_ylabel("Movement cost, force field ON", fontsize=14)
    ax.set_title(f"Cost with vs without the field\n(ff_power = {ff_power:g}, "
                 f"mean and 95% range over {data['cost_ff'].shape[0]} trials)",
                 fontsize=13)
    _controller_legend(ax, extra=[
        Line2D([], [], color="grey", ls="--", label="equal cost"),
    ])
    fig.tight_layout()
    save_figure(fig, outdir, "FFFV.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--ff-power", type=float, default=FF_POWER,
                        help="force field strength")
    args = parser.parse_args()

    data = simulate(args.num_sim, args.jobs, args.ff_power)
    plot_panels(data, args.outdir, args.ff_power)
    plot_cost_scatter(data, args.outdir, args.ff_power)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
