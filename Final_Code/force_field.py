"""
Response of the three controllers to a velocity-dependent force field.

Every repetition is simulated twice, with and without the force field, so the
two conditions share the same set of noise realisations. Produces the panel
figure (trajectories, endpoint error, velocity profiles) and the scatter of the
movement cost with against without the field.

    python Final_Code/force_field.py
    python Final_Code/force_field.py --num-sim 5 --jobs 1
"""

from common import (
    COLORS, Compute_Cartesian_Speed, Cost_function, LEGEND, NUM_CONTROLLERS,
    build_parser, compute_angles_from_cartesian, delete_axis, finish, np, pi,
    plt, run_dlqg, run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
START = [0, 45]
TARGET = [0, 60]
FF_POWER = 4e-4
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

    out = {
        "cost_ff": np.array([Cost_function(x, u, tg=target) for _, _, x, u in ff_runs]),
        "cost_free": np.array([Cost_function(x, u, tg=target) for _, _, x, u in free_runs]),
        "vel_ff": np.array([x[:, 2:4].T for _, _, x, _ in ff_runs]),
        "vel_free": np.array([x[:, 2:4].T for _, _, x, _ in free_runs]),
        "pos_error": np.array([np.abs(x[-1, :2] - target_angles) for _, _, x, _ in ff_runs]),
        "traj_ff": np.array([[X, Y] for X, Y, _, _ in ff_runs]),
        "traj_free": np.array([[X, Y] for X, Y, _, _ in free_runs]),
    }

    cart = [Compute_Cartesian_Speed(x.T) for _, _, x, _ in ff_runs]
    out["cart_xy"] = np.array([[vx, vy] for vx, vy, _ in cart])
    out["cart_abs"] = np.array([speed for _, _, speed in cart])
    return out


def simulate(num_sim, jobs, ff_power=FF_POWER):
    tasks = [(MOVEMENT_TIME, NUM_ITER, START, TARGET, ff_power)
             for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc="force field")
    return {key: np.array([r[key] for r in results]) for key in results[0]}


def percentile_cross(x, y, ax, p=95, color="black"):
    """Mean marker with percentile whiskers on both axes."""
    x, y = np.asarray(x), np.asarray(y)
    alpha = (100 - p) / 2
    x_low, x_high = np.percentile(x, [alpha, 100 - alpha])
    y_low, y_high = np.percentile(y, [alpha, 100 - alpha])
    mx, my = np.mean(x), np.mean(y)

    ax.plot([x_low, x_high], [my, my], color=color, linewidth=1.5)
    ax.plot([mx, mx], [y_low, y_high], color=color, linewidth=1.5)
    ax.scatter(mx, my, marker="o", color=color, s=30)


def _style_velocity_axis(ax):
    delete_axis(ax, sides=["top", "right"])
    ax.set_xlabel("Movement Duration [ms]", fontsize=15)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_ylabel("Angular Velocity [deg/sec]", fontsize=15)
    ax.set_yticks([-180, 0, 180])
    ax.tick_params(labelsize=14)


def _style_cartesian_axis(ax):
    delete_axis(ax, sides=["top", "right"])
    ax.set_xlabel("Movement Duration [ms]", fontsize=15)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])
    ax.set_ylabel("Velocity [cm/sec]", fontsize=15)
    ax.set_yticks([-20, 0, 20])
    ax.tick_params(labelsize=14)


def plot_panels(data, outdir):
    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)
    fig, ax = plt.subplots(12, figsize=(5, 40))

    shown = min(data["traj_ff"].shape[0], MAX_TRAJECTORIES_SHOWN)
    for rep in range(shown):
        for i in range(NUM_CONTROLLERS):
            offset = TRAJECTORY_OFFSET * i
            ax[0].plot(data["traj_ff"][rep, i, 0] + offset,
                       data["traj_ff"][rep, i, 1],
                       color=COLORS[i], linewidth=0.6, linestyle="--")
            ax[0].plot(data["traj_free"][rep, i, 0] + offset,
                       data["traj_free"][rep, i, 1],
                       color="grey", linewidth=0.6, linestyle="-.")

    # Angular velocity: force field in ax[2], no force field in ax[11]
    for idx, key in enumerate(["vel_ff", "vel_free"]):
        axis = ax[idx * 9 + 2]
        for i in range(NUM_CONTROLLERS):
            for joint, linestyle in enumerate(["-", "--"]):
                trace = data[key][:, i, joint] * 180 / pi
                mean_vel, std_vel = np.mean(trace, axis=0), np.std(trace, axis=0)
                axis.plot(time, mean_vel, color=COLORS[i], linestyle=linestyle)
                axis.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                                  color=COLORS[i], alpha=0.3)
        _style_velocity_axis(axis)

    # Cartesian speed magnitude, then the x and y components
    for axis, trace_set in ((ax[7], data["cart_abs"]),
                            (ax[8], data["cart_xy"][:, :, 0]),
                            (ax[9], data["cart_xy"][:, :, 1])):
        for i in range(NUM_CONTROLLERS):
            mean_vel = np.mean(trace_set[:, i], axis=0)
            std_vel = np.std(trace_set[:, i], axis=0)
            axis.plot(time, mean_vel, color=COLORS[i])
            axis.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                              color=COLORS[i], alpha=0.3)
        _style_cartesian_axis(axis)

    # Per-joint terminal error, grouped by joint
    for joint in range(2):
        box = ax[1].boxplot(data["pos_error"][:, :, joint] * 180 / np.pi,
                            positions=[joint * 3 + 1, joint * 3 + 2, joint * 3 + 3],
                            patch_artist=True, medianprops=dict(color="black"),
                            whiskerprops=dict(color="black"),
                            capprops=dict(color="black"), showfliers=False)
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)

    terminal_speed = data["cart_abs"][:, :, -1]
    terminal_error = np.sqrt(data["traj_ff"][:, :, 0, -1] ** 2
                             + (data["traj_ff"][:, :, 1, -1] - TARGET[1]) ** 2)
    terminal_ang_vel = np.abs(data["vel_ff"][:, :, :, -1])

    boxes = [
        ax[3].boxplot(terminal_error, patch_artist=True, showfliers=False,
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"), capprops=dict(color="black")),
        ax[4].boxplot(terminal_speed, patch_artist=True, showfliers=False,
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"), capprops=dict(color="black")),
        ax[5].boxplot(terminal_ang_vel[:, :, 0] * 180 / np.pi, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"), capprops=dict(color="black")),
        ax[6].boxplot(terminal_ang_vel[:, :, 1] * 180 / np.pi, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"), capprops=dict(color="black")),
    ]
    for box in boxes:
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)

    ax[0].set_aspect("equal")
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    delete_axis(ax[0])
    for i in (1, 3, 4):
        delete_axis(ax[i], sides=["top", "right", "bottom"])

    for i, label in ((1, "Movement Cost"),
                     (3, "Terminal Endpoint Error [cm]"),
                     (4, "Terminal Endpoint Velocity [cm/sec]")):
        ax[i].set_xticks([])
        ax[i].tick_params(axis="y", labelsize=14)
        ax[i].set_ylabel(label, fontsize=15)
    for i in (5, 6):
        ax[i].set_xticks([])
        ax[i].tick_params(axis="y", labelsize=14)

    for i in range(NUM_CONTROLLERS):
        ax[0].plot([TRAJECTORY_OFFSET * i], [TARGET[1]], marker="s", markersize=8,
                   markeredgecolor="grey", markerfacecolor="grey", zorder=0,
                   markeredgewidth=3)

    save_figure(fig, outdir, "FF3Controllers.svg")


def plot_cost_scatter(data, outdir):
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(NUM_CONTROLLERS):
        percentile_cross(data["cost_free"][:, i], data["cost_ff"][:, i],
                         ax, color=COLORS[i])
    limit = max(np.max(data["cost_free"]), np.max(data["cost_ff"]))
    ax.plot([0, limit], [0, limit], color="grey", linestyle="--")
    ax.set_xlabel("Cost without FF", fontsize=14)
    ax.set_ylabel("Cost with FF", fontsize=14)
    save_figure(fig, outdir, "FFFV.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--ff-power", type=float, default=FF_POWER,
                        help="force field strength")
    args = parser.parse_args()

    data = simulate(args.num_sim, args.jobs, args.ff_power)
    plot_panels(data, args.outdir)
    plot_cost_scatter(data, args.outdir)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
