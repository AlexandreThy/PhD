"""
Effect of the via-path cost on the feedback linearisation controller.

Compares FL with and without the via-path term on the two long movements,
sweeps the weight of that term, and breaks the movement cost into its four
components. Also plots the muscle commands of FL against those of ILQG.

    python Final_Code/path_constraint.py
    python Final_Code/path_constraint.py --num-sim 5 --jobs 1
"""

from matplotlib import gridspec

from common import (
    PERCENT, TAU_PATH, WC, WP, WR, WR_FL, WV, build_parser,
    compute_angles_from_cartesian, delete_axis, delete_ticks, finish, guarded,
    get_colors_from_colormap, longmovement_1, longmovement_2, np, pi, plt,
    run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
MAX_TRAJECTORIES_SHOWN = 15
MOVEMENTS = (longmovement_1, longmovement_2)
MOVEMENT_BY_NUMBER = {1: longmovement_1, 2: longmovement_2}
# Progressive via-path weights. Each step visibly straightens the hand path:
# the peak lateral deviation of the long movements drops by roughly
# 0 / 11 / 25 / 44 / 55 percent across these five values at TAU_PATH = 0.6.
WC_SWEEP = (0, 0.1, 0.3, 1, 3)

FREE_COLOR = "#24C7C7"  # FL without the via-path cost
PATH_COLOR = "#0072B2"  # FL with the via-path cost
COMPONENT_COLORS = np.array(["#C78624", "#14445F", "#14C25C", "#B90072"])
MUSCLE_COLORS = np.array(["#C78624", "#14445F", "#14C25C", "#B90072",
                          "#C72424", "#40CAD4"])


def ToCartesian(s, e):
    """Hand position from the shoulder and elbow angles."""
    return np.cos(s + e) * 33 + np.cos(s) * 30, np.sin(s + e) * 33 + np.sin(s) * 30


def compute_path(x0, xf, wc, percent):
    """
    Via-path cost matrix penalising deviation from the straight line to xf.

    Mirrors Controllers.FL.compute_path so the cost breakdown below scores the
    same quantity the controller optimised.
    """
    xtarget = x0 + percent * (xf - x0)
    ts, te = compute_angles_from_cartesian(xtarget[0], xtarget[1])
    ts0, te0 = compute_angles_from_cartesian(x0[0], x0[1])
    k = (te - te0) / (ts - ts0)

    Qk = np.zeros((8, 8))
    Qk[0, 0] = Qk[6, 6] = k * k
    Qk[0, 6] = Qk[6, 0] = -k * k
    Qk[1, 1] = Qk[7, 7] = 1
    Qk[1, 7] = Qk[7, 1] = -1
    Qk[0, 1] = Qk[1, 0] = -k
    Qk[0, 7] = Qk[7, 0] = k
    Qk[1, 6] = Qk[6, 1] = k
    Qk[6, 7] = Qk[7, 6] = -k
    return Qk * wc


def cost_components(x, u, dt, wc, target):
    """Split the movement cost into (position, velocity, motor, via-path)."""
    target1, target2 = compute_angles_from_cartesian(target[0], target[1])
    start = np.array(ToCartesian(x[0, 0], x[0, 1]))
    Qkwc = compute_path(start, np.asarray(target), wc, PERCENT)

    decay = np.exp(-np.arange(NUM_ITER) * dt / TAU_PATH)
    # sum_i x_i' Qkwc x_i * decay_i, as one contraction over timesteps
    states = x[:NUM_ITER]
    path = float(np.sum(np.einsum("ij,jk,ik->i", states, Qkwc, states) * decay))

    thetas, thetae, omegas, omegae = x[-1, :4]
    return (
        WP * (thetas - target1) ** 2 + WP * (thetae - target2) ** 2,
        WV * (omegas**2 + omegae**2),
        np.sum(u * u) * WR,
        path,
    )


def _worker(task):
    """FL without and with the via-path cost, plus ILQG, for one repetition."""
    start, target, duration, num_iter, wc = task
    dt = duration / num_iter

    X_free, Y_free, x_free, u_free = guarded(
        run_fl, "FL without via-path cost", duration, num_iter, start, target, wc=0)
    X_path, Y_path, x_path, u_path = guarded(
        run_fl, "FL with via-path cost", duration, num_iter, start, target, wc=wc)
    _, _, _, u_ilqg = guarded(
        run_ilqg, "ILQG", duration, num_iter, start, target)

    return {
        "traj_free": np.array([X_free, Y_free]),
        "traj_path": np.array([X_path, Y_path]),
        "vel_free": x_free[:, 2:4].T,
        "vel_path": x_path[:, 2:4].T,
        "cmd_free": u_free.T,
        "cmd_path": u_path.T,
        "cmd_ilqg": u_ilqg.T,
        "cost_free": np.array(cost_components(x_free, u_free, dt, 0, target)),
        "cost_path": np.array(cost_components(x_path, u_path, dt, wc, target)),
    }


def simulate(movement, num_sim, jobs, wc=WC):
    start, target = movement()
    tasks = [(start, target, MOVEMENT_TIME, NUM_ITER, wc) for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc=f"{movement.__name__} path cost")
    data = {key: np.array([r[key] for r in results]) for key in results[0]}
    return start, target, data


def _sweep_worker(task):
    """Mean trajectory for one via-path weight. Stays top-level."""
    start, target, duration, num_iter, wc = task
    X, Y, _, _ = run_fl(duration, num_iter, start, target, wc=wc)
    return np.array([X, Y])


def sweep_wc(movement, num_sim, jobs):
    """Mean FL trajectory for each via-path weight in WC_SWEEP."""
    start, target = movement()
    tasks = [(start, target, MOVEMENT_TIME, NUM_ITER, wc)
             for wc in WC_SWEEP for _ in range(num_sim)]
    results = run_tasks(_sweep_worker, tasks, jobs,
                        desc=f"{movement.__name__} wc sweep")
    trajectories = np.array(results).reshape(len(WC_SWEEP), num_sim, 2, NUM_ITER + 1)
    return start, target, np.mean(trajectories, axis=1)


def _mark_endpoints(ax, start, target):
    ax.add_patch(plt.Circle((start[0], start[1]), 1.5, edgecolor="grey",
                            facecolor="grey", linewidth=3))
    ax.add_patch(plt.Rectangle((target[0] - 1.5, target[1] - 1.5), 3, 3,
                               edgecolor="grey", facecolor="grey", linewidth=3))
    delete_ticks(ax)
    delete_axis(ax)
    ax.set_aspect("equal")


def plot_trajectories(ax, per_movement):
    for start, target, data in per_movement:
        shown = min(data["traj_free"].shape[0], MAX_TRAJECTORIES_SHOWN)
        for rep in range(shown):
            ax.plot(data["traj_free"][rep, 0], data["traj_free"][rep, 1],
                    color=FREE_COLOR, linewidth=1)
            ax.plot(data["traj_path"][rep, 0], data["traj_path"][rep, 1],
                    color=PATH_COLOR, linewidth=1)
        _mark_endpoints(ax, start, target)


def plot_velocities(ax, data, time):
    for key, color in (("vel_free", FREE_COLOR), ("vel_path", PATH_COLOR)):
        for joint, linestyle in enumerate(["-", "--"]):
            trace = data[key][:, joint] * 180 / pi
            mean_vel, std_vel = np.mean(trace, axis=0), np.std(trace, axis=0)
            ax.plot(time, mean_vel, color=color, linestyle=linestyle)
            ax.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                            color=color, alpha=0.3)

    ax.plot(time, np.zeros(len(time)), color="black")
    delete_axis(ax, sides=["top", "right"])
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600],
                  labels=[0, "", "", "", "", "", 600])
    ticks = [-360, -270, -180, -90, 0, 90, 180, 270, 360]
    ax.set_yticks(ticks, labels=ticks)
    ax.tick_params(labelsize=20)


def plot_sweep(ax, start, target, mean_trajectories):
    colors = get_colors_from_colormap(len(WC_SWEEP), "plasma")
    for idx in range(len(WC_SWEEP)):
        ax.plot(mean_trajectories[idx, 0], mean_trajectories[idx, 1],
                color=colors[idx], linewidth=4)
    _mark_endpoints(ax, start, target)


def plot_cost_breakdown(ax, data):
    """Cost components without the via-path term (left) and with it (right)."""
    free = np.mean(data["cost_free"], axis=0)
    path = np.mean(data["cost_path"], axis=0)
    ax.bar(np.arange(0, 4), free, color=COMPONENT_COLORS)
    ax.bar(np.arange(6, 10), path, color=COMPONENT_COLORS)
    for value in free:
        ax.plot(np.linspace(-1, 10, 10), np.ones(10) * value, color="black",
                linestyle="--", linewidth=0.5)
    ax.set_yscale("log")


def plot_commands(per_movement, outdir):
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER)
    limits = [(-3, 1.5), (-6, 2)]

    for row, (_, _, data) in enumerate(per_movement):
        for col, key in enumerate(["cmd_free", "cmd_path", "cmd_ilqg"]):
            mean_cmd = np.mean(data[key], axis=0)
            for muscle in range(6):
                ax[row, col].plot(time, mean_cmd[muscle],
                                  color=MUSCLE_COLORS[muscle])
            ax[row, col].set_ylim(*limits[row])

    save_figure(fig, outdir, "PathConstraintCommands.svg", dpi=200)


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--movements", type=int, nargs="+", choices=(1, 2),
                        default=[1, 2],
                        help="which long movements to simulate (panel order)")
    args = parser.parse_args()

    fig = plt.figure(figsize=(8, 21))
    gs = gridspec.GridSpec(5, 2)
    ax_traj = fig.add_subplot(gs[0, :])
    ax_vel = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_sweep = [fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])]
    ax_cost = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)

    movements = [MOVEMENT_BY_NUMBER[n] for n in args.movements]
    per_movement = [simulate(m, args.num_sim, args.jobs) for m in movements]
    plot_trajectories(ax_traj, per_movement)

    for idx, (_, _, data) in enumerate(per_movement):
        plot_velocities(ax_vel[idx], data, time)
        plot_cost_breakdown(ax_cost[idx], data)

    for idx, movement in enumerate(movements):
        start, target, mean_trajectories = sweep_wc(movement, args.num_sim, args.jobs)
        plot_sweep(ax_sweep[idx], start, target, mean_trajectories)

    save_figure(fig, args.outdir, "PathConstraint.svg", dpi=200)
    plot_commands(per_movement, args.outdir)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
