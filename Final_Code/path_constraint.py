"""
Effect of the straight-path cost on the feedback linearisation controller.

Compares FL with and without the path term on the two long movements, sweeps
the weight of that term from unconstrained to a straight hand path, and breaks
the movement cost into its four components. Also plots the muscle commands of
FL against those of ILQG.

    python Final_Code/path_constraint.py
    python Final_Code/path_constraint.py --num-sim 5 --jobs 1
"""

from matplotlib import gridspec

from common import (
    TAU_PATH, WC, WP, WR, WR_FL, WV, build_parser, ToCartesian, compute_path,
    compute_angles_from_cartesian, delete_axis, delete_ticks, finish, guarded,
    get_colors_from_colormap, longmovement_1, longmovement_2, np, pi, plt,
    run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
MOVEMENT_BY_NUMBER = {1: longmovement_1, 2: longmovement_2}
# Progressive path weights, from unconstrained to straight. At TAU_PATH = 0.15
# the peak lateral deviation across these five values is 13.6 / 8.0 / 4.7 / 2.2
# / 0.8 cm for the first long movement and 11.9 / 4.1 / 1.3 / 0.5 / 0.6 cm for
# the second, on a 58 cm reach.
WC_SWEEP = (0, 0.002, 0.005, 0.01, 0.1)

FREE_COLOR = "#0081a7"  # FL without the straight-path cost
PATH_COLOR = "#f07167" # FL with the straight-path cost
COMPONENT_COLORS = np.array(["#0081a7", "#00afb9", "#14C25C", "#B90072"])
HAND_COLORS = np.array(["#0081a7", "#00afb9", "#b67dc3", "#fed9b7","#f07167"])
MUSCLE_COLORS = np.array(["#C78624", "#14445F", "#14C25C", "#B90072",
                          "#C72424", "#40CAD4"])


def cost_components(x, u, dt, wc, target):
    """Split the movement cost into (position, velocity, motor, path)."""
    target1, target2 = compute_angles_from_cartesian(target[0], target[1])
    start = np.array(ToCartesian(x[0, 0], x[0, 1]))
    # The controller's own cost matrices, so this scores what it optimised. It
    # weights step t by exp(-(NUM_ITER - 1 - t) * dt / TAU_PATH), i.e. the decay
    # runs backwards from the end of the movement.
    Qkwc = compute_path(start, np.asarray(target), wc, NUM_ITER)
    decay = np.exp(-(NUM_ITER - 1 - np.arange(NUM_ITER)) * dt / TAU_PATH)

    states = x[:NUM_ITER]
    path = float(np.einsum("tj,tjk,tk->t", states, Qkwc, states) @ decay)

    thetas, thetae, omegas, omegae = x[-1, :4]
    return (
        WP * (thetas - target1) ** 2 + WP * (thetae - target2) ** 2,
        WV * (omegas**2 + omegae**2),
        np.sum(u * u) * WR,
        path,
    )


def _worker(task):
    """FL without and with the straight-path cost, plus ILQG, for one repetition."""
    start, target, duration, num_iter, wc = task
    dt = duration / num_iter

    _, _, x_free, u_free = guarded(
        run_fl, "FL without path cost", duration, num_iter, start, target, wc=0)
    _, _, x_path, u_path = guarded(
        run_fl, "FL with path cost", duration, num_iter, start, target, wc=wc)
    _, _, _, u_ilqg = guarded(
        run_ilqg, "ILQG", duration, num_iter, start, target)

    return {
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
    """Mean trajectory for one path weight. Stays top-level."""
    start, target, duration, num_iter, wc = task
    X, Y, _, _ = run_fl(duration, num_iter, start, target, wc=wc)
    return np.array([X, Y])


def sweep_wc(movement, num_sim, jobs):
    """Mean FL trajectory for each path weight in WC_SWEEP."""
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
    colors = HAND_COLORS
    for idx in range(len(WC_SWEEP)):
        ax.plot(mean_trajectories[idx, 0], mean_trajectories[idx, 1],
                color=colors[idx], linewidth=4)
    _mark_endpoints(ax, start, target)


def plot_cost_breakdown(ax, data):
    """Cost components without the path term (left) and with it (right)."""
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

    # One row per long movement for the velocity profiles and for the sweep.
    fig = plt.figure(figsize=(8, 21))
    gs = gridspec.GridSpec(5, 2)
    ax_vel = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
    ax_sweep = [fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])]
    ax_cost = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)

    movements = [MOVEMENT_BY_NUMBER[n] for n in args.movements]
    per_movement = [simulate(m, args.num_sim, args.jobs) for m in movements]

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
