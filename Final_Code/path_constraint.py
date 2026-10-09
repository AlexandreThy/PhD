"""
Effect of the straight-path cost on the feedback linearisation controller.

Compares FL with and without the path term on the two long movements, sweeps
the weight of that term from unconstrained to a straight hand path, and breaks
the movement cost into its four components. Also plots the muscle commands of
FL against those of ILQG.

    python Final_Code/path_constraint.py
    python Final_Code/path_constraint.py --num-sim 5 --jobs 1
    python Final_Code/path_constraint.py --wv-fl 100   # *_wvfl100.svg

--wv-fl is the terminal velocity weight FL optimises with (default WV = 1).
The path cost pulls the joints onto the line in the last few steps, so with
WV = 1 the straight paths end at 90-100 deg/s; at wv = 10 to 100 they stop at
under 15 deg/s with nearly the same straightening (see Final_Code/README.md).
--wc sets the path weight of the with/without comparison (default WC).
--velocities-only draws just the two angular velocity panels, with a legend:

    python Final_Code/path_constraint.py --velocities-only --wc 70 --wv-fl 100
"""

from matplotlib import gridspec
from matplotlib.lines import Line2D

from common import (
    FIGURE_SUBDIRS,
    TAU_PATH, WC, WP, WR, WR_FL, WV, build_parser, ToCartesian, compute_path,
    compute_angles_from_cartesian, delete_axis, delete_ticks, finish, guarded,
    get_colors_from_colormap, longmovement_1, longmovement_2, np, pi, plt,
    run_fl, run_ilqg, run_tasks, save_figure,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
MOVEMENT_BY_NUMBER = {1: longmovement_1, 2: longmovement_2}
# Progressive path weights, from unconstrained to straight. At TAU_PATH = 0.02
# the noiseless peak lateral deviation across these five values is 13.6 / 8.6 /
# 4.9 / 2.4 / 1.0 cm for the first long movement and 11.9 / 8.0 / 5.1 / 3.1 /
# 1.8 cm for the second, on a 58 cm reach.
WC_SWEEP = (0, 20, 40, 60, 80)

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
    # Controllers/FL.py now uses one joint-space matrix for every step, built at
    # simulate_FL's default percent = 0.75 of the way to the target.
    Qkwc = compute_path(start, np.asarray(target), wc, 0.75)
    decay = np.exp(-(NUM_ITER - 1 - np.arange(NUM_ITER)) * dt / TAU_PATH)

    states = x[:NUM_ITER]
    path = float(np.einsum("tj,jk,tk->t", states, Qkwc, states) @ decay)

    thetas, thetae, omegas, omegae = x[-1, :4]
    return (
        WP * (thetas - target1) ** 2 + WP * (thetae - target2) ** 2,
        WV * (omegas**2 + omegae**2),
        np.sum(u * u) * WR,
        path,
    )


def _worker(task):
    """FL without and with the straight-path cost, plus ILQG, for one repetition."""
    start, target, duration, num_iter, wc, wv_fl = task
    dt = duration / num_iter

    _, _, x_free, u_free = guarded(
        run_fl, "FL without path cost", duration, num_iter, start, target, wc=0,
        wv=wv_fl)
    _, _, x_path, u_path = guarded(
        run_fl, "FL with path cost", duration, num_iter, start, target, wc=wc,
        wv=wv_fl)
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


def simulate(movement, num_sim, jobs, wc=WC, wv_fl=WV):
    start, target = movement()
    tasks = [(start, target, MOVEMENT_TIME, NUM_ITER, wc, wv_fl)
             for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc=f"{movement.__name__} path cost")
    data = {key: np.array([r[key] for r in results]) for key in results[0]}
    return start, target, data


def _sweep_worker(task):
    """Mean trajectory for one path weight. Stays top-level."""
    start, target, duration, num_iter, wc, wv_fl = task
    X, Y, _, _ = run_fl(duration, num_iter, start, target, wc=wc, wv=wv_fl)
    return np.array([X, Y])


def sweep_wc(movement, num_sim, jobs, wv_fl=WV):
    """Mean FL trajectory for each path weight in WC_SWEEP."""
    start, target = movement()
    tasks = [(start, target, MOVEMENT_TIME, NUM_ITER, wc, wv_fl)
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


def plot_velocity_figure(movements, per_movement, wc, num_sim, outdir, suffix):
    """The two angular velocity panels alone, labelled."""
    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)
    fig, axes = plt.subplots(len(movements), 1, figsize=(8, 4.25 * len(movements)),
                             squeeze=False)
    for ax, movement, (_, _, data) in zip(axes[:, 0], movements, per_movement):
        plot_velocities(ax, data, time)
        ax.set_title(movement.__name__, fontsize=16)
        ax.set_ylabel("Angular velocity [deg/s]", fontsize=14)
    axes[-1, 0].set_xlabel("Time [ms]", fontsize=14)
    axes[0, 0].legend(handles=[
        Line2D([], [], color=FREE_COLOR, lw=2, label="FL, no path cost"),
        Line2D([], [], color=PATH_COLOR, lw=2, label=f"FL, path cost wc = {wc:g}"),
        Line2D([], [], color="black", ls="-", label="shoulder"),
        Line2D([], [], color="black", ls="--", label="elbow"),
    ], fontsize=11, loc="upper right", frameon=False)
    fig.suptitle(f"Joint angular velocity, mean +/- SD over {num_sim} trials",
                 fontsize=15)
    fig.tight_layout()
    save_figure(fig, outdir, f"PathConstraintVelocities{suffix}.svg", dpi=200)


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


def plot_commands(per_movement, outdir, suffix=""):
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

    save_figure(fig, outdir, f"PathConstraintCommands{suffix}.svg", dpi=200)


def main():
    parser = build_parser(__doc__, num_sim_default=100,
                          subdir=FIGURE_SUBDIRS["large_amplitude"])
    parser.add_argument("--movements", type=int, nargs="+", choices=(1, 2),
                        default=[1, 2],
                        help="which long movements to simulate (panel order)")
    parser.add_argument("--wc", type=float, default=WC,
                        help="path weight of the with/without comparison")
    parser.add_argument("--wv-fl", type=float, default=WV,
                        help="terminal velocity weight FL optimises with")
    parser.add_argument("--velocities-only", action="store_true",
                        help="only the angular velocity panels, as their own figure")
    args = parser.parse_args()
    # The defaults keep the historical file names.
    wc_tag = f"_wc{args.wc:g}"
    wv_tag = "" if args.wv_fl == WV else f"_wvfl{args.wv_fl:g}"
    suffix = ("" if args.wc == WC else wc_tag) + wv_tag

    movements = [MOVEMENT_BY_NUMBER[n] for n in args.movements]
    per_movement = [simulate(m, args.num_sim, args.jobs, args.wc, args.wv_fl)
                    for m in movements]
    if args.velocities_only:
        plot_velocity_figure(movements, per_movement, args.wc, args.num_sim,
                             args.outdir, wc_tag + wv_tag)
        finish(not args.no_show)
        return

    # One row per long movement for the velocity profiles and for the sweep.
    fig = plt.figure(figsize=(8, 21))
    gs = gridspec.GridSpec(5, 2)
    ax_vel = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
    ax_sweep = [fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])]
    ax_cost = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)

    for idx, (_, _, data) in enumerate(per_movement):
        plot_velocities(ax_vel[idx], data, time)
        plot_cost_breakdown(ax_cost[idx], data)

    for idx, movement in enumerate(movements):
        start, target, mean_trajectories = sweep_wc(movement, args.num_sim,
                                                    args.jobs, args.wv_fl)
        plot_sweep(ax_sweep[idx], start, target, mean_trajectories)

    save_figure(fig, args.outdir, f"PathConstraint{suffix}.svg", dpi=200)
    plot_commands(per_movement, args.outdir, suffix)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
