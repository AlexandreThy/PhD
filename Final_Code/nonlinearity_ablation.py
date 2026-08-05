"""
Which arm nonlinearity does ILQG need to cope with a force field?

Ported from CurrentParts/Nonlinearities.ipynb. The same reach is run four
times, each with one or two nonlinearities of the arm removed, and the removal
applies to the simulated arm as well as to the model ILQG optimises against --
so this asks what the movement costs when a nonlinearity is *absent*, not what
it costs to mis-model one.

The weights, the noise and the force field strength come from common.py and
force_field.py, so this sits on the same scale as the other figures. The
notebook used WR = 0.5 and its own copy of ILQG, whose force-length curve
rose with muscle length instead of falling; both are corrected here.

    python Final_Code/nonlinearity_ablation.py
    python Final_Code/nonlinearity_ablation.py --num-sim 5 --jobs 1
"""

from matplotlib.lines import Line2D

from common import (
    Compute_Cartesian_Speed, Cost_function, Plant, build_parser,
    compute_angles_from_cartesian, delete_axis, finish, guarded, np, pi, plt,
    run_ilqg, run_tasks, save_figure,
)
from force_field import FF_POWER

MOVEMENT_TIME = 0.6
NUM_ITER = 60
START = [0, 40]
TARGET = [0, 55]

# The four conditions of the notebook. Coriolis stays nonlinear throughout;
# --linear-coriolis switches it off in every condition instead.
CONDITIONS = (
    ("Linear muscle", dict(muscle=False)),
    ("Linear inertia", dict(inertia=False)),
    ("Linear muscle + inertia", dict(muscle=False, inertia=False)),
    ("Full nonlinearities", dict()),
)
COLORS = ["#E63946", "#4ECDC4", "#8380B6", "#009E73"]
# For panels with room to name every box on the axis instead of in a legend.
SHORT_LABELS = ("Lin.\nmuscle", "Lin.\ninertia", "Lin.\nboth", "Full\nNL")


def _worker(task):
    """One repetition of every condition, sharing nothing but the parameters."""
    duration, num_iter, start, target, ff_power, linear_coriolis = task
    target_angles = np.array(compute_angles_from_cartesian(target[0], target[1]))

    runs = []
    for label, flags in CONDITIONS:
        plant = Plant(coriolis=not linear_coriolis, **flags)
        runs.append(guarded(run_ilqg, f"ILQG ({label})", duration, num_iter,
                            start, target, ff=True, ff_power=ff_power, plant=plant))

    return {
        "traj": np.array([[X, Y] for X, Y, _, _ in runs]),
        "joint_vel": np.array([x[:, 2:4].T for _, _, x, _ in runs]),
        "speed": np.array([Compute_Cartesian_Speed(x.T)[2] for _, _, x, _ in runs]),
        "pos_error": np.array([np.abs(x[-1, :2] - target_angles)
                               for _, _, x, _ in runs]),
        "cost": np.array([Cost_function(x, u, tg=target) for _, _, x, u in runs]),
    }


def simulate(num_sim, jobs, ff_power, linear_coriolis):
    tasks = [(MOVEMENT_TIME, NUM_ITER, START, TARGET, ff_power, linear_coriolis)
             for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc="nonlinearity ablation")
    return {key: np.array([r[key] for r in results]) for key in results[0]}


def _condition_legend(ax, loc="lower left", extra=()):
    handles = [Line2D([], [], color=c, lw=3, label=name)
               for c, (name, _) in zip(COLORS, CONDITIONS)]
    handles.extend(extra)
    ax.legend(handles=handles, fontsize=10, loc=loc, frameon=False)


def _boxplot(ax, values, ylabel, title):
    """One box per condition, named on the axis rather than in a legend."""
    box = ax.boxplot(values, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black"),
                     whiskerprops=dict(color="black"),
                     capprops=dict(color="black"))
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)

    ax.set_xticks(np.arange(len(CONDITIONS)) + 1, labels=SHORT_LABELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    delete_axis(ax, sides=["top", "right", "bottom"])


def _joint_boxplot(ax, values, ylabel, title):
    """One box per condition, in a shoulder group and an elbow group."""
    for joint in range(2):
        offset = joint * (len(CONDITIONS) + 1)
        box = ax.boxplot(values[:, :, joint], patch_artist=True, showfliers=False,
                         positions=np.arange(len(CONDITIONS)) + offset,
                         medianprops=dict(color="black"),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"))
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)

    centre = (len(CONDITIONS) - 1) / 2
    ax.set_xticks([centre, centre + len(CONDITIONS) + 1])
    ax.set_xticklabels(["shoulder", "elbow"], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    delete_axis(ax, sides=["top", "right", "bottom"])
    _condition_legend(ax, loc="upper left")


def plot_panels(data, outdir, ff_power):
    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)
    fig, axes = plt.subplots(7, figsize=(6, 26), layout="constrained")
    ax_traj, ax_vel, ax_pos, ax_jvel, ax_end, ax_speed, ax_cost = axes

    mean_traj = np.mean(data["traj"], axis=0)
    for i, (name, _) in enumerate(CONDITIONS):
        ax_traj.plot(mean_traj[i, 0], mean_traj[i, 1], color=COLORS[i],
                     linewidth=4, label=name)
    ax_traj.plot(TARGET[0], TARGET[1], marker="s", markersize=16,
                 markerfacecolor="lightgrey", markeredgecolor="lightgrey", zorder=0)
    ax_traj.set_aspect("equal")
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])
    ax_traj.set_title("Mean hand path", fontsize=13)
    delete_axis(ax_traj)
    # Drawn to scale the paths leave no free space, so the key goes underneath.
    ax_traj.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 0),
                   ncol=2, frameon=False)

    for i in range(len(CONDITIONS)):
        for joint, linestyle in enumerate(["-", "--"]):
            trace = data["joint_vel"][:, i, joint] * 180 / pi
            mean_vel, std_vel = np.mean(trace, axis=0), np.std(trace, axis=0)
            ax_vel.plot(time, mean_vel, color=COLORS[i], linestyle=linestyle)
            ax_vel.fill_between(time, mean_vel - std_vel, mean_vel + std_vel,
                                color=COLORS[i], alpha=0.3)
    ax_vel.plot([0, MOVEMENT_TIME * 1000], [0, 0], color="grey", linewidth=1)
    ax_vel.set_xlabel("Time [ms]", fontsize=13)
    ax_vel.set_ylabel("Angular velocity [deg/s]", fontsize=13)
    ax_vel.set_xticks([0, 100, 200, 300, 400, 500, 600])
    ax_vel.set_title("Joint angular velocity", fontsize=13)
    ax_vel.tick_params(labelsize=12)
    delete_axis(ax_vel, sides=["top", "right"])
    _condition_legend(ax_vel, loc="upper left", extra=[
        Line2D([], [], color="black", ls="-", label="shoulder"),
        Line2D([], [], color="black", ls="--", label="elbow"),
    ])

    _joint_boxplot(ax_pos, data["pos_error"] * 180 / np.pi,
                   "Terminal error [deg]", "Terminal joint position error")
    # The target is reached at rest, so the terminal velocity is itself the error.
    _joint_boxplot(ax_jvel, np.abs(data["joint_vel"][:, :, :, -1]) * 180 / np.pi,
                   "Terminal velocity [deg/s]", "Terminal joint velocity error")

    endpoint_error = np.hypot(data["traj"][:, :, 0, -1] - TARGET[0],
                              data["traj"][:, :, 1, -1] - TARGET[1])
    _boxplot(ax_end, endpoint_error, "Terminal error [cm]",
             "Distance to target at movement end")
    _boxplot(ax_speed, data["speed"][:, :, -1], "Terminal speed [cm/s]",
             "Hand speed at movement end")
    _boxplot(ax_cost, data["cost"], "Movement cost", "Total movement cost")

    fig.suptitle(f"Cost of removing an arm nonlinearity\n(ILQG under the force "
                 f"field, ff_power = {ff_power:g}, "
                 f"{data['cost'].shape[0]} trials)", fontsize=15)
    save_figure(fig, outdir, "NonlinearityAblation.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--ff-power", type=float, default=FF_POWER,
                        help="force field strength (sign sets the side)")
    parser.add_argument("--linear-coriolis", action="store_true",
                        help="also drop the Coriolis and centrifugal torques")
    args = parser.parse_args()

    data = simulate(args.num_sim, args.jobs, args.ff_power, args.linear_coriolis)
    plot_panels(data, args.outdir, args.ff_power)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
