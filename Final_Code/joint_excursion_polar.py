"""
Joint excursion of the three controllers as a function of reach direction.

The companion of centerout_cost_polar.py, with the joint excursion in place of
the movement cost. The excursion of a joint is its range of motion over the
movement, max(theta) - min(theta); the figure shows the sum over shoulder and
elbow, mean +/- one SD over repetitions. The dashed grey curve is the joint
displacement from the start posture to the target posture (inverse kinematics),
the least excursion a movement that reaches the target can have.

LQG is linearised once at the start posture, so the excursion is how far each
movement takes the arm from the posture its model is exact at.

    python Final_Code/joint_excursion_polar.py                       # all conditions
    python Final_Code/joint_excursion_polar.py --amplitude 15 --duration 0.4
    python Final_Code/joint_excursion_polar.py --num-sim 5 --jobs 1   # quick check
    python Final_Code/joint_excursion_polar.py --replot   # redraw from saved data

The shoulder and elbow excursions are also saved separately in the .npz.
"""

from matplotlib.lines import Line2D

from centerout_cost_polar import (
    CONDITIONS, NUM_TARGETS, auto_radial_ticks, num_iter_for,
)
from common import (
    FIGURE_SUBDIRS,
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, centerout_targets,
    compute_angles_from_cartesian, finish, np, plt, run_fl, run_ilqg, run_lqg,
    run_tasks, save_figure, style_polar_axis,
)


def _worker(task):
    """Shoulder and elbow excursion [deg] of the three controllers, one repetition."""
    target, duration, num_iter, start = task
    runs = (
        run_ilqg(duration, num_iter, start, target),
        run_fl(duration, num_iter, start, target),
        run_lqg(duration, num_iter, start, target),
    )
    # Columns 0 and 1 of every controller's state are the shoulder and elbow angles
    return np.array([np.degrees(np.ptp(x[:, :2], axis=0)) for _, _, x, _ in runs])


def simulate(amplitude, duration, num_sim, jobs, start):
    """
    Returns:
        excursion : array (num_sim, NUM_TARGETS, NUM_CONTROLLERS, 2 joints) [deg]
    """
    targets = centerout_targets(start, amplitude, NUM_TARGETS)
    tasks = [
        (target, duration, num_iter_for(duration), start)
        for _ in range(num_sim)
        for target in targets
    ]
    results = run_tasks(
        _worker, tasks, jobs,
        desc=f"joint excursion {amplitude:g} cm / {duration*1000:.0f} ms",
    )
    return np.array(results).reshape(num_sim, NUM_TARGETS, NUM_CONTROLLERS, 2)


def target_displacement(amplitude, start):
    """Shoulder + elbow displacement [deg] from the start posture to each target posture."""
    start_angles = np.degrees(compute_angles_from_cartesian(*start))
    return np.array([
        np.sum(np.abs(np.degrees(compute_angles_from_cartesian(*target)) - start_angles))
        for target in centerout_targets(start, amplitude, NUM_TARGETS)
    ])


def plot(excursion, displacement, outdir, tag):
    """Mean summed excursion per direction with a +/- one SD band."""
    total = np.sum(excursion, axis=3)  # shoulder + elbow
    mean, sd = np.mean(total, axis=0), np.std(total, axis=0)

    # Close the polar curves by repeating the first direction at 2*pi.
    closed = np.vstack([mean, mean[0]])
    closed_sd = np.vstack([sd, sd[0]])
    angles = np.linspace(0, 2 * np.pi, NUM_TARGETS + 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for i in range(NUM_CONTROLLERS):
        ax.fill_between(angles, closed[:, i] - closed_sd[:, i],
                        closed[:, i] + closed_sd[:, i], color=COLORS[i],
                        alpha=0.2, linewidth=0)
        ax.plot(angles, closed[:, i], color=COLORS[i], linewidth=2.5)
    ax.plot(angles, np.append(displacement, displacement[0]), color="grey",
            linestyle="--", linewidth=2)

    rmax = (closed + closed_sd).max() * 1.05
    style_polar_axis(ax, auto_radial_ticks(rmax), NUM_TARGETS, rmax)
    ax.legend(handles=[
        *(Line2D([], [], color=c, lw=2.5, label=n) for c, n in zip(COLORS, LEGEND)),
        Line2D([], [], color="grey", ls="--", lw=2, label="start to target"),
    ], loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=12, frameon=False)
    ax.set_title("Shoulder + elbow excursion [deg]", fontsize=15, pad=30)

    save_figure(fig, outdir, f"JointExcursion_polar_{tag}.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=100,
                          subdir=FIGURE_SUBDIRS["nonlinearity_index"])
    parser.add_argument("--amplitude", type=float, default=None,
                        help="reach amplitude in cm (default: all conditions)")
    parser.add_argument("--duration", type=float, default=None,
                        help="movement duration in s (default: all conditions)")
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved JointExcursion_*.npz, no simulation")
    args = parser.parse_args()

    if args.amplitude is None and args.duration is None:
        conditions = CONDITIONS
    elif args.amplitude is not None and args.duration is not None:
        conditions = [(args.amplitude, args.duration)]
    else:
        parser.error("give both --amplitude and --duration, or neither")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for amplitude, duration in conditions:
        tag = f"{int(amplitude)}cm_{int(duration * 1000)}ms"
        path = args.outdir / f"JointExcursion_{tag}.npz"
        displacement = target_displacement(amplitude, START)
        if args.replot:
            excursion = np.load(path)["excursion"]
        else:
            excursion = simulate(amplitude, duration, args.num_sim, args.jobs, START)
            np.savez(path, excursion=excursion, displacement=displacement,
                     legend=np.array(LEGEND))
            print(f"wrote {path}", flush=True)
        plot(excursion, displacement, args.outdir, tag)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
