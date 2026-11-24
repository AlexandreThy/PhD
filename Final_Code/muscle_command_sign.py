"""
How often each controller sends a negative muscle command.

The muscle model scales each command by its force-length and force-velocity
gains and sums it through the moment arms, with no rectification, so a negative
command makes a muscle push. This script counts, over every trial, time step and
muscle, the share of negative and positive commands of the three controllers in
four tasks:

- center-out, 15 cm / 400 ms from [0, 40], 8 directions;
- the force field reach of force_field.py, with the field on;
- the two large amplitude movements of large_amplitude_reaching.py.

For ILQG the command counted is the one applied in the noisy rollout
(feedforward + feedback correction, `run_ilqg(..., applied=True)`), as FL and
LQG return, not ILQG's feedforward sequence alone.

    python Final_Code/muscle_command_sign.py
    python Final_Code/muscle_command_sign.py --num-sim 5 --jobs 1
    python Final_Code/muscle_command_sign.py --replot
"""

import warnings

from matplotlib.patches import Patch

from common import (
    FIGURE_SUBDIRS,
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, centerout_targets,
    delete_axis, finish, guarded, longmovement_1, longmovement_2, np, plt,
    run_fl, run_ilqg, run_lqg, run_tasks, save_figure,
)
from force_field import FF_POWER, MOVEMENT_TIME as FF_TIME, NUM_ITER as FF_ITER, \
    START as FF_START, TARGET as FF_TARGET

MUSCLES = ["shoulder\nflexor", "shoulder\nextensor", "elbow\nflexor",
           "elbow\nextensor", "biarticular\nflexor", "biarticular\nextensor"]
NUM_MUSCLES = len(MUSCLES)


def _tasks():
    """name -> list of (start, target, duration, num_iter, ff_power) movements."""
    long1, long2 = longmovement_1(), longmovement_2()
    return {
        "Center-out\n15 cm / 400 ms":
            [(START, t, 0.4, 40, 0.0) for t in centerout_targets(START, 15, 8)],
        "Force field ON":
            [(FF_START, FF_TARGET, FF_TIME, FF_ITER, FF_POWER)],
        "Large amplitude\nmovement 1":
            [(long1[0], long1[1], 0.6, 60, 0.0)],
        "Large amplitude\nmovement 2":
            [(long2[0], long2[1], 0.6, 60, 0.0)],
    }


def _worker(task):
    """Commands (controller, time, muscle) of the three controllers for one movement."""
    name, (start, target, duration, num_iter, ff_power) = task
    ff = ff_power != 0
    label = name.replace("\n", " ")
    # ILQG's line search tries steps whose muscle length goes negative (NaN from
    # l**1.55) and rejects them; the warnings say nothing about the result.
    warnings.simplefilter("ignore", RuntimeWarning)
    runs = (
        guarded(run_ilqg, f"ILQG, {label}", duration, num_iter, start, target,
                ff=ff, ff_power=ff_power, applied=True),
        guarded(run_fl, f"FL, {label}", duration, num_iter, start, target,
                ff=ff, ff_power=ff_power),
        guarded(run_lqg, f"LQG, {label}", duration, num_iter, start, target,
                ff=ff, ff_power=ff_power),
    )
    return np.array([u for _, _, _, u in runs])


def simulate(num_sim, jobs):
    """name -> commands (movement, controller, time, muscle), movements = trials x directions."""
    commands = {}
    for name, movements in _tasks().items():
        tasks = [(name, m) for _ in range(num_sim) for m in movements]
        commands[name] = np.array(run_tasks(_worker, tasks, jobs, desc=name.replace("\n", " ")))
    return commands


def negative_share(u):
    """Share of negative commands, (controller,) overall and (controller, muscle)."""
    neg = u < 0  # (movement, controller, time, muscle)
    return neg.mean(axis=(0, 2, 3)), neg.mean(axis=(0, 2))


def negative_magnitude_share(u):
    """Share of the summed |u| carried by negative commands, (controller,)."""
    return np.abs(np.minimum(u, 0)).sum(axis=(0, 2, 3)) / np.abs(u).sum(axis=(0, 2, 3))


def plot_overall(names, shares, magnitude, num_sim, outdir):
    """One stacked positive / negative bar per controller, grouped by task."""
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    width, span = 0.8, NUM_CONTROLLERS + 1
    for t, name in enumerate(names):
        for i in range(NUM_CONTROLLERS):
            pos = t * span + i
            neg = shares[name][i]
            ax.bar(pos, neg, width, color=COLORS[i], edgecolor="black", linewidth=0.6)
            ax.bar(pos, 1 - neg, width, bottom=neg, color=COLORS[i], alpha=0.3,
                   edgecolor="black", linewidth=0.6)
            ax.text(pos, neg - 0.02, f"{100 * neg:.0f}%", ha="center", va="top",
                    fontsize=10, color="white", fontweight="bold")
            ax.plot([pos - width / 2, pos + width / 2], [magnitude[name][i]] * 2,
                    color="black", linewidth=2)

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, zorder=0)
    ax.set_xticks([t * span + (NUM_CONTROLLERS - 1) / 2 for t in range(len(names))],
                  labels=names, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    ax.set_ylabel("Share of muscle commands [%]", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    delete_axis(ax, sides=["top", "right"])
    handles = [Patch(facecolor=c, edgecolor="black", label=n) for c, n in zip(COLORS, LEGEND)]
    handles += [Patch(facecolor="grey", edgecolor="black", label="negative (solid)"),
                Patch(facecolor="grey", alpha=0.3, edgecolor="black", label="positive (light)"),
                plt.Line2D([], [], color="black", lw=2, label="share of $\\sum|u|$ that is negative")]
    ax.legend(handles=handles, fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False)
    ax.set_title("Sign of the muscle commands, over every trial, time step and muscle "
                 f"({num_sim} trials per movement)", fontsize=13)
    save_figure(fig, outdir, "MuscleCommandSign.svg")


def plot_per_muscle(names, per_muscle, outdir):
    """Share of negative commands per muscle, one panel per task."""
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)),
                             sharex=True, layout="constrained")
    width = 0.8 / NUM_CONTROLLERS
    x = np.arange(NUM_MUSCLES)
    for ax, name in zip(axes, names):
        for i in range(NUM_CONTROLLERS):
            ax.bar(x + (i - 1) * width, per_muscle[name][i], width, color=COLORS[i],
                   edgecolor="black", linewidth=0.5, label=LEGEND[i])
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, zorder=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1], labels=["0", "50", "100"])
        ax.set_ylabel("Negative [%]", fontsize=12)
        ax.set_title(name.replace("\n", ", "), fontsize=12, loc="left")
        delete_axis(ax, sides=["top", "right"])
    axes[0].legend(fontsize=10, loc="upper right", ncol=3, frameon=False,
                   bbox_to_anchor=(1, 1.25))
    axes[-1].set_xticks(x, labels=MUSCLES, fontsize=11)
    fig.suptitle("Share of negative commands per muscle", fontsize=13)
    save_figure(fig, outdir, "MuscleCommandSign_per_muscle.svg")
    save_figure(fig, outdir, "MuscleCommandSign_per_muscle.pdf")


def main():
    parser = build_parser(__doc__, num_sim_default=50, subdir=FIGURE_SUBDIRS["muscle_commands"])
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved .npz instead of simulating")
    args = parser.parse_args()
    data_path = args.outdir / "MuscleCommandSign.npz"

    if args.replot:
        saved = np.load(data_path)
        names = list(saved["names"])
        commands = {n: saved[f"commands_{k}"] for k, n in enumerate(names)}
    else:
        commands = simulate(args.num_sim, args.jobs)
        names = list(commands)
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez(data_path, names=np.array(names), muscles=np.array(MUSCLES),
                 controllers=np.array(LEGEND),
                 **{f"commands_{k}": commands[n] for k, n in enumerate(names)})

    num_movements = {n: len(m) for n, m in _tasks().items()}
    num_sim = commands[names[0]].shape[0] // num_movements[names[0]]
    shares, per_muscle, magnitude = {}, {}, {}
    for name in names:
        shares[name], per_muscle[name] = negative_share(commands[name])
        magnitude[name] = negative_magnitude_share(commands[name])

    plot_overall(names, shares, magnitude, num_sim, args.outdir)
    plot_per_muscle(names, per_muscle, args.outdir)

    print("\nShare of negative commands (count | share of sum |u|):")
    for name in names:
        row = "   ".join(f"{c} {100 * s:4.1f}% | {100 * m:4.1f}%"
                         for c, s, m in zip(LEGEND, shares[name], magnitude[name]))
        print(f"  {name.replace(chr(10), ' '):30s} {row}")
    print("\nPer muscle (negative %), " + ", ".join(m.replace("\n", " ") for m in MUSCLES))
    for name in names:
        print(f"  {name.replace(chr(10), ' ')}")
        for c, v in zip(LEGEND, per_muscle[name]):
            print(f"    {c:5s} " + " ".join(f"{100 * x:5.1f}" for x in v))
    finish(not args.no_show)


if __name__ == "__main__":
    main()
