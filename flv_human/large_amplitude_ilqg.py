"""
ILQG on the two large amplitude movements, macaque vs human muscle parameters.

Movements and settings are Final_Code/large_amplitude_reaching.py's: 600 ms,
60 steps, motor noise and 60 ms sensory delay. The human set is human_params.py's,
linearised at the centre-out start; along the straight joint-space path of both
long movements the linearised fibre lengths stay within 0.023 L0 of the exact
Nijhof & Kouwenhoven ones.

    python flv_human/large_amplitude_ilqg.py               # 10 trials per movement
    python flv_human/large_amplitude_ilqg.py --replot

Writes flv_human/figures/long_move_ilqg_macaque_vs_human.svg and its .npz.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Final_Code"))

from human_params import apply  # noqa: E402  (before common: sets sys.path)
from common import (  # noqa: E402
    build_parser, cost_components, delete_axis, finish, guarded,
    longmovement_1, longmovement_2, np, plt, run_ilqg, run_tasks, save_figure,
)
from matplotlib.lines import Line2D  # noqa: E402

MOVEMENT_TIME, NUM_ITER = 0.6, 60
# Row order of Final_Code/large_amplitude_reaching.py: movement 2, then 1.
MOVEMENTS = {2: longmovement_2, 1: longmovement_1}
SETS = ("macaque", "human")
SET_COLORS = {"macaque": "#0072B2", "human": "#D55E00"}
NAME = "long_move_ilqg_macaque_vs_human"


def _worker(task):
    """One noisy ILQG trial. Stays top-level for the process pool."""
    params, number = task
    apply(params)
    start, target = MOVEMENTS[number]()
    X, Y, x, u = guarded(run_ilqg, f"ILQG on movement {number} ({params})",
                         MOVEMENT_TIME, NUM_ITER, start, target)
    return np.array([X, Y]), np.sum(cost_components(x, u, tg=target))


def simulate(num_sim, jobs):
    tasks = [(p, m) for p in SETS for m in MOVEMENTS for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc="ILQG long movements")
    shape = (len(SETS), len(MOVEMENTS), num_sim)
    paths = np.array([r[0] for r in results])
    return {"paths": paths.reshape(shape + paths.shape[1:]),   # (..., xy, time)
            "cost": np.array([r[1] for r in results]).reshape(shape)}


def plot(data, outdir):
    fig, axes = plt.subplots(1, len(MOVEMENTS), figsize=(16, 5.5))
    for m, (number, movement) in enumerate(MOVEMENTS.items()):
        ax = axes[m]
        start, target = movement()
        for s, name in enumerate(SETS):
            for trial in data["paths"][s, m]:
                ax.plot(*trial, color=SET_COLORS[name], lw=0.6, alpha=0.5)
            ax.plot(*data["paths"][s, m].mean(0), color=SET_COLORS[name], lw=2.5)
        ax.add_patch(plt.Circle(target, 1.5, color="grey"))
        ax.add_patch(plt.Rectangle((start[0] - 1.5, start[1] - 1.5), 3, 3, color="grey"))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        delete_axis(ax)
        ax.set_title(f"Movement {number}", fontsize=16)
        ax.legend(handles=[
            Line2D([], [], color=SET_COLORS[n], lw=3,
                   label=f"{n} muscles (median cost {np.median(data['cost'][s, m]):.2f})")
            for s, n in enumerate(SETS)], fontsize=12, loc="best", frameon=False)
    fig.suptitle(f"ILQG, large amplitude reaching ({MOVEMENT_TIME * 1000:.0f} ms, noise "
                 f"and 60 ms delay, {data['cost'].shape[-1]} trials, mean in bold)",
                 fontsize=15)
    fig.tight_layout()
    save_figure(fig, outdir, f"{NAME}.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=10)
    parser.set_defaults(outdir=HERE / "figures")
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved .npz, no simulation")
    args = parser.parse_args()
    path = args.outdir / f"{NAME}.npz"

    if args.replot:
        data = dict(np.load(path))
    else:
        data = simulate(args.num_sim, args.jobs)
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
        print(f"wrote {path}", flush=True)

    for s, name in enumerate(SETS):
        for m, number in enumerate(MOVEMENTS):
            c = data["cost"][s, m]
            print(f"{name:8s} movement {number}: cost median {np.median(c):.2f}, "
                  f"range {c.min():.2f}-{c.max():.2f}")
    plot(data, args.outdir)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
