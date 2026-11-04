"""
Centre-out 15 cm / 400 ms, LQG vs ILQG, with the macaque (WP1 as is) and the
human muscle parameters of human_params.py.

Settings are Final_Code/common.py's: start (0, 40) cm, 8 targets, 40 steps,
WP = 20000, WV = 10, WR = 0.1, motor noise 5e-4, sensory delay 60 ms. Every
trial is noisy. Both controllers are scored with Cost_function, as in
Final_Code/centerout_cost_polar.py.

    python flv_human/lqg_vs_ilqg.py                     # 30 trials per target
    python flv_human/lqg_vs_ilqg.py --num-sim 2 --jobs 1   # quick check
    python flv_human/lqg_vs_ilqg.py --replot             # redraw from the .npz

Writes in flv_human/figures/: trajectories_15cm_400ms.svg,
cost_distributions_15cm_400ms.svg, lqg_vs_ilqg_15cm_400ms.npz and a summary
table lqg_vs_ilqg_15cm_400ms.txt.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Final_Code"))

from human_params import apply, describe  # noqa: E402  (before common: sets sys.path)
from common import (  # noqa: E402
    COLORS, START, build_parser, centerout_targets, cost_components, finish,
    guarded, np, plt, run_ilqg, run_lqg, run_tasks, save_figure,
)

AMPLITUDE, DURATION, NUM_ITER, NUM_TARGETS = 15, 0.4, 40, 8
SETS = ("macaque", "human")
CONTROLLERS = ("ILQG", "LQG")
COLOR = {"ILQG": COLORS[0], "LQG": COLORS[2]}
RUNNERS = {"ILQG": run_ilqg, "LQG": run_lqg}
NAME = f"{AMPLITUDE}cm_{int(DURATION * 1000)}ms"


def _worker(task):
    """One noisy trial of both controllers. Stays top-level for the process pool."""
    params, target = task
    apply(params)  # each worker process starts from WP1's macaque constants
    out = []
    for c in CONTROLLERS:
        X, Y, x, u = guarded(RUNNERS[c], f"{c} ({params})", DURATION, NUM_ITER,
                             START, target, noise=True)
        out.append((np.asarray(X), np.asarray(Y), np.array(cost_components(x, u, tg=target))))
    return out


def simulate(num_sim, jobs):
    targets = centerout_targets(START, AMPLITUDE, NUM_TARGETS)
    tasks = [(p, t) for p in SETS for _ in range(num_sim) for t in targets]
    results = run_tasks(_worker, tasks, jobs, desc=f"LQG vs ILQG {NAME}")

    shape = (len(SETS), num_sim, NUM_TARGETS, len(CONTROLLERS))
    hand = np.array([[np.stack([X, Y], -1) for X, Y, _ in r] for r in results])
    comps = np.array([[c for _, _, c in r] for r in results])
    return {
        "hand": hand.reshape(shape + hand.shape[-2:]),  # (..., time, xy) [cm]
        "components": comps.reshape(shape + (3,)),     # position, velocity, motor
        "targets": np.array(targets),
    }


def summary(data):
    """Per set and controller: cost quantiles, endpoint error, cost components."""
    hand, comps, targets = data["hand"], data["components"], data["targets"]
    total = comps.sum(-1)
    error = np.linalg.norm(hand[..., -1, :] - targets[None, None, :, None, :], axis=-1)
    lines = [describe(), "",
             f"{NAME}, {total.shape[1]} noisy trials x {NUM_TARGETS} targets per cell",
             "set      controller  median  mean    q10     q90    | position velocity motor | end error [cm]"]
    for s, name in enumerate(SETS):
        for c, ctrl in enumerate(CONTROLLERS):
            t = total[s, ..., c].ravel()
            pos, vel, mot = comps[s, ..., c, :].reshape(-1, 3).mean(0)
            lines.append(
                f"{name:8s} {ctrl:10s} {np.median(t):7.2f} {t.mean():7.2f} "
                f"{np.quantile(t, .1):7.2f} {np.quantile(t, .9):7.2f} | "
                f"{pos:8.2f} {vel:8.2f} {mot:6.2f} | "
                f"{error[s, ..., c].mean():.2f} +- {error[s, ..., c].std():.2f}"
            )
        ratio = total[s, ..., 1] / total[s, ..., 0]
        lines.append(f"{'':8s} LQG / ILQG cost per trial: median {np.median(ratio):.2f}, "
                     f"LQG cheaper in {np.mean(ratio < 1) * 100:.0f} % of trials")
    return "\n".join(lines)


def plot_trajectories(data, outdir):
    hand, targets = data["hand"], data["targets"]
    fig, axes = plt.subplots(len(SETS), len(CONTROLLERS), figsize=(9, 9),
                             sharex=True, sharey=True)
    for s, name in enumerate(SETS):
        for c, ctrl in enumerate(CONTROLLERS):
            ax = axes[s, c]
            for k in range(NUM_TARGETS):
                for trial in hand[s, :, k, c]:
                    ax.plot(*trial.T, color=COLOR[ctrl], lw=0.5, alpha=0.25)
                ax.plot(*hand[s, :, k, c].mean(0).T, color=COLOR[ctrl], lw=2)
            ax.scatter(*targets.T, s=140, facecolors="none", edgecolors="k", lw=1.2, zorder=3)
            ax.plot(*START, "k+", ms=10)
            ax.set_aspect("equal")
            ax.set_title(f"{ctrl}, {name} muscles", fontsize=13)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[s, 0].set_ylabel("y [cm]")
    for ax in axes[-1]:
        ax.set_xlabel("x [cm]")
    fig.suptitle(f"Hand paths, {AMPLITUDE} cm / {DURATION * 1000:.0f} ms, noise and "
                 f"60 ms delay ({hand.shape[1]} trials per target, mean in bold)")
    fig.tight_layout()
    save_figure(fig, outdir, f"trajectories_{NAME}.svg")


def plot_costs(data, outdir):
    """Pooled cost distribution per set (left) and per direction (right)."""
    total = data["components"].sum(-1)
    lo, hi = np.log10(total.min()) - 0.1, np.log10(total.max()) + 0.1
    bins = np.logspace(lo, hi, 40)
    fig, axes = plt.subplots(len(SETS), 2, figsize=(13, 8),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    for s, name in enumerate(SETS):
        ax = axes[s, 0]
        for c, ctrl in enumerate(CONTROLLERS):
            t = total[s, ..., c].ravel()
            ax.hist(t, bins=bins, color=COLOR[ctrl], alpha=0.55,
                    label=f"{ctrl} (median {np.median(t):.2f})")
            ax.axvline(np.median(t), color=COLOR[ctrl], lw=2)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylabel(f"{name} muscles\ntrials")
        ax.legend(frameon=False)

        ax = axes[s, 1]
        width = 0.35
        for c, ctrl in enumerate(CONTROLLERS):
            pos = np.arange(NUM_TARGETS) + (c - 0.5) * width
            bp = ax.boxplot([total[s, :, k, c] for k in range(NUM_TARGETS)],
                            positions=pos, widths=width * 0.9, patch_artist=True,
                            showfliers=False, medianprops={"color": "k"})
            for box in bp["boxes"]:
                box.set(facecolor=COLOR[ctrl], alpha=0.6, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(NUM_TARGETS),
                      [f"{d:.0f}°" for d in np.linspace(0, 360, NUM_TARGETS + 1)[:-1]])
        ax.set_ylim(bins[0], bins[-1])
        for a in axes[s]:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("total cost (log)")
    axes[-1, 1].set_xlabel("reach direction")
    axes[0, 1].set_title("total cost per direction (box: quartiles, whiskers 1.5 IQR)")
    axes[0, 0].set_title("all directions pooled")
    fig.tight_layout()
    save_figure(fig, outdir, f"cost_distributions_{NAME}.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=30)
    parser.set_defaults(outdir=HERE / "figures")
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved .npz, no simulation")
    args = parser.parse_args()
    path = args.outdir / f"lqg_vs_ilqg_{NAME}.npz"

    if args.replot:
        data = dict(np.load(path))
    else:
        data = simulate(args.num_sim, args.jobs)
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)
        print(f"wrote {path}", flush=True)

    text = summary(data)
    print(text)
    (args.outdir / f"lqg_vs_ilqg_{NAME}.txt").write_text(text + "\n", encoding="utf-8")
    plot_trajectories(data, args.outdir)
    plot_costs(data, args.outdir)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
