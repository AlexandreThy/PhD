"""
LQG against ILQG across the six center-out conditions (10 / 15 / 20 cm x 400 / 600 ms).

Reads the costs written by centerout_cost_polar.py (Cfy40_<amp>cm_<dur>ms_cost.npz,
start [0, 40]), so run that first; nothing is simulated here.

- CostRatio_vs_amplitude.svg: LQG cost / ILQG cost against amplitude, one line
  per duration. Each direction gives one ratio (dots); the line joins their
  geometric means. One direction can dominate an arithmetic mean (20 cm /
  400 ms, 90 deg: LQG 149 against ILQG 1.3), so the axis is logarithmic.
- CostVsExcursion_conditions.svg: the 48 (condition, direction) points pooled,
  LQG cost and LQG / ILQG ratio against joint, shoulder and elbow excursion.
  The fits are of log10(y), for the same reason; r2 is that of log10(y).
- CostVsExcursion_conditions_linear.svg: the same, with linear axes and fits of y.

    python Final_Code/centerout_conditions.py
"""

from matplotlib.lines import Line2D
from scipy import stats

from common import (
    FIGURE_DIR, FIGURE_SUBDIRS, START, build_parser, centerout_targets,
    delete_axis, finish, np, plt, save_figure,
)
from cost_vs_kinematics import joint_displacement

AMPLITUDES = (10, 15, 20)
DURATIONS = (0.4, 0.6)
NUM_DIRECTIONS = 8
ILQG, LQG = 0, 2  # columns of the stored costs (ILQG, FL, LQG)
DURATION_COLORS = {0.4: "#CC3311", 0.6: "#0077BB"}
AMPLITUDE_COLORS = {10: "#fdae6b", 15: "#e6550d", 20: "#7f2704"}
DURATION_MARKERS = {0.4: "o", 0.6: "s"}
COST_DIR = FIGURE_DIR / FIGURE_SUBDIRS["cost_polar"]


def load():
    """(amplitude, duration) -> (LQG cost, ILQG cost), each (direction,)."""
    data = {}
    for amp in AMPLITUDES:
        for dur in DURATIONS:
            saved = np.load(COST_DIR / f"Cfy40_{amp}cm_{dur * 1000:.0f}ms_cost.npz")
            data[amp, dur] = saved["total"][:, LQG], saved["total"][:, ILQG]
    return data


def plot_ratio_vs_amplitude(data, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
    for k, dur in enumerate(DURATIONS):
        color, offset = DURATION_COLORS[dur], (k - 0.5) * 0.6
        geo = []
        for amp in AMPLITUDES:
            lqg, ilqg = data[amp, dur]
            ratio = lqg / ilqg
            geo.append(stats.gmean(ratio))
            ax.scatter(np.full(NUM_DIRECTIONS, amp + offset), ratio, s=28, color=color,
                       alpha=0.45, linewidths=0)
        ax.plot(np.array(AMPLITUDES) + offset, geo, "-o", color=color, markersize=10,
                linewidth=2.5, label=f"{dur * 1000:.0f} ms")
    ax.axhline(1, color="grey", linestyle=":", linewidth=1, zorder=0)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], labels=["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xticks(AMPLITUDES, labels=[f"{a} cm" for a in AMPLITUDES])
    ax.set_xlabel("Movement amplitude", fontsize=18)
    ax.set_ylabel("LQG cost / ILQG cost", fontsize=18)
    ax.tick_params(labelsize=16)
    delete_axis(ax, sides=["top", "right"])
    ax.legend(title="Movement duration", fontsize=15, title_fontsize=15, frameon=False,
              loc="upper left")
    ax.set_title("Cost of linearising the arm\n(dots: 8 directions, line: geometric mean)",
                 fontsize=16)
    save_figure(fig, outdir, "CostRatio_vs_amplitude.svg")
    save_figure(fig, outdir, "CostRatio_vs_amplitude.pdf")


def pooled_points(data):
    """Per (condition, direction): amplitude, duration, |dtheta| (3,), LQG cost, ratio."""
    rows = []
    for (amp, dur), (lqg, ilqg) in data.items():
        for k, target in enumerate(centerout_targets(START, amp, NUM_DIRECTIONS)):
            ds, de = joint_displacement(target)
            rows.append((amp, dur, np.hypot(ds, de), abs(ds), abs(de), lqg[k], lqg[k] / ilqg[k]))
    return np.array(rows)


def plot_excursion(points, outdir, log=True):
    """log: fit log10(y) on log axes; otherwise fit y on linear axes."""
    tr = np.log10 if log else (lambda v: v)
    inv = (lambda v: 10 ** v) if log else (lambda v: v)
    amp, dur, ys = points[:, 0], points[:, 1], points[:, 5:]
    excursions = [("Joint excursion $\\|\\Delta\\theta\\|$ [deg]", points[:, 2]),
                  ("Shoulder excursion $|\\Delta\\theta_s|$ [deg]", points[:, 3]),
                  ("Elbow excursion $|\\Delta\\theta_e|$ [deg]", points[:, 4])]
    ylabels = ["LQG movement cost", "LQG cost / ILQG cost"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), layout="constrained")
    fits = {}
    for row, ylabel in enumerate(ylabels):
        y = ys[:, row]
        for col, (xlabel, x) in enumerate(excursions):
            ax = axes[row, col]
            for a in AMPLITUDES:
                for d in DURATIONS:
                    sel = (amp == a) & (dur == d)
                    ax.scatter(x[sel], y[sel], s=70, color=AMPLITUDE_COLORS[a],
                               marker=DURATION_MARKERS[d], edgecolors="black", linewidths=0.4)
            res = stats.linregress(x, tr(y))
            per_dur = [stats.linregress(x[dur == d], tr(y[dur == d])).rvalue ** 2
                       for d in DURATIONS]
            rho = stats.spearmanr(x, y)[0]
            fits[ylabel, xlabel] = (res.rvalue ** 2, res.pvalue, rho, per_dur)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, inv(res.slope * xs + res.intercept), "--", color="black",
                    linewidth=1.5, label=f"r$^2$ = {res.rvalue ** 2:.2f}")
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlabel(xlabel, fontsize=18)
            if col == 0:
                ax.set_ylabel(f"{ylabel} (log)" if log else ylabel, fontsize=18)
            ax.tick_params(labelsize=16)
            ax.legend(fontsize=17, loc="upper left", frameon=False)
            delete_axis(ax, sides=["top", "right"])

    handles = [Line2D([], [], marker="o", linestyle="", color=AMPLITUDE_COLORS[a],
                      markeredgecolor="black", markersize=12, label=f"{a} cm") for a in AMPLITUDES]
    handles += [Line2D([], [], marker=DURATION_MARKERS[d], linestyle="", color="white",
                       markeredgecolor="black", markersize=12, label=f"{d * 1000:.0f} ms")
                for d in DURATIONS]
    fig.legend(handles=handles, loc="outside right center", fontsize=17, frameon=False)
    fit_on = "log$_{10}$ y" if log else "y, linear axes"
    fig.suptitle("LQG cost and LQG / ILQG ratio against joint excursion, 6 conditions x "
                 f"8 directions (fit and r$^2$ on {fit_on})", fontsize=19)
    name = "CostVsExcursion_conditions" + ("" if log else "_linear")
    save_figure(fig, outdir, f"{name}.svg")
    save_figure(fig, outdir, f"{name}.pdf")
    return fits


def main():
    parser = build_parser(__doc__, subdir=FIGURE_SUBDIRS["nonlinearity_index"])
    args = parser.parse_args()
    data = load()

    plot_ratio_vs_amplitude(data, args.outdir)
    print("\nLQG / ILQG ratio per condition: geometric mean [min, max] over directions")
    for dur in DURATIONS:
        for amp in AMPLITUDES:
            r = data[amp, dur][0] / data[amp, dur][1]
            print(f"  {amp} cm / {dur * 1000:.0f} ms: {stats.gmean(r):5.2f}  [{r.min():.2f}, {r.max():.2f}]")

    points = pooled_points(data)
    for log in (True, False):
        fits = plot_excursion(points, args.outdir, log)
        print(f"\nPooled 48 points, fit of {'log10(y)' if log else 'y (linear)'}: r2 (p), "
              "Spearman rho, r2 at 400 / 600 ms only")
        for (ylabel, xlabel), (r2, p, rho, per_dur) in fits.items():
            print(f"  {ylabel:22s} vs {xlabel.split(' $')[0]:20s} r2 = {r2:.2f} (p = {p:.1g}), "
                  f"rho = {rho:+.2f}, r2 400 / 600 ms = {per_dur[0]:.2f} / {per_dur[1]:.2f}")
    np.savez(args.outdir / "CostVsExcursion_conditions.npz",
             columns=np.array(["amplitude", "duration", "joint_excursion", "shoulder_excursion",
                               "elbow_excursion", "lqg_cost", "ratio"]), points=points)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
