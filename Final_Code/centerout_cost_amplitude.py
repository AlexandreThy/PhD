"""
Movement cost of the three controllers as a function of reach amplitude.

Reads the per-direction costs written by centerout_cost_polar.py and reduces the
eight directions of each condition to one number: either the worst direction
(maximum of the mean cost) or the average over directions. One trace per
(controller, movement duration): colour is the controller, line style the
duration.

    python Final_Code/centerout_cost_polar.py          # writes the costs first
    python Final_Code/centerout_cost_amplitude.py
    python Final_Code/centerout_cost_amplitude.py --no-show

Each statistic gives three figures: all three controllers, ILQG and FL alone
(*_ILQG_FL.svg) on a scale fitted to those two, and all three controllers on a
logarithmic cost axis (*_log.svg).
"""

from matplotlib.ticker import FuncFormatter

from centerout_cost_polar import condition_name
from common import (
    FIGURE_DIR, FIGURE_SUBDIRS,
    COLORS, LEGEND, NUM_CONTROLLERS, START, build_parser, finish, np, plt,
    save_figure,
)

AMPLITUDES = (10, 15, 20)  # cm
DURATIONS = (0.4, 0.6)  # s
LINESTYLES = {0.4: "-", 0.6: "--"}
MARKERS = {0.4: "o", 0.6: "s"}
# How the eight directions of a condition are reduced to one cost
STATISTICS = {
    "max": (np.max, "Maximum over directions of the mean cost"),
    "mean": (np.mean, "Mean cost over directions"),
}
WITHOUT_LQG = (0, 1)  # ILQG, FL
COST_DIR = FIGURE_DIR / FIGURE_SUBDIRS["cost_polar"]


def load_costs(cost_dir=COST_DIR, start=START):
    """
    Mean cost per direction of every condition.

    Returns:
        costs : array (len(DURATIONS), len(AMPLITUDES), 8 directions, NUM_CONTROLLERS)
    """
    costs = []
    for duration in DURATIONS:
        row = []
        for amplitude in AMPLITUDES:
            saved = np.load(cost_dir / f"{condition_name(amplitude, duration, start)}_cost.npz")
            row.append(saved["total"])
        costs.append(row)
    return np.array(costs)


def plot(costs, statistic, outdir, controllers=tuple(range(NUM_CONTROLLERS)), suffix="",
         log=False):
    """With `log`, the cost axis is logarithmic and the file ends in _log.svg."""
    reduce, ylabel = STATISTICS[statistic]
    # Reduce over the direction axis: (durations, amplitudes, controllers)
    summary = reduce(costs, axis=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in controllers:
        for d, duration in enumerate(DURATIONS):
            ax.plot(AMPLITUDES, summary[d, :, i], color=COLORS[i],
                    linestyle=LINESTYLES[duration], marker=MARKERS[duration],
                    markersize=8, linewidth=2,
                    label=f"{LEGEND[i]}, {duration * 1000:.0f} ms")

    ax.set_xticks(AMPLITUDES)
    ax.set_xlabel("Movement amplitude [cm]")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        # Plain numbers (1, 10, 100) rather than powers of ten
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        suffix += "_log"
    else:
        ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    save_figure(fig, outdir, f"CostVsAmplitude_{statistic}{suffix}.svg")


def main():
    parser = build_parser(__doc__, subdir=FIGURE_SUBDIRS["cost_polar"])
    args = parser.parse_args()

    costs = load_costs()
    for statistic in STATISTICS:
        plot(costs, statistic, args.outdir)
        plot(costs, statistic, args.outdir, controllers=WITHOUT_LQG, suffix="_ILQG_FL")
        plot(costs, statistic, args.outdir, log=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
