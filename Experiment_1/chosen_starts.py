"""
The two starting points for the experiment, and their hand trajectories.

Selection: among the starts whose posture and all eight targets are comfortable
(COMFORT, per amplitude), the pair whose noiseless FL trajectories differ most,
by the RMS of trajectory_rms.py averaged over the eight directions. Done for each
amplitude. Movement times, weights and segment lengths of trajectory_rms.py /
FL.py.

25 cm reaches in eight directions do not fit in the default comfortable postures
(elbow 30 to 140 deg): only four nearly identical starts do. For 25 cm the elbow
range is extended to 15 to 150 deg.

Figures (FIGURES): the pair selected for one amplitude, drawn at that amplitude
and the smaller ones, since a pair that allows the largest reach is the one an
experiment with all of them would use.

    python Experiment_1/chosen_starts.py

Writes, in Experiment_1/figures/: ChosenStarts_trajectories.svg (20 cm pair),
ChosenStarts_trajectories_25cm.svg (25 cm pair) and ChosenStarts.npz
(candidates, their mean RMS matrix and the selected pairs).
"""

import matplotlib.pyplot as plt
import numpy as np

import trajectory_rms as base
from FL import compute_angles_from_cartesian

OUTDIR = base.OUTDIR

AMPLITUDES = (10, 20, 25)
# Comfortable postures [deg] of the starts and the targets: (elbow, shoulder).
DEFAULT_COMFORT = ((30, 140), (-20, 110))
COMFORT = {10: DEFAULT_COMFORT, 20: DEFAULT_COMFORT, 25: ((15, 150), (-20, 110))}
# (pair of this amplitude, drawn at these amplitudes, file name)
FIGURES = ((20, (10, 20), "ChosenStarts_trajectories.svg"),
           (25, (10, 20, 25), "ChosenStarts_trajectories_25cm.svg"))
R_STEP, PHI_STEP = 1.0, 4.0  # candidate grid [cm, deg]
PHI_RANGE = (20, 160)
COLORS = ("#0081a7", "#f07167")


def comfortable(point, comfort=DEFAULT_COMFORT):
    (elbow_lo, elbow_hi), (shoulder_lo, shoulder_hi) = comfort
    ts, te = np.degrees(compute_angles_from_cartesian(*point))
    return (np.isfinite(ts) and elbow_lo <= te <= elbow_hi
            and shoulder_lo <= ts <= shoulder_hi)


def reach_vectors(amplitude):
    return [amplitude * np.array([np.cos(d * np.pi / 4), np.sin(d * np.pi / 4)])
            for d in range(base.NUM_DIRECTIONS)]


def select_pair(amplitude):
    """Admissible starts, their RMS matrices (direction, start, start) and the best pair."""
    vectors, comfort = reach_vectors(amplitude), COMFORT[amplitude]
    radii = np.arange(abs(base.L2 - base.L1), base.L1 + base.L2 + R_STEP, R_STEP)
    angles = np.arange(PHI_RANGE[0], PHI_RANGE[1] + PHI_STEP, PHI_STEP)
    grid = [r * np.array([np.cos(np.radians(f)), np.sin(np.radians(f))])
            for r in radii for f in angles]
    starts, paths = [], []
    for p in grid:
        if not (comfortable(p, comfort) and all(comfortable(p + v, comfort) for v in vectors)):
            continue
        try:
            paths.append([base.displacement(p, p + v, amplitude) for v in vectors])
            starts.append(p)
        except np.linalg.LinAlgError:
            pass  # FL fails at this posture
    starts, paths = np.array(starts), np.array(paths)  # paths: (start, dir, time, xy)
    rms = np.array([base.pairwise_rms(paths[:, d]) for d in range(base.NUM_DIRECTIONS)])
    best = np.unravel_index(np.argmax(rms.mean(axis=0)), rms.shape[1:])
    return starts, rms, best


def posture(point):
    return np.degrees(compute_angles_from_cartesian(*point))


def pair_rms(pair, amplitude):
    """RMS between the two starts' displacements, per direction."""
    return np.array([np.sqrt(np.mean(np.sum(
        (base.displacement(pair[0], pair[0] + v, amplitude)
         - base.displacement(pair[1], pair[1] + v, amplitude))**2, axis=1)))
        for v in reach_vectors(amplitude)])


def comfortable_region(ax, comfort):
    """Shade the hand positions reachable with a comfortable posture."""
    (elbow_lo, elbow_hi), (shoulder_lo, shoulder_hi) = comfort
    reach = base.L1 + base.L2
    x, y = np.meshgrid(np.linspace(-reach, reach, 500), np.linspace(-0.3 * reach, reach, 400))
    with np.errstate(invalid="ignore"):
        ts, te = np.degrees(compute_angles_from_cartesian(x, y))
    inside = ((ts >= shoulder_lo) & (ts <= shoulder_hi)
              & (te >= elbow_lo) & (te <= elbow_hi)).astype(float)
    ax.contourf(x, y, inside, levels=[0.5, 1.5], colors=["0.92"], zorder=0)


def plot(pair, amplitudes, comfort, filename, pair_amplitude):
    fig, axes = plt.subplots(1, len(amplitudes), figsize=(7 * len(amplitudes), 6.5),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    drawn = [np.zeros(2)]  # every point drawn, to fit the window to all of it
    for ax, amplitude in zip(axes, amplitudes):
        comfortable_region(ax, comfort)
        for name, start, color in zip("AB", pair, COLORS):
            ts, te = posture(start)
            elbow = base.L1 * np.array([np.cos(np.radians(ts)), np.sin(np.radians(ts))])
            ax.plot([0, elbow[0], start[0]], [0, elbow[1], start[1]], color=color,
                    linewidth=2, alpha=0.6)
            drawn += [elbow, start]
            for v in reach_vectors(amplitude):
                path = start + base.displacement(start, start + v, amplitude)
                ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2)
                drawn += list(path) + [start + v]
                ax.plot(*(start + v), "o", markersize=7, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=1.5, zorder=3)
            ax.plot(*start, "o", color=color, markersize=10, zorder=4,
                    label=f"{name} ({start[0]:.1f}, {start[1]:.1f}) cm: "
                          f"shoulder {ts:.0f}, elbow {te:.0f} deg")
            ax.annotate(name, start, textcoords="offset points", xytext=(-14, 8),
                        fontsize=13, color=color, fontweight="bold")
        ax.plot(0, 0, "o", color="grey", markersize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("Hand x [cm]")
        ax.set_ylabel("Hand y [cm]")
        ax.set_title(f"{amplitude} cm reaches in {base.DURATIONS[amplitude] * 1000:.0f} ms "
                     f"(mean RMS {pair_rms(pair, amplitude).mean():.2f} cm)", fontsize=13)
        ax.legend(fontsize=9, loc="lower left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    # One window for all panels, fitted to everything drawn with an 8 cm margin.
    drawn = np.array(drawn)
    low, high = drawn.min(axis=0) - 8, drawn.max(axis=0) + 8
    for ax in axes:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    (elbow_lo, elbow_hi), (shoulder_lo, shoulder_hi) = comfort
    fig.suptitle(f"Noiseless FL hand trajectories from the pair chosen for {pair_amplitude} cm "
                 f"(grey: comfortable postures, elbow {elbow_lo}-{elbow_hi}, shoulder "
                 f"{shoulder_lo}-{shoulder_hi} deg)", fontsize=13)
    path = OUTDIR / filename
    fig.savefig(path)
    print(f"wrote {path}")
    return fig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    saved, pairs = {}, {}
    for amplitude in AMPLITUDES:
        starts, rms, (a, b) = select_pair(amplitude)
        mean = rms.mean(axis=0)
        pairs[amplitude] = (starts[a], starts[b])
        dist = np.linalg.norm(starts, axis=1)
        (elbow_lo, elbow_hi), _ = COMFORT[amplitude]
        print(f"\n{amplitude} cm ({base.DURATIONS[amplitude] * 1000:.0f} ms, elbow "
              f"{elbow_lo}-{elbow_hi} deg): {len(starts)} admissible starts "
              f"({dist.min():.0f} to {dist.max():.0f} cm from the shoulder)")
        for name, k in zip("AB", (a, b)):
            ts, te = posture(starts[k])
            print(f"  {name}: ({starts[k][0]:6.1f}, {starts[k][1]:5.1f}) cm, "
                  f"shoulder {ts:5.1f}, elbow {te:5.1f} deg")
        print(f"  mean RMS over directions {mean[a, b]:.2f} cm; per direction: "
              + " ".join(f"{d * 45}:{rms[d, a, b]:.2f}" for d in range(base.NUM_DIRECTIONS)))
        # The mean over directions only (single precision): the full per-direction
        # matrices of every candidate pair would take tens of MB.
        saved.update({f"starts_{amplitude}cm": starts,
                      f"mean_rms_{amplitude}cm": mean.astype(np.float32),
                      f"pair_{amplitude}cm": np.array([starts[a], starts[b]]),
                      f"pair_rms_{amplitude}cm": rms[:, a, b]})
    np.savez(OUTDIR / "ChosenStarts.npz", **saved)

    figs = []
    for pair_amplitude, amplitudes, filename in FIGURES:
        pair = pairs[pair_amplitude]
        comfort = COMFORT[pair_amplitude]
        for amplitude in amplitudes:
            r = pair_rms(pair, amplitude)
            fits = all(comfortable(p + v, COMFORT[amplitude])
                       for p in pair for v in reach_vectors(amplitude))
            print(f"\n{pair_amplitude} cm pair at {amplitude} cm: mean RMS {r.mean():.2f} cm"
                  f"{'' if fits else ' (some targets outside the ' + str(amplitude) + ' cm comfort limits)'}"
                  f"; per direction: " + " ".join(f"{d * 45}:{x:.2f}" for d, x in enumerate(r)))
        figs.append(plot(pair, amplitudes, comfort, filename, pair_amplitude))
    return figs


if __name__ == "__main__":
    main()
    plt.show()
