"""
Pairwise trajectory RMS at 500 ms, and a refined search for the pair of starting
points whose hand trajectories differ the most.

1. The eight-point RMS matrices of trajectory_rms.py (movement time of each
   amplitude in trajectory_rms.DURATIONS).

2. For each reach direction and amplitude, a coarse-to-fine search over starting
   points for the pair that maximises the RMS distance between their hand
   displacements (same definition as trajectory_rms.py). The search space is
   the hand position in polar coordinates around the shoulder, inside the region
   where every 10 and 20 cm target is reachable (base.R_MIN to base.R_MAX from
   the shoulder, set by the segment lengths of FL.py),
   over angles 60 to 140 deg:
     - a coarse 7 x 7 grid, every pair compared;
     - the best few pairs as seeds;
     - for each seed, a 5 x 5 local grid around each of its two points, every
       point of one grid against every point of the other; the best pair is
       kept and the spacing halved, five times.

    python Experiment_1/trajectory_rms_search.py
    python Experiment_1/trajectory_rms_search.py --amplitudes 10 --no-show

Writes, in Experiment_1/figures/: the eight-point figures and npz of
trajectory_rms.py, TrajectoryRMS_search_<amp>cm.svg (the best pair
for each direction) and TrajectoryRMS_search.npz.
"""

import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import trajectory_rms as base
from FL import compute_angles_from_cartesian


R_RANGE = (base.R_MIN, base.R_MAX)  # [cm] from the shoulder: every 10 and 20 cm target reachable
PHI_RANGE = (60.0, 140.0)  # [deg] around the shoulder, 90 = straight ahead
# Targets must keep the elbow between 20 and 163 deg. Near full flexion FL produces
# large hooks, which would win the search for a mechanical reason, not a postural
# one. As distances from the shoulder [cm], for the segment lengths of FL.py.
TARGET_ELBOW = (163.0, 20.0)
TARGET_LIMITS = tuple(np.sqrt(base.L1**2 + base.L2**2 + 2 * base.L1 * base.L2
                              * np.cos(np.radians(e))) for e in TARGET_ELBOW)
COARSE = 7
SEEDS = 3
LOCAL = 5  # points per side of the local grids
LEVELS = 5
PAIR_COLORS = ("#0081a7", "#f07167")


def to_xy(polar):
    """(n, 2) array of (r [cm], phi [deg]) to hand positions [cm]."""
    polar = np.atleast_2d(polar)
    phi = np.radians(polar[:, 1])
    return np.stack([polar[:, 0] * np.cos(phi), polar[:, 0] * np.sin(phi)], axis=1)


class Displacements:
    """Hand displacement of one reach (amplitude, direction) from any start, cached."""

    def __init__(self, amplitude, direction):
        self.amplitude = amplitude
        self.steps = base.num_steps(amplitude)
        self.vector = amplitude * np.array([np.cos(direction * np.pi / 4),
                                            np.sin(direction * np.pi / 4)])
        self.cache = {}
        self.calls = 0
        self.failed = 0
        self.excluded = 0

    def __call__(self, polar):
        out = []
        for r, phi in np.atleast_2d(polar):
            key = (round(r, 6), round(phi, 6))
            if key not in self.cache:
                start = to_xy([r, phi])[0]
                if not TARGET_LIMITS[0] <= np.linalg.norm(start + self.vector) <= TARGET_LIMITS[1]:
                    self.cache[key] = np.full((self.steps + 1, 2), np.nan)
                    self.excluded += 1
                    out.append(self.cache[key])
                    continue
                try:
                    self.cache[key] = base.displacement(start, start + self.vector,
                                                        self.amplitude)
                except np.linalg.LinAlgError:
                    # FL fails at a few postures (SVD of the muscle gains does not
                    # converge): the start is left out of the search.
                    self.cache[key] = np.full((self.steps + 1, 2), np.nan)
                    self.failed += 1
                self.calls += 1
            out.append(self.cache[key])
        return np.array(out)


def cross_rms(paths_a, paths_b):
    """RMS distance between every displacement of a and every one of b: (len a, len b)."""
    diff = paths_a[:, None] - paths_b[None, :]
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1), axis=-1))


def local_grid(centre, step):
    """LOCAL x LOCAL points around centre, clipped to the search region, duplicates removed."""
    offsets = (np.arange(LOCAL) - LOCAL // 2)
    r = np.clip(centre[0] + offsets * step[0], *R_RANGE)
    phi = np.clip(centre[1] + offsets * step[1], *PHI_RANGE)
    return np.unique(np.array([[a, b] for a in r for b in phi]), axis=0)


def search(amplitude, direction):
    """Coarse grid, then refinement around the best SEEDS pairs. Returns a dict."""
    disp = Displacements(amplitude, direction)
    r = np.linspace(*R_RANGE, COARSE)
    phi = np.linspace(*PHI_RANGE, COARSE)
    coarse = np.array([[a, b] for a in r for b in phi])
    rms = cross_rms(disp(coarse), disp(coarse))
    rms = np.nan_to_num(rms, nan=-1.0)  # failed starts never win
    pairs = sorted(combinations(range(len(coarse)), 2), key=lambda p: -rms[p])
    coarse_best = (coarse[pairs[0][0]], coarse[pairs[0][1]], rms[pairs[0]])

    best = None
    for i, j in pairs[:SEEDS]:
        p, q, value = coarse[i], coarse[j], rms[i, j]
        step = np.array([r[1] - r[0], phi[1] - phi[0]]) / 2
        for _ in range(LEVELS):
            grid_p, grid_q = local_grid(p, step), local_grid(q, step)
            m = np.nan_to_num(cross_rms(disp(grid_p), disp(grid_q)), nan=-1.0)
            a, b = np.unravel_index(np.argmax(m), m.shape)
            if m[a, b] >= value:
                p, q, value = grid_p[a], grid_q[b], m[a, b]
            step = step / 2
        if best is None or value > best[2]:
            best = (p, q, value)
    p, q, value = best
    return {"p": p, "q": q, "rms": value, "coarse": coarse_best,
            "paths": disp(np.array([p, q])), "simulations": disp.calls,
            "failed": disp.failed, "excluded": disp.excluded}


def posture(polar):
    return np.degrees(compute_angles_from_cartesian(*to_xy(polar)[0]))


def plot_search(results, amplitude, eight_point_rms):
    """For each direction: the search region, the best pair with arms and hand paths."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), layout="constrained")
    phi = np.radians(np.linspace(*PHI_RANGE, 100))
    for d, ax in enumerate(axes.ravel()):
        res = results[d]
        ring = np.concatenate([np.stack([R_RANGE[0] * np.cos(phi), R_RANGE[0] * np.sin(phi)], 1),
                               np.stack([R_RANGE[1] * np.cos(phi[::-1]),
                                         R_RANGE[1] * np.sin(phi[::-1])], 1)])
        ax.fill(ring[:, 0], ring[:, 1], color="0.93", zorder=0)
        for k, (point, color) in enumerate(zip((res["p"], res["q"]), PAIR_COLORS)):
            start = to_xy(point)[0]
            ts, te = posture(point)
            elbow = base.L1 * np.array([np.cos(np.radians(ts)), np.sin(np.radians(ts))])
            ax.plot([0, elbow[0], start[0]], [0, elbow[1], start[1]], color=color,
                    linewidth=2, marker="o", markersize=4, alpha=0.8,
                    label=f"r {point[0]:.1f} cm, {point[1]:.1f} deg  "
                          f"(shoulder {ts:.0f}, elbow {te:.0f})")
            path = start + res["paths"][k]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=3)
            ax.plot(*(start + res["paths"][k][-1]), "s", color=color, markersize=6)
        ax.plot(0, 0, "o", color="grey", markersize=8)
        ax.set_aspect("equal")
        reach = base.L1 + base.L2
        ax.set_xlim(-1.05 * reach, 1.05 * reach)
        ax.set_ylim(-0.35 * reach, 1.1 * reach)
        ax.set_title(f"Direction {d * 45} deg: RMS {res['rms']:.2f} cm\n"
                     f"(coarse grid {res['coarse'][2]:.2f}, eight points "
                     f"{eight_point_rms[d]:.2f} cm)", fontsize=12)
        ax.legend(fontsize=8, loc="lower left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Most different pair of starting points, {amplitude} cm reaches in "
                 f"{base.DURATIONS[amplitude] * 1000:.0f} ms (grey: search region)",
                 fontsize=15)
    fig.savefig(base.OUTDIR / f"TrajectoryRMS_search_{amplitude}cm.svg")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=base.AMPLITUDES)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    base.OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1. The eight-point matrices at 500 ms.
    points, postures = base.starting_points()
    base.plot_starts(points, postures)
    saved = {"points": points, "postures": postures}
    off = ~np.eye(len(points), dtype=bool)
    eight_point_max = {}
    for amplitude in args.amplitudes:
        amplitude = int(amplitude) if float(amplitude).is_integer() else amplitude
        paths, rms = base.simulate(amplitude, points)
        base.plot_rms(rms, amplitude)
        base.plot_paths(paths, amplitude)
        saved[f"rms_{amplitude}cm"], saved[f"paths_{amplitude}cm"] = rms, paths
        eight_point_max[amplitude] = rms.max(axis=(1, 2))
        print(f"\n{amplitude} cm, {base.DURATIONS[amplitude] * 1000:.0f} ms, eight points: "
              f"mean RMS {rms[:, off].mean():.2f} cm, "
              f"max {rms.max():.2f} cm")
    np.savez(base.OUTDIR / "TrajectoryRMS.npz", **saved)

    # 2. The refined search.
    found = {}
    for amplitude in args.amplitudes:
        amplitude = int(amplitude) if float(amplitude).is_integer() else amplitude
        print(f"\n{amplitude} cm, {base.DURATIONS[amplitude] * 1000:.0f} ms: "
              f"most different pair per direction")
        print("  dir    RMS    coarse  8-pt   start A (r, phi | shoulder, elbow)"
              "         start B")
        results = []
        for d in range(base.NUM_DIRECTIONS):
            res = search(amplitude, d)
            results.append(res)
            pa, pb = posture(res["p"]), posture(res["q"])
            print(f"  {d * 45:3d}  {res['rms']:5.2f}  {res['coarse'][2]:5.2f}  "
                  f"{eight_point_max[amplitude][d]:5.2f}   "
                  f"({res['p'][0]:4.1f}, {res['p'][1]:5.1f} | {pa[0]:5.1f}, {pa[1]:5.1f})   "
                  f"({res['q'][0]:4.1f}, {res['q'][1]:5.1f} | {pb[0]:5.1f}, {pb[1]:5.1f})"
                  f"   [{res['simulations']} sims"
                  + (f", {res['failed']} failed" if res["failed"] else "")
                  + (f", {res['excluded']} starts excluded" if res["excluded"] else "") + "]")
        plot_search(results, amplitude, eight_point_max[amplitude])
        for key in ("p", "q", "rms", "paths"):
            found[f"{key}_{amplitude}cm"] = np.array([r[key] for r in results])
    np.savez(base.OUTDIR / "TrajectoryRMS_search.npz", r_range=R_RANGE,
             phi_range=PHI_RANGE, **found)
    print(f"\nwrote figures and npz to {base.OUTDIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
