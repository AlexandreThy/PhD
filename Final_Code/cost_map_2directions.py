"""
DLQG movement cost as a function of where the reach starts.

Sweeps a grid of starting positions and, for each, reaches 15 cm in two
directions (90 and 315 degrees), drawing one cost map per direction on a shared
colour scale. Ported from CurrentParts/2Dir.py.

    python Final_Code/cost_map_2directions.py
    python Final_Code/cost_map_2directions.py --num-sim 2 --jobs 1
"""

from matplotlib.colors import Normalize

from common import (
    Cost_function, build_parser, finish, np, plt, run_dlqg, run_tasks,
    save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
AMPLITUDE = 15
# Grid of starting positions, as in the original script
X_VALS = np.linspace(-10, 10, 12)
Y_VALS = np.linspace(30, 45, 9)
TARGET_ANGLES_DEG = [90, 315]


def _worker(task):
    """Mean DLQG cost over repetitions for one starting position and direction."""
    x0, y0, angle_deg, num_sim = task
    start = [x0, y0]
    angle = np.deg2rad(angle_deg)
    target = [start[0] + np.cos(angle) * AMPLITUDE,
              start[1] + np.sin(angle) * AMPLITUDE]

    costs = np.zeros(num_sim)
    for sim in range(num_sim):
        _, _, x, u = run_dlqg(MOVEMENT_TIME, NUM_ITER, start, target)
        costs[sim] = Cost_function(x, u, tg=target)
    return np.mean(costs)


def simulate(num_sim, jobs):
    tasks = [(x0, y0, angle, num_sim)
             for angle in TARGET_ANGLES_DEG
             for x0 in X_VALS
             for y0 in Y_VALS]
    results = run_tasks(_worker, tasks, jobs, desc="DLQG cost map")

    flat = np.array(results).reshape(len(TARGET_ANGLES_DEG), len(X_VALS), len(Y_VALS))
    return {angle: flat[i] for i, angle in enumerate(TARGET_ANGLES_DEG)}


def plot(cost_maps, outdir, num_sim):
    values = np.concatenate([cost_maps[a].flatten() for a in TARGET_ANGLES_DEG])
    shared_norm = Normalize(vmin=values.min(), vmax=values.max())

    fig, axes = plt.subplots(len(TARGET_ANGLES_DEG), 1, figsize=(14, 6))
    for ax, angle_deg in zip(np.atleast_1d(axes), TARGET_ANGLES_DEG):
        im = ax.imshow(
            cost_maps[angle_deg].T,
            origin="lower",
            extent=[X_VALS.min(), X_VALS.max(), Y_VALS.min(), Y_VALS.max()],
            aspect="auto",
            cmap="plasma",
            norm=shared_norm,
            interpolation="bicubic",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Initial hand position x [cm]", fontsize=16)
        ax.set_ylabel("Initial hand position y [cm]", fontsize=16)
        ax.set_title(f"DLQG mean movement cost, {angle_deg} deg direction",
                     fontsize=16)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([30, 35, 40, 45])
        ax.tick_params(labelsize=14)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label(f"Mean movement cost over {num_sim} trials", fontsize=14)

    save_figure(fig, outdir, "DLQG_CostMap_90_315.svg")


def main():
    parser = build_parser(__doc__, num_sim_default=10)
    args = parser.parse_args()

    cost_maps = simulate(args.num_sim, args.jobs)
    plot(cost_maps, args.outdir, args.num_sim)

    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / "DLQG_CostMap_90_315.npz"
    np.savez(path, x_vals=X_VALS, y_vals=Y_VALS,
             **{f"deg_{a}": cost_maps[a] for a in TARGET_ANGLES_DEG})
    print(f"wrote {path}", flush=True)

    finish(not args.no_show)


if __name__ == "__main__":
    main()
