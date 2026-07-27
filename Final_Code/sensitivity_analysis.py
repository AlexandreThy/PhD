"""
Sensitivity of the movement cost to the cost function weights.

Sweeps the position weight, the terminal velocity weight and the motor cost one
at a time, holding the other two at their reference values, and plots the mean
cost of each controller.

    python Final_Code/sensitivity_analysis.py
    python Final_Code/sensitivity_analysis.py --num-sim 5 --num-points 3 --jobs 1
"""

from common import (
    COLORS, Cost_function, LEGEND, NUM_CONTROLLERS, WP, WR, WR_FL, WV,
    build_parser, delete_axis, finish, np, plt, run_dlqg, run_fl, run_ilqg,
    run_tasks, save_figure,
)

MOVEMENT_TIME = 0.4
NUM_ITER = 40
START = [0, 40]
# The notebook probed a single target, down and to the right of the start
TARGET_ANGLE = 7 * np.pi / 4
TARGET_AMPLITUDE = 20

NUM_POINTS = 10
WP_RANGE = (2000, 200000)
WV_RANGE = (0.1, 10)
WR_RANGE = (0.05, 5)

SWEEPS = ("wp", "wv", "wr")
AXIS_LABELS = {
    "wp": "Position weight",
    "wv": "Terminal velocity weight",
    "wr": "Motor cost",
}


def _target(start):
    return np.array([np.cos(TARGET_ANGLE) * TARGET_AMPLITUDE + start[0],
                     np.sin(TARGET_ANGLE) * TARGET_AMPLITUDE + start[1]])


def _worker(task):
    """Run the three controllers for one weight set. Stays top-level."""
    wp, wv, wr, wr_fl, start = task
    target = _target(start)

    _, _, x_ilqg, u_ilqg = run_ilqg(MOVEMENT_TIME, NUM_ITER, start, target,
                                    wp=wp, wv=wv, wr=wr)
    _, _, x_fl, u_fl = run_fl(MOVEMENT_TIME, NUM_ITER, start, target,
                              wp=wp, wv=wv, wr=wr_fl)
    _, _, x_dlqg, u_dlqg = run_dlqg(MOVEMENT_TIME, NUM_ITER, start, target,
                                    wp=wp, wv=wv, wr=wr)

    runs = ((x_ilqg, u_ilqg), (x_fl, u_fl), (x_dlqg, u_dlqg))
    # The cost is always scored with the weights the controllers were given.
    return np.array([Cost_function(x, u, wp, wv, wr, tg=target) for x, u in runs])


def sweep(name, values, num_sim, jobs, start):
    """Mean cost per controller for each value of the swept weight."""
    tasks = []
    for value in values:
        wp = value if name == "wp" else WP
        wv = value if name == "wv" else WV
        wr = value if name == "wr" else WR
        tasks.extend([(wp, wv, wr, WR_FL, start)] * num_sim)

    results = run_tasks(_worker, tasks, jobs, desc=f"sensitivity to {name}")
    costs = np.array(results).reshape(len(values), num_sim, NUM_CONTROLLERS)
    return np.mean(costs, axis=1)


def main():
    parser = build_parser(__doc__, num_sim_default=100)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS,
                        help="values probed per weight")
    args = parser.parse_args()

    ranges = {
        "wp": np.linspace(*WP_RANGE, args.num_points),
        "wv": np.linspace(*WV_RANGE, args.num_points),
        "wr": np.linspace(*WR_RANGE, args.num_points),
    }

    fig, axes = plt.subplots(len(SWEEPS), figsize=(8, 8))
    for ax, name in zip(axes, SWEEPS):
        values = ranges[name]
        mean_cost = sweep(name, values, args.num_sim, args.jobs, START)

        for i in range(NUM_CONTROLLERS):
            ax.plot(values, mean_cost[:, i], color=COLORS[i], label=LEGEND[i])
        ax.set_yscale("log")
        ax.set_xlabel(AXIS_LABELS[name], fontsize=13)
        ax.set_ylabel("Mean movement cost", fontsize=13)
        delete_axis(ax, sides=["top", "right"])
        ax.legend(fontsize=10)

    fig.tight_layout()
    save_figure(fig, args.outdir, "SensitivityAnalysis.svg")
    finish(not args.no_show)


if __name__ == "__main__":
    main()
