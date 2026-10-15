"""
Two ways of asking the FL controller for a straight hand path, compared.

Joint-line cost (Controllers/FL.py, used in path_constraint.py)
    Penalises the distance, in joint space, from the line through the start
    posture and the posture 75 % of the way to the target, weighted by
    exp(-(time to go) / TAU_PATH). A straight line in joint space is not a
    straight hand path, so this is a proxy.

Jacobian cost (this script)
    Penalises the hand's actual lateral distance to the start-target line,
        e(theta) = n . (p(theta) - p0),
    with n the unit normal of the line. e is not quadratic in theta, so it is
    linearised at a reference posture theta*_t on the line for every step:
        e_t ~ n . J(theta*_t) (theta_t - theta*_t) = g_t . theta_t - c_t,
    which is exact to first order because p(theta*_t) lies on the line. The
    cost w * e_t**2 is then quadratic in the state plus a constant, so the
    state is augmented with a constant 1 and the usual Riccati recursion gives
    an affine feedback. theta*_t starts on a minimum-jerk progress along the
    line and is re-linearised, a few times, at the projection onto the line of
    the noiseless path the previous gains produce.

Both run in the same simulation: the FL dynamics, delay, Kalman filter and
feedback linearisation of Controllers/FL.py, with the weights of common.py.
Only the path term of the cost differs.

    python Final_Code/path_constraint_jacobian.py
    python Final_Code/path_constraint_jacobian.py --num-sim 20 --jobs 1
"""

from matplotlib.lines import Line2D

from common import (
    DELAY, FIGURE_DIR, FIGURE_SUBDIRS, MOTOR_NOISE, TAU_PATH, WP, WR_FL, WV,
    build_parser, compute_angles_from_cartesian, cost_components, delete_axis, finish,
    longmovement_1, longmovement_2, np, plt, run_fl, run_tasks, save_figure,
)
# After common, which puts the repository root on sys.path.
from Controllers.FL import (
    compute_linear_control_gains, compute_next_state, compute_path,
    nonlinear_transform_command, next_state_estimate,
)

MOVEMENT_TIME = 0.6
NUM_ITER = 60
L1, L2 = 30, 33  # upper arm and forearm [cm], as in ToCartesian
MOVEMENTS = (longmovement_1, longmovement_2)
# Path weights of each method. They are on different scales (rad**2 with the
# end-loaded decay, against cm**2 at every step), so each sweep spans its own
# range, both from unconstrained (about 13 cm off the line) to about 1-2 cm.
# Noiseless, wj = 0.01 already gives under 1 cm; past it the path is straight.
WC_SWEEP = (0, 20, 40, 60, 80)
WJ_SWEEP = (0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
RELINEARISATIONS = 3

JOINT_COLOR = "#f07167"
JACOBIAN_COLOR = "#0081a7"


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def hand(theta):
    s, e = theta[..., 0], theta[..., 1]
    return np.stack([L1 * np.cos(s) + L2 * np.cos(s + e),
                     L1 * np.sin(s) + L2 * np.sin(s + e)], axis=-1)


def jacobian(theta):
    s, e = theta
    return np.array([[-L1 * np.sin(s) - L2 * np.sin(s + e), -L2 * np.sin(s + e)],
                     [L1 * np.cos(s) + L2 * np.cos(s + e), L2 * np.cos(s + e)]])


def line_frame(start, target):
    """Start point, unit direction, unit normal and length of the reach."""
    p0, pf = np.asarray(start, float), np.asarray(target, float)
    length = np.linalg.norm(pf - p0)
    along = (pf - p0) / length
    return p0, along, np.array([-along[1], along[0]]), length


def lateral_deviation(hand_xy, start, target):
    p0, _, normal, _ = line_frame(start, target)
    return (hand_xy - p0) @ normal


def minimum_jerk(num_iter):
    tau = np.arange(num_iter) / num_iter
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def postures_on_line(start, target, progress):
    """Joint angles of the points at `progress` (0 to 1) along the line."""
    p0, along, _, length = line_frame(start, target)
    points = p0 + np.outer(np.clip(progress, 0, 1), along * length)
    return np.array([compute_angles_from_cartesian(x, y) for x, y in points])


# ----------------------------------------------------------------------------
# Feedback gains, on the 8-state FL model augmented with a constant 1
# ----------------------------------------------------------------------------
def _fl_matrices(duration, num_iter, wp, wv, r):
    dt = duration / num_iter
    A = np.identity(9)
    A[0, 2] = A[1, 3] = dt
    B = np.zeros((9, 2))
    B[2, 0] = B[3, 1] = dt
    Q = np.zeros((9, 9))
    Q[0, 0] = Q[1, 1] = Q[4, 4] = Q[5, 5] = wp
    Q[0, 4] = Q[4, 0] = Q[1, 5] = Q[5, 1] = -wp
    Q[2, 2] = Q[3, 3] = wv
    return A, B, Q, np.identity(2) * r


def joint_line_gains(start, target, wc, duration=MOVEMENT_TIME, num_iter=NUM_ITER,
                     taupath=TAU_PATH, wp=WP, wv=WV, r=WR_FL):
    """The gains of Controllers/FL.py, padded with a zero affine column."""
    Qk = compute_path(np.asarray(start), np.asarray(target), wc, 0.75)
    L8 = compute_linear_control_gains(num_iter, duration, Qk, taupath,
                                      motor_cost=r, cost_weights=[wp, wp, wv, wv])
    return np.concatenate([L8, np.zeros((num_iter, 2, 1))], axis=2)


def jacobian_gains(theta_ref, start, target, wj, duration=MOVEMENT_TIME,
                   num_iter=NUM_ITER, wp=WP, wv=WV, r=WR_FL):
    """
    Riccati recursion with the per-step cost wj * (g_t . theta_t - c_t)**2,
    g_t = J(theta*_t)^T n and c_t = g_t . theta*_t, on the augmented state.
    """
    A, B, Q_terminal, R = _fl_matrices(duration, num_iter, wp, wv, r)
    _, _, normal, _ = line_frame(start, target)
    S = Q_terminal
    L = np.zeros((num_iter, 2, 9))
    for t in range(num_iter - 1, -1, -1):
        L[t] = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
        g = jacobian(theta_ref[t]).T @ normal
        h = np.zeros(9)
        h[:2], h[8] = g, -g @ theta_ref[t]
        S = wj * np.outer(h, h) + A.T @ S @ (A - B @ L[t])
    return L


# ----------------------------------------------------------------------------
# Simulation: simulate_FL with an affine feedback
# ----------------------------------------------------------------------------
def simulate(L, start, target, noise, duration=MOVEMENT_TIME, num_iter=NUM_ITER,
             motor_noise=MOTOR_NOISE, delay=DELAY):
    """
    The loop of Controllers.FL.simulate_FL, with the command -L_t [x_hat; 1].
    With a zero last column of L it is simulate_FL exactly.
    """
    dt = duration / num_iter
    kdelay = int(delay / dt)
    st = compute_angles_from_cartesian(*start)
    tg = compute_angles_from_cartesian(*target)
    x0 = np.array([st[0], st[1], 0, 0, tg[0], tg[1], st[0], st[1]])
    true_state = estimated = np.tile(x0, kdelay + 1)
    sigma = np.zeros((8 * (kdelay + 1), 8 * (kdelay + 1)))
    states = np.zeros((num_iter + 1, 8))
    states[0] = x0
    commands = np.zeros((num_iter, 6))
    F = np.zeros(2)
    for j in range(num_iter):
        # nonlinear_transform_command only reads x[0:4], so the constant 1 can
        # ride along as a ninth component of the state it is given.
        u, v = nonlinear_transform_command(L[j], np.append(estimated[:8], 1.0))
        estimated, sigma = next_state_estimate(estimated, true_state, v, dt, noise,
                                               kdelay, sigma, motor_noise)
        new_state, F = compute_next_state(true_state[:8], u, dt, noise, False, F,
                                          0.0, motor_noise)
        true_state = np.concatenate((new_state, true_state[:-8]))
        states[j + 1] = true_state[:8]
        commands[j] = u
    return hand(states[:, :2]), states, commands


def iterated_jacobian_gains(start, target, wj, relinearisations=RELINEARISATIONS):
    """
    Gains linearised on a minimum-jerk progress along the line, then re-linearised
    at the projection onto the line of the noiseless path they produce.
    """
    p0, along, _, length = line_frame(start, target)
    progress = minimum_jerk(NUM_ITER)
    for _ in range(relinearisations + 1):
        theta_ref = postures_on_line(start, target, progress)
        L = jacobian_gains(theta_ref, start, target, wj)
        xy, _, _ = simulate(L, start, target, noise=False)
        progress = (xy[:NUM_ITER] - p0) @ along / length
    return L


def metrics(xy, states, commands, start, target):
    """
    Peak and RMS lateral deviation, endpoint error, terminal and peak joint
    velocity, effort, and the movement cost of common.py split into its three
    terms. The path term is left out: it is defined differently by each method.
    """
    dev = lateral_deviation(xy, start, target)
    vel = np.degrees(np.abs(states[:, 2:4]))
    accuracy, velocity, motor = cost_components(states, commands, tg=target)
    return {
        "peak_dev": np.abs(dev).max(),
        "rms_dev": np.sqrt(np.mean(dev**2)),
        "end_err": np.linalg.norm(xy[-1] - np.asarray(target)),
        "term_vel": vel[-1].max(),
        "peak_vel": vel.max(),
        "effort": np.sum(commands**2),
        "cost_accuracy": accuracy,
        "cost_velocity": velocity,
        "cost_motor": motor,
        "cost": accuracy + velocity + motor,
    }


METRIC_LABELS = {
    "peak_dev": "Peak lateral deviation [cm]",
    "rms_dev": "RMS lateral deviation [cm]",
    "end_err": "Endpoint error [cm]",
    "term_vel": "Terminal joint velocity [deg/s]",
    "peak_vel": "Peak joint velocity [deg/s]",
    "effort": "Muscle effort, sum u**2",
    "cost_accuracy": "Target accuracy cost",
    "cost_velocity": "Terminal velocity cost",
    "cost_motor": "Motor cost",
    "cost": "Movement cost",
}
COST_TERMS = ("cost_accuracy", "cost_velocity", "cost_motor")
COST_TERM_COLORS = ("#0081a7", "#00afb9", "#B90072")
RESULTS_FILE = FIGURE_DIR / FIGURE_SUBDIRS["large_amplitude"] / "PathConstraint_Jacobian.npz"


# ----------------------------------------------------------------------------
# Noisy runs
# ----------------------------------------------------------------------------
def _worker(task):
    """One noisy run for one (movement, method, weight). Stays top-level."""
    movement_index, L = task
    start, target = MOVEMENTS[movement_index]()
    xy, states, commands = simulate(L, start, target, noise=True)
    return xy, np.degrees(states[:, 2:4]), metrics(xy, states, commands, start, target)


def run_method(movement_index, gains, num_sim, jobs, desc):
    tasks = [(movement_index, L) for L in gains for _ in range(num_sim)]
    results = run_tasks(_worker, tasks, jobs, desc=desc)
    out = []
    for k in range(len(gains)):
        chunk = results[k * num_sim:(k + 1) * num_sim]
        out.append({
            "xy": np.mean([r[0] for r in chunk], axis=0),
            "vel": np.array([r[1] for r in chunk]),
            **{key: np.array([r[2][key] for r in chunk]) for key in METRIC_LABELS},
        })
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _sweep_colors(n, color):
    """From light to `color`, one shade per weight of a sweep."""
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    return [tuple(1 - (1 - rgb) * f) for f in np.linspace(0.25, 1, n)]


def plot_comparison(per_movement, outdir, num_sim):
    fig, axes = plt.subplots(len(MOVEMENTS), 4, figsize=(22, 5.5 * len(MOVEMENTS)),
                             layout="constrained")
    for row, (movement, (joint, jac)) in enumerate(zip(MOVEMENTS, per_movement)):
        start, target = movement()
        for col, (runs, weights, color, title, unit) in enumerate((
                (joint, WC_SWEEP, JOINT_COLOR, "Joint-line cost (FL.py)", "wc"),
                (jac, WJ_SWEEP, JACOBIAN_COLOR, "Jacobian cost", "wj"))):
            ax = axes[row, col]
            for run, w, c in zip(runs, weights, _sweep_colors(len(weights), color)):
                ax.plot(run["xy"][:, 0], run["xy"][:, 1], color=c, linewidth=3,
                        label=f"{unit} = {w:g}")
            ax.plot(*zip(start, target), color="grey", linestyle=":", linewidth=1.5)
            ax.plot(*start, "o", color="grey", markersize=10)
            ax.plot(*target, "s", color="grey", markersize=10)
            ax.set_aspect("equal")
            ax.set_title(f"{movement.__name__}: {title}", fontsize=13)
            ax.legend(fontsize=9, loc="best", frameon=False)
            delete_axis(ax, sides=["top", "right"])

        for col, ykey in ((2, "term_vel"), (3, "effort")):
            ax = axes[row, col]
            for runs, weights, color, name in ((joint, WC_SWEEP, JOINT_COLOR, "joint-line"),
                                               (jac, WJ_SWEEP, JACOBIAN_COLOR, "Jacobian")):
                x = [np.mean(r["peak_dev"]) for r in runs]
                y = [np.mean(r[ykey]) for r in runs]
                ax.plot(x, y, "-o", color=color, label=name)
                for xi, yi, w in zip(x, y, weights):
                    ax.annotate(f"{w:g}", (xi, yi), textcoords="offset points",
                                xytext=(4, 4), fontsize=8, color=color)
            ax.set_xlabel(METRIC_LABELS["peak_dev"], fontsize=12)
            ax.set_ylabel(METRIC_LABELS[ykey], fontsize=12)
            ax.set_title(f"{movement.__name__}: straightness against "
                         f"{'terminal velocity' if ykey == 'term_vel' else 'effort'}",
                         fontsize=13)
            ax.invert_xaxis()  # straighter to the right
            ax.legend(fontsize=10, frameon=False)
            delete_axis(ax, sides=["top", "right"])
    fig.suptitle(f"Straight-path cost: joint line against Jacobian, mean of {num_sim} "
                 f"noisy trials per weight", fontsize=16)
    save_figure(fig, outdir, "PathConstraint_JacobianComparison.svg", dpi=200)


def plot_velocities(per_movement, matched, outdir, num_sim):
    """Joint velocity profiles of the two methods at matched straightness."""
    time = np.linspace(0, MOVEMENT_TIME * 1000, NUM_ITER + 1)
    fig, axes = plt.subplots(len(MOVEMENTS), 1, figsize=(8, 4.25 * len(MOVEMENTS)),
                             squeeze=False)
    for ax, movement, (joint, jac), (i, k) in zip(axes[:, 0], MOVEMENTS,
                                                    per_movement, matched):
        for run, color in ((joint[i], JOINT_COLOR), (jac[k], JACOBIAN_COLOR)):
            for jnt, ls in enumerate(["-", "--"]):
                mean, sd = run["vel"][:, :, jnt].mean(0), run["vel"][:, :, jnt].std(0)
                ax.plot(time, mean, color=color, linestyle=ls)
                ax.fill_between(time, mean - sd, mean + sd, color=color, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{movement.__name__}: wc = {WC_SWEEP[i]:g} "
                     f"({np.mean(joint[i]['peak_dev']):.1f} cm) against wj = "
                     f"{WJ_SWEEP[k]:g} ({np.mean(jac[k]['peak_dev']):.1f} cm)", fontsize=12)
        ax.set_ylabel("Angular velocity [deg/s]", fontsize=12)
        delete_axis(ax, sides=["top", "right"])
    axes[-1, 0].set_xlabel("Time [ms]", fontsize=12)
    axes[0, 0].legend(handles=[
        Line2D([], [], color=JOINT_COLOR, lw=2, label="joint-line cost"),
        Line2D([], [], color=JACOBIAN_COLOR, lw=2, label="Jacobian cost"),
        Line2D([], [], color="black", ls="-", label="shoulder"),
        Line2D([], [], color="black", ls="--", label="elbow"),
    ], fontsize=10, loc="upper right", frameon=False)
    fig.suptitle(f"Joint velocity at matched straightness, mean +/- SD over "
                 f"{num_sim} trials", fontsize=14)
    fig.tight_layout()
    save_figure(fig, outdir, "PathConstraint_JacobianVelocities.svg", dpi=200)


def plot_movement_cost(per_movement, matched, outdir, num_sim,
                       filename="PathConstraint_JacobianMovementCost.svg"):
    """
    Left: total movement cost against straightness over both sweeps (mean +/- SEM).
    Right: its three terms for no path cost, the straightest joint-line weight,
    and the Jacobian weight of matched straightness.
    """
    fig, axes = plt.subplots(len(MOVEMENTS), 2, figsize=(11, 4.2 * len(MOVEMENTS)),
                             gridspec_kw=dict(width_ratios=[1.3, 1]), squeeze=False)
    for row, (movement, (joint, jac), (i, k)) in enumerate(zip(MOVEMENTS, per_movement,
                                                               matched)):
        ax = axes[row, 0]
        for runs, weights, color, name in ((joint, WC_SWEEP, JOINT_COLOR, "joint-line, $w_c$"),
                                           (jac, WJ_SWEEP, JACOBIAN_COLOR, "Jacobian, $w_J$")):
            x = [np.mean(r["peak_dev"]) for r in runs]
            y = [np.mean(r["cost"]) for r in runs]
            sem = [np.std(r["cost"]) / np.sqrt(len(r["cost"])) for r in runs]
            ax.errorbar(x, y, yerr=sem, fmt="-o", color=color, capsize=3, label=name)
            for xi, yi, w in zip(x, y, weights):
                ax.annotate(f"{w:g}", (xi, yi), textcoords="offset points",
                            xytext=(4, 5), fontsize=8, color=color)
        ax.invert_xaxis()  # straighter to the right
        ax.set_xlabel(METRIC_LABELS["peak_dev"], fontsize=11)
        ax.set_ylabel("Movement cost (mean +/- SEM)", fontsize=11)
        ax.set_title(f"Movement {row + 1}: cost against straightness", fontsize=12)
        ax.legend(fontsize=10, frameon=False)
        delete_axis(ax, sides=["top", "right"])

        ax = axes[row, 1]
        bars = ((joint[0], "no path\ncost"),
                (joint[i], f"joint-line\n$w_c$={WC_SWEEP[i]:g}"),
                (jac[k], f"Jacobian\n$w_J$={WJ_SWEEP[k]:g}"))
        bottom = np.zeros(len(bars))
        for term, color in zip(COST_TERMS, COST_TERM_COLORS):
            values = np.array([np.mean(run[term]) for run, _ in bars])
            ax.bar(range(len(bars)), values, bottom=bottom, color=color, width=0.6,
                   label=METRIC_LABELS[term])
            bottom += values
        for b, (run, _) in enumerate(bars):
            ax.text(b, bottom[b], f"{bottom[b]:.2f}\n({np.mean(run['peak_dev']):.1f} cm)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(bars)), [label for _, label in bars], fontsize=10)
        ax.set_ylim(0, bottom.max() * 1.3)
        ax.set_ylabel("Movement cost", fontsize=11)
        ax.set_title(f"Movement {row + 1}: cost terms", fontsize=12)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        delete_axis(ax, sides=["top", "right"])
    fig.suptitle(f"Movement cost of common.py (no path term), {num_sim} noisy trials "
                 f"per weight", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outdir, filename, dpi=200)


def save_results(per_movement, matched, num_sim, path=RESULTS_FILE):
    """Mean hand paths and per-trial metrics, so the figures can be redrawn."""
    arrays = {"matched": np.array(matched), "num_sim": num_sim}
    for m, (joint, jac) in enumerate(per_movement):
        for method, runs in (("joint", joint), ("jac", jac)):
            arrays[f"m{m}_{method}_xy"] = np.array([r["xy"] for r in runs])
            for key in METRIC_LABELS:
                arrays[f"m{m}_{method}_{key}"] = np.array([r[key] for r in runs])
    np.savez(path, **arrays)
    print(f"wrote {path}", flush=True)


def load_results(path=RESULTS_FILE):
    """per_movement, matched and num_sim as main() built them (without velocities)."""
    saved = np.load(path)
    per_movement = []
    for m in range(len(MOVEMENTS)):
        methods = []
        for method in ("joint", "jac"):
            xy = saved[f"m{m}_{method}_xy"]
            methods.append([{"xy": xy[w], **{key: saved[f"m{m}_{method}_{key}"][w]
                                              for key in METRIC_LABELS}}
                            for w in range(len(xy))])
        per_movement.append(tuple(methods))
    return per_movement, [tuple(p) for p in saved["matched"]], int(saved["num_sim"])


def print_table(per_movement):
    keys = ("peak_dev", "rms_dev", "end_err", "term_vel", "peak_vel", "effort", "cost")
    for movement, (joint, jac) in zip(MOVEMENTS, per_movement):
        print(f"\n{movement.__name__}   " + "  ".join(f"{k:>9}" for k in keys))
        for name, runs, weights in (("wc", joint, WC_SWEEP), ("wj", jac, WJ_SWEEP)):
            for run, w in zip(runs, weights):
                print(f"  {name} = {w:<6g}" + "  ".join(
                    f"{np.mean(run[k]):9.2f}" for k in keys))


def main():
    parser = build_parser(__doc__, num_sim_default=200,
                          subdir=FIGURE_SUBDIRS["large_amplitude"])
    args = parser.parse_args()

    # The padded joint-line gains must reproduce run_fl exactly.
    start, target = MOVEMENTS[0]()
    _, _, ref, _ = run_fl(MOVEMENT_TIME, NUM_ITER, list(start), list(target),
                          noise=False, wc=WC_SWEEP[-1])
    _, mine, _ = simulate(joint_line_gains(start, target, WC_SWEEP[-1]), start,
                          target, noise=False)
    assert np.allclose(ref[:, :4], mine[:, :4]), "simulate() differs from simulate_FL"

    per_movement, matched = [], []
    for index, movement in enumerate(MOVEMENTS):
        start, target = movement()
        joint_gains = [joint_line_gains(start, target, wc) for wc in WC_SWEEP]
        jac_gains = [iterated_jacobian_gains(start, target, wj) for wj in WJ_SWEEP]
        joint = run_method(index, joint_gains, args.num_sim, args.jobs,
                           f"{movement.__name__} joint-line")
        jac = run_method(index, jac_gains, args.num_sim, args.jobs,
                         f"{movement.__name__} Jacobian")
        per_movement.append((joint, jac))
        # The straightest joint-line weight, and the Jacobian weight closest to it.
        i = len(WC_SWEEP) - 1
        target_dev = np.mean(joint[i]["peak_dev"])
        k = int(np.argmin([abs(np.mean(r["peak_dev"]) - target_dev) for r in jac]))
        matched.append((i, k))

    print_table(per_movement)
    save_results(per_movement, matched, args.num_sim)
    plot_comparison(per_movement, args.outdir, args.num_sim)
    plot_velocities(per_movement, matched, args.outdir, args.num_sim)
    plot_movement_cost(per_movement, matched, args.outdir, args.num_sim)
    finish(not args.no_show)


if __name__ == "__main__":
    main()
