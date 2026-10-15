"""Figures of docs/path_constraint_methods.tex (movement 1 geometry, noiseless paths)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path_constraint_jacobian as pj  # noqa: E402
from common import compute_angles_from_cartesian, np, plt  # noqa: E402

plt.rcParams.update({"font.size": 12})

OUT = Path(__file__).resolve().parent
JOINT, JAC, GREY = pj.JOINT_COLOR, pj.JACOBIAN_COLOR, "0.45"


def geometry():
    start, target = pj.MOVEMENTS[0]()
    p0, pf = np.asarray(start), np.asarray(target)
    th0 = np.array(compute_angles_from_cartesian(*p0))
    thf = np.array(compute_angles_from_cartesian(*pf))
    thr = np.array(compute_angles_from_cartesian(*(p0 + 0.75 * (pf - p0))))
    k = (thr[1] - th0[1]) / (thr[0] - th0[0])

    s = np.linspace(0, 1, 200)
    hand_line_in_joints = pj.postures_on_line(start, target, s)       # straight hand line
    # The 75 % posture lies past the target in shoulder angle, so draw beyond both.
    end = min(thr[0], thf[0])
    ts = np.linspace(th0[0], end - 0.1 * (th0[0] - end), 200)
    joint_line = np.stack([ts, th0[1] + k * (ts - th0[0])], axis=1)  # your joint line
    joint_line_in_hand = pj.hand(joint_line)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    d = np.degrees
    a.plot(d(joint_line[:, 0]), d(joint_line[:, 1]), color=JOINT, lw=2.5,
           label="joint line (your cost)")
    a.plot(d(hand_line_in_joints[:, 0]), d(hand_line_in_joints[:, 1]), color=JAC, lw=2.5,
           label="straight hand line (Jacobian cost)")
    for th, mk, name, off in ((th0, "o", r"$\theta^0$", (7, -4)),
                              (thr, "D", r"$\theta^{\rm ref}$ (75%)", (7, -4)),
                              (thf, "s", r"$\theta^{\rm tg}$", (-26, -4))):
        a.plot(*d(th), mk, color="black", ms=8)
        a.annotate(name, d(th), textcoords="offset points", xytext=off, fontsize=13)
    a.set_xlabel(r"shoulder $\theta_s$ [deg]")
    a.set_ylabel(r"elbow $\theta_e$ [deg]")
    a.set_title("Joint space")
    a.legend(fontsize=10, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))

    b.plot(*joint_line_in_hand.T, color=JOINT, lw=2.5, label="image of the joint line")
    b.plot(*np.stack([p0, pf]).T, color=JAC, lw=2.5, label="straight hand line")
    for p, mk in ((p0, "o"), (p0 + 0.75 * (pf - p0), "D"), (pf, "s")):
        b.plot(*p, mk, color="black", ms=8)
    b.plot(0, 0, "o", color=GREY, ms=7)
    b.annotate("shoulder", (0, 0), textcoords="offset points", xytext=(-58, -4),
               fontsize=10, color=GREY)
    b.set_aspect("equal")
    b.set_xlabel("hand x [cm]")
    b.set_ylabel("hand y [cm]")
    b.set_title("Hand space")
    b.legend(fontsize=10, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_geometry.pdf")
    return np.degrees(th0), np.degrees(thr), np.degrees(thf), k


def paths():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    axes = axes.ravel()
    for m_idx, movement in enumerate(pj.MOVEMENTS):
        start, target = movement()
        for col, (weights, color, gains, title) in enumerate((
                (pj.WC_SWEEP, JOINT, lambda w: pj.joint_line_gains(start, target, w),
                 "your joint-line cost"),
                (pj.WJ_SWEEP, JAC, lambda w: pj.iterated_jacobian_gains(start, target, w),
                 "Jacobian cost"))):
            ax = axes[2 * m_idx + col]
            for w, c in zip(weights, pj._sweep_colors(len(weights), color)):
                xy, _, _ = pj.simulate(gains(w), start, target, noise=False)
                ax.plot(*xy.T, color=c, lw=2.2, label=f"{w:g}")
            ax.plot(*np.stack([start, target]).T, ":", color=GREY)
            ax.plot(*start, "o", color="black", ms=6)
            ax.plot(*target, "s", color="black", ms=6)
            ax.set_aspect("equal")
            ax.set_title(f"Movement {m_idx + 1}: {title}", fontsize=12)
            ax.legend(fontsize=9, frameon=False, loc="best",
                      title="$w_c$" if col == 0 else "$w_J$", title_fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paths.pdf")


def movement_cost():
    """From the 200-trial results saved by path_constraint_jacobian.py."""
    per_movement, matched, num_sim = pj.load_results()
    pj.plot_movement_cost(per_movement, matched, OUT, num_sim, filename="fig_cost.pdf")


if __name__ == "__main__":
    print(geometry())
    paths()
    movement_cost()
