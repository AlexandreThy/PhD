"""
Human muscle parameters for the WP1 arm, in place of Lillicrap & Scott's macaque ones.

WP1 (Controllers/ILQG.py, Controllers/LQGControllers.py) has a human skeleton
(Li & Todorov 2004) but the macaque muscle geometry of Lillicrap & Scott (2013):
constant moment arms MOMENT_ARM [cm], optimal lengths L0 [cm] and the angles
THETA0 at which each muscle is at L0, through

    l = 1 + MOMENT_ARM[0] (THETA0[0] - theta_s) / L0 + MOMENT_ARM[1] (THETA0[1] - theta_e) / L0

The human set comes from the six lumped muscles of Nijhof & Kouwenhoven (2000),
as implemented in MotorNet (RigidTendonArm26, Codol et al. 2024): pectoralis,
deltoid, brachioradialis, triceps lateral, biceps, triceps long, which are the
same six groups as WP1 (shoulder flexor / extensor, elbow flexor / extensor,
biarticular flexor / extensor). There the musculotendon length is a quadratic
polynomial of the joint angles and the tendon is rigid, so

    moment arm    m_ij(theta) = a1_ij + 2 a2_ij (theta_j - a3_j)     [m, + lengthens]
    fibre length  l_i(theta)  = (MTL_i(theta) - tendon_i) / optimal_i

WP1's model needs constant moment arms, so this geometry is linearised at the
centre-out starting posture (hand at START): MOMENT_ARM = -100 m(theta*), and
THETA0 is chosen so that l(theta*) is the Nijhof & Kouwenhoven fibre length.
Only the elbow moment arms vary with the angle (by about +-20 % over the 15 cm
centre-out workspace); the linearised fibre length stays within 0.02 L0 of the
exact one there. The force-length / force-velocity curves (Brown et al.), the
skeleton, and the absence of F_max are WP1's, unchanged: only MOMENT_ARM, L0
and THETA0 differ between the two sets.

    from human_params import apply
    apply("human")      # or "macaque" to restore WP1 as it is
"""

import sys
from pathlib import Path

import numpy as np

WP1_ROOT = Path(__file__).resolve().parent.parent
if str(WP1_ROOT) not in sys.path:
    sys.path.insert(0, str(WP1_ROOT))

import Controllers.FL as FL  # noqa: E402
import Controllers.ILQG as ILQG  # noqa: E402
import Controllers.LQGControllers as LQGC  # noqa: E402

START = (0.0, 40.0)  # centre-out starting hand position [cm], Final_Code/common.py

# --- Nijhof & Kouwenhoven (2000), MotorNet RigidTendonArm26 -----------------
MUSCLES = ["pectoralis", "deltoid", "brachioradialis", "tricepslat", "biceps", "tricepslong"]
NK_A0 = np.array([0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989])
NK_A1 = np.array([-.03, .03, 0, 0, -.03, .03,
                  0, 0, -.014, .025, -.016, .03]).reshape(2, 6)
NK_A2 = np.array([0, 0, 0, 0, 0, 0,
                  0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]).reshape(2, 6)
NK_A3 = np.array([np.pi / 2, 0.0])
NK_TENDON = np.array([0.039, 0.066, 0.172, 0.187, 0.204, 0.217])  # m
NK_OPTIMAL = np.array([0.134, 0.140, 0.092, 0.093, 0.137, 0.127])  # m
NK_FMAX = np.array([838, 1207, 1422, 1549, 414, 603])  # N, reported only


def nk_geometry(theta):
    """Moment arms [m, + lengthens] (2 x 6) and normalised fibre lengths (6,)."""
    p = (np.asarray(theta, dtype=float) - NK_A3)[:, None]
    moment = NK_A1 + 2 * NK_A2 * p
    mtl = NK_A0 + np.sum((NK_A1 + NK_A2 * p) * p, axis=0)
    return moment, (mtl - NK_TENDON) / NK_OPTIMAL


def human_set(start=START):
    """MOMENT_ARM [cm], L0 [cm], THETA0 [rad] linearised at the starting posture."""
    theta_star = np.array(ILQG.compute_angles_from_cartesian(*start))
    moment, l_star = nk_geometry(theta_star)
    moment_arm = -100 * moment          # WP1 sign: + shortens as the angle grows
    moment_arm[np.abs(moment_arm) < 1e-12] = 0.0
    l0 = 100 * NK_OPTIMAL

    # Shift the angles of each muscle by the same delta so that
    # sum_j MOMENT_ARM[j] delta = (l* - 1) L0; a joint the muscle does not
    # cross keeps theta0 = 0, as in WP1.
    delta = (l_star - 1) * l0 / moment_arm.sum(axis=0)
    crosses = moment_arm != 0
    theta0 = np.where(crosses, theta_star[:, None] + delta, 0.0)
    return moment_arm, l0, theta0


MACAQUE = (ILQG.MOMENT_ARM.copy(), ILQG.L0.copy(), ILQG.THETA0.copy())
HUMAN = human_set()
SETS = {"macaque": MACAQUE, "human": HUMAN}


def apply(name):
    """Patch the muscle constants that ILQG, LQG and FL read at call time."""
    moment_arm, l0, theta0 = SETS[name]
    for module in (ILQG, LQGC, FL):
        module.MOMENT_ARM = moment_arm.copy()
        module.L0 = l0.copy()
        module.THETA0 = theta0.copy()
        if hasattr(module, "DLDTS"):
            module.DLDTS = -module.MOMENT_ARM[0] / module.L0
            module.DLDTE = -module.MOMENT_ARM[1] / module.L0
        if hasattr(module, "MOMENT_ARM_PINV"):
            module.MOMENT_ARM_PINV = np.linalg.pinv(module.MOMENT_ARM)


def describe():
    """Table of both sets: moment arms, L0, r/L0 and the fibre length at START."""
    theta_star = np.array(ILQG.compute_angles_from_cartesian(*START))
    lines = [f"start posture: shoulder {np.degrees(theta_star[0]):.1f} deg, "
             f"elbow {np.degrees(theta_star[1]):.1f} deg"]
    for name, (moment_arm, l0, theta0) in SETS.items():
        l_start = 1 + (moment_arm * (theta0 - theta_star[:, None])).sum(axis=0) / l0
        lines.append(f"\n{name}")
        lines.append("muscle            r_s[cm]  r_e[cm]  L0[cm]  r_s/L0  r_e/L0  l(start)")
        for i in range(6):
            lines.append(
                f"{MUSCLES[i] if name == 'human' else f'muscle {i + 1}':16s}"
                f"{moment_arm[0, i]:8.2f} {moment_arm[1, i]:8.2f} {l0[i]:7.2f}"
                f"{moment_arm[0, i] / l0[i]:8.2f}{moment_arm[1, i] / l0[i]:8.2f}"
                f"{l_start[i]:9.3f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    # Linearisation check against the exact Nijhof & Kouwenhoven fibre lengths
    # at the eight 15 cm centre-out targets.
    moment_arm, l0, theta0 = HUMAN
    worst = 0.0
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        th = np.array(ILQG.compute_angles_from_cartesian(
            START[0] + 15 * np.cos(a), START[1] + 15 * np.sin(a)))
        lin = 1 + (moment_arm * (theta0 - th[:, None])).sum(axis=0) / l0
        worst = max(worst, np.max(np.abs(lin - nk_geometry(th)[1])))
    print(f"\nlargest linearisation error on l over the 15 cm targets: {worst:.3f} L0")
