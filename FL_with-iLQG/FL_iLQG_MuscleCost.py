"""
FL + iLQG minimising the cost of the *muscle* commands (PaperPlot.ipynb cost).

Copy of FL_iLQG_Combined.py with a different cost. Classical FL minimises the
virtual acceleration v; the paper scores every controller with

    J = WP ||theta_N - theta_target||^2 + WV ||omega_N||^2 + WR sum_k ||u_k||^2

where u_k are the 6 muscle commands. On the linearised system z_{k+1} = A z_k + B v_k
the muscle commands are a nonlinear function of the state and of v:

    u = h(z, v) = P (M(theta) v + C(theta, omega) + Bvisc omega) / (fl(z) fv(z))

(P: pseudo-inverse of the moment arms, fl, fv: force-length / force-velocity gains).
So the running cost WR ||h(z, v)||^2 is not quadratic, and iLQG has to quadratise
it around the current nominal trajectory at each iteration.

Quadratisation (Gauss-Newton), see MuscleCommandCost.l_derivatives:
  1. linearise the command around the nominal (zbar_k, vbar_k):
         h(zbar + dz, vbar + dv) ~ hbar + Hz dz + Hv dv
     Hv = P M / (fl fv) is exact (h is linear in v); Hz is obtained by finite
     differences on theta and omega (h does not depend on the target / start).
  2. plug it in the cost: WR ||hbar + Hz dz + Hv dv||^2 is exactly quadratic in
     (dz, dv), which gives the gradients and Hessians used by the backward pass:
         lx = 2 WR Hz' hbar     lxx = 2 WR Hz' Hz
         lu = 2 WR Hv' hbar     luu = 2 WR Hv' Hv
                                lux = 2 WR Hv' Hz
     These Hessians are always positive (semi-)definite, so no regularisation is
     needed. The dynamics being linear, nothing else has to be approximated.

The iLQG loop then repeats: roll out -> quadratise -> backward pass -> update v
(with a line search on the true cost), until v stops changing.

Muscle redundancy (min_norm option): FL turns the torque tau into commands by
u = pinv(MA) tau / (fl fv), which is not the smallest u producing tau: a muscle
far from its optimal length (small fl) receives a very large command. With
min_norm=True, u = pinv(MA diag(fl fv)) tau, the minimum-norm command producing
tau. The optimisation over v then reaches the optimum of the paper cost over all
muscle commands: it matches ILQG of Controllers/ILQG.py with r1 = WR.
The default (False) keeps the transform of Controllers/FL.py.
"""

from Controllers.FL import *

# Weights of Cost_function in CurrentParts/PaperPlot.ipynb (cost evaluation cells)
WP_PAPER = 20000
WV_PAPER = 1
WR_PAPER = 0.1


def linear_dynamics(dt):
    """(A, B) of the feedback-linearised double integrator, as in FL."""
    A = np.identity(8)
    A[0, 2] = dt
    A[1, 3] = dt
    B = np.zeros((8, 2))
    B[2, 0] = dt
    B[3, 1] = dt
    return A, B


def terminal_cost_matrix(wp, wv):
    """Q such that z' Q z = wp ||theta - theta_target||^2 + wv ||omega||^2."""
    Q = np.zeros((8, 8))
    Q[0, 0] = Q[4, 4] = Q[1, 1] = Q[5, 5] = wp
    Q[0, 4] = Q[4, 0] = Q[1, 5] = Q[5, 1] = -wp
    Q[2, 2] = Q[3, 3] = wv
    return Q


def muscle_command_from_virtual(v, x, min_norm=False):
    """
    u = h(x, v): muscle commands producing the virtual joint acceleration v at state x.

    min_norm=False: split of Controllers/FL.py, pinv(MA) tau / (fl fv).
    min_norm=True : smallest commands producing tau, pinv(MA diag(fl fv)) tau.
    """
    M = np.array(
        [[a1 + 2 * a2 * cos(x[1]), a3 + a2 * cos(x[1])], [a3 + a2 * cos(x[1]), a3]]
    )
    C = np.array(
        [
            -x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]),
            x[2] ** 2 * a2 * np.sin(x[1]),
        ]
    )
    fl, ff_v = muscle_force_scaling(x)
    torque = M @ v + C + Viscous @ x[2:4]
    if min_norm:
        return np.linalg.pinv(MOMENT_ARM * (fl * ff_v)) @ torque
    return MOMENT_ARM_PINV @ torque / (fl * ff_v)


# ----------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------
class MuscleCommandCost:
    """
    J = sum_k WR ||h(z_k, v_k)||^2 + z_N' Q z_N    (Cost_function of PaperPlot.ipynb)
    """

    fd_step = 1e-6

    def __init__(self, wp=WP_PAPER, wv=WV_PAPER, wr=WR_PAPER, min_norm=False):
        self.wr = wr
        self.min_norm = min_norm
        self.Q = terminal_cost_matrix(wp, wv)

    def l(self, z, v, k):
        u = muscle_command_from_virtual(v, z, self.min_norm)
        return self.wr * u @ u

    def h(self, z):
        return z @ self.Q @ z

    def command_jacobians(self, z, v):
        """hbar = h(z, v), and the Jacobians Hz (6x8) and Hv (6x2) of h."""
        h = lambda v_, z_: muscle_command_from_virtual(v_, z_, self.min_norm)
        hbar = h(v, z)

        # h is linear in v: h(z, v) = h(z, 0) + Hv v, so Hv follows from two evaluations.
        h0 = h(np.zeros(2), z)
        Hv = np.array([h(e, z) - h0 for e in np.identity(2)]).T

        # Hz by central finite differences on theta_s, theta_e, omega_s, omega_e.
        # Columns 4:8 (target and start angles) stay zero: h does not depend on them.
        Hz = np.zeros((6, 8))
        for i in range(4):
            dz = np.zeros(8)
            dz[i] = self.fd_step
            Hz[:, i] = (h(v, z + dz) - h(v, z - dz)) / (2 * self.fd_step)
        return hbar, Hz, Hv

    def l_derivatives(self, z, v, k):
        """Gauss-Newton quadratisation of WR ||h(z, v)||^2. Returns (lx, lu, lxx, luu, lux)."""
        hbar, Hz, Hv = self.command_jacobians(z, v)
        r2 = 2 * self.wr
        return (
            r2 * Hz.T @ hbar,
            r2 * Hv.T @ hbar,
            r2 * Hz.T @ Hz,
            r2 * Hv.T @ Hv,
            r2 * Hv.T @ Hz,
        )

    def h_derivatives(self, z):
        return 2 * self.Q @ z, 2 * self.Q


# ----------------------------------------------------------------------------
# iLQG on the linear system
# ----------------------------------------------------------------------------
def rollout(z0, v, A, B):
    z = np.zeros((len(v) + 1, len(z0)))
    z[0] = z0
    for k in range(len(v)):
        z[k + 1] = A @ z[k] + B @ v[k]
    return z


def total_cost(z, v, cost):
    return sum(cost.l(z[k], v[k], k) for k in range(len(v))) + cost.h(z[-1])


def backward_pass(z, v, cost, A, B):
    """
    Riccati-like backward pass of iLQG around the nominal (z, v).

    Returns the open-loop increments l_ff and feedback gains L such that the
    locally optimal update is dv_k = l_ff_k + L_k dz_k.
    """
    N, m = v.shape
    n = z.shape[1]
    l_ff = np.zeros((N, m))
    L = np.zeros((N, m, n))

    sbold, S = cost.h_derivatives(z[-1])
    for k in range(N - 1, -1, -1):
        lx, lu, lxx, luu, lux = cost.l_derivatives(z[k], v[k], k)
        g = lu + B.T @ sbold
        G = lux + B.T @ S @ A
        H = luu + B.T @ S @ B

        l_ff[k] = -np.linalg.solve(H, g)
        L[k] = -np.linalg.solve(H, G)

        S = lxx + A.T @ S @ A + G.T @ L[k]
        S = 0.5 * (S + S.T)
        sbold = lx + A.T @ sbold + G.T @ l_ff[k]
    return l_ff, L


def linear_increment(l_ff, L, A, B, alpha=1.0):
    """Control increment obtained by propagating the deviation dz (step4 of iLQGController)."""
    dz = np.zeros(A.shape[0])
    dv = np.zeros(l_ff.shape)
    for k in range(len(l_ff)):
        dv[k] = alpha * l_ff[k] + L[k] @ dz
        dz = A @ dz + B @ dv[k]
    return dv


def ilqg_linear(z0, cost, A, B, Num_iter, max_iter=100, tol=1e-6, verbose=False):
    """
    iLQG on z_{k+1} = A z_k + B v_k, starting from v = 0.

    Returns:
        zbar, vbar : nominal state (Num_iter+1, 8) and virtual command (Num_iter, 2)
        L          : feedback gains (Num_iter, 2, 8), v_k = vbar_k + L_k (z_k - zbar_k)
        info       : dict with the number of iterations, cost history and convergence flag
    """
    v = np.zeros((Num_iter, B.shape[1]))
    z = rollout(z0, v, A, B)
    J = total_cost(z, v, cost)
    history = [J]
    converged = False

    for iteration in range(max_iter):
        # 1. quadratise the cost around (z, v) and solve the resulting LQ problem
        l_ff, L = backward_pass(z, v, cost, A, B)
        dv = linear_increment(l_ff, L, A, B)
        if np.max(np.abs(dv)) < tol * (1 + np.max(np.abs(v))):
            converged = True
            break

        # 2. line search: the quadratic model is only local, keep the step if the true cost decreases
        alpha = 1.0
        while alpha > 1e-6:
            v_new = v + linear_increment(l_ff, L, A, B, alpha)
            z_new = rollout(z0, v_new, A, B)
            J_new = total_cost(z_new, v_new, cost)
            if J_new < J:
                break
            alpha *= 0.5
        else:
            # No decrease even for tiny steps: we are at the optimum up to round-off.
            converged = True
            break

        v, z, J = v_new, z_new, J_new
        history.append(J)
        if verbose:
            print(f"iteration {iteration}: cost {J:.6e}, alpha {alpha}")

    if not converged:
        print("iLQG: solution not converged")
    info = {"iterations": iteration, "cost_history": np.array(history), "converged": converged}
    return z, v, L, info


# ----------------------------------------------------------------------------
# Execution on the nonlinear arm
# ----------------------------------------------------------------------------
def initial_state(starting_point, targets):
    """z0 = [start angles, zero velocity, target angles, start angles]."""
    st1, st2 = compute_angles_from_cartesian(starting_point[0], starting_point[1])
    tg1, tg2 = compute_angles_from_cartesian(targets[0], targets[1])
    return np.array([st1, st2, 0, 0, tg1, tg2, st1, st2])


def plan_FL_iLQG_muscle(
    Duration=0.4,
    wp=WP_PAPER,
    wv=WV_PAPER,
    wr=WR_PAPER,
    targets=[0, 55],
    starting_point=[0, 40],
    Num_iter=40,
    min_norm=False,
    **ilqg_kwargs,
):
    """iLQG plan. Returns (zbar, vbar, L, info)."""
    dt = Duration / Num_iter
    z0 = initial_state(starting_point, targets)
    A, B = linear_dynamics(dt)
    zbar, vbar, L, info = ilqg_linear(
        z0, MuscleCommandCost(wp, wv, wr, min_norm), A, B, Num_iter, **ilqg_kwargs
    )
    return zbar, vbar, L, info


def simulate_FL_iLQG_muscle(
    Duration=0.4,
    wp=WP_PAPER,
    wv=WV_PAPER,
    wr=WR_PAPER,
    targets=[0, 55],
    starting_point=[0, 40],
    Activate_Noise=False,
    Num_iter=40,
    Delay=0.06,
    FF=False,
    ff_power=0.3,
    motornoise_variance=1e-3,
    return_plan=False,
    plan=None,
    min_norm=False,
    **ilqg_kwargs,
):
    """
    Same outputs as simulate_FL (X, Y, states, muscle commands), with v planned
    by iLQG on the paper cost. The weights are those of Cost_function directly
    (wp, wv, wr), not the FL ones. With return_plan=True the plan
    (zbar, vbar, L, info) is appended to the outputs. Passing a plan from
    plan_FL_iLQG_muscle skips the optimisation (useful for many noisy trials);
    it must have been computed with the same min_norm.
    """
    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    x0 = initial_state(starting_point, targets)
    if plan is None:
        plan = plan_FL_iLQG_muscle(
            Duration, wp, wv, wr, targets, starting_point, Num_iter, min_norm, **ilqg_kwargs
        )
    zbar, vbar, L, info = plan

    all_true_states = np.zeros((Num_iter + 1, 8))
    all_estimated_states = np.zeros((Num_iter + 1, 8))
    all_commands = np.zeros((Num_iter, 6))
    x0_with_delay = np.tile(x0, kdelay + 1)
    true_state, estimated_state = x0_with_delay, x0_with_delay
    all_true_states[0, :] = np.copy(x0)
    all_estimated_states[0, :] = np.copy(x0)
    sigma = np.zeros((8 * (kdelay + 1), 8 * (kdelay + 1)))
    F = np.zeros(2)
    for j in range(Num_iter):
        v = vbar[j] + L[j] @ (estimated_state[:8] - zbar[j])
        u = muscle_command_from_virtual(v, estimated_state[:8], min_norm)
        estimated_state, sigma = next_state_estimate(
            estimated_state,
            true_state,
            v,
            dt,
            Activate_Noise,
            kdelay,
            sigma,
            motornoise_variance,
        )
        new_state, F = compute_next_state(
            true_state[:8], u, dt, Activate_Noise, FF, F, ff_power, motornoise_variance
        )
        true_state = np.concatenate((new_state, true_state[:-8]))

        all_true_states[j + 1, :] = true_state[:8]
        all_estimated_states[j + 1, :] = estimated_state[:8]
        all_commands[j] = u

    s, e = all_true_states[:, 0], all_true_states[:, 1]
    X = np.cos(s + e) * 33 + np.cos(s) * 30
    Y = np.sin(s + e) * 33 + np.sin(s) * 30
    if return_plan:
        return X, Y, all_true_states, all_commands, plan
    return X, Y, all_true_states, all_commands
