"""
iLQG in the linear coordinates of feedback linearisation.

FL (Controllers/FL.py) cancels the arm and muscle nonlinearities so that the
virtual state

    z = [theta_s, theta_e, omega_s, omega_e, target_s, target_e, start_s, start_e]

obeys the double integrator z_{k+1} = A z_k + B v_k, with v the virtual joint
acceleration, and plans v with an LQR (compute_linear_control_gains).

Here v is planned with iLQG instead (same forward pass / backward pass / linear
increment scheme as Controllers/iLQGController.py), which admits any smooth
discrete-time cost

    J = sum_{k=0}^{N-1} l(z_k, v_k, k) + h(z_N).

The plan (zbar, vbar, L) is executed exactly as in simulate_FL: the feedback
v_k = vbar_k + L_k (zhat_k - zbar_k) is computed on the delayed Kalman estimate
and turned into muscle commands by the linearising transform.

With the default QuadraticFLCost -- the cost of compute_linear_control_gains --
the problem is linear-quadratic: iLQG converges in one iteration and the policy
equals FL's, v_k = -L^FL_k z_k. FL_iLQG_verif.py checks this.
"""

from Controllers.FL import *


def linear_dynamics(dt):
    """(A, B) of the feedback-linearised double integrator, as in FL."""
    A = np.identity(8)
    A[0, 2] = dt
    A[1, 3] = dt
    B = np.zeros((8, 2))
    B[2, 0] = dt
    B[3, 1] = dt
    return A, B


def terminal_cost_matrix(cost_weights):
    """Terminal state cost of compute_linear_control_gains (penalises z[0:2] - z[4:6] and z[2:4])."""
    w1, w2, w3, w4 = cost_weights
    Q = np.zeros((8, 8))
    Q[0, 0] = Q[4, 4] = w1
    Q[0, 4] = Q[4, 0] = -w1
    Q[1, 1] = Q[5, 5] = w2
    Q[1, 5] = Q[5, 1] = -w2
    Q[2, 2] = w3
    Q[3, 3] = w4
    return Q


# ----------------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------------
class Cost:
    """
    Discrete-time cost sum_k l(z_k, v_k, k) + h(z_N) on the linearised system.

    Subclasses define l and h. The derivatives default to central finite
    differences; override l_derivatives / h_derivatives when they are known in
    closed form.
    """

    fd_step = 1e-4

    def l(self, z, v, k):
        raise NotImplementedError

    def h(self, z):
        raise NotImplementedError

    def l_derivatives(self, z, v, k):
        """Returns (lx, lu, lxx, luu, lux)."""
        n = len(z)
        g, H = _fd_grad_hess(
            lambda w: self.l(w[:n], w[n:], k), np.concatenate((z, v)), self.fd_step
        )
        return g[:n], g[n:], H[:n, :n], H[n:, n:], H[n:, :n]

    def h_derivatives(self, z):
        """Returns (hx, hxx)."""
        return _fd_grad_hess(self.h, z, self.fd_step)


def _fd_grad_hess(fun, w, eps):
    """Central finite-difference gradient and Hessian of a scalar function."""
    n = len(w)
    E = np.identity(n) * eps
    grad = np.array([(fun(w + E[i]) - fun(w - E[i])) / (2 * eps) for i in range(n)])
    hess = np.zeros((n, n))
    f0 = fun(w)
    for i in range(n):
        hess[i, i] = (fun(w + E[i]) - 2 * f0 + fun(w - E[i])) / eps**2
        for j in range(i + 1, n):
            hess[i, j] = hess[j, i] = (
                fun(w + E[i] + E[j])
                - fun(w + E[i] - E[j])
                - fun(w - E[i] + E[j])
                + fun(w - E[i] - E[j])
            ) / (4 * eps**2)
    return grad, hess


class QuadraticFLCost(Cost):
    """
    The cost minimised by compute_linear_control_gains:

        l(z, v, k) = 1/2 z' Qk exp(-(N-1-k) dt / taupath) z + 1/2 v' R v
        h(z)       = 1/2 z' Q z

    (the overall 1/2 does not change the minimiser). The running state cost at
    k = 0 is irrelevant since z_0 is fixed.
    """

    def __init__(self, Num_iter, dt, Qk, taupath, motor_cost, cost_weights):
        self.N = Num_iter
        self.dt = dt
        self.Qk = Qk
        self.taupath = taupath
        self.R = np.identity(2) * motor_cost
        self.Q = terminal_cost_matrix(cost_weights)

    def running_Q(self, k):
        return self.Qk * np.exp(-(self.N - 1 - k) * self.dt / self.taupath)

    def l(self, z, v, k):
        return 0.5 * z @ self.running_Q(k) @ z + 0.5 * v @ self.R @ v

    def h(self, z):
        return 0.5 * z @ self.Q @ z

    def l_derivatives(self, z, v, k):
        Qrun = self.running_Q(k)
        return Qrun @ z, self.R @ v, Qrun, self.R, np.zeros((2, 8))

    def h_derivatives(self, z):
        return self.Q @ z, self.Q


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


def backward_pass(z, v, cost, A, B, reg=0.0):
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
        H = luu + B.T @ S @ B + reg * np.identity(m)
        if np.min(np.linalg.eigvalsh(H)) <= 0:
            print("H is not positive definite at step", k, ": increase reg")

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


def ilqg_linear(z0, cost, A, B, Num_iter, max_iter=50, tol=1e-8, reg=0.0, v_init=None, verbose=False):
    """
    iLQG on z_{k+1} = A z_k + B v_k.

    Returns:
        zbar, vbar : nominal state (Num_iter+1, 8) and virtual command (Num_iter, 2)
        L          : feedback gains (Num_iter, 2, 8), v_k = vbar_k + L_k (z_k - zbar_k)
        info       : dict with the number of iterations, cost history and convergence flag
    """
    v = np.zeros((Num_iter, B.shape[1])) if v_init is None else np.array(v_init, dtype=float)
    z = rollout(z0, v, A, B)
    J = total_cost(z, v, cost)
    history = [J]
    converged = False

    for iteration in range(max_iter):
        l_ff, L = backward_pass(z, v, cost, A, B, reg)
        dv = linear_increment(l_ff, L, A, B)
        if np.max(np.abs(dv)) < tol * (1 + np.max(np.abs(v))):
            converged = True
            break

        # Backtracking line search: exact dynamics are linear, but the cost may not be quadratic.
        alpha = 1.0
        while alpha > 1e-6:
            v_new = v + linear_increment(l_ff, L, A, B, alpha)
            z_new = rollout(z0, v_new, A, B)
            J_new = total_cost(z_new, v_new, cost)
            if J_new < J:
                break
            alpha *= 0.5
        else:
            print("iLQG: line search failed, stopping")
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
def muscle_command_from_virtual(v, x):
    """Muscle commands producing the virtual joint acceleration v at state x (as nonlinear_transform_command)."""
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
    U = MOMENT_ARM_PINV @ (M @ v + C + Viscous @ x[2:4])
    return U / (fl * ff_v)


def plan_FL_iLQG(
    Duration=0.6,
    w1=1e8,
    w2=1e8,
    w3=1e4,
    w4=1e4,
    r=1e-5,
    targets=[0, 55],
    starting_point=[0, 30],
    Num_iter=300,
    wp=0,
    taupath=0.04,
    percent=0.75,
    cost=None,
    **ilqg_kwargs,
):
    """
    Initial state and iLQG plan on the linearised system. Without `cost`, the
    FL cost built from the same arguments as simulate_FL is used.

    Returns (z0, zbar, vbar, L, info).
    """
    dt = Duration / Num_iter
    st1, st2 = compute_angles_from_cartesian(starting_point[0], starting_point[1])
    tg1, tg2 = compute_angles_from_cartesian(targets[0], targets[1])
    z0 = np.array([st1, st2, 0, 0, tg1, tg2, st1, st2])

    if cost is None:
        Qk = compute_path(np.array(starting_point), np.array(targets), wp, percent)
        cost = QuadraticFLCost(Num_iter, dt, Qk, taupath, r, [w1, w2, w3, w4])
    A, B = linear_dynamics(dt)
    zbar, vbar, L, info = ilqg_linear(z0, cost, A, B, Num_iter, **ilqg_kwargs)
    return z0, zbar, vbar, L, info


def simulate_FL_iLQG(
    Duration=0.6,
    w1=1e8,
    w2=1e8,
    w3=1e4,
    w4=1e4,
    r=1e-5,
    targets=[0, 55],
    starting_point=[0, 30],
    Activate_Noise=False,
    Num_iter=300,
    Delay=0.06,
    FF=False,
    ff_power=0.3,
    motornoise_variance=1e-3,
    wp=0,  # set to 4*1e-3 if active
    taupath=0.04,
    percent=0.75,
    cost=None,
    return_plan=False,
    **ilqg_kwargs,
):
    """
    Same arguments and outputs as simulate_FL, with the virtual command planned
    by iLQG. `cost` (a Cost instance) replaces the FL quadratic cost; extra
    keyword arguments go to ilqg_linear. With return_plan=True the plan
    (zbar, vbar, L, info) is appended to the outputs.
    """
    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    x0, zbar, vbar, L, info = plan_FL_iLQG(
        Duration, w1, w2, w3, w4, r, targets, starting_point, Num_iter,
        wp, taupath, percent, cost, **ilqg_kwargs,
    )

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
        u = muscle_command_from_virtual(v, estimated_state[:8])
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
        return X, Y, all_true_states, all_commands, (zbar, vbar, L, info)
    return X, Y, all_true_states, all_commands
