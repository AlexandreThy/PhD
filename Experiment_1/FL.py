import numpy as np
from math import *

# Biomechanical Parameters

I1 = 0.025
I2 = 0.045
m2 = 1
l1 = 0.3
l2 = 0.33
s2 = 0.16
K = 1 / 0.06
tau = 0.06

# SHOULDER PUIS ELBOW
a1 = I1 + I2 + m2 * l1 * l1
a2 = m2 * l1 * s2
a3 = I2

Viscous = np.array([[0.05, 0.025], [0.025, 0.05]])

# Muscle model constants. Hoisted to module level: they used to be rebuilt from
# nested lists on every call of the dynamics, which dominated the runtime.
MOMENT_ARM = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])
L0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
THETA0 = np.array(
    [
        [
            2 * pi / 360 * 15,
            2 * pi / 360 * 4.88,
            0,
            0,
            2 * pi / 360 * 4.5,
            2 * pi / 360 * 2.12,
        ],
        [
            0,
            0,
            2 * pi / 360 * 80.86,
            2 * pi / 360 * 109.32,
            2 * pi / 360 * 92.96,
            2 * pi / 360 * 91.52,
        ],
    ]
)
# np.linalg.pinv runs an SVD; the argument never changes so it is computed once.
MOMENT_ARM_PINV = np.linalg.pinv(MOMENT_ARM)


def muscle_force_scaling(x):
    """
    Force-length and force-velocity multipliers of the six muscles at state x.

    Returns:
        (fl, ff_v) : the length- and velocity-dependent gains
    """
    l = 1 + MOMENT_ARM[0] * (THETA0[0] - x[0]) / L0 + MOMENT_ARM[1] * (THETA0[1] - x[1]) / L0
    v = MOMENT_ARM[0] * (-x[2]) / L0 + MOMENT_ARM[1] * (-x[3]) / L0

    fl = np.exp(-(np.abs((l**1.55 - 1) / 0.81) ** 2.12))
    ff_v = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    return fl, ff_v


def compute_angles_from_cartesian(x, y, l1=30, l2=33):
    """
    Computes joint angles in radians based on cartesian coordinates.

    Parameters:
        x (float): x-coordinate of the end effector.
        y (float): y-coordinate of the end effector.
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.

    Returns:
        angles (float): Computed angle in radians.
    """
    r_squared = x**2 + y**2

    shoulder_angle = np.arctan2(y, x) - np.arccos(
        (r_squared + l1**2 - l2**2) / (2 * l1 * np.sqrt(r_squared))
    )

    elbow_angle = np.pi - np.arccos((l1**2 + l2**2 - r_squared) / (2 * l1 * l2))
    return shoulder_angle, elbow_angle


def compute_forcefield(theta, omega, coefficient):
    """
    Compute the joint angles acceleration resulting from a lateral
    velocity-dependent forcefield.

    Args:
        theta : current joint angles
        omega : current joint angular velocities
        acc : current joint angular accelerations
        coefficient : Multiplier coefficient on the force field such that yddot = 13 * coeff * xdot

    """
    D = np.array([[0, coefficient], [0, 0]])
    Jacobian = np.array(
        [
            [
                -33 * np.sin(theta[0] + theta[1]) - 30 * np.sin(theta[0]),
                -33 * np.sin(theta[0] + theta[1]),
            ],
            [
                33 * np.cos(theta[0] + theta[1]) + 30 * np.cos(theta[0]),
                33 * np.cos(theta[0] + theta[1]),
            ],
        ]
    )

    return -Jacobian.T @ D @ Jacobian @ omega


def compute_next_state(x, u, dt, activate_noise, FF, F, ff_power, motornoise_variance):
    """
    Compute one step of the dynamics of the system, composed of a two joint biomechanical model, and a nonlinear network dynamic
    \ddot{\theta} = M^{-1}(Wout gamma-B \dot{\theta} - C)
    \dot{\gamma} = tanh(W gamma) + u
    Args:
        x : x_t biomechanical state at time t [theta_s, theta_e, omega_s, omega_e, angular_acc_s, angular_acc_e]
        gamma : gamma_t vector of the newtork activity at time t
        u : command
        dt : timestep
        Wout : readout matrix
        W : internal network connectivity
    Returns:
        [x_{t+1},gamma_{t+1}]
    """
    newx = np.copy(x)
    M = np.array(
        [[a1 + 2 * a2 * cos(x[1]), a3 + a2 * cos(x[1])], [a3 + a2 * cos(x[1]), a3]]
    )
    C = np.array(
        [
            -x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]),
            x[2] ** 2 * a2 * np.sin(x[1]),
        ]
    )
    newx[0:2] += dt * x[2:4]
    fl, ff_v = muscle_force_scaling(x)
    F = np.zeros(2)
    if FF == True:
        F = compute_forcefield(x[0:2], x[2:4], ff_power)
    noise = (
        np.random.normal(0, np.sqrt(motornoise_variance), 2)
        if activate_noise
        else np.zeros(2)
    )
    newx[2:4] += (
        dt
        * np.linalg.solve(
            M, (MOMENT_ARM @ (u * fl * ff_v) - Viscous @ (x[2:4]) - C + F)
        )
        + noise
    )
    return newx, F


_delayed_dynamics_cache = {}


def _delayed_dynamics(dt, delay):
    """
    Observation and delayed-state dynamics matrices (H, A, B) of the estimator.

    These depend only on dt and the delay, so they are built once per parameter
    set rather than on every timestep. Returned read-only: callers must not
    mutate them in place.
    """
    key = (dt, delay)
    cached = _delayed_dynamics_cache.get(key)
    if cached is not None:
        return cached

    H = np.zeros((8, (delay + 1) * 8))
    H[:, delay * 8 :] = np.identity(8)

    A_basic = np.identity(8)
    A_basic[0, 2] = dt
    A_basic[1, 3] = dt

    B_basic = np.zeros((8, 2))
    B_basic[2, 0] = dt
    B_basic[3, 1] = dt

    A = np.zeros(((delay + 1) * 8, (delay + 1) * 8))
    A[:8, :8] = A_basic
    A[8:, :-8] = np.identity((delay) * 8)
    B = np.zeros(((delay + 1) * 8, 2))
    B[:8] = B_basic

    cached = (H, A, B)
    _delayed_dynamics_cache[key] = cached
    return cached


_noise_cov_cache = {}


def NoiseAndCovMatrix(M=np.identity(2), N=8, kdelay=0, motornoise_variance=1e-3):

    key = (N, kdelay, motornoise_variance)
    cached = _noise_cov_cache.get(key)
    if cached is None:
        SigmaMotor = np.zeros((N * (kdelay + 1), N * (kdelay + 1)))
        SigmaSense = np.diag(np.ones(N) * 1e-4)

        for i in range(2, 4):

            SigmaMotor[i, i] = motornoise_variance

        cached = (SigmaMotor, SigmaSense)
        _noise_cov_cache[key] = cached
    SigmaMotor, SigmaSense = cached

    sensorynoise = np.random.normal(0, 1e-2, N)

    return SigmaMotor, SigmaSense, sensorynoise


def next_state_estimate(
    est_x, true_x, u, dt, activated_noise, delay, sigma, motornoise_variance
):
    """
    Compute one step of the dynamics of the system, composed of a two joint biomechanical model, and a nonlinear network dynamic
    \ddot{\theta} = M^{-1}(Wout gamma-B \dot{\theta} - C)
    \dot{\gamma} = tanh(W gamma) + u
    Args:
        x : x_t biomechanical state at time t [theta_s, theta_e, omega_s, omega_e, angular_acc_s, angular_acc_e]
        gamma : gamma_t vector of the newtork activity at time t
        u : command
        dt : timestep
        Wout : readout matrix
        W : internal network connectivity
    Returns:
        [x_{t+1},gamma_{t+1}]
    """
    H, A, B = _delayed_dynamics(dt, delay)

    Omega_motor, Omega_measure, sensorynoise = NoiseAndCovMatrix(
        kdelay=delay, motornoise_variance=motornoise_variance
    )
    K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
    sigma = Omega_motor + (A - K @ H) @ sigma @ A.T

    y = H @ true_x
    if activated_noise:
        y += sensorynoise
    next_est_x = A @ est_x + B @ u + K @ (y - H @ est_x)
    return next_est_x, sigma


def ToCartesian(s, e):
    X = np.cos(s + e) * 33 + np.cos(s) * 30
    Y = np.sin(s + e) * 33 + np.sin(s) * 30

    return X, Y

def f(x, u, F=0):
    C = np.array(
        [-x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]), x[2] ** 2 * a2 * np.sin(x[1])]
    )

    Denominator = a3 * (a1 - a3) - a2**2 * np.cos(x[1]) ** 2
    Minv = np.array(
        [
            [a3 / Denominator, (-a2 * np.cos(x[1]) - a3) / Denominator],
            [
                (-a2 * np.cos(x[1]) - a3) / Denominator,
                (2 * a2 * np.cos(x[1]) + a1) / Denominator,
            ],
        ]
    )
    _, _, _, fl, ff_v = muscle_force_scaling(x)
    theta = Minv @ (MOMENT_ARM @ (u * fl * ff_v) - Viscous @ x[2:4] - C + F)

    return np.array([[x[2], x[3], theta[0], theta[1]]])


def l(x, u, r1, xtarg=0, w1=0, w2=0):
    return r1 * np.sum(u**2) / 2


def lx(x, u, xtarg=0, w1=0, w2=0):
    return np.zeros(4)


def lu(x, u, r1):
    return r1 * u


def lxx(w1=0, w2=0):
    return np.zeros((4, 4))


def luu(x, u, r1):
    return np.diag(np.ones(len(u))) * r1


def h(x, w1, w2, xtarg):
    return w1 / 2 * ((x[0] - xtarg[0]) ** 2 + (x[1] - xtarg[1]) ** 2) + w2 / 2 * (
        x[2] ** 2 + x[3] ** 2
    )


def hx(x, w1, w2, xtarg):
    return np.array(
        [w1 * (x[0] - xtarg[0]), w1 * (x[1] - xtarg[1]), w2 * x[2], w2 * x[3]]
    )


def hxx(x, w1, w2):
    return np.diag([w1, w1, w2, w2])


def Kalman(Omega_measure, Omega_sens, A, sigma, H):
    K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
    sigma = Omega_sens + (A - K @ H) @ sigma @ A.T
    return K, sigma


def step1(x0, u, A, B):

    K = np.shape(u)[0]
    newx = np.zeros((K + 1, len(x0)))
    newx[0] = np.copy(x0)

    for i in range(K):
        newx[i + 1] = A@newx[i] + B@u[i]

    return newx


def quadratizecost(x, u, Duration, w1, w2, r1, xtarg):
    K = np.shape(u)[0]
    dt = Duration / K
    n, m = len(x[0]), len(u[0])
    q, qbold = np.zeros(K + 1), np.zeros((K + 1, n))
    r, Q, R = np.zeros((K, m)), np.zeros((K + 1, n, n)), np.zeros((K, m, m))
    Q_step = dt * lxx(w1, w2)
    R_step = dt * luu(x[0], u[0], r1)

    for i in range(K):

        q[i] = dt * l(x[i], u[i], r1, xtarg, w1, w2)
        qbold[i] = dt * lx(x[i], u[i], xtarg, w1, w2)
        r[i] = dt * lu(x[i], u[i], r1)
        Q[i] = Q_step
        R[i] = R_step

    q[-1], qbold[-1], Q[-1] = (
        h(x[-1], w1, w2, xtarg),
        hx(x[-1], w1, w2, xtarg),
        hxx(x[-1], w1, w2),
    )
    return q, qbold, r, Q, R


def step3(A, B, C, cbold, q, qbold, r, Q, R, eps):
    K = A.shape[0]
    n, m = np.shape(B[0])
    S = np.zeros((K + 1, n, n))
    s = np.zeros(K + 1)
    sbold = np.zeros((K + 1, n))
    l = np.zeros((K, m))
    L = np.zeros((K, m, n))

    S[-1] = Q[-1]
    s[-1] = q[-1]
    sbold[-1] = qbold[-1]

    for k in range(K - 1, -1, -1):
        temp1, temp2, temp3 = 0, 0, 0

        Sk = S[k + 1]
        for i in range(m):
            Ci = C[k, i, :, :]
            cboldi = cbold[k, i, :]
            # C[k,i].T @ S is common to temp1 and temp2; @ is left-associative
            # so factoring it out leaves both products unchanged.
            CiTS = Ci.T @ Sk
            temp1 += CiTS @ cboldi
            temp2 += CiTS @ Ci
            temp3 += cboldi.T @ Sk @ cboldi

        gbold = r[k] + B[k].T @ sbold[k + 1] + temp1
        G = B[k].T @ S[k + 1] @ A[k]
        H = R[k] + B[k].T @ S[k + 1] @ B[k] + temp2

        eigenvalues, eigenvectors = np.linalg.eig(H)
        V = np.diag(eigenvalues)

        for i in range(V.shape[0]):
            if V[i, i] < (eps):
                print("H matrix poorly defined")
                V[i, i] = eps
            V[i, i] = 1 / V[i, i]
        Hinv = eigenvectors @ V @ np.linalg.inv(eigenvectors)

        S[k] = Q[k] + A[k].T @ S[k + 1] @ A[k] - G.T @ Hinv @ G
        sbold[k] = qbold[k] + A[k].T @ sbold[k + 1] - G.T @ Hinv @ gbold
        s[k] = q[k] + s[k + 1] + 0.5 * temp3 - 0.5 * gbold.T @ Hinv @ gbold

        l[k] = -Hinv @ gbold
        L[k] = -Hinv @ G

    return l, L


def compute_uincr(l, L, K, A, B):
    m, n = L[0].shape
    x = np.zeros(n)
    u_incr = np.zeros((K, m))

    for k in range(K):
        u_incr[k] = l[k] + L[k] @ x
        x = A[k] @ x + B[k] @ u_incr[k]

    return u_incr



def nonlinear_transform_command(u, L, x):
    M = np.array(
        [[a1 + 2 * a2 * cos(x[1]), a3 + a2 * cos(x[1])], [a3 + a2 * cos(x[1]), a3]]
    )

    linear_command = u + L @ x

    C = np.array(
        [
            -x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]),
            x[2] ** 2 * a2 * np.sin(x[1]),
        ]
    )
    fl, ff_v = muscle_force_scaling(x)

    U = MOMENT_ARM_PINV @ (M @ linear_command + C + Viscous @ x[2:4])
    u = U / (fl * ff_v)
    return u, linear_command

def perform_ilqg_iteration(targetangle,startingangle,Duration,A,B,cost_weights,eps,K,motornoise_variance):

    
    w1,w2,r = cost_weights
    x0 = np.array([startingangle[0], startingangle[1], 0, 0])
    m, n = 2, 4
    u = np.zeros((K, m))

    cbold = np.zeros((K, m, n))
    C = np.zeros((K, m, n, m))
    for i in range(K):
        for j in range(2):
            cbold[i, j, 2 + j] = sqrt(motornoise_variance)

    u_incr = np.ones(u.shape) * np.inf

    for iterate in range(300):
        x = step1(
            x0, u, Duration
        )  # Forward step computing the sequence of state trajectory given a sequence of input u

        if (
            np.max(np.abs(u_incr)) < 1e-14
        ): break # If the trajectory improvement is small enough, stop the iteration and perform a full simulation with feedback and potential noise

        q, qbold, r, Q, R = quadratizecost(
            x, u, Duration, w1, w2, r, targetangle
        )  # Compute the Linearizations of the dynamic
        l, L = step3(
            A, B, C, cbold, q, qbold, r, Q, R, eps
        )  # Compute the control gains improvement (feedforward and feedback)
        u_incr = compute_uincr(l, L, K, A, B)  # Compute the command sequence improvement
        u += u_incr  # Improves the command sequence
    return u, L

def simulate_FL(
    Duration=0.6,
    w1=1e8,
    w2=1e4,
    r=1e-5,
    targets=[0, 55],
    starting_point=[0, 30],
    Activate_Noise=False,
    Num_iter=300,
    Delay=0.06,
    FF=False,
    ff_power=0.3,
    motornoise_variance=1e-3,
    Cartesian = False
):
    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    num_states = 4

    all_true_states = np.zeros((Num_iter + 1, num_states + 4))
    all_estimated_states = np.zeros((Num_iter + 1, (num_states + 4)))
    all_commands = np.zeros((Num_iter, 6))
    st1, st2 = compute_angles_from_cartesian(starting_point[0], starting_point[1])

    tg1, tg2 = compute_angles_from_cartesian(targets[0], targets[1])
    x0 = np.array([st1, st2, 0, 0, tg1, tg2, st1, st2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    true_state, estimated_state = x0_with_delay, x0_with_delay
    all_true_states[0, :] = np.copy(x0)
    all_estimated_states[0, :] = np.copy(x0)
    sigma = np.zeros((8 * (kdelay + 1), 8 * (kdelay + 1)))
    F = np.zeros(2)
    A,B = _delayed_dynamics(dt, 0)[1:]

    u,L = perform_ilqg_iteration(np.array([tg1, tg2]), np.array([st1, st2]), Duration, A, B, (w1,w2,r), 1e-4, Num_iter, motornoise_variance
    )

    for j in range(Num_iter):
        u, v = nonlinear_transform_command(u[j], L[j], estimated_state[:8])
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
    return X, Y, all_true_states, all_commands
