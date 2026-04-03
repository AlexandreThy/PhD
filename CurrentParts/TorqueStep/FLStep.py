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

def compute_next_state(x, u, dt, activate_noise, FF, ff_power, motornoise_variance, torque_step):
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
    A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])
    l0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
    theta0 = np.array(
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
    l = 1 + A[0] * (theta0[0] - x[0]) / l0 + A[1] * (theta0[1] - x[1]) / l0
    v = A[0] * (-x[2]) / l0 + A[1] * (-x[3]) / l0
    fl = np.exp(np.abs((l**1.55 - 1) / 0.81))
    ff_v = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    F = np.zeros(2)
    if FF == True:
        F = compute_forcefield(x[0:2], x[2:4], ff_power)
    noise = (
        np.random.normal(0, np.sqrt(motornoise_variance), 2)
        if activate_noise
        else np.zeros(2)
    )
    
    x[4:6] = torque_step
    newx[2:4] += (
        dt * np.linalg.solve(M, (A @ (u * fl * ff_v) - Viscous @ (x[2:4]) - C + F + x[4:6]))
        + noise)
    newx[4:6] = x[4:6]
    return newx


def NoiseAndCovMatrix(M=np.identity(2), N=10, kdelay=0, motornoise_variance=1e-3):

    SigmaMotor = np.zeros((N * (kdelay + 1), N * (kdelay + 1)))
    SigmaSense = np.diag(np.ones(N) * 1e-4)

    for i in range(2, 4):

        SigmaMotor[i, i] = motornoise_variance

    sensorynoise = np.zeros(N)
    for i in range(N):
        sensorynoise[i] = np.random.normal(0, 1e-2)

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
    H = np.zeros((10, (delay + 1) * 10))
    H[:, delay * 10 :] = np.identity(10)

    A_basic = np.array(
        [
            [1, 0, dt, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, dt, 0, 0, 0, 0,0 ,0],
            [0, 0, 1, 0, 0, 0, 0, 0,0,0],
            [0, 0, 0, 1, 0, 0, 0, 0,0,0],
            [0, 0, 0, 0, 1, 0, 0, 0,0,0],
            [0, 0, 0, 0, 0, 1, 0, 0,0,0],
            [0, 0, 0, 0, 0, 0, 1, 0,0,0],
            [0, 0, 0, 0, 0, 0, 0, 1,0,0],
            [0, 0, 0, 0, 0, 0, 0, 0,1,0],
            [0, 0, 0, 0, 0, 0, 0, 0,0,1],
        ]
    )
    B_basic = np.zeros((10, 2))
    B_basic[2, 0] = dt
    B_basic[3, 1] = dt
    A = np.zeros(((delay + 1) * 10, (delay + 1) * 10))
    A[:10, :10] = A_basic
    A[10:, :-10] = np.identity((delay) * 10)
    B = np.zeros(((delay + 1) * 10, 2))
    B[:10] = B_basic

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


def compute_path(x0, xf, wp, percent):
    # percent = compute_percent(x0,xf)
    xtarget = x0 + percent * (xf - x0)
    ts, te = compute_angles_from_cartesian(xtarget[0], xtarget[1])
    ts0, te0 = compute_angles_from_cartesian(x0[0], x0[1])
    k = (te - te0) / (ts - ts0)
    Qk = np.zeros((10,10))

    Qk[0, 0] = k * k
    Qk[0, 6] = -k * k
    Qk[6, 0] = -k * k
    Qk[6, 6] = k * k
    Qk[1, 1] = 1
    Qk[1, 7] = -1
    Qk[7, 1] = -1
    Qk[7, 7] = 1
    Qk[0, 1] = -k
    Qk[1, 0] = -k
    Qk[0, 7] = k
    Qk[7, 0] = k
    Qk[1, 6] = k
    Qk[6, 1] = k
    Qk[6, 7] = -k
    Qk[7, 6] = -k

    return Qk * wp


def ToCartesian(s, e):
    X = np.cos(s + e) * 33 + np.cos(s) * 30
    Y = np.sin(s + e) * 33 + np.sin(s) * 30

    return X, Y


def compute_percent(x0, xf):
    ts, te = compute_angles_from_cartesian(xf[0], xf[1])
    ts0, te0 = compute_angles_from_cartesian(x0[0], x0[1])
    te_incr, ts_incr = (te0 + 0.001 * (te - te0)), (ts0 + 0.001 * (ts - ts0))
    x_incr, y_incr = ToCartesian(ts_incr, te_incr)
    angle_objective = np.arctan2(xf[1] - x0[1], xf[0] - x0[0]) * 180 / pi
    actual_angle = np.arctan2(y_incr - x0[1], x_incr - x0[0]) * 180 / pi
    percent = 1 if abs(angle_objective - actual_angle) < 30 else 0.7
    return percent


def compute_linear_control_gains(
    Num_iter,
    Duration,
    Qk,
    taupath,
    motor_cost=1e-4,
    cost_weights=[1e4, 1e4, 1, 1],
):
    dt = Duration / Num_iter
    Num_Var = 10

    R = np.diag(np.ones(2)) * motor_cost
    w1, w2, w3, w4 = cost_weights
    Q = np.array([[w1, 0, 0, 0, 0, 0, -w1,0,0,0],
            [0, w2, 0, 0, 0, 0, 0,-w2,0,0],
            [0, 0, w3, 0, 0, 0, 0,0,0,0],
            [0, 0, 0, w4, 0, 0,0,0,0, 0],
            [0, 0, 0, 0, 0, 0, 0,0,0,0],
            [0, 0, 0, 0, 0, 0, 0,0,0,0],
            [-w1, 0, 0, 0, 0, 0, w1, 0,0,0],
            [0, -w2, 0, 0, 0, 0, 0, w2,0,0],
            [0, 0, 0, 0, 0, 0, 0, 0,0,0],
            [0, 0, 0, 0, 0, 0, 0, 0,0,0],
        ]
    )

    A = np.array(
        [
            [1, 0, dt, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, dt, 0, 0, 0, 0,0 ,0],
            [0, 0, 1, 0, 0, 0, 0, 0,0,0],
            [0, 0, 0, 1, 0, 0, 0, 0,0,0],
            [0, 0, 0, 0, 1, 0, 0, 0,0,0],
            [0, 0, 0, 0, 0, 1, 0, 0,0,0],
            [0, 0, 0, 0, 0, 0, 1, 0,0,0],
            [0, 0, 0, 0, 0, 0, 0, 1,0,0],
            [0, 0, 0, 0, 0, 0, 0, 0,1,0],
            [0, 0, 0, 0, 0, 0, 0, 0,0,1],
        ]
    )
    B = np.zeros((10, 2))
    B[2, 0] = dt
    B[3, 1] = dt

    S = Q
    L = np.zeros((Num_iter, 2, Num_Var))

    for k in range(Num_iter):
        L[Num_iter - 1 - k] = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
        S = Qk * np.exp(-k * dt / (taupath)) + A.T @ S @ (A - B @ L[Num_iter - 1 - k])
    return L


def nonlinear_transform_command(L, x):
    M = np.array(
        [[a1 + 2 * a2 * cos(x[1]), a3 + a2 * cos(x[1])], [a3 + a2 * cos(x[1]), a3]]
    )

    linear_command = -L @ x

    C = np.array(
        [
            -x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]),
            x[2] ** 2 * a2 * np.sin(x[1]),
        ]
    )
    A = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])
    l0 = np.array([7.32, 3.26, 6.4, 4.26, 5.95, 4.04])
    theta0 = np.array(
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
    l = 1 + A[0] * (theta0[0] - x[0]) / l0 + A[1] * (theta0[1] - x[1]) / l0
    v = A[0] * (-x[2]) / l0 + A[1] * (-x[3]) / l0

    fl = np.exp(np.abs((l**1.55 - 1) / 0.81))
    ff_v = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )

    U = np.linalg.pinv(A) @ (M @ linear_command + C + Viscous @ x[2:4])
    u = np.zeros(6)
    for i in range(6):
        u[i] = U[i] / (fl[i] * ff_v[i])
    return u, linear_command


def simulate_FL(
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
    torque_val = [0, 0],
    torque_time = .2
):
    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    Qk = compute_path(np.array(starting_point), np.array(targets), wp, percent)
    L = compute_linear_control_gains(
        Num_iter,
        Duration,
        Qk,
        taupath,
        motor_cost=r,
        cost_weights=[w1, w2, w3, w4],
    )
    num_states = 6

    all_true_states = np.zeros((Num_iter + 1, num_states + 4))
    all_estimated_states = np.zeros((Num_iter + 1, (num_states + 4)))
    all_commands = np.zeros((Num_iter, 6))
    st1, st2 = compute_angles_from_cartesian(starting_point[0], starting_point[1])

    tg1, tg2 = compute_angles_from_cartesian(targets[0], targets[1])
    x0 = np.array([st1, st2, 0, 0, 0, 0, tg1, tg2, st1, st2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    true_state, estimated_state = x0_with_delay, x0_with_delay
    all_true_states[0, :] = np.copy(x0)
    all_estimated_states[0, :] = np.copy(x0)
    sigma = np.zeros((10 * (kdelay + 1), 10 * (kdelay + 1)))

    for j in range(Num_iter):
        tv = torque_val if j*dt >= torque_time else [0,0]
        u, v = nonlinear_transform_command(L[j], estimated_state[:10])
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
        new_state= compute_next_state(
            true_state[:10], u, dt, Activate_Noise, FF, ff_power, motornoise_variance, tv
        )
        true_state = np.concatenate((new_state, true_state[:-10]))

        all_true_states[j + 1, :] = true_state[:10]
        all_estimated_states[j + 1, :] = estimated_state[:10]
        all_commands[j] = u

    s, e = all_true_states[:, 0], all_true_states[:, 1]
    X = np.cos(s + e) * 33 + np.cos(s) * 30
    Y = np.sin(s + e) * 33 + np.sin(s) * 30
    return X, Y, all_true_states, all_commands
