import numpy as np
from matplotlib import pyplot as plt
from math import *
import matplotlib.gridspec as gridspec

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

Bdyn = np.array([[0.05, 0.025], [0.025, 0.05]])


def newton(f, Df, epsilon, max_iter, X, Y, x0=np.array([0.8, 1.5])):
    """
    Compute joint angles given a cartesian position X,Y

    Args:
        f: A function that describes the change of coordinate Angular -> Cartesian, giving the error between current estimate and target values .
        df: Analytical derivative of f for newton step.
        epsilon : Acceptable error threshold
        X : X targetted cartesian position
        Y : Y targetted cartesian position

    Returns:
        [theta_s,theta_e] , the joint angles producing the desired (X,Y) cartesian coordinates

    Example:
        >>> calculate_area(f,df,1e-8,1000,0,30)
        [thetas,thetae]
    """
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn, X, Y)
        if abs(np.max(np.abs(fxn))) < epsilon:
            return xn
        Dfxn = Df(xn)
        if np.max(np.abs(Dfxn)) < epsilon:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - np.linalg.inv(Dfxn) @ fxn
    print("Exceeded maximum iterations. No solution found.")
    return None


def newtonf(var, X, Y):
    u, v = var
    return np.array(
        [
            33 * np.cos(u + v) + 30 * np.cos(u) - X,
            33 * np.sin(u + v) + 30 * np.sin(u) - Y,
        ]
    )


def newtondf(var):
    u, v = var
    return np.array(
        [
            [-33 * np.sin(u + v) - 30 * np.sin(u), -33 * np.sin(u + v)],
            [33 * np.cos(u + v) + 30 * np.cos(u), 33 * np.cos(u + v)],
        ]
    )


def Compute_gamma_nu(theta, omega):
    fe = -33 * np.sin(theta[0] + theta[1])
    fs = -33 * np.sin(theta[0] + theta[1]) - 30 * np.sin(theta[0])
    ge = 33 * np.cos(theta[0] + theta[1])
    gs = 33 * np.cos(theta[0] + theta[1]) + 30 * np.cos(theta[0])
    fse = -33 * np.cos(theta[0] + theta[1])
    gse = -33 * np.sin(theta[0] + theta[1])
    fee = -33 * np.cos(theta[0] + theta[1])
    fss = -33 * np.cos(theta[0] + theta[1]) - 30 * np.cos(theta[0])
    gee = fe
    gss = fs
    gamma = (
        gs * omega[0]
        + ge * omega[1]
        - fss * omega[0] * omega[0]
        - 2 * fse * omega[0] * omega[1]
        - fee * omega[1] * omega[1]
    )
    nu = (
        -gss * omega[0] * omega[0]
        - 2 * gse * omega[0] * omega[1]
        - gee * omega[1] * omega[1]
    )
    return gamma, nu, fs, fe, gs, ge, fss, fse, fee, gss, gse, gee


def pre_Compute(theta, omega):
    fe = -33 * np.sin(theta[0] + theta[1])
    fs = -33 * np.sin(theta[0] + theta[1]) - 30 * np.sin(theta[0])
    ge = 33 * np.cos(theta[0] + theta[1])
    gs = 33 * np.cos(theta[0] + theta[1]) + 30 * np.cos(theta[0])
    fse = -33 * np.cos(theta[0] + theta[1])
    gse = -33 * np.sin(theta[0] + theta[1])
    fee = -33 * np.cos(theta[0] + theta[1])
    fss = -33 * np.cos(theta[0] + theta[1]) - 30 * np.cos(theta[0])
    gee = fe
    gss = fs
    return fs, fe, gs, ge, fss, fse, fee, gss, gse, gee


def Compute_f_new_version(theta, omega, acc, factor):
    fs, fe, gs, ge, fss, fse, fee, gss, gse, gee = pre_Compute(theta, omega)
    xddot = (
        13 * (gs * omega[0] + ge * omega[1]) * factor
        + fss * omega[0] * omega[0]
        + 2 * fse * omega[0] * omega[1]
        + fee * omega[1] * omega[1]
        + fs * acc[0]
        + fe * acc[1]
    )
    yddot = (
        gss * omega[0] * omega[0]
        + 2 * gse * omega[0] * omega[1]
        + gee * omega[1] * omega[1]
        + gs * acc[0]
        + ge * acc[1]
    )
    gamma = (
        xddot
        - fss * omega[0] * omega[0]
        - 2 * fse * omega[0] * omega[1]
        - fee * omega[1] * omega[1]
    )
    nu = (
        yddot
        - gss * omega[0] * omega[0]
        - 2 * gse * omega[0] * omega[1]
        - gee * omega[1] * omega[1]
    )
    F1 = (fe * nu - ge * gamma) / (fe * gs - ge * fs) - acc[0]
    F2 = (gs * gamma - fs * nu) / (gs * fe - ge * fs) - acc[1]
    return np.array([F1, F2])


def sysdyn(x, u, dt, activate_noise, FF, F, ff_power):
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
    # Equation (6): fl(l)
    fl = np.exp(np.abs((l**1.55 - 1) / 0.81))
    # Equation (7): ff_v(l, v)
    ff_v = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    acc = (np.linalg.solve(M, (A @ (u * fl * ff_v) - Bdyn @ (x[2:4]) - C)) + F).reshape(
        2
    )
    F = (
        Compute_f_new_version(x[0:2], x[2:4], acc, ff_power)
        if FF == True
        else np.array([0, 0])
    )
    noise = np.random.normal(0, np.sqrt(1e-3), 2) if activate_noise else np.zeros(2)
    newx[2:4] += (
        dt * np.linalg.solve(M, (A @ (u * fl * ff_v) - Bdyn @ (x[2:4]) - C)) + noise
    )
    if FF == True:
        newx[2:4] += dt * F
    return newx, F


def NoiseAndCovMatrix(M=np.identity(2), N=6, kdelay=0):

    SigmaMotor = np.zeros((N * (kdelay + 1), N * (kdelay + 1)))
    SigmaSense = np.diag(np.ones(N) * 1e-4)
    for i in range(2, 4):

        SigmaMotor[i, i] = 1e-3

    sensorynoise = np.zeros(N)
    for i in range(N):
        sensorynoise[i] = np.random.normal(0, 1e-2)

    return SigmaMotor, SigmaSense, sensorynoise


def estdyn(est_x, true_x, u, dt, activated_noise, delay, sigma):
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
    H = np.zeros((6, (delay + 1) * 6))
    H[:, delay * 6 :] = np.identity(6)

    A_basic = np.array(
        [
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    B_basic = np.zeros((6, 2))
    B_basic[2, 0] = dt
    B_basic[3, 1] = dt
    A = np.zeros(((delay + 1) * 6, (delay + 1) * 6))
    A[:6, :6] = A_basic
    A[6:, :-6] = np.identity((delay) * 6)
    B = np.zeros(((delay + 1) * 6, 2))
    B[:6] = B_basic

    Omega_motor, Omega_measure, sensorynoise = NoiseAndCovMatrix(kdelay=delay)
    K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
    sigma = Omega_motor + (A - K @ H) @ sigma @ A.T

    y = H @ true_x
    if activated_noise:
        y += sensorynoise
    next_est_x = A @ est_x + B @ u + K @ (y - H @ est_x)
    return next_est_x, sigma


def compute_control_gains(
    Num_iter, Duration, motor_cost=1e-4, cost_weights=[1e4, 1e4, 1, 1]
):
    """
    Compute the control gains L

    Args:
        Num_iter : the number of simulation steps
        Duration : The Movement Duration
        motor_cost : the motor cost in the cost function r u^2
        cost_weights : q_ij the weight of the states in the cost function : x^TQx
    Returns:
        L : The control gains
    """
    dt = Duration / Num_iter
    Num_Var = 6

    R = np.diag(np.ones(2)) * motor_cost
    w1, w2, w3, w4 = cost_weights
    Q = np.array(
        [
            [w1, 0, 0, 0, -w1, 0],
            [0, w2, 0, 0, 0, -w2],
            [0, 0, w3, 0, 0, 0],
            [0, 0, 0, w4, 0, 0],
            [-w1, 0, 0, 0, w1, 0],
            [0, -w2, 0, 0, 0, w2],
        ]
    )

    A = np.array(
        [
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    B = np.zeros((6, 2))
    B[2, 0] = dt
    B[3, 1] = dt

    S = Q
    L = np.zeros((Num_iter - 1, 2, Num_Var))

    for k in range(Num_iter - 1):
        L[Num_iter - 2 - k] = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
        S = A.T @ S @ (A - B @ L[Num_iter - 2 - k])

    return L


def compute_nonlinear_command(L, x):
    """
    Applies the feedback linearization technique to compute the nonlinear command

    Args:
        L : The control gains
        x : The state of the system at time t
        Wout : readout matrix
        W : internal network connectivity
        gamma : the state of the newtork

    Returns:
        u : the nonlinear command to be send to the network
    """
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

    # Equation (6): fl(l)
    fl = np.exp(np.abs((l**1.55 - 1) / 0.81))

    # Equation (7): ff_v(l, v)
    ff_v = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    # print(v.shape, C.shape, M.shape, Bdyn.shape, np.linalg.pinv(A).shape)
    U = np.linalg.pinv(A) @ (M @ linear_command + C + Bdyn @ x[2:4])
    u = np.zeros(6)
    for i in range(6):
        u[i] = U[i] / (fl[i] * ff_v[i])
    return u, linear_command


# now define the 8 condition reaching controller
def FL_6muscles(
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
    FF=True,
    ff_power=0.3,
):
    """Simulates an eight-condition reaching task with control gains and neural network dynamics."""

    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    L = compute_control_gains(
        Num_iter, Duration, motor_cost=r, cost_weights=[w1, w2, w3, w4]
    )
    num_states = 4

    all_true_states = np.zeros((Num_iter, num_states + 2))
    all_estimated_states = np.zeros((Num_iter, (num_states + 2)))
    all_commands = np.zeros((Num_iter - 1, 6))

    st1, st2 = newton(
        newtonf, newtondf, 1e-8, 1000, starting_point[0], starting_point[1]
    )
    tg1, tg2 = newton(newtonf, newtondf, 1e-8, 1000, targets[0], targets[1])
    x0 = np.array([st1, st2, 0, 0, tg1, tg2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    true_state, estimated_state = x0_with_delay, x0_with_delay
    all_true_states[0, :] = np.copy(x0)
    all_estimated_states[0, :] = np.copy(x0)
    sigma = np.zeros((6 * (kdelay + 1), 6 * (kdelay + 1)))
    F = np.zeros(2)
    for j in range(Num_iter - 1):
        u, v = compute_nonlinear_command(L[j], estimated_state[:6])
        estimated_state, sigma = estdyn(
            estimated_state, true_state, v, dt, Activate_Noise, kdelay, sigma
        )
        new_state, F = sysdyn(true_state[:6], u, dt, Activate_Noise, FF, F, ff_power)
        true_state = np.concatenate((new_state, true_state[:-6]))

        all_true_states[j + 1, :] = true_state[:6]
        all_estimated_states[j + 1, :] = estimated_state[:6]
        all_commands[j] = u

    s, e = all_true_states[:, 0], all_true_states[:, 1]
    X = np.cos(s + e) * 33 + np.cos(s) * 30
    Y = np.sin(s + e) * 33 + np.sin(s) * 30
    return X, Y, all_true_states, all_commands
