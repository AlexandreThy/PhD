import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from math import *
import matplotlib.gridspec as gridspec

I1 = 0.025
I2 = 0.045
m1 = 1.4
m2 = 1
l1 = 0.3
l2 = 0.33
s1 = 0.11
s2 = 0.16
K = 1 / 0.06
tau = 0.06


a1 = I1 + I2 + m2 * l1 * l1
a2 = m2 * l1 * s2
a3 = I2

Bvisc = np.array([[0.05, 0.025], [0.025, 0.05]])

# newton functions are used to implement a newton-raphson method that computes the joint angles given a desired cartesian position.


def delete_axis(ax, sides=["left", "right", "bottom", "top"]):
    for side in sides:
        ax.spines[side].set_visible(False)


def compute_angles_from_cartesian(x, y, l1=30, l2=33):
    """
    Computes h1 using the given equation.

    Parameters:
        x (float): x-coordinate of the end effector.
        y (float): y-coordinate of the end effector.
        l1 (float): Length of the first link.
        l2 (float): Length of the second link.

    Returns:
        h1 (float): Computed angle in radians.
    """

    # Compute the second term: arccos formula
    r_squared = x**2 + y**2

    # Final h1 calculation
    h1 = np.arctan2(y, x) - np.arccos(
        (r_squared + l1**2 - l2**2) / (2 * l1 * np.sqrt(r_squared))
    )

    # Compute h2
    h2 = np.pi - np.arccos((l1**2 + l2**2 - r_squared) / (2 * l1 * l2))
    return h1, h2


def ToCartesian(x, at3=False):
    elbowindex = 3 if at3 else 1
    if len(x.shape) == 1:
        s = x[0]
        e = x[elbowindex]
    else:
        s = x[:, 0]
        e = x[:, elbowindex]
    X = np.cos(s + e) * 33 + np.cos(s) * 30
    Y = np.sin(s + e) * 33 + np.sin(s) * 30

    return X, Y


def compute_absolute_velocity(x, y, dt):
    """
    Computes the absolute velocity (magnitude) in the X-Y plane.

    Parameters:
    - x (array-like): X-coordinates over time
    - y (array-like): Y-coordinates over time
    - dt (float): Time step between each sample

    Returns:
    - np.ndarray: Array of absolute velocity values
    """

    # Compute the differences between consecutive x and y points
    dx = np.diff(x)
    dy = np.diff(y)

    # Compute velocity components
    vx = dx / dt
    vy = dy / dt

    # Compute absolute velocity (Euclidean norm)
    velocity = np.sqrt(vx**2 + vy**2)
    velocity = np.insert(velocity, 0, 0.0)

    return velocity


def Linearization(x, alpha):
    """
    Parameters :
        - x : the state of the system
        - alpha : the body tilt

    return :
        The Jacobian Matrix of the dynamic of the system around the state x
    """
    TimeConstant = 1 / 0.06

    # Extract state variables
    theta1, theta2, dtheta1, dtheta2, tau1, tau2 = x[:6]
    # Gravity force
    G = np.array(
        [
            g
            * (
                m1 * s1 * np.cos(theta1 + alpha)
                + m2
                * (s2 * np.cos(theta1 + theta2 + alpha) + l1 * np.cos(theta1 + alpha))
            ),
            g * m2 * s2 * np.cos(theta1 + theta2 + alpha),
        ]
    )

    dGdts = np.array(
        [
            g
            * (
                -m1 * s1 * np.sin(theta1 + alpha)
                - m2
                * (s2 * np.sin(theta1 + theta2 + alpha) + l1 * np.sin(theta1 + alpha))
            ),
            -g * m2 * s2 * np.sin(theta1 + theta2 + alpha),
        ]
    )

    dGdte = np.array(
        [
            g * (-m2 * (s2 * np.sin(theta1 + theta2 + alpha))),
            -g * m2 * s2 * np.sin(theta1 + theta2 + alpha),
        ]
    )
    # Coriolis force
    C = np.array(
        [
            -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.sin(theta2),
            dtheta1**2 * a2 * np.sin(theta2),
        ]
    )

    dCdte = np.array(
        [
            -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.cos(theta2),
            dtheta1**2 * a2 * np.cos(theta2),
        ]
    )
    dCdos = np.array(
        [-dtheta2 * 2 * a2 * np.sin(theta2), 2 * dtheta1 * a2 * np.sin(theta2)]
    )
    dCdoe = np.array([(-2 * dtheta1 - 2 * dtheta2) * a2 * np.sin(theta2), 0])

    # Inertia matrix
    M = np.array(
        [
            [a1 + 2 * a2 * np.cos(theta2), a3 + a2 * np.cos(theta2)],
            [a3 + a2 * np.cos(theta2), a3],
        ]
    )

    Minv = np.linalg.inv(M)

    dM = np.array(
        [[-2 * a2 * np.sin(theta2), -a2 * np.sin(theta2)], [-a2 * np.sin(theta2), 0]]
    )

    # Compute acceleration dependencies
    dtheta = np.array([dtheta1, dtheta2])
    tau = np.array([tau1, tau2])

    d_accel_theta1 = -Minv @ (dGdts)
    d_accel_dtheta1 = -Minv @ (dCdos + Bvisc @ np.array([1, 0]))
    d_accel_tau = Minv @ np.array([1, 0])
    d_accel_theta2 = -Minv @ (dM @ Minv @ (tau - C - G - Bvisc @ dtheta)) - Minv @ (
        dCdte + dGdte
    )
    d_accel_dtheta2 = -Minv @ (dCdoe + Bvisc @ np.array([0, 1]))
    d_accel_tau2 = Minv @ np.array([0, 1])

    # Construct the Jacobian matrix
    A = np.zeros((6, 6))

    A[0, 2] = 1
    A[1, 3] = 1

    # Acceleration contributions
    A[2, 0] = d_accel_theta1[0]
    A[2, 2] = d_accel_dtheta1[0]
    A[2, 1] = d_accel_theta2[0]
    A[2, 3] = d_accel_dtheta2[0]
    A[2, 4] = d_accel_tau[0]
    A[2, 5] = d_accel_tau2[0]

    A[3, 0] = d_accel_theta1[1]
    A[3, 2] = d_accel_dtheta1[1]
    A[3, 1] = d_accel_theta2[1]
    A[3, 3] = d_accel_dtheta2[1]
    A[3, 4] = d_accel_tau[1]
    A[3, 5] = d_accel_tau2[1]

    # Torque dynamics
    A[4, 4] = -TimeConstant
    A[5, 5] = -TimeConstant
    return A


def f(x, u, alpha):
    C = np.array(
        [-x[3] * (2 * x[2] + x[3]) * a2 * np.sin(x[1]), x[2] ** 2 * a2 * np.sin(x[1])]
    )
    G = np.array(
        [
            g
            * (
                m1 * s1 * np.cos(x[0] + alpha)
                + m2 * (s2 * np.cos(x[0] + alpha + x[1]) + l1 * np.cos(x[0] + alpha))
            ),
            g * m2 * s2 * np.cos(x[0] + x[1] + alpha),
        ]
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

    theta = Minv @ (x[4:6] - Bvisc @ x[2:4] - C - G)
    torque = (u - x[4:6]) / 0.06

    return np.array([[x[2], x[3], theta[0], theta[1], torque[0], torque[1]]])


def fx(x, u, alpha):
    return Linearization(x, alpha)


def fu(x, u):
    tau = 0.06
    return np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1 / tau, 0], [0, 1 / tau]])


def X_l(x, u, r1, xtarg=0, w1=0, w2=0, wp=0):
    return (
        r1 * (u[0] ** 2 + u[1] ** 2) / 2
        + wp
        * (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1])) ** 2
        / 2
    )


def Y_l(x, u, r1, xtarg=0, w1=0, w2=0, wp=0):
    return (
        r1 * (u[0] ** 2 + u[1] ** 2) / 2
        + wp
        * (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1])) ** 2
        / 2
    )


def X_lx(x, u, xtarg=0, w1=0, w2=0, wp=0):
    return np.array(
        [
            wp
            * (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1]))
            * -compute_xcartesian_dts(x[0], x[1]),
            +wp
            * (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1]))
            * -compute_xcartesian_dte(x[0], x[1]),
            0,
            0,
            0,
            0,
        ]
    )


def Y_lx(x, u, xtarg=0, w1=0, w2=0, wp=0):
    return np.array(
        [
            wp
            * (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1]))
            * -compute_ycartesian_dts(x[0], x[1]),
            +wp
            * (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1]))
            * -compute_ycartesian_dte(x[0], x[1]),
            0,
            0,
            0,
            0,
        ]
    )


def X_lxx(x, wp, xtarg):
    lxx = np.zeros((6, 6))
    lxx[0, 0] += wp * (
        compute_xcartesian_dts(x[0], x[1]) ** 2
        - (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1]))
        * compute_xcartesian_dtss(x[0], x[1])
    )

    lxx[0, 1] += wp * (
        compute_xcartesian_dte(x[0], x[1]) * compute_xcartesian_dts(x[0], x[1])
        + (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1]))
        * -compute_xcartesian_dtse(x[0], x[1])
    )

    lxx[1, 0] += wp * (
        compute_xcartesian_dte(x[0], x[1]) * compute_xcartesian_dts(x[0], x[1])
        + (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1]))
        * -compute_xcartesian_dtse(x[0], x[1])
    )

    lxx[1, 1] += wp * (
        compute_xcartesian_dte(x[0], x[1]) ** 2
        - (compute_xcartesian(xtarg[0], xtarg[1]) - compute_xcartesian(x[0], x[1]))
        * compute_xcartesian_dtee(x[0], x[1])
    )
    return lxx


def Y_lxx(x, wp, xtarg):
    lxx = np.zeros((6, 6))
    lxx[0, 0] += wp * (
        compute_ycartesian_dts(x[0], x[1]) ** 2
        - (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1]))
        * compute_ycartesian_dtss(x[0], x[1])
    )

    lxx[0, 1] += wp * (
        compute_ycartesian_dte(x[0], x[1]) * compute_ycartesian_dts(x[0], x[1])
        + (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1]))
        * -compute_ycartesian_dtse(x[0], x[1])
    )

    lxx[1, 0] += wp * (
        compute_ycartesian_dte(x[0], x[1]) * compute_ycartesian_dts(x[0], x[1])
        + (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1]))
        * -compute_ycartesian_dtse(x[0], x[1])
    )

    lxx[1, 1] += wp * (
        compute_ycartesian_dte(x[0], x[1]) ** 2
        - (compute_ycartesian(xtarg[0], xtarg[1]) - compute_ycartesian(x[0], x[1]))
        * compute_ycartesian_dtee(x[0], x[1])
    )
    return lxx


def lu(x, u, r1):
    return np.array([u[0] * r1, u[1] * r1])


def luu(x, u, r1):
    return np.array([[r1, 0], [0, r1]])


def compute_xcartesian(ts, te):
    return 30 * np.sin(ts) + 33 * np.sin(ts + te)


def compute_xcartesian_dts(ts, te):
    return 30 * np.cos(ts) + 33 * np.cos(ts + te)


def compute_xcartesian_dte(ts, te):
    return 33 * np.cos(ts + te)


def compute_xcartesian_dtee(ts, te):
    return -33 * np.sin(ts + te)


def compute_xcartesian_dtse(ts, te):
    return -33 * np.sin(ts + te)


def compute_xcartesian_dtss(ts, te):
    return -compute_xcartesian(ts, te)


def compute_ycartesian(ts, te):
    return 30 * np.cos(ts) + 33 * np.cos(ts + te)


def compute_ycartesian_dts(ts, te):
    return -30 * np.sin(ts) - 33 * np.sin(ts + te)


def compute_ycartesian_dte(ts, te):
    return -33 * np.sin(ts + te)


def compute_ycartesian_dtee(ts, te):
    return -33 * np.cos(ts + te)


def compute_ycartesian_dtse(ts, te):
    return -33 * np.cos(ts + te)


def compute_ycartesian_dtss(ts, te):
    return -compute_ycartesian(ts, te)


def h(x, w1, w2, xtarg):
    return w1 / 2 * ((x[0] - xtarg[0]) ** 2 + (x[1] - xtarg[1]) ** 2) + w2 / 2 * (
        x[2] ** 2 + x[3] ** 2
    )


def hx(x, w1, w2, xtarg):
    return np.array(
        [w1 * (x[0] - xtarg[0]), w1 * (x[1] - xtarg[1]), w2 * x[2], w2 * x[3], 0, 0]
    )


def hxx(x, w1, w2):
    Hxx = np.diag([w1, w1, w2, w2, 0, 0])

    return Hxx


def Kalman(Omega_measure, Omega_sens, A, sigma, H):
    K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
    sigma = Omega_sens + (A - K @ H) @ sigma @ A.T
    return K, sigma


def step1(x0, u, Duration, alpha):
    K = np.shape(u)[0] + 1
    dt = Duration / (K - 1)
    newx = np.zeros((K, len(x0)))
    newx[0] = np.copy(x0)

    for i in range(K - 1):
        newx[i + 1] = newx[i] + dt * f(newx[i], u[i], alpha)

    return newx


def step2(x, u, Duration, w1, w2, wp, r1, xtarg, alpha, direction):
    K = np.shape(u)[0] + 1
    dt = Duration / K
    n, m = len(x[0]), len(u[0])

    A, B = np.zeros((K - 1, n, n)), np.zeros((K - 1, n, m))
    q, qbold = np.zeros(K), np.zeros((K, n))
    r, Q, R = np.zeros((K - 1, m)), np.zeros((K, n, n)), np.zeros((K - 1, m, m))

    for i in range(K - 1):
        if direction == "Horizontal":
            q[i] = dt * X_l(x[i], u[i], r1, xtarg, w1, w2, wp)
            qbold[i] = dt * X_lx(x[i], u[i], xtarg, w1, w2, wp)
            r[i] = dt * lu(x[i], u[i], r1)
            Q[i] = dt * X_lxx(x[i], wp, xtarg)
            R[i] = dt * luu(x[i], u[i], r1)

        elif direction == "Vertical":
            q[i] = dt * Y_l(x[i], u[i], r1, xtarg, w1, w2, wp)
            qbold[i] = dt * Y_lx(x[i], u[i], xtarg, w1, w2, wp)
            r[i] = dt * lu(x[i], u[i], r1)
            Q[i] = dt * Y_lxx(x[i], wp, xtarg)
            R[i] = dt * luu(x[i], u[i], r1)
        else:
            raise ValueError
        A[i] = np.identity(n) + dt * fx(x[i], u[i], alpha)
        B[i] = dt * fu(x[i], u[i])

    q[-1], qbold[-1], Q[-1] = (
        h(x[-1], w1, w2, xtarg),
        hx(x[-1], w1, w2, xtarg),
        hxx(x[-1], w1, w2),
    )
    return A, B, q, qbold, r, Q, R


def step3(A, B, C, cbold, q, qbold, r, Q, R):
    K = A.shape[0] + 1
    n, m = np.shape(B[0])
    S = np.zeros((K, n, n))
    s = np.zeros(K)
    sbold = np.zeros((K, n))
    l = np.zeros((K - 1, m))
    L = np.zeros((K - 1, m, n))

    S[-1] = Q[-1]
    s[-1] = q[-1]
    sbold[-1] = qbold[-1]

    for k in np.arange(K - 2, -1, -1):
        temp1, temp2, temp3 = 0, 0, 0

        for i in range(m):
            temp1 += C[k, i, :, :].T @ S[k + 1] @ cbold[k, i, :]
            temp2 += C[k, i, :, :].T @ S[k + 1] @ C[k, i, :, :]
            temp3 += cbold[k, i, :].T @ S[k + 1] @ cbold[k, i, :]

        gbold = r[k] + B[k].T @ sbold[k + 1] + temp1
        G = B[k].T @ S[k + 1] @ A[k]
        H = R[k] + B[k].T @ S[k + 1] @ B[k] + temp2
        Hinv = np.linalg.inv(H)

        S[k] = Q[k] + A[k].T @ S[k + 1] @ A[k] - G.T @ Hinv @ G
        sbold[k] = qbold[k] + A[k].T @ sbold[k + 1] - G.T @ Hinv @ gbold
        s[k] = q[k] + s[k + 1] + 0.5 * temp3 - 0.5 * gbold.T @ Hinv @ gbold

        l[k] = -Hinv @ gbold
        L[k] = -Hinv @ G

    return l, L


def step4(l, L, K, A, B):
    m, n = L[0].shape
    x = np.zeros(n)
    u_incr = np.zeros((K - 1, m))

    for k in range(K - 1):
        u_incr[k] = l[k] + L[k] @ x
        x = A[k] @ x + B[k] @ u_incr[k]

    return u_incr


def step5(
    x0,
    l,
    L,
    Duration,
    Noise,
    A,
    B,
    Num_steps,
    bestu,
    kdelay,
    motornoise_variance,
    alpha,
    cbold,
    C,
):
    dt = Duration / (Num_steps - 1)
    Num_Var = len(x0)

    x0 = np.tile(x0, kdelay + 1)
    xref = np.zeros((Num_steps, Num_Var * (kdelay + 1)))
    xref[0] = np.copy(x0)
    newx = np.zeros((Num_steps, Num_Var * (kdelay + 1)))
    newx[0] = np.copy(x0)
    xhat = np.zeros((Num_steps, Num_Var * (kdelay + 1)))

    H = np.zeros((Num_Var, (kdelay + 1) * Num_Var))
    H[:, (kdelay) * Num_Var :] = np.identity(Num_Var)

    sigma = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    Omega_measure = np.diag(np.ones(6)) * 1e-6
    sigmax = np.zeros((len(x0), len(x0)))
    mx = np.zeros(len(x0))
    temp1, temp2, temp3 = 0, 0, 0
    ##

    for i in range(Num_steps - 1):

        Extended_A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
        Extended_A[:Num_Var, :Num_Var] = A[i]
        Extended_A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)
        Extended_B = np.zeros(((kdelay + 1) * Num_Var, 2))
        Extended_B[:Num_Var] = B[i]

        deltau = l[i] + L[i] @ xhat[i, :Num_Var]
        u = bestu[i] + deltau

        for j in range(len(u)):
            temp1 += cbold[i, j, :] @ cbold[i, j, :].T
            temp2 += C[i, j, :, :] @ (l[i] + L[i] @ mx) @ cbold[i, j, :].T
            temp3 += (
                C[i, j, :, :]
                @ (
                    l[i] @ l[i].T
                    + l[i] @ (mx.T @ L[i].T)
                    + L[i] @ mx @ l[i].T
                    + L[i] @ sigmax @ L[i].T
                )
                @ C[i, j, :, :].T
            )

        Omega_sens = np.zeros((len(x0), len(x0)))
        Omega_sens[5, 5] = motornoise_variance
        Omega_sens[4, 4] = motornoise_variance
        mx = (Extended_A + Extended_B @ L[i]) @ mx + Extended_B @ l[i]
        Omega_sens += temp1 + temp2 + temp3
        K, sigma = Kalman(Omega_measure, Omega_sens, Extended_A, sigma, H)

        passed_newx = np.copy(newx[i, :-Num_Var])
        newx[i + 1, :Num_Var] = newx[i, :Num_Var] + dt * f(newx[i, :Num_Var], u, alpha)
        newx[i + 1, Num_Var:] = passed_newx

        passed_xref = np.copy(xref[i, :-Num_Var])
        xref[i + 1, :Num_Var] = xref[i, :Num_Var] + dt * f(xref[i, :Num_Var], u, alpha)
        xref[i + 1, Num_Var:] = passed_xref

        if Noise:
            newx[i + 1, 4 : 4 + len(u)] += np.random.normal(
                0, np.sqrt(motornoise_variance), len(u)
            )

        y = H @ (newx[i] - xref[i])
        if Noise:
            y += np.random.normal(0, 1e-3, len(y))

        xhat[i + 1] = Extended_A @ xhat[i] + Extended_B @ deltau + K @ (y - H @ xhat[i])
        sigmax = (
            (Extended_A + Extended_B @ L[i])
            @ sigmax
            @ (Extended_A + Extended_B @ L[i]).T
            + K @ H @ sigma @ Extended_A.T
            + ((Extended_A + Extended_B @ L[i]) @ mx) @ (l[i].T @ Extended_B.T)
            + Extended_B @ l[i] @ ((Extended_A + Extended_B @ L[i]) @ mx).T
            + (Extended_B @ l[i]) @ (l[i].T @ Extended_B.T)
        )
    return newx


def Compute_Cartesian_Speed(X, Y, dt):
    V = np.zeros(X.shape)
    Vx = np.diff(X) / dt
    Vy = np.diff(Y) / dt
    V[1:] = np.sqrt(Vx * Vx + Vy * Vy)
    return V


def ILQG(
    Duration=0.5,
    w1=1e4,
    w2=1,
    wp=1e-2,
    r1=1e-2,
    targets=[0, 50],
    start=[0, 30],
    K=120,
    plot=True,
    Noise=False,
    delay=0,
    alpha=0,
    filename="test.pdf",
    direction="Horizontal",
):
    """
    Parameters :
        - Duration : Movement Duration in sec
        - w1,w2,r1 : Weight of the costs function associated to distance penalty  to the target, end velocity, and motor costs respectively
        - targets : Position of the target in cartesian coordinates
        - k : Number of iterations
        - start : Starting position of the hand
        - plot : Boolean, True ==> plotting enabled
        - Noise : Boolean, True ==> Noise Activated in the simulation
        - Delay : Sensory Delay in sec
        - motornoise_variance : Variance of the motor noise
        - alpha : Body Tilt in radiant

    return :
        - X,Y : Cartesian coordinates of the hand trajectory
        - u : Input command
        - x : Vector state of the trajectory
    """

    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])
    st1, st2 = compute_angles_from_cartesian(start[0], start[1])

    tau1, tau2 = np.array(
        [
            g
            * (
                m1 * s1 * np.cos(st1 + alpha)
                + m2 * (s2 * np.cos(st1 + alpha + st2) + l1 * np.cos(st1 + alpha))
            ),
            g * m2 * s2 * np.cos(st1 + st2 + alpha),
        ]
    )

    x0 = np.array(
        [st1, st2, 0, 0, tau1, tau2]
    )  # theta_shoulder,theta_elbow,omega_shoulder,omega_elbow,shoulder_torque,elbow_torque

    m, n = 2, 6  # Command and State dimension
    u = np.zeros((K - 1, m))
    dt = Duration / K
    kdelay = int(delay / dt)

    cbold = np.zeros((K - 1, m, n))
    C = np.zeros((K - 1, m, n, m))

    motornoise_variance = (
        1e-4 * K / 60
    )  # Play with it to change the motornoise variance, K/60 is to scale it withthe number of iteration steps

    for i in range(K - 1):
        for j in range(m):
            cbold[i, j, 4 + j] = sqrt(motornoise_variance)

    oldX = np.ones(K) * np.inf
    oldY = np.ones(K) * np.inf
    u_incr = np.ones(u.shape) * np.inf

    for iter in range(100):
        x = step1(
            x0, u, Duration, alpha
        )  # Forward step computing the sequence of state trajectory given a sequence of input u
        X = np.cos(x[:, 0] + x[:, 1]) * 33 + np.cos(x[:, 0]) * 30
        Y = np.sin(x[:, 0] + x[:, 1]) * 33 + np.sin(x[:, 0]) * 30

        if (
            np.max(np.abs(u_incr)) < 1e-7
        ):  # If the trajectory improvement is small enough, stop the iteration and perform a full simulation with feedback and potential noise
            print("Solution found at iteration ", iter)
            x = step5(
                x0,
                l,
                L,
                Duration,
                Noise,
                A,
                B,
                K,
                u - u_incr,
                kdelay,
                motornoise_variance,
                alpha,
                cbold,
                C,
            )
            X = np.cos(x[:, 0] + x[:, 1]) * 33 + np.cos(x[:, 0]) * 30
            Y = np.sin(x[:, 0] + x[:, 1]) * 33 + np.sin(x[:, 0]) * 30

            break

        A, B, q, qbold, r, Q, R = step2(
            x,
            u,
            Duration,
            w1,
            w2,
            wp,
            r1,
            np.array([obj1, obj2]),
            alpha,
            direction,
        )  # Compute the Linearizations of the dynamic
        l, L = step3(
            A, B, C, cbold, q, qbold, r, Q, R
        )  # Compute the control gains improvement (feedforward and feedback)
        u_incr = step4(l, L, K, A, B)  # Compute the command sequence improvement
        u += u_incr  # Improves the command sequence
        oldX = np.copy(X)
        oldY = np.copy(Y)

    return X, Y, x


if __name__ == "__main__":

    ALL_DIRECTIONS = ["Vertical", "Horizontal"]
    SIMULATED_MOVEMENT_DIRECTION = ALL_DIRECTIONS[0]

    ACTIVATE_PATH_CONSTRAINT = True
    ACTIVATE_Gravity = True
    MOVEMENT_DURATION = 0.3  # in seconds
    MOVEMENT_LENGTH = 20  # in cm
    NUM_ITER = 600
    g = 9.81 if ACTIVATE_Gravity else 0

    led_dl = int(MOVEMENT_LENGTH / 10)

    dt = MOVEMENT_DURATION / NUM_ITER
    time = np.linspace(0, MOVEMENT_DURATION * 1000, NUM_ITER)

    if SIMULATED_MOVEMENT_DIRECTION == "Horizontal":
        LED = np.array([20, 30, 40, 50])
        HEIGHT = -30
        starting_positions = np.column_stack(
            (LED[: (4 - led_dl)], [HEIGHT] * (4 - led_dl))
        )
        ending_positions = np.column_stack(
            (LED[(4 - led_dl) :], [HEIGHT] * (4 - led_dl))
        )
    else:
        LED = np.array([-20, -10, 0, 10])
        DEPTH = 40
        starting_positions = np.column_stack(
            ([DEPTH] * (4 - led_dl), LED[: (4 - led_dl)])
        )
        ending_positions = np.column_stack(
            ([DEPTH] * (4 - led_dl), LED[(4 - led_dl) :])
        )

    fig = plt.figure(figsize=(8, 14))
    gs = gridspec.GridSpec(4, (4 - led_dl), hspace=0.5)
    colors = plt.cm.viridis(np.linspace(0, 1, 6))[:3]
    colors2 = plt.cm.viridis(np.linspace(0, 1, 6))[3:]
    dotcolor = np.array(["blue", "red", "#f932f0"])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])

    WP = 1e-2 * 2 if ACTIVATE_PATH_CONSTRAINT else 0  # Path Constraint Cost

    for u in range((4 - led_dl)):

        start = starting_positions[u]
        end = ending_positions[u]

        ax1 = fig.add_subplot(gs[0, u])
        ax2 = fig.add_subplot(gs[1, u])

        ax1.scatter(end[0], end[1], marker="s", color="grey", s=300)
        delete_axis(ax1)
        ax1.set_aspect("equal")
        if SIMULATED_MOVEMENT_DIRECTION == "Horizontal":
            ax1.set_ylim(HEIGHT - 3, HEIGHT + 3)
            ax1.set_yticks([])
        else:
            ax1.set_xlim(DEPTH - 3, DEPTH + 3)
            ax1.set_xticks([])

        X, Y, states = ILQG(
            Duration=MOVEMENT_DURATION,
            K=NUM_ITER,
            start=start,
            targets=end,
            delay=0,
            wp=WP,
            direction=SIMULATED_MOVEMENT_DIRECTION,
        )
        X2, Y2, states2 = ILQG(
            Duration=MOVEMENT_DURATION,
            K=NUM_ITER,
            start=end,
            targets=start,
            delay=0,
            wp=WP,
            direction=SIMULATED_MOVEMENT_DIRECTION,
        )

        ax1.plot(X, Y, color=colors[u], label="Forward")
        ax1.plot(X2, Y2, color=colors2[u], label="Backward")
        ax1.legend(fontsize=6)

        if u == 0:
            ax2.set_ylabel("Velocity [cm/sec]")
        ax2.set_xlabel("Time [sec]")

        ax2.plot(
            time,
            compute_absolute_velocity(X, Y, dt),
            color=colors[u],
            label="Forward",
            linewidth=2,
        )
        ax2.plot(
            time,
            compute_absolute_velocity(X2, Y2, dt),
            color=colors2[u],
            label="Backward",
            linewidth=2,
        )

        ax3.set_ylabel("Delta rtpv")
        ax3.set_xlabel("3 Movements")
        ax3.plot(
            np.linspace(0, (3 - led_dl), 100),
            np.zeros(100),
            color="grey",
            linestyle="--",
        )
        ax3.set_xticks([])

        ax3.scatter(
            u,
            (
                np.argmax(compute_absolute_velocity(X2, Y2, dt))
                - np.argmax(compute_absolute_velocity(X, Y, dt))
            )
            / NUM_ITER,
            color=dotcolor[u],
        )

        ax4.plot(
            states[:, 0] * 180 / pi,
            states[:, 1] * 180 / pi,
            color=colors[u],
            linewidth=3,
        )
        ax4.plot(
            states2[:, 0] * 180 / pi,
            states2[:, 1] * 180 / pi,
            color=colors2[u],
            linewidth=3,
        )

        ax4.set_ylabel(r"$\theta_e$")
        ax4.set_xlabel(r"$\theta_s$")

    plt.savefig("Gravitypath.png", dpi=200)
    plt.show()
