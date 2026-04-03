import numpy as np
from math import *

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


def Linearization_6dof(dt, x, u):
    """
    Parameters :
        - x : the state of the system
        - alpha : the body tilt

    return :
        The Jacobian Matrix of the dynamic of the system around the state x
    """

    theta1, theta2, dtheta1, dtheta2 = x[:4]
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

    Moment_Arm = np.array([[2, -2, 0, 0, 1.5, -2], [0, 0, 2, -2, 2, -1.5]])

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
    l = (
        1
        + Moment_Arm[0] * (theta0[0] - x[0]) / l0
        + Moment_Arm[1] * (theta0[1] - x[1]) / l0
    )
    dldts = -Moment_Arm[0] / l0
    dldte = -Moment_Arm[1] / l0

    v = Moment_Arm[0] * (-x[2]) / l0 + Moment_Arm[1] * (-x[3]) / l0
    dvdos = -Moment_Arm[0] / l0
    dvdoe = -Moment_Arm[1] / l0
    # Equation (6): fl(l)
    fl = np.exp(np.abs((l**1.55 - 1) / 0.81))

    dfldl = fl * np.sign((l**1.55 - 1) / 0.81) * (1.55 * l**0.55 / 0.81)
    # Equation (7): ff_v(l, v)
    fv = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    dfvdl = np.where(v <= 0, 0, v * (-4.21 + 5.34 * l) / (0.62 + v))

    dfvdv = np.where(
        v <= 0,
        7.39 * (1 + 0.96) / (-7.39 + 0.96 * v) ** 2,
        -0.62 * (-3.12 + 4.21 * l - 2.67 * l**2 + 1) / (0.62 + v) ** 2,
    )

    dfldts = dfldl * dldts
    dfldte = dfldl * dldte
    dfvdts = dfvdl * dldts
    dfvdte = dfvdl * dldte
    dfvdos = dfvdv * dvdos
    dfvdoe = dfvdv * dvdoe

    # Compute acceleration dependencies
    dtheta = np.array([dtheta1, dtheta2])

    d_accel_theta1 = Minv @ (Moment_Arm @ (u * (dfldts * fv + fl * dfvdts)))
    d_accel_dtheta1 = Minv @ (
        Moment_Arm @ (u * dfvdos * fl) - dCdos - Viscous @ np.array([1, 0])
    )
    d_accel_theta2 = -Minv @ (
        dM @ Minv @ (Moment_Arm @ (u * fl * fv) - C - Viscous @ dtheta)
    ) + Minv @ (Moment_Arm @ (u * (dfldte * fv + fl * dfvdte)) - dCdte)
    d_accel_dtheta2 = Minv @ (
        Moment_Arm @ (u * dfvdoe * fl) - dCdoe - Viscous @ np.array([0, 1])
    )

    # Construct the Jacobian matrix
    A = np.zeros((4, 4))

    A[0, 2] = 1
    A[1, 3] = 1

    # Acceleration contributions
    A[2, 0] = d_accel_theta1[0]
    A[2, 2] = d_accel_dtheta1[0]
    A[2, 1] = d_accel_theta2[0]
    A[2, 3] = d_accel_dtheta2[0]

    A[3, 0] = d_accel_theta1[1]
    A[3, 2] = d_accel_dtheta1[1]
    A[3, 1] = d_accel_theta2[1]
    A[3, 3] = d_accel_dtheta2[1]
    FinalA = np.identity(8)
    FinalA[:4, :4] += dt * A
    return FinalA


def f(x, u, F=0, torque_val = [0,0]):
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
    x[4:6] = torque_val
    theta = Minv @ (A @ (u * fl * ff_v) - Viscous @ x[2:4] - C + F + x[4:6])

    return np.array([[x[2], x[3], theta[0], theta[1],0,0, 0, 0]])


def fu(dt, x, u):
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
    fv = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    sol = np.zeros((6, 6))
    for i in range(6):
        du = np.zeros(6)
        du[i] = 1
        sol[2:4, i] = Minv @ (A @ (du * fl * fv))
    return dt * sol


def DLQG_6Muscles(
    Duration=0.6,
    w1=1e4,
    w2=1e4,
    w3=1,
    w4=1,
    r1=1e-5,
    targets=[0, 55],
    starting_point=[0, 20],
    Delay=0,
    Num_iter=60,
    Activate_Noise=False,
    motornoise_variance=1e-3,
    FF=False,
    ff_power=0.3,
    torque_val = [0,0],
    torque_time = 0.15
):

    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])  # Defini les targets
    st1, st2 = compute_angles_from_cartesian(starting_point[0], starting_point[1])  # Defini le point de depart

    x0 = np.array([st1, st2, 0, 0, 0, 0, obj1, obj2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    Num_Var = 8

    R = np.diag(np.ones(6) * r1)

    Q = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    Q[:Num_Var, :Num_Var] = np.array(
        [
            [w1, 0, 0, 0, 0, 0, -w1, 0],
            [0, w2, 0, 0, 0, 0, 0, -w2],
            [0, 0, w3, 0, 0, 0, 0, 0],
            [0, 0, 0, w4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-w1, 0, 0, 0, 0, 0, w1, 0],
            [0, -w2, 0, 0, 0, 0, 0, w2],
        ]
    )

    H = np.zeros((Num_Var, (kdelay + 1) * Num_Var))
    H[:, (kdelay) * Num_Var :] = np.identity(Num_Var)

    A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)

    B = np.zeros(((kdelay + 1) * Num_Var, 6))

    array_x = np.zeros((Num_iter + 1, Num_Var))
    array_xhat = np.zeros((Num_iter + 1, Num_Var))
    array_u = np.zeros((Num_iter, 6))
    y = np.zeros((Num_iter, Num_Var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()

    xhat = np.copy(x0_with_delay)
    x = np.copy(x0_with_delay)

    sigma = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    J = 0
    u = np.zeros(6)
    for k in range(Num_iter):
        tv = torque_val if k*dt >= torque_time else [0,0]

        xcopy = np.copy(x)
        F = (
            compute_forcefield(x[0:2], x[2:4], ff_power)
            if FF == True
            else np.array([0, 0])
        )

        A[:Num_Var, :Num_Var] = Linearization_6dof(dt, xcopy, 0)
        B[:6] = fu(dt, xcopy, 0)

        S = Q
        for _ in range(Num_iter - k):
            L = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
            S = A.T @ S @ (A - B @ L)
        u = -L @ xhat
        J += u.T @ R @ u

        Omega_motor = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
        Omega_measure = np.diag(np.ones(Num_Var) * 1e-4)
        for i in range(2, 4):

            Omega_motor[i, i] = motornoise_variance
        y[k] = (H @ x).flatten()
        if Activate_Noise == True:
            y[k] += np.random.normal(0, 1e-2, Num_Var)

        K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
        sigma = Omega_motor + (A - K @ H) @ sigma @ A.T

        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)

        x_new = (x[:Num_Var] + dt * (f(x, u, F, tv))).reshape(8)

        # Concatenate with remaining x values
        x = np.concatenate((x_new, x[:-Num_Var]))

        if Activate_Noise:

            x[[2, 3]] += np.random.normal(0, np.sqrt(motornoise_variance), 2)

        array_xhat[k + 1] = xhat[:Num_Var].flatten()
        array_x[k + 1] = x[:Num_Var].flatten()
        array_u[k] = u

        # print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])

    # Plot
    J += x.T @ Q @ x

    x_nonlin = array_x.T[:, :][:, ::1]
    X = np.cos(x_nonlin[0] + x_nonlin[1]) * 33 + np.cos(x_nonlin[0]) * 30
    Y = np.sin(x_nonlin[0] + x_nonlin[1]) * 33 + np.sin(x_nonlin[0]) * 30

    return X, Y, array_u, x_nonlin

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
