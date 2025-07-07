from Helpers.Linearization import *
from Helpers.Environment import *


def LQG(
    Duration=0.6,
    w1=1e8,
    w2=1e8,
    w3=1e4,
    w4=1e4,
    r1=1e-5,
    r2=1e-5,
    targets=[0, 55],
    starting_point=[0, 20],
    FF=False,
    Side="Right",
    plot=True,
    Delay=0,
    plotEstimation=False,
    Showu=False,
    newtonfunc=newtonf,
    newtondfunc=newtondf,
    Num_iter=300,
    Activate_Noise=False,
):

    dt = Duration / Num_iter
    kdelay = int(Delay / dt)

    obj1, obj2 = newton(
        newtonfunc, newtondfunc, 1e-8, 1000, targets[0], targets[1]
    )  # Defini les targets
    st1, st2 = newton(
        newtonfunc, newtondfunc, 1e-8, 1000, starting_point[0], starting_point[1]
    )

    xstart = np.array([st1, 0, 0, st2, 0, 0, obj1, 0, obj2, 0])
    x0 = np.array([st1, 0, 0, st2, 0, 0, obj1, obj2])
    x0_with_delay = np.copy(x0)
    for _ in range(kdelay):
        x0_with_delay = np.concatenate((x0_with_delay, x0))
    Num_Var = 8

    # Define Weight Matrices

    R = np.array([[r1, 0], [0, r2]])
    Q = np.array(
        [
            [w1, 0, 0, 0, 0, 0, -w1, 0],
            [0, w2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, w3, 0, 0, 0, -w3],
            [0, 0, 0, 0, w4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-w1, 0, 0, 0, 0, 0, w1, 0],
            [0, 0, 0, -w3, 0, 0, 0, w3],
        ]
    )

    # Define Dynamic Matrices

    A_basic = Linearization(dt, [pi / 4, 0, 0, pi / 2, 0, 0])

    B_basic = np.transpose(
        [[0, 0, dt / tau, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, dt / tau, 0, 0]]
    )

    NewQ = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    NewQ[:Num_Var, :Num_Var] = Q
    Q = NewQ

    H = np.zeros((Num_Var, (kdelay + 1) * Num_Var))
    H[:, (kdelay) * Num_Var :] = np.identity(Num_Var)

    A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    A[:Num_Var, :Num_Var] = A_basic
    A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)
    B = np.zeros(((kdelay + 1) * Num_Var, 2))
    B[:Num_Var] = B_basic

    S = Q

    array_L = np.zeros((Num_iter - 1, 2, Num_Var * (kdelay + 1)))
    array_S = np.zeros((Num_iter, Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    array_S[-1] = Q
    for k in range(Num_iter - 1):
        L = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
        array_L[Num_iter - 2 - k] = L
        S = A.T @ S @ (A - B @ L)
        array_S[Num_iter - 2 - k] = S

    # print(array_L[0])
    # Feedback
    L = array_L

    array_x = np.zeros((Num_iter, Num_Var))
    array_xhat = np.zeros((Num_iter, Num_Var))
    array_u = np.zeros((Num_iter - 1, 2))
    y = np.zeros((Num_iter - 1, Num_Var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()
    xhat = np.copy(x0_with_delay)
    x = np.copy(x0_with_delay)

    sigma = np.identity(Num_Var * (kdelay + 1)) * 10**-6
    J = 0
    F = [0, 0]
    for k in range(Num_iter - 1):

        acc = (
            np.array([array_x[k][2], array_x[k][5]])
            - np.array([array_x[k - 1][2], array_x[k - 1][5]])
        ) / dt
        if (np.sin(x[0] + x[3]) * 33 + np.sin(x[0]) * 30 > 35) and (FF == True):

            F = Compute_f_new_version(
                np.array([x[0], x[3]]), np.array([x[1], x[4]]), acc, 1
            )
            if Side == "Left":
                F *= -1

        else:
            F = [0, 0]
        Omega_sens, Omega_measure, motor_noise, measure_noise = NoiseAndCovMatrix(
            N=Num_Var, kdelay=kdelay, Linear=True, Var=1e-3
        )
        y[k] = (H @ x).flatten()
        if Activate_Noise == True:
            y[k] += measure_noise
        K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
        sigma = Omega_sens + (A - K @ H) @ sigma @ A.T
        u = -L[k].reshape(np.flip(B.shape)) @ xhat
        array_u[k] = u
        J += u.T @ R @ u
        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)
        x = (
            A @ x
            + B @ u
            + np.concatenate(
                ([0, dt * F[0], 0, 0, dt * F[1], 0, 0, 0], np.zeros(Num_Var * kdelay))
            ).flatten()
        )
        if Activate_Noise:
            for j, i in enumerate([2, 5]):
                x[i] += motor_noise[j]
        array_xhat[k + 1] = xhat[:Num_Var].flatten()
        array_x[k + 1] = x[:Num_Var].flatten()

        # print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])

    # Plot
    J += x.T @ Q @ x
    x0 = xstart

    x_nonlin = array_x.T[:, 1:][:, ::1]
    X = np.cos(x_nonlin[0] + x_nonlin[3]) * 33 + np.cos(x_nonlin[0]) * 30
    Y = np.sin(x_nonlin[0] + x_nonlin[3]) * 33 + np.sin(x_nonlin[0]) * 30

    if plot:
        plt.plot(X, Y, color="green", label="LQG", linewidth=0.8)
        plt.axis("equal")
        plt.scatter([targets[0]], [targets[1]], color="black")
        if plotEstimation:
            x_nonlin = array_xhat.T[:, 1:][:, ::1]
            X2 = np.cos(x_nonlin[0] + x_nonlin[3]) * 33 + np.cos(x_nonlin[0]) * 30
            Y2 = np.sin(x_nonlin[0] + x_nonlin[3]) * 33 + np.sin(x_nonlin[0]) * 30
            plt.plot(
                X2,
                Y2,
                color="black",
                label="Estimation",
                linewidth=0.8,
                linestyle="--",
                alpha=0.5,
            )
    if Showu:
        return X, Y, array_u
    return X, Y, J, x_nonlin


def BestLQG(
    Duration=0.6,
    w1=1e4,
    w2=1e4,
    w3=1,
    w4=1,
    r1=1e-5,
    r2=1e-5,
    targets=[0, 55],
    starting_point=[0, 20],
    plot=True,
    Delay=0,
    Num_iter=60,
    Activate_Noise=False,
    plotEstimation=False,
    ClassicLQG=False,
    filter=[1, 0, 0, 1, 0, 0],
    AdditionalDynamics={},
):

    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    obj1, obj2 = newton(
        newtonf, newtondf, 1e-8, 1000, targets[0], targets[1]
    )  # Defini les targets
    st1, st2 = newton(
        newtonf, newtondf, 1e-8, 1000, starting_point[0], starting_point[1]
    )
    TimeConstant = 1 / 0.06

    x0 = np.array([st1, 0, 0, st2, 0, 0, obj1, obj2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    Num_Var = 8

    R = np.array([[r1, 0], [0, r2]])

    Q = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    Q[:Num_Var, :Num_Var] = np.array(
        [
            [w1, 0, 0, 0, 0, 0, -w1, 0],
            [0, w3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, w2, 0, 0, 0, -w2],
            [0, 0, 0, 0, w4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-w1, 0, 0, 0, 0, 0, w1, 0],
            [0, 0, 0, -w2, 0, 0, 0, w2],
        ]
    )

    H = np.zeros((Num_Var, (kdelay + 1) * Num_Var))
    H[:, (kdelay) * Num_Var :] = np.identity(Num_Var)

    A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)

    B = np.zeros(((kdelay + 1) * Num_Var, 2))
    B[:Num_Var] = np.transpose(
        [[0, 0, dt / tau, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, dt / tau, 0, 0]]
    )

    array_x = np.zeros((Num_iter, Num_Var))
    array_xhat = np.zeros((Num_iter, Num_Var))
    y = np.zeros((Num_iter - 1, Num_Var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()

    xhat = np.copy(x0_with_delay)
    x = np.copy(x0_with_delay)

    sigma = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    J = 0
    omega = np.zeros(2)
    acc = np.zeros(2)
    for k in range(Num_iter - 1):
        F = EnvironmentDynamics(AdditionalDynamics, x, acc)
        if ClassicLQG:
            A[:Num_Var, :Num_Var] = Linearization(
                dt, np.array([pi / 4, 0, 0, pi / 2, 0, 0])
            )
        else:
            xcopy = np.copy(xhat)
            for i in range(6):
                xcopy[i] *= filter[i]
            A[:Num_Var, :Num_Var] = Linearization(dt, xcopy)
        S = Q
        for _ in range(Num_iter - 1 - k):
            L = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
            S = A.T @ S @ (A - B @ L)
        u = -L @ xhat
        J += u.T @ R @ u

        C = np.array(
            [
                -x[4] * (2 * x[1] + x[4]) * a2 * np.sin(x[3]),
                x[1] * x[1] * a2 * np.sin(x[3]),
            ]
        )
        M = np.array(
            [
                [a1 + 2 * a2 * np.cos(x[3]), a3 + a2 * np.cos(x[3])],
                [a3 + a2 * np.cos(x[3]), a3],
            ]
        )

        Omega_sens, Omega_measure, motor_noise, measure_noise = NoiseAndCovMatrix(
            M, Num_Var, kdelay, Linear=True, Var=1e-4
        )
        # Omega_sens = np.diag(np.ones(Num_Var))*1e-6
        # Omega_measure = np.diag(np.ones(Num_Var))*1e-7
        y[k] = (H @ x).flatten()
        if Activate_Noise == True:
            y[k] += measure_noise

        sigma = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
        for _ in range(Num_iter - 1):

            K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
            sigma = Omega_sens + (A - K @ H) @ sigma @ A.T

        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)
        # print(xhat[:8])
        acc = np.linalg.solve(M, ((x[[2, 5]]) - C - Bdyn @ omega)) + F
        omega += dt * acc
        x_new = np.array(
            [
                x[0] + dt * x[1],
                omega[0],
                x[2] + dt * TimeConstant * (u[0] - x[2]),
                x[3] + dt * x[4],
                omega[1],
                x[5] + dt * TimeConstant * (u[1] - x[5]),
                x[6],
                x[7],
            ]
        )

        # Concatenate with remaining x values
        x = np.concatenate((x_new, x[:-Num_Var]))

        if Activate_Noise:

            x[[2, 5]] += motor_noise

        array_xhat[k + 1] = xhat[:Num_Var].flatten()
        array_x[k + 1] = x[:Num_Var].flatten()

        # print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])

    # Plot
    J += x.T @ Q @ x

    x_nonlin = array_x.T[:, :][:, ::1]
    X = np.cos(x_nonlin[0] + x_nonlin[3]) * 33 + np.cos(x_nonlin[0]) * 30
    Y = np.sin(x_nonlin[0] + x_nonlin[3]) * 33 + np.sin(x_nonlin[0]) * 30

    if plot:
        color = "magenta" if ClassicLQG else "green"
        label = "LQG" if ClassicLQG else "DLQG"
        plt.plot(X, Y, color=color, label=label, linewidth=0.8)
        plt.axis("equal")
        tg = np.array([obj1, obj2])
        plt.scatter(
            np.array([ToCartesian(tg)[0]]),
            np.array([ToCartesian(tg)[1]]),
            color="black",
        )
    if plotEstimation:
        x_nonlin2 = array_xhat.T[:, 1:][:, ::1]
        X2 = np.cos(x_nonlin2[0] + x_nonlin2[3]) * 33 + np.cos(x_nonlin2[0]) * 30
        Y2 = np.sin(x_nonlin2[0] + x_nonlin2[3]) * 33 + np.sin(x_nonlin2[0]) * 30
        plt.plot(
            X2,
            Y2,
            color="black",
            label="Estimation",
            linewidth=0.8,
            linestyle="--",
            alpha=0.5,
        )
    return X, Y, u, x_nonlin


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
        Moment_Arm @ (u * dfvdos * fl) - dCdos - Bdyn @ np.array([1, 0])
    )
    d_accel_theta2 = -Minv @ (
        dM @ Minv @ (Moment_Arm @ (u * fl * fv) - C - Bdyn @ dtheta)
    ) + Minv @ (Moment_Arm @ (u * (dfldte * fv + fl * dfvdte)) - dCdte)
    d_accel_dtheta2 = Minv @ (
        Moment_Arm @ (u * dfvdoe * fl) - dCdoe - Bdyn @ np.array([0, 1])
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
    FinalA = np.identity(6)
    FinalA[:4, :4] += dt * A
    return FinalA


def f(x, u):
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
    theta = Minv @ (A @ (u * fl * ff_v) - Bdyn @ x[2:4] - C)

    return np.array([[x[2], x[3], theta[0], theta[1], 0, 0]])


def fx(x, u):
    return Linearization_6dof(x, u)


def fu(x, u):
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
    sol = np.zeros((4, 6))
    for i in range(6):
        du = np.zeros(6)
        du[i] = 1
        sol[2:, i] = Minv @ (A @ (du * fl * fv))
    return sol


def DLQG_6Muscles(
    Duration=0.6,
    w1=1e4,
    w2=1e4,
    w3=1,
    w4=1,
    r1=1e-5,
    targets=[0, 55],
    starting_point=[0, 20],
    plot=True,
    Delay=0,
    Num_iter=60,
    Activate_Noise=False,
    plotEstimation=False,
    ClassicLQG=False,
):

    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    obj1, obj2 = newton(
        newtonf, newtondf, 1e-8, 1000, targets[0], targets[1]
    )  # Defini les targets
    st1, st2 = newton(
        newtonf, newtondf, 1e-8, 1000, starting_point[0], starting_point[1]
    )

    x0 = np.array([st1, st2, 0, 0, obj1, obj2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    Num_Var = 6

    R = np.diag(np.ones(6) * r1)

    Q = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    Q[:Num_Var, :Num_Var] = np.array(
        [
            [w1, 0, 0, 0, -w1, 0],
            [0, w2, 0, 0, 0, -w2],
            [0, 0, w3, 0, 0, 0],
            [0, 0, 0, w4, 0, 0],
            [-w1, 0, 0, 0, w1, 0],
            [0, -w2, 0, 0, 0, w2],
        ]
    )

    H = np.zeros((Num_Var, (kdelay + 1) * Num_Var))
    H[:, (kdelay) * Num_Var :] = np.identity(Num_Var)

    A = np.zeros(((kdelay + 1) * Num_Var, (kdelay + 1) * Num_Var))
    A[Num_Var:, :-Num_Var] = np.identity((kdelay) * Num_Var)

    B = np.zeros(((kdelay + 1) * Num_Var, 6))

    array_x = np.zeros((Num_iter, Num_Var))
    array_xhat = np.zeros((Num_iter, Num_Var))
    array_u = np.zeros((Num_iter - 1, 6))
    y = np.zeros((Num_iter - 1, Num_Var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()

    xhat = np.copy(x0_with_delay)
    x = np.copy(x0_with_delay)

    sigma = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    J = 0
    u = np.zeros(6)
    for k in range(Num_iter - 1):
        xcopy = np.copy(x)
        A[:Num_Var, :Num_Var] = Linearization_6dof(dt, xcopy, 0)
        B[:4] = dt * fu(xcopy, u)

        S = Q
        for _ in range(Num_iter - 1 - k):
            L = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
            S = A.T @ S @ (A - B @ L)
        u = -L @ xhat
        J += u.T @ R @ u

        Omega_motor = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
        Omega_measure = np.diag(np.ones(Num_Var) * 1e-3)
        for i in range(2, 4):

            Omega_motor[i, i] = 1e-4 * 9
        y[k] = (H @ x).flatten()
        if Activate_Noise == True:
            y[k] += np.random.normal(0, 1e-2 * 3, Num_Var)

        K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
        sigma = Omega_motor + (A - K @ H) @ sigma @ A.T

        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)

        x_new = (x[:Num_Var] + dt * f(x, u)).reshape(6)

        # Concatenate with remaining x values
        x = np.concatenate((x_new, x[:-Num_Var]))

        if Activate_Noise:

            x[[2, 3]] += np.random.normal(0, np.sqrt(1e-3), 2)

        array_xhat[k + 1] = xhat[:Num_Var].flatten()
        array_x[k + 1] = x[:Num_Var].flatten()
        array_u[k] = u

        # print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])

    # Plot
    J += x.T @ Q @ x

    x_nonlin = array_x.T[:, :][:, ::1]
    X = np.cos(x_nonlin[0] + x_nonlin[1]) * 33 + np.cos(x_nonlin[0]) * 30
    Y = np.sin(x_nonlin[0] + x_nonlin[1]) * 33 + np.sin(x_nonlin[0]) * 30

    if plot:
        color = "magenta" if ClassicLQG else "green"
        label = "LQG" if ClassicLQG else "DLQG"
        plt.plot(X, Y, color=color, label=label, linewidth=0.8)
        plt.scatter(X, Y, color=color, s=10)
        plt.axis("equal")
        tg = np.array([obj1, obj2])
        plt.scatter(
            np.array([ToCartesian(tg)[0]]),
            np.array([ToCartesian(tg)[1]]),
            color="black",
        )
        plt.show()
        time = np.linspace(0, Duration, Num_iter)
        plt.plot(time, x_nonlin[2], color="green", linestyle="--")
        plt.plot(time, x_nonlin[3], color="green")
    return X, Y, array_u, x_nonlin
