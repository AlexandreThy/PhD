from Helpers.Linearization import *
from Helpers.Environment import *

# Muscle model constants. Hoisted to module level: they used to be rebuilt from
# nested lists on every call of Linearization_6dof / f / fu, once per timestep.
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
# Derivatives of the muscle length / velocity w.r.t. the joint state are constant.
DLDTS = -MOMENT_ARM[0] / L0
DLDTE = -MOMENT_ARM[1] / L0


def muscle_force_scaling(x):
    """
    Force-length and force-velocity multipliers of the six muscles at state x.

    Returns:
        (l, v, temp, fl, fv) : normalised length, normalised velocity, the
        (l**1.55 - 1) / 0.81 intermediate, and the two force gains.
    """
    l = 1 + MOMENT_ARM[0] * (THETA0[0] - x[0]) / L0 + MOMENT_ARM[1] * (THETA0[1] - x[1]) / L0
    v = MOMENT_ARM[0] * (-x[2]) / L0 + MOMENT_ARM[1] * (-x[3]) / L0

    temp = (l**1.55 - 1) / 0.81
    fl = np.exp(-(np.abs(temp) ** 2.12))
    fv = np.where(
        v <= 0,
        (-7.39 - v) / (-7.39 + (-3.21 + 4.17) * v),
        (0.62 - (-3.12 + 4.21 * l - 2.67 * l**2) * v) / (0.62 + v),
    )
    return l, v, temp, fl, fv


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

    l, v, temp, fl, fv = muscle_force_scaling(x)
    dldts = DLDTS
    dldte = DLDTE
    dvdos = DLDTS
    dvdoe = DLDTE

    dfldl = (
        -fl
        * 2.12
        * np.abs(temp)**1.12
        * np.sign(temp)
        * (1.55 * l**0.55 / 0.81)
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

    d_accel_theta1 = Minv @ (MOMENT_ARM @ (u * (dfldts * fv + fl * dfvdts)))
    d_accel_dtheta1 = Minv @ (
        MOMENT_ARM @ (u * dfvdos * fl) - dCdos - Bdyn @ np.array([1, 0])
    )
    d_accel_theta2 = -Minv @ (
        dM @ Minv @ (MOMENT_ARM @ (u * fl * fv) - C - Bdyn @ dtheta)
    ) + Minv @ (MOMENT_ARM @ (u * (dfldte * fv + fl * dfvdte)) - dCdte)
    d_accel_dtheta2 = Minv @ (
        MOMENT_ARM @ (u * dfvdoe * fl) - dCdoe - Bdyn @ np.array([0, 1])
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
    theta = Minv @ (MOMENT_ARM @ (u * fl * ff_v) - Bdyn @ x[2:4] - C + F)

    return np.array([[x[2], x[3], theta[0], theta[1], 0, 0]])


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
    _, _, _, fl, fv = muscle_force_scaling(x)
    # Column i is the response to a unit command on muscle i. The one-hot vector
    # is reused across iterations rather than reallocated.
    sol = np.zeros((4, 6))
    du = np.zeros(6)
    for i in range(6):
        du[i] = 1
        sol[2:, i] = Minv @ (MOMENT_ARM @ (du * fl * fv))
        du[i] = 0
    return dt * sol

def LQG(
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
):

    dt = Duration / Num_iter
    kdelay = int(Delay / dt)
    obj1, obj2 = compute_angles_from_cartesian(targets[0], targets[1])  # Defini les targets
    st1, st2 = compute_angles_from_cartesian(starting_point[0], starting_point[1])

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

    # Constant across timesteps, so built once instead of on every iteration.
    Omega_motor = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    Omega_measure = np.diag(np.ones(Num_Var) * 1e-4)
    for i in range(2, 4):

        Omega_motor[i, i] = motornoise_variance
    
    A[:Num_Var, :Num_Var] = Linearization_6dof(dt, x0, 0)
    B[:4] = fu(dt, x0, 0)
    # One backward Riccati pass on the model linearised at x0. The cost is
    # terminal only, so the gain depends on the time to go: step k uses the gain
    # computed Num_iter - k steps from the end, as in DLQG. Keeping only the
    # last gain of the pass (the step-0 gain) for every step undershoots.
    array_L = np.zeros((Num_iter, 6, (kdelay + 1) * Num_Var))
    S = Q
    for i in range(Num_iter):
        # B.T @ S is shared with the gain expression below; @ is
        # left-associative so this is the same product, computed once.
        BtS = B.T @ S
        L = np.linalg.inv(R + BtS @ B) @ B.T @ S @ A
        S = A.T @ S @ (A - B @ L)
        array_L[Num_iter - 1 - i] = L

    for k in range(Num_iter):
        L = array_L[k]
        
        F = (
            compute_forcefield(x[0:2], x[2:4], ff_power)
            if FF == True
            else np.array([0, 0])
        )
        u = -L @ xhat
        J += u.T @ R @ u

        y[k] = (H @ x).flatten()
        if Activate_Noise == True:
            y[k] += np.random.normal(0, 1e-2, Num_Var)

        K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
        sigma = Omega_motor + (A - K @ H) @ sigma @ A.T

        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)

        x_new = (x[:Num_Var] + dt * (f(x, u, F))).reshape(6)

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

def DLQG(
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
    motornoise_variance=1e-3,
    ClassicLQG=False,
    FF=False,
    ff_power=0.3,
    delta_state=False,
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

    # Constant across timesteps, so built once instead of on every iteration.
    Omega_motor = np.zeros((Num_Var * (kdelay + 1), Num_Var * (kdelay + 1)))
    Omega_measure = np.diag(np.ones(Num_Var) * 1e-4)
    for i in range(2, 4):

        Omega_motor[i, i] = motornoise_variance

    for k in range(Num_iter):
        xcopy = np.copy(x)
        F = (
            compute_forcefield(x[0:2], x[2:4], ff_power)
            if FF == True
            else np.array([0, 0])
        )

        A[:Num_Var, :Num_Var] = Linearization_6dof(dt, xcopy, 0)
        B[:4] = fu(dt, xcopy, 0)

        S = Q
        for _ in range(Num_iter - k):
            # B.T @ S is shared with the gain expression below; @ is
            # left-associative so this is the same product, computed once.
            BtS = B.T @ S
            L = np.linalg.inv(R + BtS @ B) @ B.T @ S @ A
            S = A.T @ S @ (A - B @ L)
        if delta_state:
            # The paper defines delta x around the current operating point.
            x0_local = x[:Num_Var]
            delta_xhat = xhat - np.tile(x0_local, kdelay + 1)
            # u0 = 0 in the paper; delta x is a local linearization
            # coordinate, not a target error or a delayed-state reference.
            u = -L @ (delta_xhat + np.tile(x0_local, kdelay + 1))
        else:
            u = -L @ xhat
        J += u.T @ R @ u

        y[k] = (H @ x).flatten()
        if Activate_Noise == True:
            y[k] += np.random.normal(0, 1e-2, Num_Var)

        K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
        sigma = Omega_motor + (A - K @ H) @ sigma @ A.T

        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)

        x_new = (x[:Num_Var] + dt * (f(x, u, F))).reshape(6)

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
