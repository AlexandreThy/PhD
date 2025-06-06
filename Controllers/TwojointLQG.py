
import numpy as np
from matplotlib import pyplot as plt
from math import *

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

def linearization_of_dynamics(dt, x,N = 8,Euler = True):
    TimeConstant = 1 / 0.06  # Torque dynamics coefficient

    # Extract state variables according to the given order
    theta1, dtheta1, tau1, theta2, dtheta2, tau2 = x[:6]

    # Coriolis force
    C = np.array([
        -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.sin(theta2),
        dtheta1**2 * a2 * np.sin(theta2)
    ])
    
    # Partial derivatives of C
    dCdte = np.array([
        -dtheta2 * (2 * dtheta1 + dtheta2) * a2 * np.cos(theta2),
        dtheta1**2 * a2 * np.cos(theta2)
    ])
    dCdos = np.array([
        -dtheta2 * 2 * a2 * np.sin(theta2),
        2 * dtheta1 * a2 * np.sin(theta2)
    ])
    dCdoe = np.array([
        (-2 * dtheta1 - 2 * dtheta2) * a2 * np.sin(theta2),
        0
    ])
    
    # Inertia matrix
    M = np.array([
        [a1 + 2 * a2 * np.cos(theta2), a3 + a2 * np.cos(theta2)],
        [a3 + a2 * np.cos(theta2), a3]
    ])
    
    Minv = np.linalg.inv(M)
    
    # Derivative of inertia matrix
    dM = np.array([
        [-2 * a2 * np.sin(theta2), -a2 * np.sin(theta2)],
        [-a2 * np.sin(theta2), 0]
    ])
    
    # Compute acceleration dependencies
    dtheta = np.array([dtheta1, dtheta2])
    tau = np.array([tau1, tau2])

    d_accel_dtheta1 = -Minv @ (dCdos + Bdyn @ np.array([1, 0]))
    d_accel_tau = Minv @ np.array([1, 0])
    d_accel_theta2 = -Minv @ (dM @ Minv @ (tau - C - Bdyn @ dtheta)) - Minv @ dCdte
    d_accel_dtheta2 = -Minv @ (dCdoe + Bdyn @ np.array([0, 1]))
    d_accel_tau2 = Minv @ np.array([0, 1])

    # Construct the Jacobian matrix
    A = np.zeros((N,N))

    # Assign known structure
    A[0, 1] = 1  # d(theta1)/d(dtheta1)
    A[3, 4] = 1  # d(theta2)/d(dtheta2)

    # Acceleration contributions
    A[1, 1] = d_accel_dtheta1[0]  # d(dtheta1)/d(dtheta1)
    A[1, 3] = d_accel_theta2[0]
    A[1, 4] = d_accel_dtheta2[0]  # d(dtheta1)/d(dtheta2)
    A[1, 2] = d_accel_tau[0]  # d(dtheta1)/d(tau1)
    A[1, 5] = d_accel_tau2[0]  # d(dtheta1)/d(tau2)

    A[4, 1] = d_accel_dtheta1[1]  # d(dtheta2)/d(dtheta1)
    A[4, 3] = d_accel_theta2[1]
    A[4, 4] = d_accel_dtheta2[1]  # d(dtheta2)/d(dtheta2)
    A[4, 2] = d_accel_tau[1]  # d(dtheta2)/d(tau1)
    A[4, 5] = d_accel_tau2[1]  # d(dtheta2)/d(tau2)

    # Torque dynamics
    A[2, 2] = -TimeConstant
    A[5, 5] = -TimeConstant


    if Euler : A = np.identity(N)+dt*A
    return A

def noiseandcovmatrix( N=6, kdelay=0, Var=1e-6):

    motornoise = np.random.normal(0, np.sqrt(Var), 2)
    Omegasenslinear = np.zeros((N * (kdelay + 1), N * (kdelay + 1)))
    Omegasenslinear[2, 2] = Var
    Omegasenslinear[5, 5] = Var

    return (
            Omegasenslinear,
            np.diag(np.ones(N) * 1e-6),
            motornoise,
            np.random.normal(0, np.sqrt(Var), N),
        )

def twojointLQG(
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
    num_iter=60,
    activate_noise=False,
):
    """
    Parameters
    ----------
    Duration : float, optional
        Total duration of the simulated reaching movement in seconds. Default is 0.6.

    w1 : float, optional
        Weight on final shoulder angle error in the LQG cost function.

    w2 : float, optional
        Weight on final elbow angle error in the LQG cost function.

    w3 : float, optional
        Weight on final shoulder angular velocity error

    w4 : float, optional
        Weight on final elbow angular velocity error

    r1 : float, optional
        Weight on control effort for joint 1 (e.g., torque cost). Smaller values allow stronger control signals. Default is 1e-5.

    r2 : float, optional
        Weight on control effort for joint 2 (e.g., torque cost). Default is 1e-5.

    targets : list of float, optional
        Target position in cartesian space to reach, specified as [x, y]. Default is [0, 55].

    starting_point : list of float, optional
        Initial position of the arm in cartesian space. Default is [0, 20].

    plot : bool, optional
        If True, plot trajectories.

    Delay : int, optional
        Sensorimotor feedback delay in seconds. Used to simulate biological latency. Default is 0.

    num_iter : int, optional
        Number of iterations (or discrete time steps) in the simulation. Default is 60.

    activate_noise : bool, optional
        If True, includes process and/or observation noise in the simulation to model uncertainty. Default is False.
    """


    dt = Duration / num_iter
    kdelay = int(Delay / dt)
    obj1, obj2 = compute_angles_from_cartesian(targets[0],targets[1])  # Defini les targets
    st1, st2 = compute_angles_from_cartesian(starting_point[0],starting_point[1]) 

    x0 = np.array([st1, 0, 0, st2, 0, 0, obj1, obj2])
    x0_with_delay = np.tile(x0, kdelay + 1)
    num_var = 8

    R = np.array([[r1, 0], [0, r2]])

    Q = np.zeros(((kdelay + 1) * num_var, (kdelay + 1) * num_var))
    Q[:num_var, :num_var] = np.array(
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

    H = np.zeros((num_var, (kdelay + 1) * num_var))
    H[:, (kdelay) * num_var :] = np.identity(num_var)

    A = np.zeros(((kdelay + 1) * num_var, (kdelay + 1) * num_var))
    A[num_var:, :-num_var] = np.identity((kdelay) * num_var)

    B = np.zeros(((kdelay + 1) * num_var, 2))
    B[:num_var] = np.transpose(
        [[0, 0, dt / tau, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, dt / tau, 0, 0]]
    )

    array_x = np.zeros((num_iter, num_var))
    array_xhat = np.zeros((num_iter, num_var))
    y = np.zeros((num_iter - 1, num_var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()

    xhat = np.copy(x0_with_delay)
    x = np.copy(x0_with_delay)

    J = 0

    A[:num_var, :num_var] = linearization_of_dynamics(
                dt, np.array([pi / 4, 0, 0, pi / 2, 0, 0])
            )
    
    S = Q
    L = np.zeros((num_iter-1,2,num_var*(kdelay+1)))
    
    for i in range(num_iter - 1):
        L[num_iter-2-i] = np.linalg.inv(R + B.T @ S @ B) @ B.T @ S @ A
        S = A.T @ S @ (A - B @ L[num_iter-2-i])

    sigma = np.zeros((num_var * (kdelay + 1), num_var * (kdelay + 1)))

    for k in range(num_iter - 1):

        u = -L[k] @ xhat
        J += u.T @ R @ u

        Omega_sens, Omega_measure, motor_noise, measure_noise = noiseandcovmatrix(
            num_var, kdelay, Var=1e-3
        )
        y[k] = (H @ x).flatten()
        if activate_noise == True:
            y[k] += measure_noise


        K = A @ sigma @ H.T @ np.linalg.inv(H @ sigma @ H.T + Omega_measure)
        sigma = Omega_sens + (A - K @ H) @ sigma @ A.T

        xhat = A @ xhat + B @ u + K @ (y[k] - H @ xhat)
        x = A@x + B@u

        if activate_noise:

            x[[2, 5]] += motor_noise

        array_xhat[k + 1] = xhat[:num_var].flatten()
        array_x[k + 1] = x[:num_var].flatten()

        # print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])

    # Plot
    J += x.T @ Q @ x

    X = np.cos(array_x[:,0] + array_x[:,3]) * 33 + np.cos(array_x[:,0]) * 30
    Y = np.sin(array_x[:,0] + array_x[:,3]) * 33 + np.sin(array_x[:,0]) * 30

    if plot:
        color = "magenta"
        label = "LQG"

        plt.plot(X, Y, color=color, label=label, linewidth=0.8)
        plt.axis("equal")
        plt.scatter(
                [targets[0]],[targets[1]],
            color="grey",s = 200,marker = "s"
        )

    output = {}
    output["X"] = X
    output["Y"] = Y
    output["u"] = u 
    output["state"] = array_x
    output["cost"] = J

    return output

def centeroutreaching(movement_length = 10, num_targets = 8, starting_point = np.array([0,30]),activate_noise = False, num_sim = 0):
    for _ in range(num_sim):
        for angles in np.linspace(0,2*pi,num_targets+1)[:-1]:

            sol = twojointLQG(starting_point=starting_point,targets=starting_point+np.array([movement_length*cos(angles),movement_length*sin(angles)]),activate_noise=activate_noise)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=14,
        title="",
        title_fontsize=12,
        frameon=True,
        shadow=True,
        fancybox=True,
        loc="upper left",
    )
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for side in ["left", "right", "bottom", "top"]:
        ax.spines[side].set_visible(False)

    plt.plot(np.linspace(starting_point[0]-5-movement_length,starting_point[0]-movement_length, 100), np.ones(100) * (starting_point[1]-5-movement_length), color="black")
    plt.plot(np.ones(100) * (starting_point[0]-5-movement_length),np.linspace(starting_point[1]-5-movement_length,starting_point[1]-movement_length, 100), color="black")
    plt.text(starting_point[0]-movement_length-5 + 1, starting_point[1]-movement_length+ 1-5, "5 cm", fontsize=12)
    plt.show()

centeroutreaching(activate_noise=False,num_sim=1)

centeroutreaching(activate_noise=True,num_sim=16)