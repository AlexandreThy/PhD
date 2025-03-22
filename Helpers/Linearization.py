from Helpers.Helpers import *
def Linearization(dt, x,N = 8,Euler = True):
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