import numpy as np
from matplotlib import pyplot as plt
from math import *
import casadi as ca
import matplotlib.gridspec as gridspec

#I1 = 0.025
#I2 = 0.045
#m1 = 1.4
#m2 = 1
#l1 = 0.3
#l2 = 0.33
#s1 = 0.11
#s2 = 0.16

def kinematic_params(body_mass,body_height):
    m1 = body_mass*0.028
    m2 = body_mass * 0.022
    l1 = 0.186*body_height
    l2 = 0.254*body_height
    s1 = 0.436*l1 
    s2 = 0.682*l2 
    I1 = m1*(l1*0.322)**2
    I2 = m2*(l2*0.468)**2
    print("The person has segments masses of ",m1," kg and ",m2," kg.\n The person has segments lengts of ",l1," m and ",l2," m")
    return I1,I2,m1,m2,l1,l2,s1,s2

I1,I2,m1,m2,l1,l2,s1,s2 = kinematic_params(70,1.7)


K = 1 / 0.06
tau = 0.06

# SHOULDER PUIS ELBOW

a1 = I1 + I2 + m2 * l1 * l1 + m2*s2*s2 + m1*s1*s1
a2 = m2 * l1 * s2
a3 = I2 + m2*s2*s2


def delete_axis(ax, sides=["left", "right", "bottom", "top"]):
    for side in sides:
        ax.spines[side].set_visible(False)


def compute_angles_from_cartesian(x, y,l1 = l1* 100 , l2 = l2* 100):
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

    h1 = np.arctan2(y, x) - np.arccos(
        (r_squared + l1**2 - l2**2) / (2 * l1 * np.sqrt(r_squared))
    )

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
    X = np.cos(s + e) * l2 * 100 + np.cos(s) * l1 * 100
    Y = np.sin(s + e) * l2 * 100 + np.sin(s) * l1 * 100

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

    dx = np.diff(x)
    dy = np.diff(y)

    vx = dx / dt
    vy = dy / dt

    velocity = np.sqrt(vx**2 + vy**2)
    velocity = np.insert(velocity, 0, 0.0)

    return velocity


def optimizationofmpcproblem(
    dt, Horizon, wp, r1, r2, end, estimate_now, direction, endmass, opts
):

    theta = ca.SX.sym("theta", 2)
    omega = ca.SX.sym("omega", 2)
    tau = ca.SX.sym("tau", 2)
    state = ca.vertcat(theta, omega, tau)

    u = ca.SX.sym("u", 2)
    control = ca.vertcat(u)

    cos_elbow = ca.cos(theta[1])
    sin_elbow = ca.sin(theta[1])
    cos_shoulder = ca.cos(theta[0])
    cos_both = ca.cos(theta[1] + theta[0])
    if endmass:
        m = 0.8 if g == 0 else 0.4
        a4 = m * (l1 * l1 + l2 * l2)
        a5 = m * (l1 * l2 )
        a6 = m * (l2 * l2)
        Minv = ca.inv(
            ca.vertcat(
                ca.horzcat(
                    a1 + a4 + 2 * (a2 + a5) * cos_elbow,
                    (a3 + a6) + (a2 + a5) * cos_elbow,
                ),
                ca.horzcat((a3 + a6) + (a2 + a5) * cos_elbow, (a3 + a6)),
            )
        )

        C = ca.SX(
            np.array(
                [
                    -omega[1] * (2 * omega[0] + omega[1]) * (a2 + a5) * sin_elbow,
                    omega[0] * omega[0] * (a2 + a5) * sin_elbow,
                ]
            )
        )
        G = ca.SX(
            np.array(
                [
                    g
                    * (
                        m1 * s1 * cos_shoulder
                        + m2 * (s2 * cos_both + l1 * cos_shoulder)
                        + m * (l2 * cos_both + l1 * cos_shoulder)
                    ),
                    g * (m2 * s2 + m * l2) * cos_both,
                ]
            )
        )
    else:
        Minv = ca.inv(
            ca.vertcat(
                ca.horzcat(a1 + 2 * a2 * cos_elbow, a3 + a2 * cos_elbow),
                ca.horzcat(a3 + a2 * cos_elbow, a3),
            )
        )

        C = ca.SX(
            np.array(
                [
                    -omega[1] * (2 * omega[0] + omega[1]) * a2 * sin_elbow,
                    omega[0] * omega[0] * a2 * sin_elbow,
                ]
            )
        )
        G = ca.SX(
            np.array(
                [
                    g
                    * (
                        m1 * s1 * cos_shoulder
                        + m2 * (s2 * cos_both + l1 * cos_shoulder)
                    ),
                    g * m2 * s2 * cos_both,
                ]
            )
        )

    Bdyn = ca.SX(np.array([[0.05, 0.025], [0.025, 0.05]]))
    jerk = Minv @ (tau - C - Bdyn @ omega - G)  # Accelerations
    taudot = (u - tau) / 0.06
    xdot = ca.vertcat(omega, jerk, taudot)

    # CasADi function for system dynamics
    f = ca.Function("f", [state, control], [xdot])

    # Define optimization variables
    opti = ca.Opti()
    X = opti.variable(6, Horizon)
    U = opti.variable(2, Horizon - 1)
    # Test by Simon
    # Acc = opti.variable(2, Horizon)

    # Initial and target states
    X0 = opti.parameter(6)
    Xend = l1 * 100 * np.cos(end[0]) + l2 * 100 * np.cos(end[0] + end[1])
    Yend = l1 * 100 * np.sin(end[0]) + l2 * 100 * np.sin(end[0] + end[1])
    X_targ = np.array([end[0], end[1],0,0])

    # Objective function and constraints
    cost = 0
    for k in range(Horizon - 1):

        x_k = X[:, k]
        u_k = U[:, k]

        # acc_k = Acc[:,k]

        # Current cartesian positions
        Xcurr = l1 * 100 * np.cos(x_k[0]) + l2 * 100 * np.cos(x_k[0] + x_k[1])
        Ycurr = l1 * 100 * np.sin(x_k[0]) + l2 * 100 * np.sin(x_k[0] + x_k[1])

        # Current acceleration
        cos_elbow_k = ca.cos(x_k[1])
        sin_elbow_k = ca.sin(x_k[1])
        cos_shoulder_k = ca.cos(x_k[0])
        cos_both_k = ca.cos(x_k[1] + x_k[0])

        if endmass:
            m = 0.8 if g == 0 else 0.4
            a4 = m * (l1 * l1 + l2 * l2)
            a5 = m * (l1 * l2)
            a6 = m * (l2 * l2)
            Minv_k = ca.inv(
                ca.reshape(
                    ca.vertcat(
                        a1 + a4 + 2 * (a2 + a5) * cos_elbow_k,
                        (a3 + a6) + (a2 + a5) * cos_elbow_k,
                        (a3 + a6) + (a2 + a5) * cos_elbow_k,
                        (a3 + a6),
                    ),
                    2,
                    2,
                )
            )

            C_k = ca.vertcat(
                -x_k[3] * (2 * x_k[2] + x_k[3]) * (a2 + a5) * sin_elbow_k,
                x_k[2] * x_k[2] * (a2 + a5) * sin_elbow_k,
            )
            G_k = ca.vertcat(
                g
                * (
                    m1 * s1 * cos_shoulder_k
                    + m2 * (s2 * cos_both_k + l1 * cos_shoulder_k)
                    + m * (l2 * cos_both_k + l1 * cos_shoulder_k)
                ),
                g * (m2 * s2 + m * l2) * cos_both_k,
            )

        else:
            Minv_k = ca.inv(
                ca.reshape(
                    ca.vertcat(
                        a1 + 2 * a2 * cos_elbow_k,
                        a3 + a2 * cos_elbow_k,
                        a3 + a2 * cos_elbow_k,
                        a3,
                    ),
                    2,
                    2,
                )
            )

            C_k = ca.vertcat(
                -x_k[3] * (2 * x_k[2] + x_k[3]) * a2 * sin_elbow_k,
                x_k[2] * x_k[2] * a2 * sin_elbow_k,
            )
            G_k = ca.vertcat(
                g
                * (
                    m1 * s1 * cos_shoulder_k
                    + m2 * (s2 * cos_both_k + l1 * cos_shoulder_k)
                ),
                g * m2 * s2 * cos_both_k,
            )
        Bdyn_k = ca.MX(np.array([[0.05, 0.025], [0.025, 0.05]]))
        omega_k = x_k[2:4]
        tau_k = x_k[4:6]

        Acc_k = Minv_k @ (tau_k - C_k - Bdyn_k @ omega_k - G_k)

        # Classical cost expression for motor commands : cost += r*ca.sumsqr(u_k)
        # cost += r*ca.sumsqr(u_k)

        # Cost penalizing absolute work of torque and motor commands
        cost += r1 * (
            ca.fabs(X[2, k] * X[4, k]) + ca.fabs(X[3, k] * X[5, k])
        ) + r2 * ca.sumsqr(Acc_k)
        # r*((ca.fabs(X[2,k]*X[4,k]) + ca.fabs(X[3,k]*X[5,k])) + 0.2*(xdot[0]*xdot[0] + xdot[1]*xdot[1]))
        if direction == "Horizontal":
            cost += wp * (Yend - Ycurr) ** 2
        else:
            cost += wp * (Xend - Xcurr) ** 2

        x_next = X[:, k] + dt * f(x_k, u_k)
        opti.subject_to(X[:, k + 1] == x_next)

        """ acc_next = Acc_k
        opti.subject_to(acc_k[:, k + 1] == acc_next) """

    opti.subject_to(X[:, 0] == X0)
    opti.subject_to(X[:4, -1] == X_targ)

    opti.minimize(cost)

    # Initial condition constraint
    

    """ opts = {
    "print_time": 0,
    "ipopt.hessian_approximation": "exact",
    "ipopt.tol": 1e-8,                      
    "ipopt.acceptable_tol": 1e-8,            
    "ipopt.max_iter": 1000,               
    "ipopt.constr_viol_tol": 1e-6,          
    "ipopt.dual_inf_tol": 1e-6,             
    "ipopt.compl_inf_tol": 1e-6             
    } """
    opti.solver("ipopt" , opts)
    opti.set_value(X0, estimate_now)
    sol = opti.solve()
    return sol, U, f


def MPC(
    Duration,
    start,
    end,
    wp=1e-2,
    r1=1e-4,
    r2=1e-5,
    num_iter=100,
    direction="Horizontal",
    endmass=False,
    opts = {}
):

    end = compute_angles_from_cartesian(end[0], end[1])
    start = compute_angles_from_cartesian(start[0], start[1])
    print(start,end)

    dt = Duration / num_iter
    states = np.zeros((6, num_iter))
    controls = np.zeros((2, num_iter - 1))
    if endmass:
        m = 0.8 if g == 0 else 0.4
        G = np.array(
            [
                g
                * (
                    m1
                    * s1
                    * +m2
                    * (s2 * np.cos(start[0] + start[1]) + l1 * np.cos(start[0]))
                    + m * (l2 * np.cos(start[0] + start[1]) + l1 * np.cos(start[0]))
                ),
                g * (m2 * s2 + m * l2) * np.cos(start[0] + start[1]),
            ]
        )
    else:
        G = np.array(
            [
                g
                * (
                    m1 * s1 * np.cos(start[0])
                    + m2 * (s2 * np.cos(start[0] + start[1]) + l1 * np.cos(start[0]))
                ),
                g * m2 * s2 * np.cos(start[0] + start[1]),
            ]
        )

    state_now = np.array([start[0], start[1], 0, 0, G[0], G[1]])
    states[:, 0] = state_now
    sol, U, f = optimizationofmpcproblem(
        dt, num_iter, wp, r1, r2, end, state_now, direction, endmass, opts
    )
    for t in range(num_iter - 1):

        u_opt = sol.value(U[:, t])
        controls[:, t] = u_opt

        state_now = state_now + dt * f(state_now, u_opt).full().flatten()
        states[:, t + 1] = state_now

    s = states[0]
    e = states[1]
    X = np.cos(s + e) * l2 * 100 + np.cos(s) * l1 * 100
    Y = np.sin(s + e) * l2 * 100 + np.sin(s) * l1 * 100
    return X, Y, states


if __name__ == "__main__":
    ALL_DIRECTIONS = ["Vertical", "Horizontal"]
    SIMULATED_MOVEMENT_DIRECTION = ALL_DIRECTIONS[0]

    ACTIVATE_PATH_CONSTRAINT = False
    ENDPOINTMASS = True
    ACTIVATE_Gravity = True
    MOVEMENT_DURATION = 0.45  # in seconds
    MOVEMENT_LENGTH = 20  # in cm
    NUM_ITER = 450
    EFFORT_R = 1e-4
    SMOOTH_R = 1e-5
    OPTS = {
        
        "print_time": 0,
        "ipopt.tol": 1e-5*5,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.max_iter": 5000,
    }
    FILENAME = ALL_DIRECTIONS[0] + ".png"

    g = 9.81 if ACTIVATE_Gravity else 0
    led_dl = int(MOVEMENT_LENGTH / 10)

    dt = MOVEMENT_DURATION / NUM_ITER
    time = np.linspace(0, MOVEMENT_DURATION * 1000, NUM_ITER)

    if SIMULATED_MOVEMENT_DIRECTION == "Horizontal":
        LED = np.array([20, 30, 40, 50])
        HEIGHT = -25
        starting_positions = np.column_stack(
            (LED[: (4 - led_dl)], [HEIGHT] * (4 - led_dl))
        )
        ending_positions = np.column_stack((LED[(led_dl):], [HEIGHT] * (4 - led_dl)))
    else:
        LED = np.array([-20, -10, 0, 10])
        DEPTH = 50
        starting_positions = np.column_stack(
            ([DEPTH] * (4 - led_dl), LED[: (4 - led_dl)])
        )
        ending_positions = np.column_stack(([DEPTH] * (4 - led_dl), LED[(led_dl):]))

    fig = plt.figure(figsize=(8, 14))
    gs = gridspec.GridSpec(3, (4 - led_dl), hspace=0.5)
    colors = plt.cm.viridis(np.linspace(0, 1, 6))[:3]
    colors2 = plt.cm.viridis(np.linspace(0, 1, 6))[3:]
    dotcolor = np.array(["blue", "red", "#f932f0"])
    ax3 = fig.add_subplot(gs[2, :])

    WP = 5e-4 if ACTIVATE_PATH_CONSTRAINT else 0  # Path Constraint Cost

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

        X, Y, states = MPC(
            Duration=MOVEMENT_DURATION,
            start=start,
            end=end,
            num_iter=NUM_ITER,
            direction=SIMULATED_MOVEMENT_DIRECTION,
            wp=WP,
            endmass=ENDPOINTMASS,
            r1 = EFFORT_R,
            r2 = SMOOTH_R,
            opts = OPTS
            
        )
        X2, Y2, states2 = MPC(
            Duration=MOVEMENT_DURATION,
            start=end,
            end=start,
            num_iter=NUM_ITER,
            direction=SIMULATED_MOVEMENT_DIRECTION,
            wp=WP,
            endmass=ENDPOINTMASS,
            r1 = EFFORT_R,
            r2 = SMOOTH_R,
            opts = OPTS
        )

        if SIMULATED_MOVEMENT_DIRECTION == "Horizontal":
            ax1.plot(X, Y, color=colors[u], label="Forward", linestyle="--")
            ax1.plot(X2, Y2, color=colors2[u], label="Backward")
            ax1.legend(fontsize=6, loc="lower right")
        else:
            ax1.plot(X, Y, color=colors[u], label="Upward", linestyle="--")
            ax1.plot(X2, Y2, color=colors2[u], label="Downward")
            ax1.legend(fontsize=6, loc="lower right")

        if u == 0:
            ax2.set_ylabel("Velocity [cm/sec]")
        ax2.set_xlabel("Time [sec]")

        ax2.plot(
            time,
            (
                np.abs(np.insert(np.diff(X) / dt, 0, 0.0))
                if SIMULATED_MOVEMENT_DIRECTION == "Horizontal"
                else np.abs(np.insert(np.diff(Y) / dt, 0, 0.0))
            ),
            color="grey",
            label=(
                "Forward" if SIMULATED_MOVEMENT_DIRECTION == "Horizontal" else "Upward"
            ),
            linewidth=2,
            linestyle="--",
            # compute_absolute_velocity(X, Y, dt)
        )
        ax2.plot(
            time,
            (
                np.abs(np.insert(np.diff(X2) / dt, 0, 0.0))
                if SIMULATED_MOVEMENT_DIRECTION == "Horizontal"
                else np.abs(np.insert(np.diff(Y2) / dt, 0, 0.0))
            ),
            color="black",
            label=(
                "Backward"
                if SIMULATED_MOVEMENT_DIRECTION == "Horizontal"
                else "Downward"
            ),
            linewidth=2,
        )
        ax2.legend(fontsize=6)

        ax3.set_ylabel("Delta rtpv")
        ax3.set_xlabel("3 Movements")
        ax3.plot(
            np.linspace(0, (3 - led_dl), 100),
            np.zeros(100),
            color="grey",
            linestyle="--",
        )
        ax3.set_xticks([])
        ax3.set_ylim((-0.15, 0.15))

        ax3.scatter(
            u,
            (
                np.argmax(np.abs(np.insert(np.diff(X2) / dt, 0, 0.0)))
                - np.argmax(np.abs(np.insert(np.diff(X) / dt, 0, 0.0)))
                if SIMULATED_MOVEMENT_DIRECTION == "Horizontal"
                else np.argmax(np.abs(np.insert(np.diff(Y2) / dt, 0, 0.0)))
                - np.argmax(np.abs(np.insert(np.diff(Y) / dt, 0, 0.0)))
            )
            / NUM_ITER,
            color=dotcolor[u],
        )

    plt.savefig(FILENAME, dpi=200)
    plt.show()
