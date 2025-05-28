

import numpy as np
from matplotlib import pyplot as plt
from math import *
import casadi as ca
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

# SHOULDER PUIS ELBOW

a1 = I1 + I2 + m2 * l1 * l1
a2 = m2 * l1 * s2
a3 = I2

Bdyn = np.array([[0.05, 0.025], [0.025, 0.05]])


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

    dx = np.diff(x)
    dy = np.diff(y)

    vx = dx / dt
    vy = dy / dt

    velocity = np.sqrt(vx**2 + vy**2)
    velocity = np.insert(velocity, 0, 0.0)

    return velocity


def optimizationofmpcproblem(dt,Horizon,w1,w2,wp,r,end,estimate_now,direction):

    theta = ca.SX.sym("theta",2)
    omega = ca.SX.sym("omega",2)
    tau = ca.SX.sym("tau",2)
    state = ca.vertcat(theta,omega,tau)

    u = ca.SX.sym("u",2)
    control = ca.vertcat(u)

    cos_elbow = ca.cos(theta[1])
    sin_elbow = ca.sin(theta[1])
    cos_shoulder = ca.cos(theta[0])
    cos_both = ca.cos(theta[1]+theta[0])
    DetM = a1*a3-a3*a3-a2*a2*cos_elbow*cos_elbow
    Minv = np.array([[a3,-a3-a2*cos_elbow],
                [-a3-a2*cos_elbow,a1+2*a2*cos_elbow]])/DetM


    C = ca.SX(np.array([-omega[1]*(2*omega[0]+omega[1])*a2*sin_elbow,omega[0]*omega[0]*a2*sin_elbow]))
    G = ca.SX(np.array(
        [
            g
            * (
                m1 * s1 * cos_shoulder
                + m2
                * (s2 * cos_both + l1 * cos_shoulder)
            ),
            g * m2 * s2 * cos_both]
    ))

    Bdyn = ca.SX(np.array([[0.05,0.025],[0.025,0.05]]))           
    jerk = Minv @ (tau-C-Bdyn@omega-G)
    taudot = (u-tau)/.06
    xdot = ca.vertcat(omega,jerk,taudot)

    # CasADi function for system dynamics
    f = ca.Function("f", [state, control], [xdot])

    # Define optimization variables
    opti = ca.Opti()
    X = opti.variable(6, Horizon )  
    U = opti.variable(2, Horizon-1) 

    # Initial and target states
    X0 = opti.parameter(6)
    Xend = 30*np.cos(end[0]) + 33*np.cos(end[0]+end[1])
    Yend = 30*np.sin(end[0]) + 33*np.sin(end[0]+end[1])
    X_targ = np.array([end[0],end[1]])  

    # Objective function and constraints
    cost = 0
    for k in range(Horizon-1):

        x_k = X[:, k]
        u_k = U[:, k]

        Xcurr = 30*np.cos(x_k[0]) + 33*np.cos(x_k[0]+x_k[1])
        Ycurr = 30*np.sin(x_k[0]) + 33*np.sin(x_k[0]+x_k[1])

        cost += r*ca.sumsqr(u_k)  
        if direction == "Horizontal" : 
            cost += wp* (Yend-Ycurr)**2
        else : 
            cost += wp* (Xend-Xcurr)**2
        
        x_next = X[:, k] + dt * f(x_k, u_k)
        opti.subject_to(X[:, k + 1] == x_next)

    # Final state cost
    cost += w1 * ca.sumsqr(X[:2, -1] - X_targ[:2])
    cost += w2*ca.sumsqr(X[2:4, -1])

    opti.minimize(cost)

    # Initial condition constraint
    opti.subject_to(X[:, 0] == X0)

    # Solver setup
    opts = {
    "print_time": 0,
    "ipopt.hessian_approximation": "exact",
    "ipopt.tol": 1e-8,                      
    "ipopt.acceptable_tol": 1e-8,            
    "ipopt.max_iter": 1000,               
    "ipopt.constr_viol_tol": 1e-6,          
    "ipopt.dual_inf_tol": 1e-6,             
    "ipopt.compl_inf_tol": 1e-6             
    }
    opti.solver("ipopt", opts)
    opti.set_value(X0, estimate_now)
    sol = opti.solve()
    return sol,U,f

def MPC(Duration,start,end,w1 = 1e4,w2 = 1,wp = 1e-2,r = 1e-4,num_iter = 60,direction = "Horizontal"):


    
    end = compute_angles_from_cartesian(end[0],end[1])
    start = compute_angles_from_cartesian(start[0],start[1])


    dt = Duration/num_iter
    states = np.zeros((6, num_iter))
    controls = np.zeros((2, num_iter-1))
    G = np.array(
        [
            g
            * (
                m1 * s1 * np.cos(start[0])
                + m2
                * (s2 * np.cos(start[0] +start[1] ) + l1 * np.cos(start[0] ))
            ),
            g * m2 * s2 * np.cos(start[0] + start[1] ),
        ]
    )

    state_now = np.array([start[0], start[1],0, 0,G[0],G[1]])  
    states[:,0] = state_now
    sol,U,f = optimizationofmpcproblem(dt,num_iter,w1,w2,wp,r,end,state_now,direction)
    for t in range(num_iter-1):
            
        u_opt = sol.value(U[:, t])
        controls[:, t] = u_opt

        state_now = state_now + dt * f(state_now, u_opt).full().flatten()
        states[:, t+1] = state_now

    s = states[0]
    e = states[1]
    X = np.cos(s+e)*33+np.cos(s)*30
    Y = np.sin(s+e)*33+np.sin(s)*30
    return X,Y,states


if __name__ == "__main__":

    ALL_DIRECTIONS = ["Vertical", "Horizontal"]
    SIMULATED_MOVEMENT_DIRECTION = ALL_DIRECTIONS[0]

    ACTIVATE_PATH_CONSTRAINT = True
    ACTIVATE_Gravity = True
    MOVEMENT_DURATION = 0.4  # in seconds
    MOVEMENT_LENGTH = 20  # in cm
    NUM_ITER = 60
    FILENAME = "HorizontalGP.png"

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
        ending_positions = np.column_stack(
            (LED[(led_dl) :], [HEIGHT] * (4 - led_dl))
        )
    else:
        LED = np.array([-20, -10, 0, 10])
        DEPTH = 40
        starting_positions = np.column_stack(
            ([DEPTH] * (4 - led_dl), LED[: (4 - led_dl)])
        )
        ending_positions = np.column_stack(
            ([DEPTH] * (4 - led_dl), LED[( led_dl) :])
        )

    fig = plt.figure(figsize=(8, 14))
    gs = gridspec.GridSpec(4, (4 - led_dl), hspace=0.5)
    colors = plt.cm.viridis(np.linspace(0, 1, 6))[:3]
    colors2 = plt.cm.viridis(np.linspace(0, 1, 6))[3:]
    dotcolor = np.array(["blue", "red", "#f932f0"])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])

    WP = 1e-4 if ACTIVATE_PATH_CONSTRAINT else 0  # Path Constraint Cost

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
            num_iter = NUM_ITER,
            direction=SIMULATED_MOVEMENT_DIRECTION,
            wp = WP
        )
        X2, Y2, states2 = MPC(
            Duration=MOVEMENT_DURATION,
            start=end,
            end=start,
            num_iter = NUM_ITER,
            direction=SIMULATED_MOVEMENT_DIRECTION,
            wp = WP
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

    plt.savefig(FILENAME, dpi=200)
    plt.show()
