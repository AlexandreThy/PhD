import os

import scipy

from Controllers.FL import *
from Controllers.ILQG import *
from Controllers.LQGControllers import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec


XST = 0
YST = 45
st = [XST, YST]
MovementTime = 0.4
NUM_ITERATIONS = 40
dt = 0.01
Time = np.linspace(0, MovementTime * 1000, NUM_ITERATIONS)
MN = 1e-3
WP = 100
WV = 0.05
WR = 0.1
WR_FL = 1e-8
colors = ["#009E73", "#0072B2", "#E69F00"]
legend = ["ILQG", "FL", "DLQG"]


def compute_angles_from_cartesian(x, y, l1=30, l2=33):

    r_squared = x**2 + y**2

    shoulder_angle = np.arctan2(y, x) - np.arccos(
        (r_squared + l1**2 - l2**2) / (2 * l1 * np.sqrt(r_squared))
    )

    elbow_angle = np.pi - np.arccos((l1**2 + l2**2 - r_squared) / (2 * l1 * l2))
    return shoulder_angle, elbow_angle


def time_derivative(data, dt):
    derivative = np.zeros_like(data)
    derivative[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    derivative[0] = (data[1] - data[0]) / dt
    derivative[-1] = (data[-1] - data[-2]) / dt
    return derivative


def compute_torque(x, u, F=0):
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

    return A @ (u * fl * ff_v)


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
    ang_acc = Minv @ (A @ (u * fl * ff_v) - Bdyn @ x[2:4] - C + F)

    return np.array([[x[2], x[3], ang_acc[0], ang_acc[1], 0, 0]])


def dfdx(x, u):

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
    fl = np.exp(np.abs((l**1.55 - 1) / 0.81))

    dfldl = fl * np.sign((l**1.55 - 1) / 0.81) * (1.55 * l**0.55 / 0.81)
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
    return A


def dfdu(x, u):
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
    B = np.zeros((4, 6))
    for i in range(6):
        du = np.zeros(6)
        du[i] = 1
        B[2:, i] = Minv @ (A @ (du * fl * fv))
    return B


def compute_nl_index(x, u, neglect_u=True):

    N = np.zeros(x.shape[0] - 1)
    for i in range(1, x.shape[0]):
        uforlin = np.zeros(6) if neglect_u else u[i - 1]
        x_true = x[i]
        x_pred = (
            x[i - 1]
            + dt * dfdx(x[i - 1], uforlin) @ x[i - 1]
            + dt * dfdu(x[i - 1], uforlin) @ u[i - 1]
        )
        N[i - 1] = np.linalg.norm(x_true[2:4] - x_pred[2:4])
    return N


def compute_effort(x, u):
    N = np.zeros(x.shape[0] - 1)
    for i in range(x.shape[0] - 1):
        torque = compute_torque(x[i], u[i])
        N[i] = torque[0] * x[i, 2] + torque[1] * x[i, 3]
    return N


def compute_correlation(x):

    y = np.load("Costdata.npz")["my_array"]
    result = scipy.stats.linregress(y[:8, 2], x)

    slope = result.slope
    intercept = result.intercept
    r = result.rvalue
    r2 = r**2
    print(r2)

    return y[:, 2], r2, slope, intercept


def simulate_traj(num_sim, noise):

    N_FL = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))
    N_DLQG = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))
    N_ILQG = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))

    E_ILQG = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))
    E_DLQG = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))
    E_FL = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))

    for num_sim in range(NUM_SIM):

        for move, angles in enumerate(np.linspace(0, 2 * pi, 9)[:-1]):

            tg = [cos(angles) * 15, 45 + sin(angles) * 15]

            _, _, data_ilqg, command_ilqg = simulate_ILQG(
                MovementTime,
                100,
                0.05,
                0.1,
                tg,
                st,
                NUM_ITERATIONS,
                delay=0.06,
                Noise=noise,
                print_iterations=False,
            )
            N_ILQG[num_sim, move, :] = compute_nl_index(data_ilqg, command_ilqg)
            E_ILQG[num_sim, move, :] = compute_effort(data_ilqg, command_ilqg)

            _, _, data_FL, command_FL = simulate_FL(
                Duration=MovementTime,
                w1=WP,
                w2=WP,
                w3=WV,
                w4=WV,
                r=WR_FL,
                FF=False,
                Num_iter=NUM_ITERATIONS,
                starting_point=st,
                targets=tg,
                Delay=0.06,
                Activate_Noise=noise,
            )

            N_FL[num_sim, move, :] = compute_nl_index(data_FL[:, :4], command_FL)
            E_FL[num_sim, move, :] = compute_effort(data_FL[:, :4], command_FL)
            _, _, command_DLQG, data_DLQG = DLQG_6Muscles(
                w1=WP,
                w2=WP,
                w3=WV,
                w4=WV,
                Duration=MovementTime,
                r1=WR,
                Num_iter=NUM_ITERATIONS,
                starting_point=st,
                targets=tg,
                plot=False,
                Delay=0.06,
                Activate_Noise=noise,
            )
            N_DLQG[num_sim, move, :] = compute_nl_index(
                data_DLQG.T[:, :4], command_DLQG
            )
            E_DLQG[num_sim, move, :] = compute_effort(data_DLQG.T[:, :4], command_DLQG)

    return (
        N_ILQG,
        N_FL,
        N_DLQG,
        np.max(E_ILQG, axis=2),
        np.max(E_FL, axis=2),
        np.max(E_DLQG, axis=2),
    )


if __name__ == "__main__":
    NUM_SIM = 100
    NOISE = True
    N_ILQG, N_FL, N_DLQG, E_ILQG, E_FL, E_DLQG = simulate_traj(NUM_SIM, NOISE)

    N_ILQG_mean = np.mean(N_ILQG, axis=(0, 2))
    N_FL_mean = np.mean(N_FL, axis=(0, 2))
    N_DLQG_mean = np.mean(N_DLQG, axis=(0, 2))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.bar(
        np.linspace(0, 2 * pi, 9)[:-1] - 0.2,
        N_ILQG_mean / np.max(N_ILQG_mean),
        width=0.2,
        color=colors[0],
        label="NL Index " + legend[0],
        edgecolor="black",
    )
    ax.bar(
        np.linspace(0, 2 * pi, 9)[:-1],
        N_FL_mean / np.max(N_FL_mean),
        width=0.2,
        color=colors[1],
        label="NL Index " + legend[1],
        edgecolor="black",
    )
    # ax.bar(
    #    np.linspace(0, 2 * pi, 9)[:-1] + 0.2,
    #    N_DLQG_mean,
    #    width=0.2,
    #    color=colors[2],
    #    label=legend[2],
    #    edgecolor="black",
    # )
    cost, rfl, _, _ = compute_correlation(N_FL_mean)
    _, rilqg, _, _ = compute_correlation(N_ILQG_mean)
    ax.plot(
        np.linspace(0, 2 * pi, 9),
        cost / np.max(cost),
        color=colors[2],
        linewidth=4,
        label="Cost DLQG",
    )
    ax.text(
        0,
        1.1,
        f"FL : r² = {rfl:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0,
        1.05,
        f"ILQG : r² = {rilqg:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.11, 1.1), fontsize=10)
    ax.set_title("Nonlinearity Index vs Cost Function", fontsize=13)
    ax.set_yticks([])
    plt.savefig("Corr_Plots_1_Noisy.svg", dpi=300, bbox_inches="tight")
    plt.show()

    E_ILQG_mean = np.mean(E_ILQG, axis=(0))
    E_FL_mean = np.mean(E_FL, axis=(0))
    E_DLQG_mean = np.mean(E_DLQG, axis=(0))

    E_ILQG_mean = np.append(E_ILQG_mean, [E_ILQG_mean[0]])
    E_FL_mean = np.append(E_FL_mean, [E_FL_mean[0]])
    E_DLQG_mean = np.append(E_DLQG_mean, [E_DLQG_mean[0]])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    ax.plot(
        np.linspace(0, 2 * pi, 9),
        E_ILQG_mean / np.max(E_ILQG_mean),
        color=colors[0],
        label="Peak Joint Power " + legend[0],
    )
    ax.plot(
        np.linspace(0, 2 * pi, 9),
        E_FL_mean / np.max(E_FL_mean),
        color=colors[1],
        label="Peak Joint Power " + legend[1],
    )

    ax.scatter(
        np.linspace(0, 2 * pi, 9),
        E_ILQG_mean / np.max(E_ILQG_mean),
        color=colors[0],
        label="Peak Joint Power " + legend[0],
    )
    ax.scatter(
        np.linspace(0, 2 * pi, 9),
        E_FL_mean / np.max(E_FL_mean),
        color=colors[1],
        label="Peak Joint Power " + legend[1],
    )
    # ax.bar(
    #    np.linspace(0, 2 * pi, 9)[:-1] + 0.2,
    #    E_DLQG_mean,
    #    width=0.2,
    #    color=colors[2],
    #    label=legend[2],
    #    edgecolor="black",
    # )
    # ax.plot(
    #    np.linspace(0, 2 * pi, 9),
    #    cost / np.max(cost),
    #    color=colors[2],
    #    linewidth=4,
    #    label="Cost DLQG",
    # )

    _, rfl, _, _ = compute_correlation(E_FL_mean[:8])
    _, rilqg, _, _ = compute_correlation(E_ILQG_mean[:8])

    ax.text(
        0,
        1.1,
        f"FL : r² = {rfl:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0,
        1.05,
        f"ILQG : r² = {rilqg:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.11, 1.1), fontsize=10)
    ax.set_title("Peak Joint Power vs Cost Function", fontsize=13)
    plt.savefig("Corr_Plots_2_Noisy.svg", dpi=300, bbox_inches="tight")
    plt.show()

    angles = np.linspace(0, 2 * pi, 9)[:-1]
    angles_deg = np.degrees(angles)

    fig, ax = plt.subplots(2, figsize=(8, 8))
    cost, rfl, slopefl, interceptfl = compute_correlation(N_FL_mean)
    _, rilqg, slopeilqg, interceptilqg = compute_correlation(N_ILQG_mean)
    cost = cost[:8]
    ax[0].scatter(
        cost,
        N_ILQG_mean,
        marker="o",
        color=colors[0],
        linewidth=2,
        label="NL Index " + legend[0],
    )

    ax[1].plot(
        cost,
        slopefl * cost + interceptfl,
        color=colors[1],
        linestyle="--",
        label="FL Fit",
    )
    ax[0].plot(
        cost,
        slopeilqg * cost + interceptilqg,
        color=colors[0],
        linestyle="--",
        label="ILQG Fit",
    )

    ax[1].scatter(
        cost,
        N_FL_mean,
        marker="o",
        color=colors[1],
        linewidth=2,
        label="NL Index " + legend[1],
    )

    ax[1].text(
        0.02,
        0.9,
        f"FL : r² = {rfl:.2f}",
        transform=ax[1].transAxes,
        fontsize=12,
    )

    ax[0].text(
        0.02,
        0.90,
        f"ILQG : r² = {rilqg:.2f}",
        transform=ax[0].transAxes,
        fontsize=12,
    )

    for i in range(2):

        ax[i].set_xlabel("DLQG Movement Cost")
        ax[i].set_ylabel("Nonlinearity Index")
        ax[i].grid(True)

    plt.savefig("Corr_Plots_3_Noisy.svg", dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(2, figsize=(8, 8))

    ax[0].scatter(
        cost,
        E_ILQG_mean[:8],
        marker="o",
        color=colors[0],
        linewidth=2,
        label="Peak Joint Power " + legend[0],
    )

    ax[1].scatter(
        cost,
        E_FL_mean[:8],
        marker="o",
        color=colors[1],
        linewidth=2,
        label="Peak Joint Power " + legend[1],
    )

    _, rfl, slopefl, interceptfl = compute_correlation(E_FL_mean[:8])
    _, rilqg, slopeilqg, interceptilqg = compute_correlation(E_ILQG_mean[:8])

    ax[1].plot(
        cost,
        slopefl * cost + interceptfl,
        color=colors[1],
        linestyle="--",
        label="FL Fit",
    )
    ax[0].plot(
        cost,
        slopeilqg * cost + interceptilqg,
        color=colors[0],
        linestyle="--",
        label="ILQG Fit",
    )

    ax[1].text(
        0.02,
        0.9,
        f"FL : r² = {rfl:.2f}",
        transform=ax[1].transAxes,
        fontsize=12,
    )

    ax[0].text(
        0.02,
        0.90,
        f"ILQG : r² = {rilqg:.2f}",
        transform=ax[0].transAxes,
        fontsize=12,
    )

    for i in range(2):
        ax[i].set_xlabel("DLQG Movement Cost")
        ax[i].set_ylabel("Peak Joint Power")
        ax[i].grid(True)

    plt.savefig("Corr_Plots_4_Noisy.svg", dpi=300, bbox_inches="tight")
    plt.show()
