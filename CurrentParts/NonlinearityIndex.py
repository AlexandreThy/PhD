import os

import scipy

from Controllers.FL import *
from Controllers.ILQG import *
from Controllers.LQGControllers import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec


XST = 0
YST = 40
st = [XST, YST]
MovementTime = 0.4
NUM_ITERATIONS = 40
dt = 0.01
Time = np.linspace(0, MovementTime * 1000, NUM_ITERATIONS)
MN = 5e-4
WP = 20000
WV = 1
WR = 0.5
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


def compute_torque(x, u):
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

def compute_effort(x, u):
    N = np.zeros(x.shape[0] - 1)
    for i in range(x.shape[0] - 1):
        torque = compute_torque(x[i], u[i])
        N[i] = torque[0] * x[i, 2] + torque[1] * x[i, 3]
    return N


def compute_correlation_peak_joint_torque(x, controller=2):

    y = np.load("Costdata.npz")["my_array"]
    result = scipy.stats.linregress(y[:8, controller], x)

    slope = result.slope
    intercept = result.intercept
    r = result.rvalue
    r2 = r**2

    return y[:, controller], r2, slope, intercept

def compute_correlation_motor_cost(x, controller=2):

    y = np.load("Costr.npz")["my_array"]
    result = scipy.stats.linregress(y[:8, controller], x)

    slope = result.slope
    intercept = result.intercept
    r = result.rvalue
    r2 = r**2

    return y[:, controller], r2, slope, intercept


def simulate_traj(num_sim, noise):

    E_ILQG = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))
    E_DLQG = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))
    E_FL = np.zeros((NUM_SIM, 8, NUM_ITERATIONS))

    for num_sim in range(NUM_SIM):

        for move, angles in enumerate(np.linspace(0, 2 * pi, 9)[:-1]):

            tg = [st[0] + cos(angles) * 15, st[1] + sin(angles) * 15]

            _, _, data_ilqg, command_ilqg = simulate_ILQG(
                MovementTime,
                WP,
                WV,
                WR,
                tg,
                st,
                NUM_ITERATIONS,
                delay=0.06,
                Noise=noise,
                print_iterations=False,
                motornoise_variance=MN,
            )

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
                motornoise_variance=MN,
            )

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
                motornoise_variance=MN,
            )

            E_DLQG[num_sim, move, :] = compute_effort(data_DLQG.T[:, :4], command_DLQG)

    return (
        E_ILQG,
        E_FL,
        E_DLQG,
    )


def append_last(arr):
    return np.append(arr, arr[0])


if __name__ == "__main__":
    NUM_SIM = 100
    NOISE = True
    peak_joint_torque_ILQG, peak_joint_torque_FL, peak_joint_torque_DLQG= simulate_traj(NUM_SIM, NOISE)

    peak_joint_torque_ILQG_mean = np.mean(peak_joint_torque_ILQG, axis=(0))
    peak_joint_torque_FL_mean = np.mean(peak_joint_torque_FL, axis=(0))
    peak_joint_torque_DLQG_mean = np.mean(peak_joint_torque_DLQG, axis=(0))

    peak_joint_torque_ILQG_mean = np.max(peak_joint_torque_ILQG_mean, axis=(1))
    peak_joint_torque_FL_mean = np.max(peak_joint_torque_FL_mean, axis=(1))
    peak_joint_torque_DLQG_mean = np.max(peak_joint_torque_DLQG_mean, axis=(1))

    peak_joint_torque_ILQG_mean = append_last(peak_joint_torque_ILQG_mean)
    peak_joint_torque_FL_mean = append_last(peak_joint_torque_FL_mean)
    peak_joint_torque_DLQG_mean = append_last(peak_joint_torque_DLQG_mean)

    motorcost_ilqg, n_rilqg, n_slope_ilqg, n_intercept_ilqg = compute_correlation_motor_cost(peak_joint_torque_ILQG_mean[:8], i=2)
    motorcost_fl, n_rfl, n_slope_fl, n_intercept_fl = compute_correlation_motor_cost(peak_joint_torque_FL_mean[:8], i=2)
    motorcost_dlqg, n_rdlqg, n_slope_dlqg, n_intercept_dlqg = compute_correlation_motor_cost(peak_joint_torque_DLQG_mean[:8], i=2)

    cost_ilqg, e_rilqg, e_slope_ilqg, e_intercept_ilqg = compute_correlation_peak_joint_torque(peak_joint_torque_ILQG_mean[:8], i=2)
    cost_fl, e_rfl, e_slope_fl, e_intercept_fl = compute_correlation_peak_joint_torque(peak_joint_torque_FL_mean[:8], i=2)
    cost_dlqg, e_rdlqg, e_slope_dlqg, e_intercept_dlqg = compute_correlation_peak_joint_torque(peak_joint_torque_DLQG_mean[:8], i=2)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    ax.plot(
        np.linspace(0, 2 * pi, 9),
        peak_joint_torque_ILQG_mean,
        color=colors[0],
    )
    ax.plot(
        np.linspace(0, 2 * pi, 9),
        peak_joint_torque_FL_mean,
        color=colors[1],
    )

    ax.plot(
        np.linspace(0, 2 * pi, 9),
        peak_joint_torque_DLQG_mean,
        color=colors[2],
    )

    ax.scatter(
        np.linspace(0, 2 * pi, 9),
        peak_joint_torque_ILQG_mean,
        color=colors[0],
    )
    ax.scatter(
        np.linspace(0, 2 * pi, 9),
        peak_joint_torque_FL_mean,
        color=colors[1],
    )
    ax.scatter(
        np.linspace(0, 2 * pi, 9),
        peak_joint_torque_DLQG_mean,
        color=colors[2],
    )

    ax.text(
        0,
        1.1,
        f"FL : r² = {n_rfl:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0,
        1.05,
        f"ILQG : r² = {n_rilqg:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0,
        1.00,
        f"DLQG : r² = {n_rdlqg:.2f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.11, 1.1), fontsize=10)
    ax.set_title("Peak Nonlinearity Index vs Cost Function", fontsize=13)
    plt.savefig("Corr_Plots_1DLQG.svg", dpi=300, bbox_inches="tight")
    plt.show()


    angles = np.linspace(0, 2 * pi, 9)[:-1]
    angles_deg = np.degrees(angles)

    fig, ax = plt.subplots(3, figsize=(8, 8))

    
    ax[0].scatter(
        motorcost_ilqg[:8],
        peak_joint_torque_ILQG_mean[:8],
        marker="o",
        color=colors[0],
        linewidth=2,
        label="NL Index " + legend[0],
    )

    ax[1].scatter(
        motorcost_fl[:8],
        peak_joint_torque_FL_mean[:8],
        marker="o",
        color=colors[1],
        linewidth=2,
        label="NL Index " + legend[1],
    )

    ax[2].scatter(
        motorcost_dlqg[:8],
        peak_joint_torque_DLQG_mean[:8],
        marker="o",
        color=colors[2],
        linewidth=2,
        label="NL Index " + legend[2],
    )

    ax[0].plot(
        motorcost_ilqg[:8],
        n_slope_ilqg * motorcost_ilqg[:8] + n_intercept_ilqg,
        color=colors[0],
        linestyle="--",
        label="ILQG Fit",
    )

    ax[1].plot(
        motorcost_fl[:8],
        n_slope_fl * motorcost_fl[:8] + n_intercept_fl,
        color=colors[1],
        linestyle="--",
        label="FL Fit",
    )

    ax[2].plot(
        motorcost_dlqg[:8],
        n_slope_dlqg * motorcost_dlqg[:8] + n_intercept_dlqg,
        color=colors[2],
        linestyle="--",
        label="DLQG Fit",
    )

    ax[1].text(
        0.02,
        0.9,
        f"FL : r² = {n_rfl:.2f}",
        transform=ax[1].transAxes,
        fontsize=12,
    )

    ax[2].text(
        0.02,
        0.9,
        f"DLQG : r² = {n_rdlqg:.2f}",
        transform=ax[2].transAxes,
        fontsize=12,
    )

    ax[0].text(
        0.02,
        0.90,
        f"ILQG : r² = {n_rilqg:.2f}",
        transform=ax[0].transAxes,
        fontsize=12,
    )

    for i in range(3):

        ax[i].set_xlabel("DLQG Motor Cost")
        ax[i].set_ylabel("Peak Joint Power Index")
        ax[i].grid(True)

    plt.savefig("Corr_Plots_2DLQG.svg", dpi=300, bbox_inches="tight")
    plt.show()
    fig, ax = plt.subplots(3, figsize=(8, 8))

    ax[0].scatter(
        cost_ilqg[:8],
        peak_joint_torque_ILQG_mean[:8],
        marker="o",
        color=colors[0],
        linewidth=2,
        label="NL Index " + legend[0],
    )

    ax[1].scatter(
        cost_fl[:8],
        peak_joint_torque_FL_mean[:8],
        marker="o",
        color=colors[1],
        linewidth=2,
        label="NL Index " + legend[1],
    )

    ax[2].scatter(
        cost_dlqg[:8],
        peak_joint_torque_DLQG_mean[:8],
        marker="o",
        color=colors[2],
        linewidth=2,
        label="NL Index " + legend[2],
    )

    ax[0].plot(
        cost_ilqg[:8],
        e_slope_ilqg * cost_ilqg[:8] + e_intercept_ilqg,
        color=colors[0],
        linestyle="--",
        label="ILQG Fit",
    )

    ax[1].plot(
        cost_fl[:8],
        e_slope_fl * cost_fl[:8] + e_intercept_fl,
        color=colors[1],
        linestyle="--",
        label="FL Fit",
    )

    ax[2].plot(
        cost_dlqg[:8],
        e_slope_dlqg * cost_dlqg[:8] + e_intercept_dlqg,
        color=colors[2],
        linestyle="--",
        label="DLQG Fit",
    )

    ax[1].text(
        0.02,
        0.9,
        f"FL : r² = {e_rfl:.2f}",
        transform=ax[1].transAxes,
        fontsize=12,
    )

    ax[2].text(
        0.02,
        0.9,
        f"DLQG : r² = {e_rdlqg:.2f}",
        transform=ax[2].transAxes,
        fontsize=12,
    )

    ax[0].text(
        0.02,
        0.90,
        f"ILQG : r² = {e_rilqg:.2f}",
        transform=ax[0].transAxes,
        fontsize=12,
    )

    for i in range(3):

        ax[i].set_xlabel("DLQG Cost")
        ax[i].set_ylabel("Peak Joint Power Index")
        ax[i].grid(True)

    plt.savefig("Corr_Plots_3DLQG.svg", dpi=300, bbox_inches="tight")
    plt.show()
