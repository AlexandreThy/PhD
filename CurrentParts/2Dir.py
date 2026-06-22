from Controllers.FL import *
from Controllers.ILQG import *
from Controllers.LQGControllers import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

NUM_SIM = 100

WP = 20000
WV = 1
WR = 0.5

NOISE = True
MovementTime = 0.4
NumIteration = 40

MN = 5e-4

colors = ["#009E73", "#0072B2", "#E69F00"]
legend = ["ILQG", "FL", "DLQG"]


def Cost_function(x, u, w1=WP, w2=WV, r=WR, tg=[0, 0]):
    target1, target2 = compute_angles_from_cartesian(tg[0], tg[1])
    thetas, thetae, omegas, omegae = x[-1, :4]
    return (
        w1 * (thetas - target1) ** 2
        + w1 * (thetae - target2) ** 2
        + w2 * (omegas**2 + omegae**2)
        + np.sum(u * u) * r
    )


# grid of initial conditions
x_vals = np.linspace(-10, 10, 12)
y_vals = np.linspace(30, 45, 9)

# Two target angles: 90° and 315°
target_angles_deg = [90, 315]
target_angles_rad = [np.deg2rad(a) for a in target_angles_deg]

# One cost map per direction
cost_maps = {angle: np.zeros((len(x_vals), len(y_vals))) for angle in target_angles_deg}

for ix, x0 in enumerate(x_vals):
    for iy, y0 in enumerate(y_vals):

        st = [x0, y0]

        for angle_deg, angle_rad in zip(target_angles_deg, target_angles_rad):

            tg = [st[0] + np.cos(angle_rad) * 15, st[1] + np.sin(angle_rad) * 15]

            costs = np.zeros(NUM_SIM)

            for sim in range(NUM_SIM):

                xLQG, yLQG, uDLQG, z = DLQG_6Muscles(
                    w1=WP,
                    w2=WP,
                    w3=WV,
                    w4=WV,
                    Duration=MovementTime,
                    r1=WR,
                    Num_iter=NumIteration,
                    starting_point=st,
                    plot=False,
                    targets=tg,
                    Delay=0.06,
                    Activate_Noise=NOISE,
                    motornoise_variance=MN,
                )

                costs[sim] = Cost_function(z.T, uDLQG, tg=tg)

            cost_maps[angle_deg][ix, iy] = np.mean(costs)
all_values = np.concatenate([cost_maps[a].flatten() for a in target_angles_deg])
global_min = all_values.min()
global_max = all_values.max()
shared_norm = Normalize(vmin=global_min, vmax=global_max)

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

for ax, angle_deg in zip(axes, target_angles_deg):

    data = cost_maps[angle_deg]

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        aspect="auto",
        cmap="plasma",
        norm=shared_norm,
        interpolation="bicubic",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Initial x", fontsize=16)
    ax.set_ylabel("Initial y", fontsize=16)
    ax.set_title(f"iLQG mean cost — {angle_deg}° direction", fontsize=16)
    ax.set_yticks([30, 35, 40, 45])
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels([-10, -5, 0, 5, 10], fontsize=14)
    ax.set_yticklabels([30, 35, 40, 45], fontsize=14)

# Single shared colorbar on the right
cbar = fig.colorbar(im, ax=axes, shrink=0.8)
cbar.set_label("Mean cost (iLQG)", fontsize=14)

plt.savefig("iLQG_CostMap_90_315.svg", dpi=300)
plt.show()
