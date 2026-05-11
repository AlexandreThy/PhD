

from Controllers.FL import *
from Controllers.ILQG import *
from Controllers.LQGControllers import *

import numpy as np
import matplotlib.pyplot as plt

NUM_SIM = 100

WP = 20000
WV = 1
WR = 0.5

NOISE = True
MovementTime = .4
NumIteration = 40

st = [0,40]
MN = 5e-4

colors = ["#009E73","#0072B2","#E69F00"]
legend = ["ILQG","FL","DLQG"]

def Cost_function(x,u,w1 = WP,w2 = WV,r = WR,tg = [0,0] ):
    target1,target2 = compute_angles_from_cartesian(tg[0],tg[1])
    thetas,thetae,omegas,omegae = x[-1,:4]
    
    return w1*(thetas-target1)**2 + w1*(thetae-target2)**2 + w2*(omegas**2+omegae**2)+ np.sum(u*u) * r
# grid of initial conditions
x_vals = np.linspace(-10, 10, 20)
y_vals = np.linspace(30, 45, 15)

n_dirs = 8

direction_map = np.zeros((len(x_vals), len(y_vals)))
for ix, x0 in enumerate(x_vals):
    for iy, y0 in enumerate(y_vals):

        st = [x0, y0]

        mean_costs = np.zeros(n_dirs)

        for d, angle in enumerate(np.linspace(0, 2*np.pi, n_dirs, endpoint=False)):

            tg = [st[0] + np.cos(angle)*15,
                  st[1] + np.sin(angle)*15]

            costs = np.zeros(NUM_SIM)

            for sim in range(NUM_SIM):

                xLQG, yLQG, uDLQG, z = DLQG_6Muscles(
                    w1=WP, w2=WP, w3=WV, w4=WV,
                    Duration=MovementTime,
                    r1=WR,
                    Num_iter=NumIteration,
                    starting_point=st,
                    targets=tg,
                    plot=False,
                    Delay=0.06,
                    Activate_Noise=NOISE,
                    motornoise_variance=5e-4
                )

                costs[sim] = Cost_function(z.T, uDLQG, tg=tg)

            mean_costs[d] = np.mean(costs)

        # direction with WORST cost
        direction_map[ix, iy] = np.argmax(mean_costs)
fig, ax = plt.subplots(figsize=(8, 6))

cmap = plt.cm.get_cmap("tab10", 8)

im = ax.imshow(
    direction_map.T,
    origin="lower",
    extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
    aspect="auto",
    cmap=cmap,
    vmin=-0.5,   # add these two lines
    vmax=7.5
)



cbar = plt.colorbar(im, ticks=np.arange(8))
cbar.set_label("Worst-cost direction (DLQG)")
ax.set_xlabel("Initial x", fontsize=16)
ax.set_ylabel("Initial y", fontsize=16)
ax.set_title("Worst DLQG direction across initial conditions", fontsize=16)
ax.set_aspect("equal")
ax.set_yticks([30,35,40,45])
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_xticklabels([-10, -5, 0, 5, 10], fontsize=16)
ax.set_yticklabels([30,35,40,45], fontsize=16)
plt.savefig("DLQG_WorstDirection_400ms_v2.svg", dpi=300)
plt.show()