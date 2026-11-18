from FL import *
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable





shoulder_lim = np.array([-20,110])
elbow_lim = np.array([15,150])

def valid_directions(shoulder, elbow, mov_length=10, n_directions=8, n_path=50):

    start = np.array(ToCartesian(shoulder, elbow))
    angles = np.linspace(0, 2*pi, n_directions, endpoint=False)

    for angle in angles:
        # Check n_path points along the movement, from the start (0) to the target (1)
        for fraction in np.linspace(0, 1, n_path):
            x = start[0] + fraction * mov_length * np.cos(angle)
            y = start[1] + fraction * mov_length * np.sin(angle)

            with np.errstate(invalid="ignore"):  # unreachable points give NaN
                shoulder_angle, elbow_angle = np.degrees(compute_angles_from_cartesian(x, y))

            if np.isnan(shoulder_angle) or np.isnan(elbow_angle):
                return False  # the arm cannot reach this point
            if not shoulder_lim[0] <= shoulder_angle <= shoulder_lim[1]:
                return False  # shoulder outside its limits
            if not elbow_lim[0] <= elbow_angle <= elbow_lim[1]:
                return False  # elbow outside its limits

    return True

n_discretization = 18  # step doubled from the 35-point grid ((35-1)/2+1 points, keeps every other old point)

shoulder_discretization = np.linspace(shoulder_lim[0], shoulder_lim[1], n_discretization)*pi/180
elbow_discretization = np.linspace(elbow_lim[0], elbow_lim[1], n_discretization)*pi/180
  

cmap = plt.get_cmap("viridis")
mov_durations = [.4, .5, .6]
maxdiff_pairs = {}  # amplitude -> pair of starts with the largest summed rtpv difference
for idx,mov_length in enumerate([8, 15,22]):

    mov_duration = mov_durations[idx]
    dt = 0.005  # same 5 ms time step for every amplitude
    K = int(round(mov_duration/dt))  # number of time steps: 80, 100, 120 for 0.4, 0.5, 0.6 s

    rtpv_values = np.full((n_discretization, n_discretization, 8), np.nan)
    y_values = np.full((n_discretization, n_discretization, 8, K+1), np.nan)
    x_values = np.full((n_discretization, n_discretization, 8, K+1), np.nan)
    start_values = np.zeros((n_discretization, n_discretization, 2))
    for i,shoulder in enumerate(shoulder_discretization):
        for j,elbow in enumerate(elbow_discretization):
            # Only the starts from which all 8 directions are feasible at 22 cm, for every amplitude
            if valid_directions(shoulder, elbow, mov_length=22, n_directions=8):
                start = ToCartesian(shoulder,elbow)
                for k,angles in enumerate(np.linspace(0,2*pi,8,endpoint=False)):

                    end = start + np.array([mov_length*np.cos(angles),mov_length*np.sin(angles)])
                    tg1,tg2 = np.array(compute_angles_from_cartesian(end[0],end[1]))*180/pi
                    X,Y,_,_ = simulate_FL(starting_point=start, targets=end, Duration = mov_duration,wp = 2e4, wv = 10, wr = 6e-5,Num_iter = K,Delay = 0)
                    V = SpeedProfil(X,Y,dt = dt)
                    rtpv_values[i,j,k] = rtpv(V)
                    x_values[i,j,k] = X
                    y_values[i,j,k] = Y
                start_values[i,j] = start
    n_points = 100
    shoulder_range = np.radians(np.linspace(shoulder_lim[0], shoulder_lim[1], n_points))
    elbow_range = np.radians(np.linspace(elbow_lim[0], elbow_lim[1], n_points))

    border_shoulder = np.concatenate([shoulder_range,                        # edge 1: shoulder goes up, elbow at min
                                      np.full(n_points, shoulder_range[-1]), # edge 2: shoulder at max, elbow goes up
                                      shoulder_range[::-1],                  # edge 3: shoulder goes down, elbow at max
                                      np.full(n_points, shoulder_range[0])]) # edge 4: shoulder at min, elbow goes down
    border_elbow = np.concatenate([np.full(n_points, elbow_range[0]),
                                   elbow_range,
                                   np.full(n_points, elbow_range[-1]),
                                   elbow_range[::-1]])

    border_x, border_y = ToCartesian(border_shoulder, border_elbow)

    # One map per spread measure of rtpv across the 8 directions: (values, colorbar label, file suffix)
    spreads = [(np.nanstd(rtpv_values, axis=2), "rtpv standard deviation across directions", ""),
               (np.nanvar(rtpv_values, axis=2), "rtpv variance across directions", "var_")]
    for spread, label, suffix in spreads:
        fig,ax = plt.subplots(1,1)
        norm = Normalize(vmin=np.nanmin(spread), vmax=np.nanmax(spread))

        for i in range(n_discretization):
            for j in range(n_discretization):
                if np.all(np.isnan(rtpv_values[i,j])):
                    continue
                ax.scatter(start_values[i,j][0],start_values[i,j][1], c=spread[i,j], cmap="viridis", norm=norm)

        # The two starts with the largest spread, at least 5 cm apart: the largest one, then the
        # largest among the starts at 5 cm or more from it
        order = np.argsort(np.nan_to_num(spread, nan=-np.inf), axis=None)[::-1]
        order = [np.unravel_index(o, spread.shape) for o in order if not np.isnan(spread.flat[o])]
        best = [order[0]]
        best.append(next(o for o in order if np.linalg.norm(start_values[o] - start_values[best[0]]) >= 5))
        for o in best:
            ax.scatter(*start_values[o], marker="*", s=25, c="red", edgecolors="black", linewidths=0.4, zorder=3)
            print(f"{mov_length} cm, {label}: {spread[o]:.5f} at ({start_values[o][0]:.2f}, {start_values[o][1]:.2f}) cm, "
                  f"shoulder {np.degrees(shoulder_discretization[o[0]]):.1f}, elbow {np.degrees(elbow_discretization[o[1]]):.1f} deg")
        if mov_length == 8 and suffix == "":
            # The pair found at 8 cm (std) is the one used for the trajectories at every amplitude
            chosen_starts = [start_values[o].copy() for o in best]

        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label=label)
        ax.fill(border_x, border_y, color="lightgrey", zorder=0)  # zorder=0 puts it behind the points

        ax.set_aspect("equal")  # same scale in x and y, so the arm geometry is not distorted
        ax.axis("off")          # remove the axes, ticks and frame
        plt.savefig("Experiment_1/figures/rtpv_map_" + suffix + str(mov_length) + "cm.png", dpi=300, bbox_inches="tight")
        #plt.show()

    # The two feasible starts (all 8 directions feasible at 22 cm) farthest apart in the hand plane
    feasible = start_values[~np.all(np.isnan(rtpv_values), axis=2)]
    distances = np.linalg.norm(feasible[:, None] - feasible[None, :], axis=2)
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    farthest_starts = [feasible[a], feasible[b]]

    # The two feasible starts whose rtpv differ most: |rtpv difference| per direction, summed over the 8 directions
    feasible_rtpv = rtpv_values[~np.all(np.isnan(rtpv_values), axis=2)]  # (n feasible starts, 8 directions)
    rtpv_diff = np.abs(feasible_rtpv[:, None] - feasible_rtpv[None, :]).sum(axis=2)
    c, d = np.unravel_index(np.argmax(rtpv_diff), rtpv_diff.shape)
    maxdiff_starts = [feasible[c], feasible[d]]
    maxdiff_pairs[mov_length] = maxdiff_starts
    n_ties = (np.sum(rtpv_diff == rtpv_diff[c, d]) // 2)
    print(f"{mov_length} cm, largest summed rtpv difference: {rtpv_diff[c, d]:.4f} between {feasible[c]} and {feasible[d]} "
          f"({distances[c, d]:.2f} cm apart, {n_ties} pair(s) with this value)")
    for start in maxdiff_starts:
        print(f"  {start}: joint configuration ", np.array(compute_angles_from_cartesian(*start))*180/pi)

    # Trajectories from each pair of starts, colored by their rtpv: (starts, figure name)
    for pair_starts, figname in [(chosen_starts, "rtpv_ideal"), (farthest_starts, "rtpv_farthest"), (maxdiff_starts, "rtpv_maxdiff")]:
        chosen_x = np.zeros((2, 8, K+1))
        chosen_y = np.zeros((2, 8, K+1))
        chosen_rtpv = np.zeros((2, 8))
        for p, start in enumerate(pair_starts):
            for k, angle in enumerate(np.linspace(0, 2*pi, 8, endpoint=False)):
                end = start + mov_length*np.array([np.cos(angle), np.sin(angle)])
                X,Y,_,_ = simulate_FL(starting_point=start, targets=end, Duration=mov_duration, wp=2e4, wv=10, wr=6e-5, Num_iter=K, Delay=0)
                chosen_x[p,k] = X
                chosen_y[p,k] = Y
                chosen_rtpv[p,k] = rtpv(SpeedProfil(X,Y,dt=dt))

        fig,ax = plt.subplots(1,1)

        traj_norm = Normalize(vmin=chosen_rtpv.min(), vmax=chosen_rtpv.max())

        for p in range(2):
            for k in range(8):
                ax.plot(chosen_x[p,k], chosen_y[p,k], color=cmap(traj_norm(chosen_rtpv[p,k])))

        fig.colorbar(ScalarMappable(norm=traj_norm, cmap=cmap), ax=ax, label="rtpv")
        ax.fill(border_x, border_y, color="lightgrey", zorder=0)  # zorder=0 puts it behind the points

        ax.set_aspect("equal")  # same scale in x and y, so the arm geometry is not distorted
        ax.axis("off")          # remove the axes, ticks and frame

        plt.savefig("Experiment_1/figures/" + figname + str(mov_length) + "cm.png", dpi=300, bbox_inches="tight")
        #plt.show()

for name, start in zip(["first", "second"], chosen_starts):
    print(f"The {name} start {start} has joint configuration: ", np.array(compute_angles_from_cartesian(*start))*180/pi)
print(f"Farthest feasible starts: {farthest_starts[0]} and {farthest_starts[1]}, {distances[a, b]:.2f} cm apart")
for start in farthest_starts:
    print(f"  {start}: joint configuration ", np.array(compute_angles_from_cartesian(*start))*180/pi)

# Where the max-difference pairs lie among the feasible starts, one panel per amplitude
fig, axes = plt.subplots(1, len(maxdiff_pairs), figsize=(5*len(maxdiff_pairs), 4.5))
for ax, (amp, pair) in zip(axes, maxdiff_pairs.items()):
    ax.fill(border_x, border_y, color="lightgrey", zorder=0)
    ax.scatter(feasible[:, 0], feasible[:, 1], s=20, c="dimgrey", label="feasible starts (22 cm)")
    pair = np.array(pair)
    ax.plot(pair[:, 0], pair[:, 1], "--", color="red", linewidth=1, zorder=2)
    ax.scatter(pair[:, 0], pair[:, 1], marker="*", s=150, c="red", edgecolors="black", linewidths=0.5, zorder=3,
               label="max summed rtpv difference")
    for start in pair:
        ts, te = np.degrees(compute_angles_from_cartesian(*start))
        left = start[0] == pair[:, 0].min()  # left point: label on its left, away from the other starts
        ax.annotate(f"({ts:.1f}°, {te:.1f}°)", start, textcoords="offset points",
                    xytext=(-9, 0) if left else (0, 9), ha="right" if left else "center",
                    va="center" if left else "bottom", fontsize=8, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_title(f"{amp} cm")
    ax.set_aspect("equal")
    ax.axis("off")
axes[0].legend(loc="lower left", fontsize=8, frameon=False, title="(shoulder, elbow)", title_fontsize=8)
plt.savefig("Experiment_1/figures/rtpv_maxdiff_starts.png", dpi=300, bbox_inches="tight")
