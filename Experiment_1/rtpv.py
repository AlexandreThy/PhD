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

n_discretization = 10

shoulder_discretization = np.linspace(shoulder_lim[0], shoulder_lim[1], n_discretization)*pi/180
elbow_discretization = np.linspace(elbow_lim[0], elbow_lim[1], n_discretization)*pi/180



fig,ax = plt.subplots(1,1)      

cmap = plt.get_cmap("viridis")

mov_duration = .4
mov_length = 10
rtpv_values = np.full((n_discretization, n_discretization, 8), np.nan)
start_values = np.zeros((n_discretization, n_discretization, 2))
for i,shoulder in enumerate(shoulder_discretization):
    for j,elbow in enumerate(elbow_discretization):
        if valid_directions(shoulder, elbow, mov_length=mov_length, n_directions=8):
            start = ToCartesian(shoulder,elbow)
            for k,angles in enumerate(np.linspace(0,2*pi,8)):
                
                end = start + np.array([mov_length*np.cos(angles),mov_length*np.sin(angles)])
                tg1,tg2 = np.array(compute_angles_from_cartesian(end[0],end[1]))*180/pi
                X,Y,_,_ = simulate_FL(starting_point=start, targets=end, Duration = mov_duration,wp = 2e4, wv = 10, wr = 6e-5,Num_iter = 40,Delay = 0)
                V = SpeedProfil(X,Y,dt = 0.01)
                rtpv_values[i,j,k] = rtpv(V)
            start_values[i,j] = start
norm = Normalize(vmin=np.nanmin(np.nanmean(rtpv_values, axis=2)), vmax=np.nanmax(np.nanmean(rtpv_values, axis=2)))

for i in range(n_discretization):
    for j in range(n_discretization):
        if np.all(np.isnan(rtpv_values[i,j])):
            continue
        ax.scatter(start_values[i,j][0],start_values[i,j][1], c=np.nanmean(rtpv_values[i,j]), cmap="viridis", norm=norm)

sm = ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=ax, label="value")

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
ax.fill(border_x, border_y, color="lightgrey", zorder=0)  # zorder=0 puts it behind the points

ax.set_aspect("equal")  # same scale in x and y, so the arm geometry is not distorted
ax.axis("off")          # remove the axes, ticks and frame
plt.show()