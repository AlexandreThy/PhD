import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Perpendicular vector direction (normal to line)
normal_vec = np.array([.2, 0])

# Points for animation
x_vals = np.zeros(100)
y_vals = np.linspace(np.pi/4,np.pi/2,100)

# Set up plot
fig, ax = plt.subplots()

# Set limits
ax.set_xlim(-1, 1)
ax.set_ylim(0.5, 2)

# Major ticks and labels
ax.set_yticks([np.pi/4, np.pi/2])
ax.set_yticklabels([r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"], fontsize=15)

ax.set_xticks([-np.pi/4, np.pi/4])
ax.set_xticklabels([r"$\frac{-\pi}{4}$", r"$\frac{\pi}{4}$"], fontsize=15)

# Minor ticks for denser grid
ax.set_xticks(np.linspace(-1, 1, 9), minor=True)   # 9 evenly spaced minor x-ticks
ax.set_yticks(np.linspace(0.5, 2, 16), minor=True) # 16 minor y-ticks

# Grid on both major and minor ticks
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')

ax.set_aspect('equal')
# Plot the line
ax.plot(x_vals, y_vals, 'k--', label='Line')
ax.set_xlabel(r"Shoulder angle $\theta_s$",labelpad = -20)
ax.set_ylabel(r"Elbow angle $\theta_e$")

# Initialize moving point and normal vector
point, = ax.plot([], [], 'ro')
arrow = ax.annotate("", xy=(0,0), xytext=(0,0), arrowprops=dict(arrowstyle="->", color='blue', lw=2))
label = ax.text(0, 0, "", fontsize=10, color='blue')

def init():
    point.set_data([], [])
    arrow.set_position((0, 0))
    label.set_text("")
    return point, arrow, label

def animate(i):
    x = x_vals[i]
    y = np.linspace(np.pi/4,np.pi/2,100)[i]
    
    # Point on line
    point.set_data([x], [y])

    # Compute normal vector at that point
    scale = 1.0  # Length of the normal vector
    dx, dy = normal_vec * scale

    # Update arrow (normal vector)
    arrow.xy = (x + dx, y + dy)
    arrow.set_position((x, y))

    # Update label with coordinates of normal
    label.set_position((x + dx - 0.1, y + dy + 0.1))
    label.set_text(f"($\\theta_s$ = 1.00, $\\theta_e$ = 0.00)")

    return point, arrow, label

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(x_vals), interval=50, blit=True)

plt.title("Normal Vector in Joint Angles coordinates")
plt.legend()
plt.show()

normal_vec = np.array([1, 0])

# Points for animation
x_vals = np.zeros(100)
y_vals = np.linspace(np.pi/4,np.pi/2,100)
def tocart(x_vals,y_vals):
    X= 30*np.cos(x_vals)+33*np.cos(x_vals+y_vals)
    Y= 30*np.sin(x_vals)+33*np.sin(x_vals+y_vals)
    return X,Y 

X,Y = tocart(x_vals,y_vals)
# Set up plot
fig, ax = plt.subplots()
# Plot the line
ax.plot(X, Y, 'k--', label='Line')

# Initialize moving point and normal vector
point, = ax.plot([], [], 'ro')
arrow = ax.annotate("", xy=(0,0), xytext=(0,0), arrowprops=dict(arrowstyle="->", color='blue', lw=2))
label = ax.text(0, 0, "", fontsize=10, color='blue')
ax.set_xlim(25,60)
ax.set_ylim(20,40)

ax.set_yticks([20,30,40])

ax.set_xticks([30,40,50,60])

ax.set_xlabel("X [cm]")
ax.set_ylabel("Y [cm]")

# Minor ticks for denser grid
ax.set_xticks(np.linspace(25, 60, 9), minor=True)   # 9 evenly spaced minor x-ticks
ax.set_yticks(np.linspace(20, 40, 16), minor=True) # 16 minor y-ticks

ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray')

ax.set_aspect('equal')
def init():
    point.set_data([], [])
    arrow.set_position((0, 0))
    label.set_text("")
    return point, arrow, label

def animate(i):
    x = X[i]
    y = Y[i]
    
    # Point on line
    point.set_data([x], [y])

    # Compute normal vector at that point
    scale = 1.0  # Length of the normal vector
    dx, dy = normal_vec * scale

    # Update arrow (normal vector)
    arrow.xy = (x + 3, y + 0)
    arrow.set_position((x, y))

    # Update label with coordinates of normal
    label.set_position((x + dx - 0.5, y + dy + 0.5))
    label.set_text(f"($\\theta_s$ = 1.00, $\\theta_e$ = 0.00)")

    return point, arrow, label

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(x_vals), interval=50, blit=True)

plt.title("Normal Vector in Cartesian coordinates")
plt.legend()
plt.show()
