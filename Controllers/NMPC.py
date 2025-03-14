from Helpers.Helpers import *

def optimizationofmpcproblem42(dt,Horizon,w1,w2,r,end,estimate_now):
        # State variables: x (cart position), x_dot, theta (pendulum angle), theta_dot
    theta = ca.SX.sym("theta",2,1)
    omega = ca.SX.sym("omega",2,1)
    a = ca.SX.sym("a",6,1)
    state = ca.vertcat(theta,omega,a)

    # Control input: Force applied to the cart
    u = ca.SX.sym("u",6)
    control = ca.vertcat(u)

    # Equations of motion (nonlinear dynamics)
    cos_elbow = ca.cos(theta[1])
    sin_elbow = ca.sin(theta[1])
    Minv = ca.inv(ca.vertcat(
    ca.horzcat(a1+2*a2*cos_elbow, a3 +a2 * cos_elbow),
    ca.horzcat(a3 +a2 * cos_elbow, a3)
))

    C = ca.vertcat(
    -omega[1] * (2 * omega[0] + omega[1]) * a2 * sin_elbow,
    omega[0] * omega[0] * a2 * sin_elbow
) 


    

    M = ca.DM([
    [0.04, 0.04, 0, 0, 0.028, 0.028],
    [0, 0, 0.025, 0.025, 0.035, 0.035]
]).T  
    lrest = ca.SX([0.09, 0.04, 0.06, 0.1, 0.19, 0.14])
    Fmax = ca.SX([1142, 260, 987, 624, 430, 798])
    vmax = 6.28*lrest
    adot = (u-a)/0.06
    l = lrest-M@theta
    v = -M@omega
    Vsh = .3
    fl = ca.exp(-(((l-lrest)/lrest)/.5)**2)
    fv = (vmax-v)/(vmax+.3*v)

    

    T = a*fl*fv*Fmax
    tau = -M.T@T
    Bdyn = ca.SX(np.array([[0.05,0.025],[0.025,0.05]]))  
    acc = Minv @ (tau-C-Bdyn@omega)
    
    xdot = ca.vertcat(omega,acc,adot)

    # CasADi function for system dynamics
    f = ca.Function("f", [state, control], [xdot])

    # Define optimization variables
    opti = ca.Opti()
    X = opti.variable(10, Horizon )  # State trajectory
    U = opti.variable(6, Horizon-1)  # Control inputs

    # Initial and target states
    X0 = opti.parameter(10)
    X_targ = ca.vertcat(end[0], end[1], 0.0, 0,0,0,0,0,0,0)

    # Objective function and constraints
    cost = 0
    for k in range(Horizon-1):
        x_k = X[:, k]
        u_k = U[:, k]
        for l in range(6):
            cost += r[l]*u_k[l]*u_k[l]  # Quadratic cost
        
        # System dynamics constraint using Euler discretization
        x_next = X[:, k] + dt * f(x_k, u_k)
        opti.subject_to(X[:, k + 1] == x_next)

    # Final state cost
    cost += w1 * ca.sumsqr(X[:2, -1] - X_targ[:2])
    cost += w2*ca.sumsqr(X[2:4, -1] - X_targ[2:4])

    opti.minimize(cost)

    # Initial condition constraint
    opti.subject_to(X[:, 0] == X0)

    # Solver setup
    opts = {'ipopt.print_level': 0,
    "print_time": 0,
    "ipopt.tol": 1e-10,                      # Decrease tolerance for better precision
    "ipopt.acceptable_tol": 1e-12,           # More stringent acceptable tolerance
    "ipopt.max_iter": 5000,                 # Increase max iterations          # Constraint violation tolerance            # Complementarity tolerance
    }
    opti.solver("ipopt",opts)
    opti.set_value(X0, estimate_now)
    sol = opti.solve()
    return sol,U,f

def MPC42(Duration,start,end,w1 = 1e4,w2 = 1,r = 1e-4,Horizon = 0,n_steps = K,stepupdate = 0,plotTraj = True,plotVel = False):


    

    # Simulation
    r = np.ones(6)*r
    end_angular= newton(newtonf,newtondf,1e-8,1000,end[0],end[1]) #Defini les targets
    start = newton(newtonf,newtondf,1e-8,1000,start[0],start[1])
    Duration
    dt = Duration / n_steps
    if Horizon == 0: Horizon = n_steps
    if stepupdate == 0: stepupdate = n_steps
    states = np.zeros((10, n_steps))

    controls = np.zeros((6, n_steps-1))
    state_now = np.array([start[0], start[1],0,0,0,0,0,0,0,0])    # Slightly off-balance initial condition
    states[:,0] = state_now
    ecart = -stepupdate
    for t in range(n_steps-1):
        if t%stepupdate ==0: 
            ecart +=stepupdate
            sol,U,f = optimizationofmpcproblem42(dt,Horizon-ecart,w1,w2,r,end_angular,state_now)
    
        u_opt = sol.value(U[:, t-ecart])
        controls[:, t] = u_opt
        # Update state using the first control action
        state_now = state_now + dt * f(state_now, u_opt).full().flatten()
        states[:, t+1] = state_now

    if plotVel :

        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, Duration, n_steps), states[2, :], label="Shoulder ",color = "#007399")
        plt.plot(np.linspace(0, Duration, n_steps), states[3,:], label="Elbow",linestyle = "-.",color = "#007399")
        plt.axhline(0, color='grey', linestyle='--')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.subplot(2, 1, 2)
    if plotTraj :
        
        s = states[0]
        e = states[1]
        X = np.cos(s+e)*33+np.cos(s)*30
        Y = np.sin(s+e)*33+np.sin(s)*30
        plt.plot(X,Y,color = "#007399",label = "NMPC 6 Muscles")
        #plt.grid()
        plt.axis("equal")
        plt.scatter([end[0]],[end[1]],color = "red")
    return states