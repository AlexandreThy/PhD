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


def optimizationofmpctodorov(dt,Horizon,w1,w2,r,end,estimate_now,l0,lopt,Q,theta0):
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


    def Todorov_fl(l):
        return ca.exp(-((l**1.92 - 1) / 1.03)**2)

    def Todorov_fv( v):
        fv_neg = (-5.72 - v) / (-5.72 + (1.38 + 2.09) * v)
        fv_pos = (0.62 - (-3.12 + 4.21 - 2.67) * v) / (0.62 + v)
        return ca.if_else(v <= 0, fv_neg, fv_pos)

    def Nf(l):
        return 2.11 + 4.16 * (1 / l - 1)

    def Todorov_A(a):
        nf_val = 2.11
        return 1 - ca.exp(-(a / (0.56 * nf_val))**nf_val)

    def Todorov_fp(l):
        return -0.02 * ca.exp(13.8 - 18.7 * l)
    
    def t(u,a):
        return ca.if_else(u > a, 0.066+u*(0.05-0.066), 0.066)
    
    Fmax = ca.SX([572.4,445.2,699.6,381.6,159,318])
    adot = (u-a)/.066 #t(u,a)
    v = Q@omega/lopt
    l = l0 + Q@(theta-theta0)/lopt
    T = Fmax*a*(Todorov_fl(l)*Todorov_fv(v))
    tau = Q.T@T
    Bdyn = ca.SX([[0.05, 0.025], [0.025, 0.05]])
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
    X_targ = ca.vertcat(end[0], end[1], 0, 0,0,0,0,0,0,0)

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
    opts = {'ipopt.print_level': 0, 'print_time': 0,
    "ipopt.tol": 1e-6,                      # Decrease tolerance for better precision
    "ipopt.acceptable_tol": 1e-6,           # More stringent acceptable tolerance
    "ipopt.max_iter": 50000,                 # Increase max iterations          # Constraint violation tolerance            # Complementarity tolerance
    }
    opti.solver("ipopt",opts)
    opti.set_value(X0, estimate_now)
    sol = opti.solve()
    status = sol.stats()

    if status == 'Solve_Succeeded':
        print("Solver found a solution!")
        print("Optimal value of x:")
    else:
        print("Solver did not find a solution.")
    return sol,U,f

def initial_muscle_length(theta):
    # Extract angles
    ts, te = theta[0], theta[1]

    # Constants
    a4 = 0.03
    a11, a22, a33 = 0.055, 0.055, 0.03  # for clarity in l computation
    a51, a52 = 0.04, 0.045
    a61, a62 = 0.04, 0.045
    b1, b2, b3, b4 = 0.08, 0.08, 0.12, 0.12
    l1 = 0.3
    l1_opt = ca.sqrt(a11**2 + b1**2 + 2*a11*b1)/1.1
    l2_opt = ca.sqrt(a22**2 + b2**2 + 2*a22*b2)/1.25
    l3_opt = ca.sqrt(a33**2 + b3**2 + 2*a33*b3)/1.2
    l4_opt = ca.sqrt(a4**2  + b4**2 + 2*a4*b4)/1.1
    l5_opt = ca.sqrt(a51**2 + a52**2 + l1**2 +
               2*a51*l1*ca.cos(ts) + 2*a52*l1*ca.cos(te) + 2*a51*a52)/1.1
    l6_opt = ca.sqrt(a61**2 + a62**2 + l1**2 -
               2*a61*l1*ca.cos(ts) - 2*a62*l1*ca.cos(te) + 2*a61*a62)/1.2

    # Muscle lengths (l)
    l1_ = ca.sqrt(a11**2 + b1**2 + 2*a11*b1*ca.cos(ts))
    l2_ = ca.sqrt(a22**2 + b2**2 - 2*a22*b2*ca.cos(ts))
    l3_ = ca.sqrt(a33**2 + b3**2 + 2*a33*b3*ca.cos(te))
    l4_ = ca.sqrt(a4**2  + b4**2 - 2*a4*b4*ca.cos(te))
    l5_ = ca.sqrt(a51**2 + a52**2 + l1**2 +
               2*a51*l1*ca.cos(ts) + 2*a52*l1*ca.cos(te) + 2*a51*a52*ca.cos(ts + te))
    l6_ = ca.sqrt(a61**2 + a62**2 + l1**2 -
               2*a61*l1*ca.cos(ts) - 2*a62*l1*ca.cos(te) + 2*a61*a62*ca.cos(ts + te))

    l = ca.vertcat(l1_/l1_opt, l2_/l2_opt, l3_/l3_opt, l4_/l4_opt, l5_/l5_opt, l6_/l6_opt)
    l_opt = ca.vertcat(l1_opt, l2_opt, l3_opt, l4_opt, l5_opt, l6_opt)

    # Trig terms
    cos1 = ca.cos(ts)
    cos2 = ca.cos(te)
    cos12 = ca.cos(ts + te)
    sin1 = ca.sin(ts)
    sin2 = ca.sin(te)
    sin12 = ca.sin(ts + te)

    # Denominators
    denom_plus = ca.sqrt(
        a51**2 + a52**2 + l1**2 +
        2*a51*l1*cos1 + 2*a52*l1*cos2 + 2*a51*a52*cos12
    )
    denom_minus = ca.sqrt(
        a61**2 + a62**2 + l1**2 -
        2*a61*l1*cos1 - 2*a62*l1*cos2 + 2*a61*a62*cos12
    )

    # Q matrix elements
    q11 = -a1 * b1 * sin1 / sqrt(a1**2 + b1**2 + 2*a1*b1*cos1)
    q12 =  a2 * b2 * sin1 / sqrt(a2**2 + b2**2 - 2*a2*b2*cos1)
    q23 = -a3 * b3 * sin2 / sqrt(a3**2 + b3**2 + 2*a3*b3*cos2)
    q24 =  a4 * b4 * sin2 / sqrt(a4**2 + b4**2 - 2*a4*b4*cos2)

    q15 = (-a51 * l1 * sin1 - a51 * a52 * sin12) / denom_plus
    q25 = (-a52 * l1 * sin2 - a51 * a52 * sin12) / denom_plus
    q16 = (a61 * l1 * sin1 - a61 * a62 * sin12) / denom_minus
    q26 = (a62 * l1 * sin2 - a61 * a62 * sin12) / denom_minus

    # Assemble Q using CasADi horzcat/vertcat
    Q_row1 = ca.horzcat(q11, q12, 0, 0, q15, q16)
    Q_row2 = ca.horzcat(0, 0, q23, q24, q25, q26)
    Q = ca.vertcat(Q_row1, Q_row2)
    print("l is",l)
    return l, l_opt, Q.T
    
def MPCTodorov(Duration,start,end,w1 = 1e4,w2 = 1,r = 1e-4,Horizon = 0,n_steps = 60,stepupdate = 0,plotTraj = True,plotVel = False):


    

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
            l,lopt,Q = initial_muscle_length(start)
            sol,U,f = optimizationofmpctodorov(dt,Horizon-ecart,w1,w2,r,end_angular,state_now,l,lopt,Q,start)
    
        u_opt = sol.value(U[:, t-ecart])
        controls[:, t] = u_opt
        # Update state using the first control action
        state_now = state_now + dt * f(state_now, u_opt).full().flatten()
        print(l + Q@(state_now[:2]-start))
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