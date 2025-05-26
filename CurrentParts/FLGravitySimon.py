import numpy as np
from matplotlib import pyplot as plt
from math import *
import matplotlib.cm as cm
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

def NoiseAndCovMatrix(M=np.identity(2), N=6, kdelay=0, Var=1e-6, Linear=False):

    K = 1 / 0.06
    M = np.linalg.inv(M)
    Sigmau = np.array([[Var, 0], [0, Var]])
    Sigmav = K * K * M @ Sigmau @ M.T
    SigmaMotor = np.zeros((N * (kdelay + 1), N * (kdelay + 1)))
    Sigma = np.zeros((N, N))
    SigmaSense = np.diag(np.ones(N) * 1e-6)
    for S in [Sigma, SigmaSense]:
        S[2, 2] = Sigmav[0, 0]
        S[2, 5] = Sigmav[0, 1]
        S[5, 2] = Sigmav[1, 0]
        S[5, 5] = Sigmav[1, 1]
    SigmaMotor[:N, :N] = Sigma

    motornoise, sensorynoise = np.zeros(2), np.zeros(N)
    for i in range(N):
        sensorynoise[i] = np.random.normal(0, np.sqrt(SigmaSense[i, i]))
    motornoise = np.random.normal(0, np.sqrt(Var), 2)
    Omegasenslinear = np.zeros((N * (kdelay + 1), N * (kdelay + 1)))
    Omegasenslinear[2, 2] = Var
    Omegasenslinear[5, 5] = Var
    if Linear:
        return (
            Omegasenslinear,
            np.diag(np.ones(N) * 1e-6),
            motornoise,
            np.random.normal(0, np.sqrt(Var), N),
        )
    return SigmaMotor, SigmaSense, motornoise, sensorynoise

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

    # Final h1 calculation
    h1 = np.arctan2(y, x) - np.arccos(
        (r_squared + l1**2 - l2**2) / (2 * l1 * np.sqrt(r_squared))
    )

    # Compute h2
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
    
    # Compute the differences between consecutive x and y points
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Compute velocity components
    vx = dx / dt
    vy = dy / dt
    
    # Compute absolute velocity (Euclidean norm)
    velocity = np.sqrt(vx**2 + vy**2)
    velocity = np.insert(velocity, 0, 0.0)
    
    return velocity


def Path_Constraint(x0,xf,wp,percent):
    xtarget = x0 + percent*(xf-x0)
    ts,te= compute_angles_from_cartesian(xtarget[0],xtarget[1])
    ts0,te0 = compute_angles_from_cartesian(x0[0],x0[1])
    k = (te-te0)/(ts-ts0)
    Qk=np.zeros((10,10))

    Qk[0,0] = k*k
    Qk[0,8] = -k*k
    Qk[8,0] = -k*k
    Qk[8,8] = k*k
    Qk[3,3] = 1
    Qk[3,9] = -1
    Qk[9,3] = -1
    Qk[9,9] = 1
    Qk[0,3] = -k 
    Qk[3,0] = -k 
    Qk[0,9] = k 
    Qk[9,0] = k 
    Qk[3,8] = k 
    Qk[8,3] = k
    Qk[8,9] = -k
    Qk[9,8] = -k

    return Qk*wp

def Feedback_Linearization_witPathConstraintG(Duration = .6,w1 = 1e8,w2 = 1e8,w3 = 1e4,w4 = 1e4,wp = 1e4,r1 = 1e-6,r2 = 1e-6,g = 9.81,targets = [0,55],starting_point = [0,30] ,Activate_Noise = False,Num_iter = 300,delay = .06,taupath = .1,path_constraint_length = .3):
    
    """
    Duration (float) : Duration of the movement

    w1 (float) : Weight associated to the penalty on shoulder angle 
    
    w2 (float) : Weight associated to the penalty on elbow angle 

    w3 (float) : Weight associated to the penalty on shoulder angular velocity

    w4 (float) : Weight associated to the penalty on elbow angular velocity

    wp (float) : Weight associated to the penalty on path constrained

    r1 (float) : Weight associated to the motor cost on shoulder torques

    r2 (float) : Weight associated to the motor cost on elbow torques

    targets (array of float of size 2): X,Y Position of the end point of the movement
        must be a biomechanically feasible end point considering that the shoulder is at (0,0)

    starting_point (array of float of size 2): X,Y Position of the starting point of the movement
        must be a biomechanically feasible starting point considering that the shoulder is at (0,0)
    
    Activate_Noise (boolean) : Gaussian variance associated to the white noises in the model

    Num_iter (int) : Number of simulations steps

    delay(float) : Internal Delay of the system in seconds

    taupath (float) : Time_Constant for Path Constraint 

    path_constraint_length (float between 0 and 1) : Normalized reaching point of path constraint
    """

    dt = Duration/Num_iter 
    kdelay = int(delay/dt)
    obj1,obj2 = compute_angles_from_cartesian(targets[0],targets[1])
    st1,st2 = compute_angles_from_cartesian(starting_point[0],starting_point[1])
    Qk = Path_Constraint(np.array(starting_point),np.array(targets),wp,path_constraint_length)

    Initial_torque = np.array(
        [
            g
            * (
                m1 * s1 * np.cos(st1)
                + m2
                * (s2 * np.cos(st1+st2) + l1 * np.cos(st1 ))
            ),
            g * m2 * s2 * np.cos(st1 + st2),
        ]
        ) # To have 0 acceleration at start, we set the initial torque equal to gravitational effect

    x0 = np.array([st1,0,0,st2,0,0,obj1,obj2,st1,st2])
    x0_with_delay = np.tile(x0, kdelay + 1) 
    
    Num_Var = 10
    
    #Define Weight Matrices of the cost function
    R = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0,0,0],[0,w3,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
               [0,0,0,w2,0,0,0,-w2,0,0],[0,0,0,0,w4,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
               [-w1,0,0,0,0,0,w1,0,0,0],[0,0,0,-w2,0,0,0,w2,0,0],
               [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])

    
    
    #Define the Dynamic of the linear system 

    Kfactor = 1/0.06

    A_basic = np.array([[1,dt,0,0,0,0,0,0,0,0],[0,1,dt,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,dt,0,0,0,0,0],[0,0,0,0,1,dt,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0]
                        ,[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
    B_basic = np.array([[0,0],[0,0],[dt,0],[0,0],[0,0],[0,dt],[0,0],[0,0],[0,0],[0,0]])
    sigma = np.zeros((Num_Var*(kdelay+1),Num_Var*(kdelay+1)))

    #Incorporation of delay 

    H = np.zeros((Num_Var,(kdelay+1)*Num_Var))
    H[:,(kdelay)*Num_Var:]= np.identity(Num_Var)

    A = np.zeros(((kdelay+1)*Num_Var,(kdelay+1)*Num_Var))
    A[:Num_Var,:Num_Var] = A_basic
    A[Num_Var:,:-Num_Var] = np.identity((kdelay)*Num_Var)
    B = np.zeros(((kdelay+1)*Num_Var,2))
    B[:Num_Var] = B_basic

    #Compute the Feedback Gain of the Control law
    S = Q

    array_L = np.zeros((Num_iter-1,2,Num_Var))   
    array_S = np.zeros((Num_iter,Num_Var,Num_Var)) 
    array_S[-1] = S
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+B_basic.T@S@B_basic)@B_basic.T@S@A_basic
        array_L[Num_iter-2-k] = L
        S = Qk*np.exp(-k*dt/taupath)+A_basic.T@S@(A_basic-B_basic@L)
        array_S[Num_iter-2-k] = S
        
    L = array_L


    #Initialize the arrays to stock the simulations 
    array_zhat = np.zeros((Num_iter,Num_Var))
    array_x = np.zeros((Num_iter,Num_Var-4)) 
    array_z = np.zeros((Num_iter,Num_Var)) 
    array_u = np.zeros((Num_iter-1,2))
    y = np.zeros((Num_iter-1,Num_Var))

    #Initialize the state vectors of the linear system
    array_zhat[0] = x0.flatten()
    array_z[0] = x0.flatten()
    zhat = np.copy(x0_with_delay)
    z =  np.copy(x0_with_delay)

    #Initialize the state vectors of the nonlinear system
    x = np.zeros(Num_Var-4)
    x[0],x[1] = x0[0],x0[3]
    x[4],x[5] = Initial_torque
    new_x = np.copy(x)
    array_x[0] = x

    angular_acceleration = np.zeros(2)
    for k in range(Num_iter-1):
        #Compute the matrices of the FL technique in function of the current estimate state 
        

        C = np.array([-zhat[4]*(2*zhat[1]+zhat[4])*a2*np.sin(zhat[3]),zhat[1]*zhat[1]*a2*np.sin(zhat[3])])

        M = np.array([[a1+2*a2*cos(zhat[3]),a3+a2*cos(zhat[3])],[a3+a2*cos(zhat[3]),a3]])

        Cdot = np.array([-a2*zhat[5]*(2*zhat[1]+zhat[4])*sin(zhat[3])-a2*zhat[4]*(2*zhat[2]+zhat[5])*sin(zhat[3])
                         -a2*zhat[4]*zhat[4]*(2*zhat[1]+zhat[4])*cos(zhat[3]),2*zhat[1]*zhat[2]*a2*sin(zhat[3])+zhat[1]*zhat[1]*a2*cos(zhat[3])*zhat[4]])

        Mdot = np.array([[-2*a2*sin(zhat[3])*zhat[4],-a2*sin(zhat[3])*zhat[4]],[-a2*sin(zhat[3])*zhat[4],0]])

        G = np.array([g* (m1 * s1 * np.cos(zhat[0])+ m2 * (s2 * np.cos(zhat[0] + zhat[3]) + l1 * np.cos(zhat[0] ))),
            g * m2 * s2 * np.cos(zhat[0] +  zhat[3] )])  

        Gdot = np.array( [g* (m1 * s1 * -np.sin(zhat[0])*zhat[1]+ m2* (s2 * -np.sin(zhat[0] + zhat[3])*(zhat[1]+zhat[4]) - l1 * np.sin(zhat[0] )*zhat[1])),g * m2 * s2 * -np.sin(zhat[0] +  zhat[3] )*(zhat[1]+zhat[4])]) 
        
        Omega_motor,Omega_measure,motornoise,sensorynoise = NoiseAndCovMatrix(M,Num_Var,kdelay,Var = 1e-3*4)

        # Compute the command through the FL technique
        
        v = -L[k].reshape(np.flip(B_basic.shape))@zhat[:Num_Var]
        u = 1/Kfactor*M@(v)+1/Kfactor*Mdot@(np.array([zhat[2],zhat[5]]))+M@(np.array([zhat[2],zhat[5]]))+C+G+Bdyn@np.array([zhat[1],zhat[4]])+1/Kfactor*Cdot+1/Kfactor*Bdyn@np.array([zhat[2],zhat[5]])+1/Kfactor*Gdot 
        array_u[k] = u

        # Delayed Observation of the Nonlinear system expressed in linear coordinates
        
        y[k] = (H@z).flatten()
        if Activate_Noise : 

            y[k]+=sensorynoise
        
        # Kalman Filter Gains 

        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_motor + (A - K@H)@sigma@A.T

        # Compute the Estimation of the system in the linear system
        
        zhat = A@zhat + B@v + K@(y[k]-H@zhat)
        
        # Simulate the nonlinear plant 
        
        C = np.array([-x[3]*(2*x[2]+x[3])*a2*np.sin(x[1]),x[2]*x[2]*a2*np.sin(x[1])])
        
        M = np.array([[a1+2*a2*cos(x[1]),a3+a2*cos(x[1])],[a3+a2*cos(x[1]),a3]])

        
        Cdot = np.array([-a2*angular_acceleration[1]*(2*x[2]+x[3])*sin(x[1])-a2*x[3]*(2*angular_acceleration[0]+angular_acceleration[1])*sin(x[1])
                         -a2*x[3]*x[3]*(2*x[2]+x[3])*cos(x[1]),2*x[2]*angular_acceleration[0]*a2*sin(x[1])+x[2]*x[2]*a2*cos(x[1])*x[3]])

        G = np.array([g*(m1 * s1 * np.cos(x[0])+ m2* (s2 * np.cos(x[0] + x[1]) + l1 * np.cos(x[0] ))),
            g * m2 * s2 * np.cos(x[0] + x[1])])  

        new_x[0:2] += dt*x[2:4]
        new_x[2:4] += dt*(np.linalg.solve(M,(x[4:6]-Bdyn@(x[2:4])-C-G)))
        new_x[4:6] += dt*Kfactor*(u-x[4:6])

        if Activate_Noise : 

            new_x[4:6]+=motornoise
        
        z = np.concatenate((np.array([new_x[0],new_x[2],angular_acceleration[0],new_x[1],new_x[3],angular_acceleration[1],z[6],z[7],z[8],z[9]]),z[:-Num_Var]))
        

        C = np.array([-z[4]*(2*z[1]+z[4])*a2*np.sin(z[3]),z[1]*z[1]*a2*np.sin(z[3])])

        M = np.array([[a1+2*a2*cos(z[3]),a3+a2*cos(z[3])],[a3+a2*cos(z[3]),a3]])

        angular_acceleration = np.linalg.solve(M,(new_x[4:6]-Bdyn@(new_x[2:4])-C-G))

        #Stock the true and estimated states

        array_z[k+1] = z[:Num_Var].flatten()
        array_zhat[k+1] = zhat[:Num_Var].flatten()
        array_x[k+1] = new_x[:Num_Var-4].flatten()
        
        x = np.copy(new_x)

    X,Y = ToCartesian(array_z,at3 = True)

    return X,Y,array_x

if __name__ == "__main__":


    ACTIVATE_PATH_CONSTRAINT = True
    HEIGHT = -33   
    MOVEMENT_DURATION = .3 #in seconds
    NUM_ITER = 600

    dt = MOVEMENT_DURATION/NUM_ITER
    time = np.linspace(0,MOVEMENT_DURATION*1000,NUM_ITER)
    WP = 1e4 if ACTIVATE_PATH_CONSTRAINT else 0 

    LED = np.array([20,30,40,50])
    starting_positions = np.array([[LED[0],HEIGHT],[LED[1],HEIGHT],[LED[2],HEIGHT]])
    ending_positions = np.array([[LED[1],HEIGHT],[LED[2],HEIGHT],[LED[3],HEIGHT]])

    

    fig = plt.figure(figsize = (8,14))
    gs = gridspec.GridSpec(4, 3, hspace=0.5)
    colors = plt.cm.viridis(np.linspace(0, 1, 6))[:3]
    colors2 = plt.cm.viridis(np.linspace(0, 1, 6))[3:]
    dotcolor= np.array(["blue","red","#f932f0"])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])

    for u in range(3):

        start = starting_positions[u]
        end = ending_positions[u]

        ax1 = fig.add_subplot(gs[0, u])
        ax2 = fig.add_subplot(gs[1, u])

        ax1.scatter(end[0],end[1],marker = "s",color="grey",s = 300)
        delete_axis(ax1)
        ax1.set_aspect("equal")
        ax1.set_ylim(HEIGHT - 5,HEIGHT + 5)

        X,Y,states = Feedback_Linearization_witPathConstraintG(Duration=.3,wp = WP,Num_iter=NUM_ITER,starting_point=start,targets=end,path_constraint_length = .8,taupath = .04,delay = 0)
        X2,Y2,states2 = Feedback_Linearization_witPathConstraintG(Duration=.3,wp = WP,Num_iter=NUM_ITER,starting_point=end,targets=start,path_constraint_length = .8,taupath = .04,delay = 0)
        
        ax1.plot(X,Y,color = colors[u], label = "Forward")
        ax1.plot(X2,Y2,color = colors2[u], label = "Backward")
        ax1.legend()
        
        if u == 0: ax2.set_ylabel("Velocity [cm/sec]")
        ax2.set_xlabel("Time [sec]")

        ax2.plot(time,compute_absolute_velocity(X,Y,dt),color = colors[u], label = "Forward",linewidth = 2)
        ax2.plot(time,compute_absolute_velocity(X2,Y2,dt),color = colors2[u], label = "Backward",linewidth = 2)

        ax3.set_ylabel("Delta rtpv")
        ax3.set_xlabel("3 Movements")
        ax3.plot(np.linspace(0,2,100),np.zeros(100),color = "grey",linestyle = "--")
        ax3.set_xticks([])

        ax3.scatter(u,(np.argmax(compute_absolute_velocity(X2,Y2,dt))-np.argmax(compute_absolute_velocity(X,Y,dt)))/NUM_ITER,color = dotcolor[u])    
        
        ax4.plot(states[:,0]*180/pi,states[:,1]*180/pi,color = colors[u],linewidth = 3)
        ax4.plot(states2[:,0]*180/pi,states2[:,1]*180/pi,color = colors2[u],linewidth = 3)

        ax4.set_ylabel(r"$\theta_e$")
        ax4.set_xlabel(r"$\theta_s$")
    plt.show()
