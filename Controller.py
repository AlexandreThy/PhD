from Helpers import *

def Feedback_Linearization(Duration,w1,w2,w3,w4,r1,r2,targets = [0,55],starting_point = [0,30],plot = True,Noise_Variance = 1e-6,ForceField = [0,0],ForceFieldSpan = [0,0],newtonfunc = f,newtondfunc = df,Num_iter = 600):
    
    """
    Duration (float) : Duration of the movement

    w1 (float) : Weight associated to the penalty on shoulder angle 
    
    w2 (float) : Weight associated to the penalty on elbow angle 

    w3 (float) : Weight associated to the penalty on shoulder angular velocity

    w4 (float) : Weight associated to the penalty on elbow angular velocity

    r1 (float) : Weight associated to the motor cost on shoulder torques

    r2 (float) : Weight associated to the motor cost on elbow torques

    targets (array of float of size 2): X,Y Position of the end point of the movement
        must be a biomechanically feasible end point considering that the shoulder is at (0,0)

    starting_point (array of float of size 2): X,Y Position of the starting point of the movement
        must be a biomechanically feasible starting point considering that the shoulder is at (0,0)
    
    plot (boolean) : Allowing plot outputs of the hand trajectory for the experiment

    Noise_Variance (float) : Gaussian variance associated to the white noise in the model

    ForceField (array of float of size 2) : Shoulder and Elbow Perturbations Torques applied during the movement 

    ForceFieldSpan (array of float of size 2) : The time span in seconds of the lateral forcefield (to the right)
    """

    dt = Duration/Num_iter 

    obj1,obj2 = newton(newtonfunc,newtondfunc,1e-8,1000,targets[0],targets[1]) #Defini les targets
    st1,st2 = newton(newtonfunc,newtondfunc,1e-8,1000,starting_point[0],starting_point[1])

    x0 = np.array([st1,0,0,st2,0,0,obj1,obj2])

    Num_Var = 8
    
    #Define Weight Matrices of the cost function
    R = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
               [0,0,0,w2,0,0,0,-w2],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
               [0-w1,0,0,0,0,0,w1,0],[0,0,0,-w2,0,0,0,w2]])

    
    
    #Define the Dynamic of the linear system 

    Kfactor = 1/0.06
    A = np.array([[1,dt,0,0,0,0,0,0],[0,1,dt,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,dt,0,0,0],[0,0,0,0,1,dt,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    B = np.array([[0,0],[0,0],[dt,0],[0,0],[0,0],[0,dt],[0,0],[0,0]])
    
    
    #Compute the Feedback Gain of the Control law
    S = Q

    
    array_L = np.zeros((Num_iter-1,2,Num_Var))   
    array_S = np.zeros((Num_iter,Num_Var,Num_Var)) 
    array_S[-1] = Q
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+B.T@S@B)@B.T@S@A
        array_L[Num_iter-2-k] = L
        S = A.T@S@(A-B@L)
        array_S[Num_iter-2-k] = S
        
    #Initialize matrices 

    sigma = np.identity(Num_Var)*10**-6 
    H,L = np.identity(8),array_L

    #Initialize the arrays to stock the simulations 
    array_xhat = np.zeros((Num_iter,Num_Var))
    array_x_nonlin = np.zeros((Num_iter,Num_Var-2)) 
    y = np.zeros((Num_iter,Num_Var))

    #Initialize the state vectors of the linear system
    array_xhat[0] = x0.flatten()
    xhat = np.copy(x0)
    x_true = np.copy(x0)

    #Initialize the state vectors of the nonlinear system
    x_nonlin = np.zeros(Num_Var-2)
    x_nonlin[0],x_nonlin[1] = x0[0],x0[3]

    new_x_nonlin = np.copy(x_nonlin)
    x_nonlinhat = np.copy(x_nonlin)
    new_x_nonlinhat = np.copy(x_nonlin)

    #Perform the task
    J = 0
    for k in range(Num_iter-1):

        #Compute the matrices of the system in function of the current state 

        F = ForceField if ((k*dt > ForceFieldSpan[0]) and (k*dt < ForceFieldSpan[1])) else 0
        
        Omega_sens,motor_noise,Omega_measure,measure_noise = Compute_Noise(Num_Var,Noise_Variance)
            
        C = np.array([-x_nonlin[3]*(2*x_nonlin[2]+x_nonlin[3])*a2*np.sin(x_nonlin[1]),x_nonlin[2]*x_nonlin[2]*a2*np.sin(x_nonlin[1])])
        Denominator = a3*(a1-a3)-a2*a2*np.cos(x_nonlin[1])*np.cos(x_nonlin[1])
        Minv = np.array([[a3/Denominator,(-a2*np.cos(x_nonlin[1])-a3)/Denominator],[(-a2*np.cos(x_nonlin[1])-a3)/Denominator,(2*a2*np.cos(x_nonlin[1])+a1)/Denominator]])
        
        Denominator = a3*(a1-a3)-a2*a2*np.cos(xhat[3])*np.cos(xhat[3])
        M = np.array([[a1+2*a2*cos(xhat[3]),a3+a2*cos(xhat[3])],[a3+a2*cos(xhat[3]),a3]])
        Minvdot = np.array([[-a3*a2*a2*sin(2*xhat[3])*xhat[4]/(Denominator*Denominator),
                             (a2*sin(xhat[3])*xhat[4]*Denominator+(a2*cos(xhat[3])+a3)*a2*a2*sin(2*xhat[3])*xhat[4])/(Denominator*Denominator)],
                            [(a2*sin(xhat[3])*xhat[4]*Denominator+(a2*cos(xhat[3])+a3)*a2*a2*sin(2*xhat[3])*xhat[4])/(Denominator*Denominator),
                            (-2*a2*sin(xhat[3])*xhat[4]*Denominator+(2*a2*cos(xhat[3])+a1)*a2*a2*sin(2*xhat[3])*xhat[4])/(Denominator*Denominator)]])
            
        Cdot = np.array([-a2*xhat[5]*(2*xhat[1]+xhat[4])*sin(xhat[3])-a2*xhat[4]*(2*xhat[2]+xhat[5])*sin(xhat[3])
                         -a2*xhat[4]*xhat[4]*(2*xhat[1]+xhat[4])*cos(xhat[3]),2*xhat[1]*xhat[2]*a2*sin(xhat[3])+xhat[1]*xhat[1]*a2*cos(xhat[3])*xhat[4]])
        
        COLORS = "red"
        LABEL = "Feedback Linearization"


        # Compute the command 

        v = -L[k].reshape(np.flip(B.shape))@xhat
        u = 1/Kfactor*M@(v)-1/Kfactor*M@Minvdot@M@(np.array([xhat[2],xhat[5]]))+M@(np.array([xhat[2],xhat[5]]))+C+Bdyn@np.array([xhat[1],xhat[4]])+1/Kfactor*Cdot+1/Kfactor*Bdyn@np.array([xhat[2],xhat[5]])
        J+= u.T@R@u
        # True state of the system
        new_x_nonlin[0:2] += dt*x_nonlin[2:4]+motor_noise[:2]
        new_x_nonlin[2:4] += dt*(Minv@(x_nonlin[4:6]-Bdyn@(x_nonlin[2:4])-C))  
        new_x_nonlin[4:6] += dt*Kfactor*(u-x_nonlin[4:6]+F)

        #Estimation of the state of the system
        new_x_nonlinhat[0:2] += dt*x_nonlinhat[2:4]
        new_x_nonlinhat[2:4] += dt*(Minv@(x_nonlinhat[4:6]-Bdyn@(x_nonlinhat[2:4])-C))  
        new_x_nonlinhat[4:6] += dt*Kfactor*(u-x_nonlinhat[4:6])
            
        array_xhat[k+1] = xhat.flatten()
        array_x_nonlin[k+1] = new_x_nonlin.flatten()

        y[k] = (H@x_true+measure_noise).flatten()
        #Change of variable 
        

        #Perform Iteration 
        
        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_sens + (A - K@H)@sigma@A.T
        #print(y[k]-H@x_internalmodel)
        xhat = A@xhat+B@v + K@(y[k]-H@xhat)

        x_true = np.array([new_x_nonlin[0],new_x_nonlin[2],x_true[2]+dt*v[0],new_x_nonlin[1],new_x_nonlin[3],x_true[5]+dt*v[1],x_true[6],x_true[7]])

        x_nonlin = np.copy(new_x_nonlin)
        x_nonlinhat = np.copy(new_x_nonlinhat)
        #print(array_x_true[k-1,2],((array_x_true[k]-array_x_true[k-1])/dt)[1])   

    #Change of coordinates from angular to cartesian 
    J+= x_true.T@Q@x_true
    print("Total cost of FL: "+str(J)[:7])
    x_nonlin = array_x_nonlin.T[:,1:][:,::1]
    X = np.cos(x_nonlin[0]+x_nonlin[1])*33+np.cos(x_nonlin[0])*30
    Y = np.sin(x_nonlin[0]+x_nonlin[1])*33+np.sin(x_nonlin[0])*30

    #Plotting
    if plot : 
        plt.grid(linestyle='--')
        plt.axis("equal")
        plt.plot(X,Y,color = COLORS,label = LABEL,linewidth = .8)
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.scatter([starting_point[0],targets[0]],[starting_point[1],targets[1]],color = "black")
    #print("Optimum values " + str(J1)[:8]+" and "+str(J2)[:8])

    
    return X,Y

def LQG(Duration,w1,w2,w3,w4,r1,r2,targets = [0,55],starting_point = [0,20],ForceField = [0,0],plot = True,ForceFieldSpan = [0,0.6],Noise_Variance = 1e-6,newtonfunc = f,newtondfunc = df,Num_iter = 600):

    """
    Duration (float) : Duration of the movement

    w1 (float) : Weight associated to the penalty on shoulder angle 
    
    w2 (float) : Weight associated to the penalty on elbow angle 

    w3 (float) : Weight associated to the penalty on shoulder angular velocity

    w4 (float) : Weight associated to the penalty on elbow angular velocity

    r1 (float) : Weight associated to the motor cost on shoulder torques

    r2 (float) : Weight associated to the motor cost on elbow torques

    targets (array of float of size 2): X,Y Position of the end point of the movement
        must be a biomechanically feasible end point considering that the shoulder is at (0,0)

    starting_point (array of float of size 2): X,Y Position of the starting point of the movement
        must be a biomechanically feasible starting point considering that the shoulder is at (0,0)
    
    plot (boolean) : Allowing plot outputs of the hand trajectory for the experiment

    Noise_Variance (float) : Gaussian variance associated to the white noise in the model

    ForceField (array of float of size 2) : Shoulder and Elbow Perturbations Torques applied during the movement 

    ForceFieldSpan (array of float of size 2) : The time span in seconds of the lateral forcefield (to the right)
    """
        
    
    dt = Duration/Num_iter

    obj1,obj2 = newton(newtonfunc,newtondfunc,1e-8,1000,targets[0],targets[1]) #Defini les targets
    st1,st2 = newton(newtonfunc,newtondfunc,1e-8,1000,starting_point[0],starting_point[1])

    xstart = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    x0 = np.array([st1,0,0,st2,0,0,obj1,obj2])
    Num_Var = 8
    
    #Define Weight Matrices

    R = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w2,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
             [0,0,0,w3,0,0,0,-w3],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
             [-w1,0,0,0,0,0,w1,0],[0,0,0,-w3,0,0,0,w3]])
    
    
    #Define Dynamic Matrices  

    A = np.array([[1,dt,0,0,0,0,0,0],[0,1+dt*(-0.5*a1+0.025*a3)/((a1-a3)*a3),dt*a1/((a1-a3)*a3),0,dt*(-0.025*a1+0.5*a3)/((a1-a3)*a3),dt/(a3-a1),0,0],
     [0,0,1-dt/tau,0,0,0,0,0],[0,0,0,1,dt,0,0,0],[0,dt*0.475/(a1-a3),-dt/(a1-a3),0,1-dt*0.475/(a1-a3),dt/(a1-a3),0,0],
     [0,0,0,0,0,1-dt/tau,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

    B = np.transpose([[0,0,dt/tau,0,0,0,0,0],[0,0,0,0,0,dt/tau,0,0]])
    
    S = Q

    
    array_L = np.zeros((Num_iter-1,2,Num_Var))   
    array_S = np.zeros((Num_iter,Num_Var,Num_Var)) 
    array_S[-1] = Q
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+B.T@S@B)@B.T@S@A
        array_L[Num_iter-2-k] = L
        S = A.T@S@(A-B@L)
        array_S[Num_iter-2-k] = S
        
    #print(array_L[0])
    #Feedback
    H,L= np.identity(8),array_L
        
    array_x = np.zeros((Num_iter,Num_Var))
    array_xhat = np.zeros((Num_iter,Num_Var))
    array_x_nonlin = np.zeros((Num_iter,Num_Var-2)) 
    y = np.zeros((Num_iter,Num_Var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()
    xhat = x0
    x = x0

    x_nonlin = np.zeros(Num_Var-2)
    x_nonlin[0] = x[0]
    x_nonlin[1] = x[3]
    new_x_nonlin = x_nonlin

    sigma = np.identity(Num_Var)*10**-6 #Espérance de (erreur erreur^) avec erreur = x - xhat
    J = 0
    for k in range(Num_iter-1):
        F = ForceField if ((k*dt > ForceFieldSpan[0]) and (k*dt < ForceFieldSpan[1])) else 0
        

        x[0],x[1],x[3],x[4] = x_nonlin[0],x_nonlin[2],x_nonlin[1],x_nonlin[3]        
        Omega_sens,motor_noise,Omega_measure,measure_noise = Compute_Noise(Num_Var,Noise_Variance)
        y[k] = (H@x+measure_noise).flatten()
        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_sens + (A - K@H)@sigma@A.T
        xhat = A@xhat - B@L[k].reshape(np.flip(B.shape))@xhat + K@(y[k]-H@xhat)
        x = A@x-B@L[k].reshape(np.flip(B.shape))@xhat+motor_noise
        u = -L[k].reshape(np.flip(B.shape))@xhat
        J += u.T@R@u
        C = np.array([-x_nonlin[3]*(2*x_nonlin[2]+x_nonlin[3])*a2*np.sin(x_nonlin[1]),x_nonlin[2]*x_nonlin[2]*a2*np.sin(x_nonlin[1])])
        Denominator = a3*(a1-a3)-a2*a2*np.cos(x_nonlin[1])*np.cos(x_nonlin[1])
        Minv = np.array([[a3/Denominator,(-a2*np.cos(x_nonlin[1])-a3)/Denominator],[(-a2*np.cos(x_nonlin[1])-a3)/Denominator,(2*a2*np.cos(x_nonlin[1])+a1)/Denominator]])
        new_x_nonlin[0:2] += dt*x_nonlin[2:4]+motor_noise[:2]
        new_x_nonlin[2:4] += dt*(Minv@(x_nonlin[4:6]-Bdyn@(x_nonlin[2:4])-C))  
        K = 1/0.06
        new_x_nonlin[4:6] += dt*K*(u-x_nonlin[4:6]+F)
            
        array_xhat[k+1] = xhat.flatten()
        array_x[k+1]= x.flatten()
        array_x_nonlin[k+1] = new_x_nonlin.flatten()
        x_nonlin = new_x_nonlin 
        #print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])   

#Plot
    x0 = xstart
    x_nonlin = array_x_nonlin.T[:,1:][:,::1]
    J+= x.T@Q@x
    print("Total cost of LQG : "+str(J)[:7])

    X = np.cos(x_nonlin[0]+x_nonlin[1])*33+np.cos(x_nonlin[0])*30
    Y = np.sin(x_nonlin[0]+x_nonlin[1])*33+np.sin(x_nonlin[0])*30

    if plot:
        plt.plot(X,Y,color = "green",label = "LQG",linewidth = .8)

    return X,Y
def f(x,u):
    tau = 0.06
    C = np.array([-x[3]*(2*x[2]+x[3])*a2*np.sin(x[1]),x[2]*x[2]*a2*np.sin(x[1])])
    Denominator = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Minv = np.array([[a3/Denominator,(-a2*np.cos(x[1])-a3)/Denominator],[(-a2*np.cos(x[1])-a3)/Denominator,(2*a2*np.cos(x[1])+a1)/Denominator]])
    theta = Minv@(x[4:6]-Bdyn@x[2:4]-C)
    torque = (u-x[4:6])/tau
    return np.array([[x[2],x[3],theta[0],theta[1],torque[0],torque[1]]])

def complex_derivative_1(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Dprime = 2*a2*a2*np.cos(x[1])*sin(x[1])
    F1 = x[4]-Bdyn[0]@x[2:4]
    F2 = x[5]-Bdyn[1]@x[2:4]
    C1 = -x[3]*(2*x[2]+x[3])*a2*np.sin(x[1])
    C1prime = -x[3]*(2*x[2]+x[3])*a2*np.cos(x[1])
    C2 = x[2]*x[2]*a2*np.sin(x[1])
    C2prime = x[2]*x[2]*a2*np.cos(x[1])
    Sol = -a3*Dprime/(D*D)*F1-a3/(D*D)*(C1prime*D-C1*Dprime)-((F2-C2)/(D*D)*((-a2*sin(x[1]))*D-(a2*cos(x[1])+a3)*Dprime)-C2prime*(a2*cos(x[1])+a3)/D)
    return Sol

def complex_derivative_2(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    C1prime = -x[3]*2*a2*np.sin(x[1])
    C2prime = x[2]*2*a2*np.sin(x[1])
    Sol = -a3/D*(Bdyn[0,0]+C1prime)+(a2*cos(x[1])+a3)/D*(Bdyn[1,0]+C2prime)
    return Sol

def complex_derivative_3(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    C1prime =(-2*x[2]-2*x[3])*a2*np.sin(x[1])
    Sol = -a3/D*(Bdyn[0,1]+C1prime)+(a2*cos(x[1])+a3)/D*(Bdyn[1,1])
    return Sol

def complex_derivative_4(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Dprime = 2*a2*a2*np.cos(x[1])*sin(x[1])
    F1 = x[4]-Bdyn[0]@x[2:4]
    F2 = x[5]-Bdyn[1]@x[2:4]
    C1 = -x[3]*(2*x[2]+x[3])*a2*np.sin(x[1])
    C1prime = -x[3]*(2*x[2]+x[3])*a2*np.cos(x[1])
    C2 = x[2]*x[2]*a2*np.sin(x[1])
    C2prime = x[2]*x[2]*a2*np.cos(x[1])
    Sol = ((a2*sin(x[1]))*D+(a2*cos(x[1])+a3)*Dprime)/(D*D)*(F1-C1)+(-a2*cos(x[1])-a3)/D*(-C1prime)+(-2*a2*sin(x[1])*D-(2*a2*cos(x[1])+a1)*Dprime)/(D*D)*(F2-C2)+(2*a2*cos(x[1])+a2)/D*(-C2prime)
    return Sol

def complex_derivative_5(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    C1prime = -x[3]*2*a2*np.sin(x[1])
    C2prime = 2*x[2]*a2*np.sin(x[1])
    Sol = (a2*cos(x[1])+a3)/D*(Bdyn[0,0]+C1prime)+(2*a2*cos(x[1])+a1)/D*(-Bdyn[1,0]-C2prime)
    return Sol

def complex_derivative_6(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    C1prime = (-2*x[2]-2*x[3])*a2*np.sin(x[1])
    Sol = (a2*cos(x[1])+a3)/D*(Bdyn[0,1]+C1prime)+(2*a2*cos(x[1])+a1)/D*(-Bdyn[1,1])
    return Sol

def easy_derivative_1(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Sol = a3/D
    return Sol

def easy_derivative_2(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Sol = -(a2*cos(x[1])+a3)/D
    return Sol

def easy_derivative_3(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Sol = (2*a2*cos(x[1])+a1)/D
    return Sol



def fx(x,u):
    tau = 0.06
    return np.array([[0,0,1,0,0,0],
                     [0,0,0,1,0,0],
                     [0,complex_derivative_1(x),complex_derivative_2(x),complex_derivative_3(x),easy_derivative_1(x),easy_derivative_2(x)],
                     [0,complex_derivative_4(x),complex_derivative_5(x),complex_derivative_6(x),easy_derivative_2(x),easy_derivative_3(x)],
                     [0,0,0,0,-1/tau,0],
                     [0,0,0,0,0,-1/tau]])

def fu(x,u):
    tau = 0.06
    return np.array([[0,0],
                     [0,0],
                     [0,0],
                     [0,0],
                     [1/tau,0],
                     [0,1/tau]])

def l(x,u,r1):
    return r1*(u[0]*u[0]+u[1]*u[1])

def lx(x,u):
    return np.zeros(6)

def lu(x,u,r1):
    return np.array([2*u[0]*r1,2*u[1]*r1])

def lxx(x,u):
    return np.zeros((6,6))

def luu(x,u,r1):
    return np.array([[2*r1,0],[0,2*r1]])

def h(x,w1,w2,xtarg):
    return w1/2*((x[0]-xtarg[0])*(x[0]-xtarg[0])+(x[1]-xtarg[1])*(x[1]-xtarg[1])) + w2/2*(x[2]*x[2]+x[3]*x[3])

def hx(x,w1,w2,xtarg):
    return np.array([w1*(x[0]-xtarg[0]),w1*(x[1]-xtarg[1]),w2*x[2],w2*x[3],0,0])

def hxx(x,w1,w2):
    return np.array([[w1,0,0,0,0,0],
                     [0,w1,0,0,0,0],
                     [0,0,w2,0,0,0],
                     [0,0,0,w2,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])

def step1(x0,u,Duration):
    K = np.shape(u)[0]+1
    dt = Duration/(K-1)
    newx = np.zeros((K,len(x0)))
    newx[0] = x0
    for i in range(K-1):
        newx[i+1] = newx[i] + dt*f(newx[i],u[i])
    return newx

def step2(x,u,Duration,w1,w2,r1,xtarg):
    n = len(x[0])
    m = len(u[0])
    K = np.shape(u)[0]+1
    dt = Duration/(K-1)
    A = np.zeros((K-1,n,n))
    B = np.zeros((K-1,n,m))
    q = np.zeros(K)
    qbold = np.zeros((K,n))
    r = np.zeros((K-1,m))
    Q = np.zeros((K,n,n))
    R = np.zeros((K-1,m,m))
    for i in range(K-1):
        A[i] = np.identity(n)+dt*fx(x[i],u[i])
        B[i] = dt*fu(x[i],u[i])
        q[i] = dt*l(x[i],u[i],r1)
        qbold[i] = dt*lx(x[i],u[i])
        r[i] = dt*lu(x[i],u[i],r1)
        Q[i] = dt*lxx(x[i],u[i])
        R[i] = dt*luu(x[i],u[i],r1)

    q[K-1] = h(x[K-1],w1,w2,xtarg)
    qbold[K-1] = hx(x[K-1],w1,w2,xtarg)
    Q[K-1] = hxx(x[K-1],w1,w2)
    return A,B,q,qbold,r,Q,R

def step3(A,B,C,cbold,q,qbold,r,Q,R):
    # C should be nxm 
    # c should be nx1

    K = A.shape[0]+1
    n,m = np.shape(B[0])
    S = np.zeros((K,n,n))
    s = np.zeros(K)
    sbold = np.zeros((K,n))
    l = np.zeros((K-1,m))
    L = np.zeros((K-1,m,n))

    S[-1] = Q[-1]
    s[-1] = q[-1]
    sbold[-1] = qbold[-1]

    for k in np.arange(K-2,-1,-1):
        gbold = r[k] + B[k].T@sbold[k+1]+np.sum(C[k,:].T@S[k+1]@cbold[k,:])
        G = B[k].T@S[k+1]@A[k]
        H = R[k] + B[k].T@S[k+1]@B[k]+np.sum(C[k,:].T@S[k+1]@C[k,:])
        Hinv = np.linalg.inv(H)


        S[k] = Q[k] + A[k].T@S[k+1]@A[k]-G.T@Hinv@G
        sbold[k] = qbold[k]+A[k].T@sbold[k+1]-G.T@Hinv@gbold
        s[k] = q[k] + s[k+1] + 0.5*np.sum(cbold[k,:].T@S[k+1]@cbold[k,:]) - .5*gbold.T@Hinv@gbold

        l[k] = -Hinv@gbold
        L[k] = -Hinv@G
    
    return l,L

def step4(l,L,K,A,B):
    m,n = L[0].shape
    x = np.zeros(n)
    u_incr = np.zeros((K-1,m))
    for k in range(K-1):
        u_incr[k] = l[k]+L[k]@x
        x = A[k]@x+B[k]@u_incr[k]
    return u_incr

def fnewton(var,X,Y):
    u,v = var
    return np.array([33*np.cos(u+v)+30*np.cos(u)-X,33*np.sin(u+v)+30*np.sin(u)-Y])

def dfnewton(var):
    u,v = var
    return np.array([[-33*np.sin(u+v)-30*np.sin(u),-33*np.sin(u+v)],[33*np.cos(u+v)+30*np.cos(u),33*np.cos(u+v)]])

def ILQG_SingleArm(Duration,w1,w2,r1,xtarg,K,x0=np.array([pi/2,0]),m=1):
    n = len(x0)
    u = np.zeros((K-1,m))
    C = np.zeros((K,n,m))
    cbold = np.zeros((K,n))
    u_incr = [1]

    #while np.max(u_incr) > 1e-12: 
    for _ in range(50):    
        x = step1(x0,u,Duration)
        A,B,q,qbold,r,Q,R = step2(x,u,Duration,w1,w2,r1,xtarg)
        l,L = step3(A,B,C,cbold,q,qbold,r,Q,R)
        u_incr = step4(l,L,K,A,B)
        u += u_incr
    x = step1(x0,u,Duration)
    return x,u


def ILQG(Duration,w1,w2,r1,targets,K,start,plot = True):
    obj1,obj2 = newton(fnewton,dfnewton,1e-8,1000,targets[0],targets[1]) #Defini les targets
    st1,st2 = newton(fnewton,dfnewton,1e-8,1000,start[0],start[1])

    x0 = np.array([st1,st2,0,0,0,0])

    m = 2
    n = 6
    u = np.zeros((K-1,m))
    C = np.zeros((K,n,m))
    cbold = np.zeros((K,n))
    u_incr = [1]

    #while np.max(u_incr) > 1e-12: 
    for _ in range(50):    
        x = step1(x0,u,Duration)
        A,B,q,qbold,r,Q,R = step2(x,u,Duration,w1,w2,r1,np.array([obj1,obj2]))
        l,L = step3(A,B,C,cbold,q,qbold,r,Q,R)
        u_incr = step4(l,L,K,A,B)
        u += u_incr
    x = step1(x0,u,Duration)

    X = np.cos(x[:,0]+x[:,1])*33+np.cos(x[:,0])*30
    Y = np.sin(x[:,0]+x[:,1])*33+np.sin(x[:,0])*30


    Xtg = np.cos(x[0]+x[1])*33+np.cos(x[0])*30
    Ytg = np.sin(x[0]+x[1])*33+np.sin(x[0])*30

        #Plotting
    if plot :
        plt.grid(linestyle='--')
        plt.axis("equal")
        plt.plot(X,Y,linewidth = .8,color = "blue",label = "ILQG")
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.scatter([Xtg],[Ytg],color = "black")
    return X,Y,u

def SingleArmLQG(Duration,w1,w2,r1,targ,K,Noise_Variance = 1e-12):

    Num_iter = K
    dt = Duration/Num_iter
    x0 = np.array([pi/2,0,targ])
    
    Num_Var = 3
    
    #Define Weight Matrices

    R = np.array([r1])
    Q = np.array([[w1,0,-w1],[0,w2,0],[-w1,0,w1]])
    
    
    #Define Dynamic Matrices  
    g = 9.81
    L = .5
    A = np.identity(3)+dt*np.array([[0,1,0],[-g/L,0,0],[0,0,0]])

    B = np.transpose([[0],[dt],[0]]).reshape((Num_Var,1))
    
    S = Q

    
    array_L = np.zeros((Num_iter-1,1,Num_Var))   
    array_S = np.zeros((Num_iter,Num_Var,Num_Var)) 
    array_S[-1] = Q
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+B.T@S@B)@B.T@S@A
        array_L[Num_iter-2-k] = L
        S = A.T@S@(A-B@L)
        array_S[Num_iter-2-k] = S
        
    #print(array_L[0])
    #Feedback
    L= array_L
        
    array_u = np.zeros((Num_iter-1,1))
    array_x_nonlin = np.zeros((Num_iter,Num_Var)) 
    array_x_nonlin[0] = x0
    x = np.copy(x0)

    x_nonlin = np.copy(x0)
    new_x_nonlin = np.copy(x0)

    g = 9.81
    ArmLength = .5
    for k in range(Num_iter-1):
        
        
        x = np.copy(x_nonlin)     

        u = -L[k].reshape(np.flip(B.shape))@x
        
        array_u[k] = u[0]
        new_x_nonlin[0] += dt*x_nonlin[1]
        new_x_nonlin[1] += dt*u[0]-dt*g/ArmLength*sin(x_nonlin[0])
        new_x_nonlin[2] = x_nonlin[2]
        #new_x_nonlin = A@x_nonlin + B@u
            
        array_x_nonlin[k+1] = new_x_nonlin.flatten()
        x_nonlin = new_x_nonlin   

#Plot

    return array_x_nonlin,array_u