from Helpers import *


def LQG(Duration = .6,w1 = 1e8,w2 = 1e8,w3 = 1e4,w4 = 1e4,r1 = 1e-5,r2 = 1e-5,targets = [0,55],starting_point = [0,20],FF = False,Side = "Right",plot = True,Delay = 0,plotEstimation = False,alpha = .1,newtonfunc = newtonf,newtondfunc = newtondf,Num_iter = 300,Activate_Noise = False):


    
    dt = Duration/Num_iter
    kdelay = int(Delay/dt)

    obj1,obj2 = newton(newtonfunc,newtondfunc,1e-8,1000,targets[0],targets[1]) #Defini les targets
    st1,st2 = newton(newtonfunc,newtondfunc,1e-8,1000,starting_point[0],starting_point[1])

    xstart = np.array([st1,0,0,st2,0,0,obj1,0,obj2,0])
    x0 = np.array([st1,0,0,st2,0,0,obj1,obj2])
    x0_with_delay = np.copy(x0)
    for _ in range(kdelay):
        x0_with_delay = np.concatenate((x0_with_delay,x0))
    Num_Var = 8
    
    #Define Weight Matrices

    R = np.array([[r1,0],[0,r2]])
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w2,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
             [0,0,0,w3,0,0,0,-w3],[0,0,0,0,w4,0,0,0],[0,0,0,0,0,0,0,0],
             [-w1,0,0,0,0,0,w1,0],[0,0,0,-w3,0,0,0,w3]])
    
    
    #Define Dynamic Matrices  

    A_basic = np.array([[1,dt,0,0,0,0,0,0],[0,1+dt*(-0.5*a1+0.025*a3)/((a1-a3)*a3),dt*a1/((a1-a3)*a3),0,dt*(-0.025*a1+0.5*a3)/((a1-a3)*a3),dt/(a3-a1),0,0],
     [0,0,1-dt/tau,0,0,0,0,0],[0,0,0,1,dt,0,0,0],[0,dt*0.475/(a1-a3),-dt/(a1-a3),0,1-dt*0.475/(a1-a3),dt/(a1-a3),0,0],
     [0,0,0,0,0,1-dt/tau,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])

    B_basic = np.transpose([[0,0,dt/tau,0,0,0,0,0],[0,0,0,0,0,dt/tau,0,0]])

    NewQ = np.zeros(((kdelay+1)*Num_Var,(kdelay+1)*Num_Var))
    NewQ[:Num_Var,:Num_Var] = Q 
    Q = NewQ

    H = np.zeros((Num_Var,(kdelay+1)*Num_Var))
    H[:,(kdelay)*Num_Var:]= np.identity(Num_Var)

    A = np.zeros(((kdelay+1)*Num_Var,(kdelay+1)*Num_Var))
    A[:Num_Var,:Num_Var] = A_basic
    A[Num_Var:,:-Num_Var] = np.identity((kdelay)*Num_Var)
    B = np.zeros(((kdelay+1)*Num_Var,2))
    B[:Num_Var] = B_basic
    
    S = Q

    
    array_L = np.zeros((Num_iter-1,2,Num_Var*(kdelay+1)))   
    array_S = np.zeros((Num_iter,Num_Var*(kdelay+1),Num_Var*(kdelay+1))) 
    array_S[-1] = Q
    for k in range(Num_iter-1):
        L = np.linalg.inv(R+B.T@S@B)@B.T@S@A
        array_L[Num_iter-2-k] = L
        S = A.T@S@(A-B@L)
        array_S[Num_iter-2-k] = S
        
    #print(array_L[0])
    #Feedback
    L=array_L
        
    array_x = np.zeros((Num_iter,Num_Var))
    array_xhat = np.zeros((Num_iter,Num_Var))
    y = np.zeros((Num_iter-1,Num_Var))

    array_x[0] = x0.flatten()
    array_xhat[0] = x0.flatten()
    xhat = np.copy(x0_with_delay)
    x = np.copy(x0_with_delay)

    sigma = np.identity(Num_Var*(kdelay+1))*10**-6 
    J = 0
    F = [0,0]
    for k in range(Num_iter-1):
        
        #C = np.array([-x[4]*(2*x[1]+x[4])*a2*np.sin(x[3]),x[1]*x[1]*a2*np.sin(x[3])])
        
        #M = np.array([[a1+2*a2*cos(x[3]),a3+a2*cos(x[3])],[a3+a2*cos(x[3]),a3]])
        #acc = np.linalg.solve(M,(np.array([x[2],x[5]])-Bdyn@np.array([x[1],x[4]])-C))+F
        acc = (np.array([array_x[k][2],array_x[k][5]])-np.array([array_x[k-1][2],array_x[k-1][5]]))/dt
        if (np.sin(x[0]+x[3])*33+np.sin(x[0])*30 > 35) and (FF == True):

            F = Compute_f_new_version(np.array([x[0],x[3]]),np.array([x[1],x[4]]),acc,1)
            if Side == "Left": F*=-1

        else : 
            F = [0,0]
        Omega_sens,motor_noise,Omega_measure,measure_noise = Compute_Noise(Num_Var,alpha,B,kdelay)
        y[k] = (H@x).flatten()
        if Activate_Noise == True : y[k]+=measure_noise
        K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
        sigma = Omega_sens + (A - K@H)@sigma@A.T
        u = - L[k].reshape(np.flip(B.shape))@xhat
        J+= u.T@R@u
        xhat = A@xhat + B@u + K@(y[k]-H@xhat)
        x = A@x+B@u+np.concatenate(([0,dt*F[0],0,0,dt*F[1],0,0,0],np.zeros(Num_Var*kdelay))).flatten()
        if Activate_Noise : x[:6]+=motor_noise[:6]
        array_xhat[k+1] = xhat[:Num_Var].flatten()
        array_x[k+1] = x[:Num_Var].flatten()

        #print(array_x[k-1,2],((array_x[k]-array_x[k-1])/dt)[1])   

#Plot
    J+= x.T@Q@x
    x0 = xstart
    
    x_nonlin = array_x.T[:,1:][:,::1]
    X = np.cos(x_nonlin[0]+x_nonlin[3])*33+np.cos(x_nonlin[0])*30
    Y = np.sin(x_nonlin[0]+x_nonlin[3])*33+np.sin(x_nonlin[0])*30

    if plot:
        plt.plot(X,Y,color = "green",label = "LQG",linewidth = .8)
        plt.axis("equal")
        plt.scatter([targets[0]],[targets[1]],color = "black")
        if plotEstimation :
            x_nonlin = array_xhat.T[:,1:][:,::1]
            X2 = np.cos(x_nonlin[0]+x_nonlin[3])*33+np.cos(x_nonlin[0])*30
            Y2 = np.sin(x_nonlin[0]+x_nonlin[3])*33+np.sin(x_nonlin[0])*30
            plt.plot(X2,Y2,color = "black",label = "Estimation",linewidth = .8,linestyle ="--",alpha = .5)
    return X,Y,J