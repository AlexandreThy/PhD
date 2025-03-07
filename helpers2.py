import numpy as np
from matplotlib import pyplot as plt
from math import *
from scipy.linalg import expm

from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

I1 = 0.025
I2 = 0.045
m2 = 1
l1 = 0.3
l2 = 0.33
s2 = 0.16
K = 1/0.06
tau = 0.06

#SHOULDER PUIS ELBOW

a1 = I1 + I2 + m2*l1*l1
a2 = m2*l1*s2
a3 = I2

Bdyn2 = np.array([[0.5,0.025],[0.025,0.5]])

def Compute_Multiplicative_Noise(NbreVar,alpha,B,mult_var):
    #B = B[:NbreVar]
    Omega_sens = alpha*B@B.T
    m = B.shape[1]
    n = B.shape[0]
    #Ok que si omegasens est diag
    motor_noise = np.zeros(n)
    totF = np.zeros((m,m))
    
    F =  np.zeros((m,m,m))
    C = np.zeros((m,n,m))
    eps = np.random.normal(0,1,m)
    for i in range(n):
        motor_noise[i] = np.random.normal(0,np.sqrt(Omega_sens[i,i]))
    for i in range(m):
        F[i,i,i] = mult_var
        totF += F[i]*eps[i]
        C[i] = B@F[i]
        
    mult_noise = B@(totF)
    Omega_measure = np.diag(np.ones(NbreVar)*1e-6)
    measure_noise = np.random.normal(0,np.sqrt(1e-6),NbreVar).T
    

    return Omega_sens,motor_noise.T,Omega_measure,measure_noise,C,mult_noise
def f1(a,Nf):
    Td = 0.066
    return np.exp(-(a/(0.56*Nf))**Nf)*Nf/Td*(1/(0.56*Nf))**Nf*a**(Nf-1)

def f2(a,Nf,u):
    Td = 0.066 + u*(0.05-0.066)
    return np.exp(-(a/(0.56*Nf))**Nf)*Nf/Td*(1/(0.56*Nf))**Nf*a**(Nf-1)

def g(a,Nf):
    return -f1(a,Nf)*a

def g2(a,Nf,u):
    return -f2(a,Nf,u)*a

def newton(f,Df,epsilon,max_iter,X,Y,x0 = np.array([0.8,1.5])):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn,X,Y)
        if abs(np.max(np.abs(fxn))) < epsilon:
            return xn
        Dfxn = Df(xn)
        if np.max(np.abs(Dfxn)) < epsilon:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - np.linalg.inv(Dfxn)@fxn
    print('Exceeded maximum iterations. No solution found.')
    return None

def newtonf(var,X,Y):
    u,v = var
    return np.array([33*np.cos(u+v)+30*np.cos(u)-X,33*np.sin(u+v)+30*np.sin(u)-Y])

def newtondf(var):
    u,v = var
    return np.array([[-33*np.sin(u+v)-30*np.sin(u),-33*np.sin(u+v)],[33*np.cos(u+v)+30*np.cos(u),33*np.cos(u+v)]])

def Compute_gamma_nu(theta,omega):
    fe = -33*np.sin(theta[0]+theta[1])
    fs = -33*np.sin(theta[0]+theta[1]) - 30*np.sin(theta[0])
    ge = 33*np.cos(theta[0]+theta[1])
    gs = 33*np.cos(theta[0]+theta[1]) + 30*np.cos(theta[0])
    fse = -33*np.cos(theta[0]+theta[1])
    gse = -33*np.sin(theta[0]+theta[1])
    fee = -33*np.cos(theta[0]+theta[1])
    fss = -33*np.cos(theta[0]+theta[1]) - 30*np.cos(theta[0])
    gee = fe
    gss = fs
    gamma = gs*omega[0] + ge*omega[1] - fss*omega[0]*omega[0] - 2*fse*omega[0]*omega[1] - fee * omega[1] * omega[1]
    nu = - gss*omega[0]*omega[0] - 2*gse*omega[0]*omega[1] - gee * omega[1] * omega[1]
    return gamma,nu,fs,fe,gs,ge,fss,fse,fee,gss,gse,gee


def pre_Compute(theta,omega):
    fe = -33*np.sin(theta[0]+theta[1])
    fs = -33*np.sin(theta[0]+theta[1]) - 30*np.sin(theta[0])
    ge = 33*np.cos(theta[0]+theta[1])
    gs = 33*np.cos(theta[0]+theta[1]) + 30*np.cos(theta[0])
    fse = -33*np.cos(theta[0]+theta[1])
    gse = -33*np.sin(theta[0]+theta[1])
    fee = -33*np.cos(theta[0]+theta[1])
    fss = -33*np.cos(theta[0]+theta[1]) - 30*np.cos(theta[0])
    gee = fe
    gss = fs
    return fs,fe,gs,ge,fss,fse,fee,gss,gse,gee

def Compute_f_new_version(theta,omega,acc,factor):
    fs,fe,gs,ge,fss,fse,fee,gss,gse,gee = pre_Compute(theta,omega)
    xddot = 13*(gs*omega[0] + ge*omega[1])*factor + fss*omega[0]*omega[0] + 2*fse*omega[0]*omega[1] + fee * omega[1] * omega[1] + fs*acc[0] + fe*acc[1]
    yddot = gss*omega[0]*omega[0] + 2*gse*omega[0]*omega[1] + gee * omega[1] * omega[1] + gs*acc[0] + ge*acc[1]
    gamma = xddot - fss*omega[0]*omega[0] - 2*fse*omega[0]*omega[1] - fee * omega[1] * omega[1]
    nu = yddot - gss*omega[0]*omega[0] - 2*gse*omega[0]*omega[1] - gee * omega[1] * omega[1]
    F1 = (fe*nu-ge*gamma)/(fe*gs-ge*fs) - acc[0]
    F2 = (gs*gamma-fs*nu)/(gs*fe-ge*fs) - acc[1]
    return np.array([F1,F2])

def Compute_constant_force(theta,omega,acc,factor):
    fs,fe,gs,ge,fss,fse,fee,gss,gse,gee = pre_Compute(theta,omega)
    xddot = factor + fss*omega[0]*omega[0] + 2*fse*omega[0]*omega[1] + fee * omega[1] * omega[1] + fs*acc[0] + fe*acc[1]
    yddot = gss*omega[0]*omega[0] + 2*gse*omega[0]*omega[1] + gee * omega[1] * omega[1] + gs*acc[0] + ge*acc[1]
    gamma = xddot - fss*omega[0]*omega[0] - 2*fse*omega[0]*omega[1] - fee * omega[1] * omega[1]
    nu = yddot - gss*omega[0]*omega[0] - 2*gse*omega[0]*omega[1] - gee * omega[1] * omega[1]
    F1 = (fe*nu-ge*gamma)/(fe*gs-ge*fs) - acc[0]
    F2 = (gs*gamma-fs*nu)/(gs*fe-ge*fs) - acc[1]
    return np.array([F1,F2])

def MultipleLabel(title = "Controllers",side = "right"):
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize = 16,title = title,title_fontsize = 15,frameon = True,shadow = True,fancybox = True,loc = "upper "+side)

def Cov_Matrix(M,N,Var = 1e-6):
    K = 1/0.06
    M = np.linalg.inv(M)
    Sigmau = np.array([[Var,0],[0,Var]])
    Sigmav = K*K*M@Sigmau@M.T
    Sigma = np.zeros((N,N))
    Sigmam = np.diag(np.ones(N)*Var)
    for S in [Sigma,Sigmam]:
        S[2,2] = Sigmav[0,0]
        S[2,5] = Sigmav[0,1]
        S[5,2] = Sigmav[1,0]
        S[5,5] = Sigmav[1,1]
    return Sigma,Sigmam

def NoiseAndCovMatrix(M=np.identity(2),N=6,kdelay=0,Var = 1e-6,Linear = False):

    K = 1/0.06
    M = np.linalg.inv(M)
    Sigmau = np.array([[Var,0],[0,Var]])
    Sigmav = K*K*M@Sigmau@M.T
    SigmaMotor = np.zeros((N*(kdelay+1),N*(kdelay+1)))
    Sigma = np.zeros((N,N))
    SigmaSense = np.diag(np.ones(N)*Var)
    for S in [Sigma,SigmaSense]:
        S[2,2] = Sigmav[0,0]
        S[2,5] = Sigmav[0,1]
        S[5,2] = Sigmav[1,0]
        S[5,5] = Sigmav[1,1]
    SigmaMotor[:N,:N] = Sigma

    motornoise,sensorynoise = np.zeros(2),np.zeros(N)
    for i in range(N):
        sensorynoise[i] = np.random.normal(0,np.sqrt(SigmaSense[i,i]))
    motornoise = np.random.normal(0,np.sqrt(Var),2)
    if Linear : return np.diag(np.ones(N*(kdelay+1))*Var),np.diag(np.ones(N)*Var),motornoise,np.random.normal(0,np.sqrt(Var),N)
    return SigmaMotor,SigmaSense,motornoise,sensorynoise

def ToCartesian(x,at3 = False):
    elbowindex = 3 if at3 else 1
    if len(x.shape) == 1 : 
        s = x[0]
        e = x[elbowindex]
    else : 
        s = x[:,0]
        e = x[:,elbowindex]
    X = np.cos(s+e)*33+np.cos(s)*30
    Y = np.sin(s+e)*33+np.sin(s)*30

    return X,Y

def get_Gravity_Matrix(x,g = 9.81,alpha = 0,additional_mass = 0):
    """
    return the gravity matrices terms
    """
    Ms = 52
    Ls = 1.57
    m1 = Ms*0.028
    m2 = Ms*0.022
    l1 = Ls*0.186
    l2 = Ls*(0.146)
    lc1 = l1*0.436
    lc2 = l2*0.682
    G = np.array([g*(m1*lc1*np.cos(x[0]+alpha)+m2*(lc2*np.cos(x[0]+alpha+x[3])+l1*np.cos(x[0]+alpha))+additional_mass*(l2*np.cos(x[0]+alpha+x[3])+l1*np.cos(x[0]+alpha))),
                    g*(m2+additional_mass*l2/lc2)*lc2*np.cos(x[0]+x[3]+alpha)])
    Gdot = np.array([g*(-m1*lc1*np.sin(x[0]+alpha)*x[1]+m2*(-lc2*np.sin(x[0]+alpha+x[3])*(x[1]+x[4])-l1*np.sin(x[0]+alpha)*x[1])+additional_mass*(-l2*np.sin(x[0]+alpha+x[3])*(x[1]+x[4])-l1*np.sin(x[0]+alpha)*x[1])),
                    -g*(m2+l2/lc2*additional_mass)*lc2*np.sin(x[0]+alpha+x[3])*(x[1]+x[4])])
    return G,Gdot

def M1(te,ObjectMass=0,Ms = 52,Ls = 1.57):
    m1 = Ms*0.028
    m2 = Ms*0.022
    l1 = Ls*0.186
    l2 = Ls*(0.146)
    lc1 = l1*0.436
    lc2 = l2*0.682
    I1 = m1*(l1*0.322)**2
    I2 = m2*(l2*0.468)**2
    return np.array([[m1*lc1*lc1+I1+I2+m2*(lc2*lc2+l1*l1+2*l1*lc2*cos(te))+ObjectMass*(l1*l1+2*l1*l2*cos(te)+l2*l2),
                      I2+m2*(lc2*lc2+l1*lc2*cos(te))+ObjectMass*(l1*l2*cos(te)+l2*l2)],
                      [I2+m2*(lc2*lc2+l1*lc2*cos(te))+ObjectMass*(l1*l2*cos(te)+l2*l2),
                       m2*lc2*lc2+I2+ObjectMass*l2*l2]])

def dM1(te,oe,ObjectMass=0,Ms=52,Ls=1.57):
    m2 = Ms*0.022
    l1 = Ls*0.186
    l2 = Ls*(0.146)
    lc2 = l2*0.682
    return np.array([[m2*(-2*l1*lc2*sin(te)*oe)+ObjectMass*(-2*l1*l2*sin(te)*oe),
                      m2*(-l1*lc2*sin(te)*oe)+ObjectMass*(-l1*l2*sin(te)*oe)],
                      [m2*(-l1*lc2*sin(te)*oe)+ObjectMass*(-l1*l2*sin(te)*oe),
                       0]])


def C1(te,os,oe,ObjectMass = 0,Ms=52,Ls=1.57):
    
    m2 = Ms*0.022
    l1 = Ls*0.186
    l2 = Ls*(0.146)
    lc2 = l2*0.682
    c = m2*l1*lc2*sin(te)+ObjectMass*l1*l2*sin(te)
    C=np.array([[-2*oe*c,-oe*c],[os*c,0]])
    return C@np.array([os,oe])

def dC1(te,os,oe,accs,acce,ObjectMass=0,Ms=52,Ls=1.57):
    
    m2 = Ms*0.022
    l1 = Ls*0.186
    l2 = Ls*(0.146)
    lc2 = l2*0.682
    c = m2*l1*lc2*sin(te)+ObjectMass*l1*l2*sin(te)
    dc = m2*l1*lc2*cos(te)*oe+ObjectMass*l1*l2*cos(te)*oe
    C=np.array([[-2*oe*c,-oe*c],[os*c,0]])
    Cdot = np.array([[-2*(oe*dc+acce*c),-(oe*dc+acce*c)],[(os*dc+accs*c),0]])
    return C@np.array([accs,acce])+Cdot@np.array([os,oe])

def f(x,u):
    tau = 0.06
    C = np.array([-x[3]*(2*x[2]+x[3])*a2*np.sin(x[1]),x[2]*x[2]*a2*np.sin(x[1])])
    Denominator = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Minv = np.array([[a3/Denominator,(-a2*np.cos(x[1])-a3)/Denominator],[(-a2*np.cos(x[1])-a3)/Denominator,(2*a2*np.cos(x[1])+a1)/Denominator]])
    theta = Minv@(x[4:6]-Bdyn2@x[2:4]-C)
    torque = (u-x[4:6])/tau
    return np.array([[x[2],x[3],theta[0],theta[1],torque[0],torque[1]]])

def complex_derivative_1(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Dprime = 2*a2*a2*np.cos(x[1])*sin(x[1])
    F1 = x[4]-Bdyn2[0]@x[2:4]
    F2 = x[5]-Bdyn2[1]@x[2:4]
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
    Sol = -a3/D*(Bdyn2[0,0]+C1prime)+(a2*cos(x[1])+a3)/D*(Bdyn2[1,0]+C2prime)
    return Sol

def complex_derivative_3(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    C1prime =(-2*x[2]-2*x[3])*a2*np.sin(x[1])
    Sol = -a3/D*(Bdyn2[0,1]+C1prime)+(a2*cos(x[1])+a3)/D*(Bdyn2[1,1])
    return Sol

def complex_derivative_4(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    Dprime = 2*a2*a2*np.cos(x[1])*sin(x[1])
    F1 = x[4]-Bdyn2[0]@x[2:4]
    F2 = x[5]-Bdyn2[1]@x[2:4]
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
    Sol = (a2*cos(x[1])+a3)/D*(Bdyn2[0,0]+C1prime)+(2*a2*cos(x[1])+a1)/D*(-Bdyn2[1,0]-C2prime)
    return Sol

def complex_derivative_6(x):
    D = a3*(a1-a3)-a2*a2*np.cos(x[1])*np.cos(x[1])
    C1prime = (-2*x[2]-2*x[3])*a2*np.sin(x[1])
    Sol = (a2*cos(x[1])+a3)/D*(Bdyn2[0,1]+C1prime)+(2*a2*cos(x[1])+a1)/D*(-Bdyn2[1,1])
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

def l(x,u,r1,xtarg=0,w1=0,w2=0):
    totalcost = 0
    return totalcost + r1*(u[0]*u[0]+u[1]*u[1])/2

def lx(x,u,xtarg=0,w1=0,w2=0):
    totalcost = np.zeros(6)
    return totalcost

def lu(x,u,r1):
    return np.array([u[0]*r1,u[1]*r1])

def lxx(w1=0,w2=0):
    totalcost =np.zeros((6,6))
    return totalcost

def luu(x,u,r1):
    return np.array([[r1,0],[0,r1]])

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
def Kalman(Omega_measure,Omega_sens,A,sigma,H):
    K = A@sigma@H.T@np.linalg.inv(H@sigma@H.T+Omega_measure)
    sigma = Omega_sens + (A - K@H)@sigma@A.T
    return K,sigma
def step1(x0,u,Duration,Noise):
    
    K = np.shape(u)[0]+1
    dt = Duration/(K-1)
    newx = np.zeros((K,len(x0)))
    newx[0] = np.copy(x0)
    for i in range(K-1):
        newx[i+1] = newx[i] + dt*f(newx[i],u[i])
    return newx
def Compute_acc(x,F):
    C = np.array([-x[3]*(2*x[2]+x[3])*a2*np.sin(x[1]),x[2]*x[2]*a2*np.sin(x[1])])
        
    M = np.array([[a1+2*a2*cos(x[1]),a3+a2*cos(x[1])],[a3+a2*cos(x[1]),a3]])

    return np.linalg.solve(M,(x[4:6]-Bdyn2@(x[2:4])-C))+F
def step5(x0,l,L,Duration,Noise,A,B,Num_steps,bestu,FF,Side,kdelay,Variance):
    
    dt = Duration/(Num_steps-1)
    Num_Var = len(x0)
    

    x0 = np.tile(x0, kdelay + 1) 
    xref = np.zeros((Num_steps,Num_Var*(kdelay + 1)))
    xref[0] = np.copy(x0)
    newx = np.zeros((Num_steps,Num_Var*(kdelay + 1)))
    newx[0] = np.copy(x0)
    xhat = np.zeros((Num_steps,Num_Var*(kdelay + 1)))
    
    H = np.zeros((Num_Var,(kdelay+1)*Num_Var))
    H[:,(kdelay)*Num_Var:]= np.identity(Num_Var)

    sigma = np.zeros((Num_Var*(kdelay+1),Num_Var*(kdelay+1)))
    for i in range(Num_steps-1):
        Extended_A = np.zeros(((kdelay+1)*Num_Var,(kdelay+1)*Num_Var))
        Extended_A[:Num_Var,:Num_Var] = A[i]
        Extended_A[Num_Var:,:-Num_Var] = np.identity((kdelay)*Num_Var)
        Extended_B = np.zeros(((kdelay+1)*Num_Var,2))
        Extended_B[:Num_Var] = B[i]

        if FF == True:
            if i == 0 : acc = np.zeros(2)
            else : acc = Compute_acc(newx[i],F)
            F=Compute_f_new_version(newx[i,0:2],newx[i,2:4],acc,.3)
            if Side == "Left": F*=-1
            
        else : 
            F = np.array([0,0])

        deltau = l[i]+L[i]@xhat[i,:Num_Var]
        u = bestu[i] + deltau
        Omega_sens=np.zeros((len(x0),len(x0)))
        Omega_sens[5,5] = Variance
        Omega_sens[4,4] = Variance
        Omega_measure = np.diag(np.ones(6))*Variance
        K,sigma = Kalman(Omega_measure,Omega_sens,Extended_A,sigma,H)

        passed_newx = np.copy(newx[i,:-Num_Var])
        newx[i+1,:Num_Var] = newx[i,:Num_Var] + dt*f(newx[i,:Num_Var],u)
        newx[i+1,Num_Var:] = passed_newx
        newx[i+1,2:4]+=dt*F

        passed_xref = np.copy(xref[i,:-Num_Var])
        xref[i+1,:Num_Var] = xref[i,:Num_Var] + dt*f(xref[i,:Num_Var],u)
        xref[i+1,Num_Var:] = passed_xref
        
        if Noise: 
            newx[i,4:4+len(u)]+=np.random.normal(0,np.sqrt(Variance),len(u))
            #newx[i+1]+= motor_noise #+ mult_noise@u
        y = H@(newx[i]-xref[i])
        if Noise : y+=np.random.normal(0,np.sqrt(Variance),len(y))
        xhat[i+1] = Extended_A@xhat[i] + Extended_B@deltau + K@(y-H@xhat[i])
        
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
        q[i] = dt*l(x[i],u[i],r1,xtarg,w1,w2)
        qbold[i] = dt*lx(x[i],u[i],xtarg,w1,w2)
        r[i] = dt*lu(x[i],u[i],r1)
        Q[i] = dt*lxx(w1,w2)
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
        temp1 = 0
        temp2 = 0
        temp3 = 0
        for i in range(m):
            temp1+=C[k,i,:,:].T@S[k+1]@cbold[k,i,:]
            temp2+=C[k,i,:,:].T@S[k+1]@C[k,i,:,:]
            temp3+=cbold[k,i,:].T@S[k+1]@cbold[k,i,:]
        gbold = r[k] + B[k].T@sbold[k+1]+temp1
        G = B[k].T@S[k+1]@A[k]
        H = R[k] + B[k].T@S[k+1]@B[k]+temp2
        Hinv = np.linalg.inv(H)


        S[k] = Q[k] + A[k].T@S[k+1]@A[k]-G.T@Hinv@G
        sbold[k] = qbold[k]+A[k].T@sbold[k+1]-G.T@Hinv@gbold
        s[k] = q[k] + s[k+1] + 0.5*temp3 - .5*gbold.T@Hinv@gbold

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



def GetNoise(alpha,multvar,dt,N,kdelay):
    B_basic = np.array([[0,0],[0,0],[0,0],[0,0],[dt,0],[0,dt]])
    B = np.zeros((N*(kdelay+1),2))
    B[:N] = B_basic 
    return Compute_Multiplicative_Noise(N,alpha,B,multvar)

def ILQGa(Duration,w1,w2,r1,targets,K,start,plot = True,Noise = False,Delay = 0,FF = False,Side = "Left",Variance = 1e-6):
    obj1,obj2 = newton(fnewton,dfnewton,1e-8,1000,targets[0],targets[1]) #Defini les targets
    st1,st2 = newton(fnewton,dfnewton,1e-8,1000,start[0],start[1])

    x0 = np.array([st1,st2,0,0,0,0])
    O=np.zeros((6,6))
    O[5,5] = Variance
    O[4,4] = Variance
    m = 2
    n = 6
    u = np.zeros((K-1,m))
    dt = Duration/K
    kdelay = int(Delay/dt)
    newcbold = np.zeros((K,m,n))
    C = np.zeros((K,m,n,m))
    for i in range(K):
        for j in range(m):
            newcbold[i,j] = np.diag(O)[j]
    cbold = newcbold
    u_incr = [1]
    oldx = np.ones(K)*100
    # Create an array of 50 colors from the colormap

    for _ in range(50):     
        x = step1(x0,u,Duration,False)
        X = np.cos(x[:,0]+x[:,1])*33+np.cos(x[:,0])*30
        Y = np.sin(x[:,0]+x[:,1])*33+np.sin(x[:,0])*30
        if np.max(np.abs(oldx-X))<1e-3:
            x = step5(x0,l,L,Duration,Noise,A,B,K,u-u_incr,FF,Side,kdelay,Variance)
            
            X = np.cos(x[:,0]+x[:,1])*33+np.cos(x[:,0])*30
            Y = np.sin(x[:,0]+x[:,1])*33+np.sin(x[:,0])*30
            break

        A,B,q,qbold,r,Q,R = step2(x,u,Duration,w1,w2,r1,np.array([obj1,obj2]))
        l,L = step3(A,B,C,cbold,q,qbold,r,Q,R)
        u_incr = step4(l,L,K,A,B)
        u += u_incr
        oldx = np.copy(X)
        
    Xtg = targets[0]
    Ytg = targets[1]

        #Plotting
    if plot :
        plt.grid(linestyle='--')
        plt.axis("equal")
        plt.plot(X,Y,linewidth = 1.4,color = "#009933",label = "old values")
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        plt.scatter([Xtg],[Ytg],color = "black")
    return X,Y,u,x

def fnewton(var,X,Y):
    u,v = var
    return np.array([33*np.cos(u+v)+30*np.cos(u)-X,33*np.sin(u+v)+30*np.sin(u)-Y])

def dfnewton(var):
    u,v = var
    return np.array([[-33*np.sin(u+v)-30*np.sin(u),-33*np.sin(u+v)],[33*np.cos(u+v)+30*np.cos(u),33*np.cos(u+v)]])
