import numpy as np
from matplotlib import pyplot as plt
from math import *
from scipy.linalg import expm

from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg
import matplotlib.cm as cm
import casadi as ca
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

Bdyn = np.array([[0.05,0.025],[0.025,0.05]])

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

def MultipleLabel(title = "Controllers",side = "right",fontsize = 16,title_fontsize = 15):
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),fontsize = fontsize,title = title,title_fontsize = title_fontsize,frameon = True,shadow = True,fancybox = True,loc = side)

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
    Omegasenslinear = np.zeros((N*(kdelay+1),N*(kdelay+1)))
    Omegasenslinear[2,2] = Var
    Omegasenslinear[5,5] = Var
    if Linear : return Omegasenslinear,np.diag(np.ones(N)*Var),motornoise,np.random.normal(0,np.sqrt(Var),N)
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

def get_colors_from_colormap(N, cmap_name):
    """
    Generates an array of N colors from the given colormap.
    
    Parameters:
        N (int): Number of colors to extract.
        colormap_name (str): Name of the Matplotlib colormap.
    
    Returns:
        List of N RGB tuples.
    """
    colormap = cm.get_cmap(cmap_name, N)  # Get colormap with N discrete levels
    colors = [colormap(i) for i in range(N)]  # Extract N colors
    return colors

def add_scale_cm(X,Y,L,fontsize = 10):
    plt.plot(np.linspace(X,X+L,100),np.ones(100)*Y,color = "black")
    plt.plot(np.ones(100)*X,np.linspace(Y,Y+L,100),color = "black")
    plt.text(X+1,Y+1,str(L)+" cm",fontsize = fontsize)

def delete_axis(ax,sides = ["left","right","bottom","top"] ):
    for side in sides : ax.spines[side].set_visible(False)   

def delete_ticks(ax):
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

def Compute_Cost_function(r,w1,w2,u,xf,target,at3=False):
    if at3:
        return np.sum(u**2)*r+w1*(xf[0]-target[0])**2+w1*(xf[3]-target[1])**2+w2*(xf[1]**2+xf[4]**2)
    else: 
        return np.sum(u**2)*r+w1*(xf[0]-target[0])**2+w1*(xf[1]-target[1])**2+w2*(xf[2]**2+xf[3]**2)