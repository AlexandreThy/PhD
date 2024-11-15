import numpy as np
from matplotlib import pyplot as plt
from math import *

I1 = 0.025
I2 = 0.045
m2 = 1
l1 = 0.3
l2 = 0.33
s2 = 0.16
K = 1/0.06
tau = 0.06

#SHOULDER PUIS ELBOW

a1 = I1 + I2 + m2*l2*l2
a2 = m2*l1*s2
a3 = I2

Bdyn = np.array([[0.5,0.025],[0.025,0.5]])


def Compute_Noise(NbreVar,Variance,B,kdelay = 0):

    Omega_sens = np.diag(np.ones(NbreVar*(kdelay+1))*Variance)
    motor_noise = np.random.normal(0,np.sqrt(Variance),NbreVar*(kdelay+1)).T
    Omega_measure = np.diag(np.ones(NbreVar)*Variance)
    measure_noise = np.random.normal(0,np.sqrt(Variance),NbreVar).T
    print("cc",Omega_sens[:NbreVar,:NbreVar],Omega_measure)
    return Omega_sens,motor_noise,Omega_measure,measure_noise

def Compute_Noise2(NbreVar,alpha,B,kdelay):
    B = B[:NbreVar]
    O_sens = alpha*B@B.T

    O_sens = np.diag(np.ones(NbreVar)*O_sens[2,2])
    Omega_sens = np.zeros((NbreVar*(kdelay+1),NbreVar*(kdelay+1)))
    Omega_sens[:NbreVar,:NbreVar] = O_sens
    #Ok que si omegasens est diag
    motor_noise = np.zeros(NbreVar)
    for i in range(NbreVar):
        motor_noise[i] = np.random.normal(0,np.sqrt(Omega_sens[i,i]))
    Omega_measure = np.diag(np.ones(NbreVar)*1e-7)
    measure_noise = np.random.normal(0,np.sqrt(1e-7),NbreVar).T

    return Omega_sens,motor_noise.T,Omega_measure,measure_noise

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
