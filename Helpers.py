import numpy as np
from scipy.linalg import *
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
from math import *
import random
from sympy.solvers import solve
from sympy import Symbol
import sympy as sym
import pandas as pd
from scipy.io import loadmat
import scipy as scipy
import plotly.express as px
import plotly.graph_objects as go

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
B = np.array([[0.5,0.025],[0.025,0.5]])

Bdyn = np.array([[0.5,0.025],[0.025,0.5]])
np.random.seed(0)

Bruit = True


def Compute_Noise(NbreVar,Variance):

    Omega_sens = np.diag(np.concatenate((np.ones(int(NbreVar/2)),np.zeros(int(NbreVar/2)))))
    motor_noise = np.concatenate((np.random.normal(0,np.sqrt(Variance),int(NbreVar/2)),np.zeros(int(NbreVar/2)))).T
    Omega_measure = np.diag(np.ones(NbreVar)*Variance)
    measure_noise = np.concatenate((np.random.normal(0,np.sqrt(Variance),int(NbreVar/2)),np.zeros(int(NbreVar/2)))).T

    return Omega_sens,motor_noise,Omega_measure,measure_noise

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

def f(var,X,Y):
    u,v = var
    return np.array([33*np.cos(u+v)+30*np.cos(u)-X,33*np.sin(u+v)+30*np.sin(u)-Y])

def df(var):
    u,v = var
    return np.array([[-33*np.sin(u+v)-30*np.sin(u),-33*np.sin(u+v)],[33*np.cos(u+v)+30*np.cos(u),33*np.cos(u+v)]])