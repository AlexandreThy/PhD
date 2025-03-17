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

from sklearn.decomposition import PCA
import seaborn as sns
# set the random seed for reproducibility
np.random.seed(25) # to reprodce the same results
#np.random.seed(10) # trying out a different seed


class joint2Ddyn:
    def __init__(self, W,nodes_number):
        self.dt = 0.01
        self.M = 6
        # the body is just a first order system
        # x_dot = -k*x + Wout*u
        # hence A = -k, B = Wout = np.array([1, -1])
        self.Wout = np.random.uniform(0,1,(2,nodes_number))
 
        # define network parameters of a 2D network
        self.N = nodes_number
        self.Win = np.zeros((self.N, 1)) # Win multiplies with the body state x

        self.W = W

        # initial state of the body
        self.states = np.zeros((self.N+self.M, 1)) # z = [x, r1, r2, x_targetpos, 0, 0].T
 
        # determine the noise covariance properties of the body and measurement noise
        self.SigmaXi = 1e-4
 
        return
 
    @staticmethod
    def sysdyn(x,gamma, u,dt,Wout,W):
        M = np.array([[a1+2*a2*cos(x[1]),a3+a2*cos(x[1])],[a3+a2*cos(x[1]),a3]])
        C = np.array([-x[3]*(2*x[2]+x[3])*a2*np.sin(x[1]),x[2]*x[2]*a2*np.sin(x[1])])

        x[0:2] += dt*x[2:4]
        x[2:4] += dt*x[4:6]
        x[4:6] = np.linalg.solve(M,(Wout@gamma-Bdyn@(x[2:4])-C))
        gamma += dt*(np.tanh(W@gamma)+u)
        return np.concatenate((x,gamma))
    

  
def compute_control_gains(Num_iter,Duration,motor_cost = 1e-4,cost_weights = [1e4,1e4,1,1]):
    
    dt = Duration/Num_iter 
    Num_Var = 8
    
    #Define Weight Matrices of the cost function
    R = np.array([[motor_cost,0],[0,motor_cost]])
    w1,w2,w3,w4 = cost_weights
    Q = np.array([[w1,0,0,0,0,0,-w1,0],[0,w2,0,0,0,0,0,-w2],
               [0,0,w3,0,0,0,0,0],[0,0,0,w4,0,0,0,0],
               [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],
               [-w1,0,0,0,0,0,w1,0],[0,-w2,0,0,0,0,0,w2]])
    
    #Define the Dynamic of the linear system 

    A = np.array([[1,0,dt,0,0,0,0,0],[0,1,0,dt,0,0,0,0],[0,0,1,0,dt,0,0,0],[0,0,0,1,0,dt,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
    B = np.array([[0,0],[0,0],[0,0],[0,0],[dt,0],[0,dt],[0,0],[0,0]])

    #Compute the Feedback Gain of the Control law
    S = Q

    L = np.zeros((Num_iter-1,2,Num_Var))   

    for k in range(Num_iter-1):
        L[Num_iter-2-k]  = np.linalg.inv(R+B.T@S@B)@B.T@S@A
        S = A.T@S@(A-B@L[Num_iter-2-k])
        
    #Initialize matrices 
    return L

def compute_nonlinear_command(L,x,Wout,W,gamma):

    M = np.array([[a1+2*a2*cos(x[1]),a3+a2*cos(x[1])],[a3+a2*cos(x[1]),a3]])
    Cdot = np.array([-a2*x[5]*(2*x[2]+x[3])*sin(x[1])-a2*x[3]*(2*x[4]+x[5])*sin(x[1])
                     -a2*x[3]*x[3]*(2*x[2]+x[3])*cos(x[1]),2*x[2]*x[4]*a2*sin(x[1])+x[2]*x[2]*a2*cos(x[1])*x[3]])
    Mdot = np.array([[-2*a2*sin(x[1])*x[3],-a2*sin(x[1])*x[3]],[-a2*sin(x[1])*x[3],0]])

    # Compute the command through the FL technique
    v = -L@x
    gammadot = M@(v)+Mdot@(np.array([x[4],x[5]]))+Cdot+Bdyn@np.array([x[4],x[5]])
    u = np.linalg.pinv(Wout)@gammadot-np.tanh(W@gamma)
    return u ,v   


 
# now define the 8 condition reaching controller
def eightCondReach(params):

    num_steps = params["num_iter"]
    duration = params["duration"]
    
    bodyins = joint2Ddyn(params['W'], params['N'])
    
    # get the feedback gains
    L = compute_control_gains(num_steps,duration)

    # define the target state as 8 values between -1 and 1
    
    num_targconditions = 8
    num_states = int(params['N']) + 6
    dt = duration/num_steps
    # initialize the state vector
    all_states = np.zeros((num_steps, num_targconditions, num_states+2))
 
    # initialize the network readout command
    all_readout = np.zeros((num_steps, num_targconditions,2))
 
    # now simulate the system
    for i in range(num_targconditions):
        angles = np.linspace(0,2*pi,num_targconditions+1)[:-1]
        st1,st2 = newton(newtonf,newtondf,1e-8,1000,0,30)
        print("Starting Position : ",st1,st2,np.array([33*np.cos(st1+st2)+30*np.cos(st1),33*np.sin(st1+st2)+30*np.sin(st1)]))
        tg1,tg2 = newton(newtonf,newtondf,1e-8,1000,10*cos(angles[i]),30+10*sin(angles[i]))
        all_states[0,i,:] = np.concatenate(([st1,st2,0,0,0,0,tg1,tg2],np.zeros(params['N'])))
        for j in range(num_steps-1):
            cur_state = all_states[j, i, :]
            # get the control input
            u,v = compute_nonlinear_command(L[j],cur_state[:8],bodyins.Wout,bodyins.W,cur_state[8:])
 
            # get the next state
            next_state = bodyins.sysdyn(cur_state[:8],cur_state[8:],u,dt,bodyins.Wout,bodyins.W)
            print("Position at time ",j*dt,": ",np.array([33*np.cos(next_state[0]+next_state[1])+30*np.cos(next_state[0]),33*np.sin(next_state[0]+next_state[1])+30*np.sin(next_state[0])]))

            # update the state vector
            all_states[j+1, i, :] = next_state 
            all_readout[j+1, i] = bodyins.Wout @ next_state[8:]
 
    results = {'states': all_states, 'L': L, 'readout': all_readout}
    return results
 
def setParams(W = None):

    rtime = 0.6
    num_iter = 60
    
    params = {'num_iter': num_iter, 'W': W, 'duration': rtime, 'N': 6}
 
    return params
 
def compute_pca_and_projection(neuron_states, time_window_analysis):
    # first apply 2 preproceesiing steps on the neural data
    # 1. normalize the data with range + 5
    # 2. center the data around the condition mean
 
    # apply those steps on the entire dataset and not just train and test data
    neuron_states_normalized = neuron_states / (np.max(neuron_states, axis=(0, 1)) - np.min(neuron_states, axis=(0, 1)) + 5)
    
    # compute the mean at each time point for each neuron across conditions and subtract
    neuron_states_mean_centered = np.zeros(neuron_states.shape)
    for n in range(neuron_states.shape[2]):
        for t in range(neuron_states.shape[0]):
            neuron_states_mean_centered[t, :, n] = neuron_states_normalized[t, :, n] - np.mean(neuron_states_normalized[t, :, n])
 
    # split the data into train and test data
    train_neuron_states = neuron_states_mean_centered[time_window_analysis[0]:time_window_analysis[1], :, :]
    test_neuron_states = neuron_states_mean_centered[time_window_analysis[0]:time_window_analysis[1], :, :]
 
    # reshape the data to fit the PCA input
    train_neuron_states_reshaped = train_neuron_states.swapaxes(0, 1).reshape(train_neuron_states.shape[0]*train_neuron_states.shape[1], train_neuron_states.shape[2])
    test_neuron_states_reshaped = test_neuron_states.swapaxes(0, 1).reshape(test_neuron_states.shape[0]*test_neuron_states.shape[1], test_neuron_states.shape[2])
 
    # apply PCA on the train data
    n_pcs = 2
    pca = PCA(n_components=n_pcs)
    pca.fit(train_neuron_states_reshaped)
 
    # compute the projection of the neural data onto the first 2 PCs
    pc_states = pca.transform(test_neuron_states_reshaped)
 
    # reshape the data to fit the original shape
    reshaped_pc_states = pc_states.reshape(test_neuron_states.shape[1], test_neuron_states.shape[0], n_pcs).swapaxes(0, 1)
 
    return reshaped_pc_states, pca.explained_variance_ratio_
 
def runOnce(params):
    results = eightCondReach(params)
    behavior = results['states'][:, :, :2]
    network = results['states'][:, :, 8:]
 
    # also get control gains
    L = results['L']
    behavior_gains = L[:, 0, :]
    readout = results['readout']
    return behavior, network, behavior_gains, readout
 
# define the main script to set the params and run the controller
if __name__ == '__main__':
    
    # run the eight condition reaching controller for 10 times in a loop and record all the behavior and network states
    num_runs = 20
    all_behavior = []
    all_network = []
    all_behavior_gains = []
    all_network_gains = []
    all_readout = []
    for i in range(num_runs):
        W =np.random.uniform(0,1,(6,6))
        # generate a random matrix for W with adjustable parameters
        #p = i * (10/num_runs)
        #W = np.array([[0, 0], [p, 0]])
        params = setParams(W) # set diff random params for each run
        behavior, network, behavior_gains, readout = runOnce(params)
        all_behavior.append(behavior)
        all_network.append(network)
        all_behavior_gains.append(behavior_gains)
        all_readout.append(readout)
    all_behavior = np.array(all_behavior)
    all_network = np.array(all_network)
    all_behavior_gains = np.array(all_behavior_gains)
    all_network_gains = np.array(all_network_gains)
    all_readout = np.array(all_readout)
    
    for i in range(8): 
        thetas = (behavior[:,i,0])
        thetae = (behavior[:,i,1])
        plt.plot(33*np.cos(thetas+thetae)+30*np.cos(thetas),33*np.sin(thetas+thetae)+30*np.sin(thetas))
    plt.show()
    
    
    # perform PCA on each of the runs of network dynamics and extract the first two components and place them in all_pcs
    all_pcs = []
    window = 50
    for i in range(num_runs):
        pc_states, explained_variance_ratio = compute_pca_and_projection(all_network[i], [0, window])
        all_pcs.append(pc_states)
    all_pcs = np.array(all_pcs)
 

 
    # compute the difference in PCs across runs
    pc_diffs = []
    for i in range(1, num_runs):
        pc_diffs.append(all_pcs[i, :, :, :] - all_pcs[i-1, :, :, :])
    pc_diffs = np.array(pc_diffs)
    # compute the squared sum of the differences in PCs across runs
    pc_diffs_sqsum = np.sum(all_pcs**2, axis=(1, 2)) # squared sum across time and target conditions gives a num_runs X 2 matrix
    
    # normalize the squared sum of differences in PCs across runs and plot them
    pc_diffs_sqsum_normalized = pc_diffs_sqsum / np.max(pc_diffs_sqsum, axis=0)

 
 
    sns.set(style="white")
 
    # plot the X-Y position of the body in dot and bar tasks in one panel
    # create a figure
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(12*cm, 5*cm), constrained_layout=True)
 
    # Set global linewidth
    # Style the Plot
    plt.rcParams.update({
        "font.size": 7,
        "font.family": "Arial",
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        'lines.linewidth': .8,
        'axes.linewidth': .8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': .5,
        'xtick.minor.width': .5,
        'ytick.major.width': .5,
        'ytick.minor.width': .5,
        "axes.titlesize": 7,  # Title font size
        "axes.titleweight": "bold",  # Title font weight
    })
    # plot the X-Y position of the body in dot and bar tasks in one panel
    colors = ['#0c121d', '#222c3b', '#37455a', '#4d5f78', '#627896', '#7892b4', '#8dabd3', '#a3c5f1']
    #colors = sns.color_palette('viridis', 8)
    colors = sns.color_palette('RdBu', 8)
    #colors = plt.cm.ocean(np.linspace(0., .8, 8))
    #colors = plt.cm.RdBu(np.linspace(0, 1, 8))
    #colors = np.concatenate((plt.cm.RdBu(np.linspace(0, 0.3, 4)), plt.cm.RdBu(np.linspace(0.7, 1.0, 4))), axis=0)
 
 
    ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2, 5), (0, 1), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 5), (0, 2), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((2, 5), (1, 0), colspan=1, rowspan=1)
    ax5 = plt.subplot2grid((2, 5), (1, 1), colspan=1, rowspan=1)
    ax6 = plt.subplot2grid((2, 5), (1, 2), colspan=1, rowspan=1)
    ax7 = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)
 
    sns.despine(top=True, right=True)
 
    for c in range(8):
        ax1.plot(all_pcs[0, :, c, 0], color = colors[c])
    ax1.set_ylim([-0.6, 0.6])
    ax1.set_ylabel("PC 1")
    #ax1.set_xlabel("Time (ms)")
    ax1.set_xticks([0, window])
    #ax1.set_xticklabels(['0', '0.2'])
    ax1.tick_params(labelbottom=False, bottom=True,labelleft=True, left=True)
 
    for c in range(8):
        ax2.plot(all_pcs[5, :, c, 0], color = colors[c])
    ax2.set_ylim([-0.6, 0.6])
    ax2.set_xticks([0, window])
    #ax2.set_xticklabels(['0', '0.2'])
    ax2.tick_params(labelbottom=False, bottom=True,labelleft=False, left=True)
 
    for c in range(8):
        ax3.plot(all_pcs[10, :, c, 0], color = colors[c])
    ax3.set_ylim([-0.6, 0.6])
    ax3.set_xticks([0, window])
    #ax3.set_xticklabels(['0', '0.2'])
    ax3.tick_params(labelbottom=False, bottom=True,labelleft=False, left=True)
 
    for c in range(8):
        ax4.plot(all_pcs[0, :, c, 1], color = colors[c])
    ax4.set_ylim([-0.3, 0.3])
    ax4.set_xticks([0, window])
    ax4.set_xticklabels(['0', '0.2'])
    ax4.set_ylabel("PC 2")
    ax4.set_xlabel("Time (s)")
    ax4.tick_params(labelbottom=True, bottom=True,labelleft=True, left=True)
    
    for c in range(8):
        ax5.plot(all_pcs[5, :, c, 1], color = colors[c])
    ax5.set_ylim([-0.3, 0.3])
    ax5.set_xticks([0, window])
    ax5.set_xticklabels(['0', '0.2'])
    ax5.set_xlabel("Time (s)")
    ax5.tick_params(labelbottom=True, bottom=True,labelleft=False, left=True)
    
    for c in range(8):
        ax6.plot(all_pcs[10, :, c, 1], color = colors[c])
    ax6.set_ylim([-0.3, 0.3])
    ax6.set_xticks([0, window])
    ax6.set_xticklabels(['0', '0.2'])
    ax6.set_xlabel("Time (s)")
    ax6.tick_params(labelbottom=True, bottom=True,labelleft=False, left=True)
 
    ax7.plot(pc_diffs_sqsum_normalized[:, 0], '-o', color='black')
    ax7.plot(pc_diffs_sqsum_normalized[:, 1], '-o', color='lightgray')
    ax7.set_ylabel("Normalized change")
    ax7.set_xlabel("Connectivity parameter")
    ax7.set_xticks([0, 5, 10, 15, 20])
    ax7.set_xticklabels(['0', 'pi/4', 'pi/2', '3pi/4', 'pi'])
    ax7.tick_params(labelbottom=True, bottom=True,labelleft=True, left=True)
 
    #plt.tight_layout()
    plt.show()
 
