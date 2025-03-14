import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.integrate import odeint
from sklearn.decomposition import PCA
import seaborn as sns
# set the random seed for reproducibility
np.random.seed(25) # to reprodce the same results
#np.random.seed(10) # trying out a different seed
 
# Define the network-body dynamics of a simple 2D network with a 1D body state
class mass1Ddyn:
    def __init__(self, W, k=1):
        self.k = k
        self.dt = 0.01
        self.naugment = 2
 
        # the body is just a first order system
        # x_dot = -k*x + Wout*u
        # hence A = -k, B = Wout = np.array([1, -1])
        self.A = np.array([[-k]])
        self.Wout = np.array([[1, 1]])
 
        # define network parameters of a 2D network
        self.N = 2
        self.Win = np.zeros((self.N, 1)) # Win multiplies with the body state x
        #self.W = np.zeros((self.N, self.N)) # W multiplies with the network state z
        # set W as a skew-symmetric matrix
        #self.W[0, 1] = 1
        #self.W[1, 0] = -1
        self.W = W
 
        # get the total network-body matrices before augmentation
        Acont_preaug, Bcont_preaug = self.appendNetworkMatrices()
        self.Adisc, self.Bdisc, self.Acont, self.Bcont = self.cont2disc(Acont_preaug, Bcont_preaug)
 
        # initial state of the body
        self.states = np.zeros((self.naugment*(self.N+1), 1)) # z = [x, r1, r2, x_targetpos, 0, 0].T
 
        # determine the noise covariance properties of the body and measurement noise
        self.SigmaXi = 0.5 * self.Bcont @ (self.Bcont.T)
 
        return
 
    def appendNetworkMatrices(self):
        temprow1 = np.concatenate((self.A, self.Wout), axis=1)
        temprow2 = np.concatenate((self.Win, self.W), axis=1)
        Atot = np.concatenate((temprow1, temprow2), axis=0)
 
        B = np.eye(self.N)
        Btot = np.concatenate((np.zeros((1, self.N)), B), axis = 0)
        return Atot, Btot
    
    def cont2disc(self, Acont, Bcont):
        ''' to append target states, and to perform cont2disc conversion'''
        nstates = Acont.shape[0]
        ncontrols = Bcont.shape[1]
        #Augment matrices A and B with target state
        tempA = np.zeros(shape=(self.naugment*nstates,self.naugment*nstates))
        tempA[0:nstates,0:nstates] = Acont + 0.0
        Adisc = np.eye(self.naugment * nstates) + (self.dt * tempA)
        
        tempB = np.zeros(shape=(self.naugment*nstates,ncontrols))
        tempB[0:nstates, 0:ncontrols] = Bcont + 0.0
        Bdisc = self.dt * tempB
        return Adisc, Bdisc, tempA, tempB
 
    def disc2cont(self, a):
        na = a.shape[0]
        A = (a - np.eye(na))/self.dt
        return A
    
    def getSystemMatrices(self):
        return self.Adisc, self.Bdisc
 
    @staticmethod
    def sysdyn(z, t, u, A, B):
        dzdt = A @ z
        dzdt += B @ u
        return dzdt
    
    def nextState(self, Tor): # Tor is supposed to be torques (xtorque; ytorque)
        # Here we take Torques as inputs and compute the state update using (x_dot = A*x + B*Tor)
        cur_state = self.states + 0.0
 
        # updating next state with the naive method - CAUTION when using with network dynamics!
        #next_state = (self.Atot @ cur_state) + (self.Btot @ Tor) + xi.T
        # updating the next state using better continuous differential equ. solvers (ss, odeint etc.,)
        next_state = odeint(self.sysdyn, cur_state, np.linspace(0,0.01,2), args=(Tor, self.Acont, self.Bcont))[-1]
 
 
        # update the state of the point-mass
        self.states = next_state + 0.0
        return next_state
    
    def reset(self, initstate):
        # z_total = [x, r1, r2, x_targetpos, 0, 0].T
        self.states = initstate + 0.0
        return
    
 
# here set the cost matrices for the given task parameters (A, B, r, w)
def setCostmatrices(Adisc, Bdisc, r, w):
    '''
    A,B,R,Q = setupMatrice(A0,B0,delta,w)
    Takes as input the matrices of continuous timme differential equation,
    timestep and weights for the definition of the cost matrices
    
    Outputs: discrete time matrices and cost matrices
    '''
    nstates = Adisc.shape[0]
    ncontrols = Bdisc.shape[1]
 
    #Setup cost matrices
    nStep = w.shape[1] - 1
    nHold = 20
    # Control cost
    R = np.zeros(shape=(nStep, ncontrols, ncontrols))
    for i in range(nStep):
        R[i, :, :] = r*np.identity(ncontrols)
    # State cost
    Q = np.zeros(shape=(nStep+1, int(2*nstates/2), int(2*nstates/2)))
    vec = np.zeros(shape=(int(2*nstates/2), 1))
    id = np.identity(int(nstates/2))
    for i in range(0, nStep - nHold +1):
        for j in range(int(nstates/2)):
            id_temp = id[:, j]
            vec[0:int(nstates/2), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(nstates/2):int(2*nstates/2), 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            Q[i, :, :] = Q[i, :, :] + w[j, i] * vec @ np.transpose(vec)
 
    for i in range(nStep - nHold +1, nStep+1):
        for j in range(int(nstates/2)):
            id_temp = id[:, j]
            vec[0:int(nstates/2), 0] = np.transpose(id_temp) # Q matrix for the actual hand state
            vec[int(nstates/2):int(2*nstates/2), 0] = -np.transpose(id_temp) # Q entries for the target state (negative)
            Q[i, :, :] = Q[i, :, :] +  w[j, i] * vec @ np.transpose(vec)
 
    return R, Q
 
 
# Definition of the function basicLQG
def basicLQG(Adisc, Bdisc, Q, R, SigmaXi):
  '''
  L,K = basicLQG(A,B,Q,R,H,SigmaXi,SigmaOmega)  
    Takes as input the matrices corresponding to the state space representation,
    the cost matrices, and the noie covariance matrices,
 
    Returns: time series of feedback gains and Kalman gains for simulations of
    LQG control
  '''
  # Calculation of the optimal feedback gains
  # Dimension parameters
  nStep = R.shape[0]
  nstates = Adisc.shape[0]
  ncontrols = Bdisc.shape[1]
  #Recurrence
  S = np.zeros(shape=(nStep+1, nstates, nstates))
  S[nStep, :, :] = Q[nStep, :, :]
  L = np.zeros(shape=(nStep, ncontrols, nstates))
  sadd = 0
  for i in reversed(range(nStep)):
    L[i, :, :] = np.linalg.inv(R[i, :, :] + np.transpose(Bdisc) @ S[i+1, :, :] @ Bdisc) @ np.transpose(Bdisc) @ S[i+1, :, :] @ Adisc
    S[i, :, :] = Q[i, :, :] + np.transpose(Adisc) @ S[i+1, :, :] @ (Adisc - Bdisc @ L[i, :, :])
    sadd = sadd + np.trace(S[i+1, :, :] @ SigmaXi)
  return L
 
 
# now define the 8 condition reaching controller
def eightCondReach(params):
    bodyins = mass1Ddyn(params['W'], params['k'])
    # instantiate a fake bodyins to get different system matrices
    W_fake = np.zeros((2, 2))
    bodyins_fake = mass1Ddyn(W_fake, params['k'])
    # get the system matrices
    Adisc, Bdisc = bodyins.getSystemMatrices()
    Adisc_fake, Bdisc_fake = bodyins_fake.getSystemMatrices()
    SigmaXi = bodyins.SigmaXi + 0.0
    # set the cost matrices
    R, Q = setCostmatrices(Adisc, Bdisc, params['r'], params['w'])
    R_fake, Q_fake = setCostmatrices(Adisc_fake, Bdisc_fake, params['r'], params['w'])
    # get the feedback gains
    L = basicLQG(Adisc, Bdisc, Q, R, SigmaXi)
    L_fake = basicLQG(Adisc_fake, Bdisc_fake, Q_fake, R_fake, SigmaXi)
    #L[:, :, 0:6] = L_fake[:, :, 0:6] + 0.0
    #L[:, :, 3] = L_fake[:, :, 3] + 0.0
    #L = L_fake + 0.0
    # now simulate the system with the controller
    # define simtime
    simtime = np.arange(0, params['rtime'] + params['delta'], params['delta'])
    # define the target state as 8 values between -1 and 1
    num_steps = len(simtime)
    num_targconditions = 8
    num_states = Adisc.shape[0]
    target_states = np.zeros((num_targconditions, num_states))
    #target_states[:, 3] = np.linspace(-1, 1, num_targconditions) # 3 is the index of the target state in the augmented state vector
 
    # initialize the state vector
    all_states = np.zeros((num_steps, num_targconditions, num_states))
    for i in range(num_steps):
        all_states[i, :, 3] = np.linspace(-1, 1, num_targconditions)
 
    # initialize the network readout command
    all_readout = np.zeros((num_steps, num_targconditions, 1))
    bodyins.reset(all_states[0, 0, :])
 
    # now simulate the system
    for i in range(num_targconditions):
        bodyins.reset(all_states[0, i, :])
        for j in range(num_steps-1):
            cur_state = all_states[j, i, :] + 0
 
            # get the control input
            u = -L[j, :, :] @ cur_state
 
            # get the next state
            next_state = bodyins.nextState(u)
 
            # update the state vector
            all_states[j+1, i, :] = next_state + 0
            all_readout[j+1, i, 0] = bodyins.Wout @ next_state[1:3]
 
    results = {'states': all_states, 'L': L, 'readout': all_readout}
    return results
 
def setParams(W = None):
    k = 1
    #W = np.zeros((2, 2))
    # generate a random matrix for W with adjustable parameters
    #p = 10
    #W = np.random.normal(0, p, (2, 2))
    # generate a random skew-symmetric matrix with parameter p
    #W = p * (np.random.rand(2, 2) - 0.5)
    #W = W - np.transpose(W)
    # genetate a random symmetric matrix with parameter p
    #W = p * (np.random.rand(2, 2) - 0.5)
    #W = W + np.transpose(W)
    #alpha = 2
    #omega = np.random.normal(20, 15)
    # generate a 2D matrix with omega frequency and alpha damping
    #W = np.array([[0, omega], [-omega, -alpha]])
    rtime = 0.5
    delta = 0.01
 
    state_penalty = 1
    control_penalty = 1e-10
    nStep = int(np.round((rtime)/delta))
    w = np.zeros((3, nStep+1))
    for t in range(nStep+1):
        w[:3, t] = (1/nStep) * state_penalty * np.array([1, 0, 0]) * (t / nStep)**2
 
    r = control_penalty + 0.0
    params = {'k': k, 'W': W, 'rtime': rtime, 'delta': delta, 'r': r, 'w': w}
 
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
    behavior = results['states'][:, :, 0]
    network = results['states'][:, :, 1:3]
    target = results['states'][:, :, 3]
 
    # also get control gains
    L = results['L']
    behavior_gains = L[:, 0, :]
    network_gains = L[:, 1:3, :]
    readout = results['readout']
    return behavior, network, behavior_gains, network_gains, readout
 
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
        p = i * (np.pi/num_runs)
        W = np.array([[0, -p], [p, 0]])
        # generate a random matrix for W with adjustable parameters
        #p = i * (10/num_runs)
        #W = np.array([[0, 0], [p, 0]])
        params = setParams(W) # set diff random params for each run
        behavior, network, behavior_gains, network_gains, readout = runOnce(params)
        all_behavior.append(behavior)
        all_network.append(network)
        all_behavior_gains.append(behavior_gains)
        all_network_gains.append(network_gains)
        all_readout.append(readout)
    all_behavior = np.array(all_behavior)
    all_network = np.array(all_network)
    all_behavior_gains = np.array(all_behavior_gains)
    all_network_gains = np.array(all_network_gains)
    all_readout = np.array(all_readout)
    
    # perform PCA on each of the runs of network dynamics and extract the first two components and place them in all_pcs
    all_pcs = []
    window = 20
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
 
