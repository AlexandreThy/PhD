import os
os.chdir('..')

from Controllers.FL import *
from Controllers.ILQG import *
from Controllers.LQGControllers import *


st = [0,40]
WP = 20000
WV =  1
WR = .5
NUM_SIM = 100
MN = 5e-4
MOVEMENT_LENGTH = 15

colors = ["#009E73","#0072B2","#E69F00"]
legend = ["ILQG","FL","DLQG"]

def Compute_Cartesian_Speed(X, Y, dt):
    V = np.zeros(X.shape)
    Vx = np.diff(X) / dt
    Vy = np.diff(Y) / dt
    V[1:] = np.sqrt(Vx * Vx + Vy * Vy)
    return V

def Cost_function(x,u,w1 = WP,w2 = WV,r = WR,tg = [0,0] ):
    target1,target2 = compute_angles_from_cartesian(tg[0],tg[1])
    thetas,thetae,omegas,omegae = x[-1,:4]
    
    return w1*(thetas-target1)**2 + w1*(thetae-target2)**2 + w2*(omegas**2+omegae**2)+ np.sum(u*u) * r


def compute_cost_slow(movement_time = .6):
    num_iterations = int(movement_time*100)
    return compute_cost(movement_time,num_iterations)

def compute_cost_fast(movement_time = .4):
    num_iterations = int(movement_time*100)
    return compute_cost(movement_time,num_iterations)


def compute_cost(MovementTime, NumIteration):

    Cost_value = np.zeros((NUM_SIM,9,3))
    
    for num_sim in range(NUM_SIM):

        for iter,angles in enumerate(np.linspace(0,2*pi,9)[:-1]) :
            tg = [st[0]+cos(angles)*MOVEMENT_LENGTH,st[1]+sin(angles)*MOVEMENT_LENGTH]
            xILQG,yILQG,zilqg,uILQG=simulate_ILQG(MovementTime,WP,WV,WR,tg,st,NumIteration,delay = 0.06,Noise=True,print_iterations=False,motornoise_variance=MN)
            xFL,yFL,xfl,uFL = simulate_FL(Duration=MovementTime,w1=WP,w2=WP,w3=WV,w4=WV,r=1e-8,FF=False,Num_iter=NumIteration,starting_point=st,targets=tg,Delay = 0.06,Activate_Noise=True,motornoise_variance=MN)
            xLQG,yLQG,uDLQG,z=DLQG_6Muscles(w1 = WP,w2 = WP,w3=WV,w4=WV,Duration=MovementTime,r1 = WR,Num_iter=NumIteration,starting_point=st,targets=tg,plot = False,Delay = 0.06,Activate_Noise=True,motornoise_variance=MN) 
            Cost_value[num_sim,iter] = np.array([Cost_function(zilqg,uILQG,tg = tg),Cost_function(xfl,uFL,tg = tg),Cost_function(z.T,uDLQG,tg = tg)])

    Cost_value = np.mean(Cost_value,axis = 0)
    Cost_value[8] = Cost_value[0]
    return Cost_value


if __name__ == "__main__":

    fig,ax = plt.subplots(2,figsize = (8,8),subplot_kw={'projection': 'polar'})
    angles = np.linspace(0, 2*np.pi, 9)

    Cost_value = compute_cost_slow()

    for i in range(3):
        ax[0].scatter(angles, Cost_value[:,i], color=colors[i], s = 20, zorder = 10)
        ax[0].plot(angles, Cost_value[:,i], color=colors[i], linewidth=2,linestyle = "--",label = legend[i])
    ax[0].set_xticklabels([])

    Cost_value = compute_cost_fast()

    for i in range(3):
        ax[1].scatter(angles, Cost_value[:,i], color=colors[i], s = 20, zorder = 10)
        ax[1].plot(angles, Cost_value[:,i], color=colors[i], linewidth=2,linestyle = "--",label = legend[i])
    ax[1].set_xticklabels([])

    for a in ax:
        a.set_yticks([2,4,6])
        a.set_xticks(np.linspace(0, 2*np.pi, 9)[:-1], labels=['', '', '', '', '', '', '', ''])

    plt.savefig("Figure2.svg",dpi = 300)


    plt.show()