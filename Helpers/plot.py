from Helpers.Helpers import *

def plotFL(Pert,FSpan,Noise,Feedback_Linearization):
    """
    Pert (array of size 2 of float): torque perturbation applied

    FSpan (array of size 2 of float): Timespan of the perturbation

    Noise (float) : variance of the noise in the system (all noise has the same variance)

    """
    fig,ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    X,Y = Feedback_Linearization(0.6,1e7,1e7,1e5,1e5,1e-5,1e-5,targets = [0,55],starting_point = [0,20],ForceField = Pert,ForceFieldSpan= FSpan,plot = True,Noise_Variance=Noise)
    ax.text(
        X[50]-15,
        50,
        "Feedback\n Linearization",
        color="red",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center",
    )
    if FSpan[0] != FSpan[1]:
        ax.text(
        -20,
        Y[int(FSpan[0]*1000)]+3,
        "Perturbation \n onset",
        color="black",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize = 8

        )
        plt.plot(np.linspace(-20,30,1000),np.ones(1000)*Y[int(FSpan[0]/0.001)],color = "black",alpha = .8)
        plt.xlim(-20,30)
        
    plt.savefig("img/FL1.png",dpi = 300)
    plt.show()

    _,ax = plt.subplots()
    plt.grid(alpha = .5)
    plt.xlabel("Time [sec]")
    plt.ylabel("Velocity of the Movement [cm/sec]")
    plt.plot(np.arange(0,0.598,0.001),(Y[1:]-Y[0:len(X)-1])/0.001)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig("img/FL2.png",dpi = 300)
    plt.show()
    
def plotSimpleMovements(Pert,FSpan,Noise,Feedback_Linearization,LQG,ILQG,K = 4000):
    fig,ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _,_ = LQG(0.6,1e6,1e6,1e6,1e6,1e-5,1e-5,targets = [0,55],starting_point = [0,20],ForceField = Pert,ForceFieldSpan= FSpan,plot = True,Noise_Variance=Noise)
    X,Y = Feedback_Linearization(0.6,1e8,1e8,1e5,1e5,1e-6,1e-6,targets = [0,55],starting_point = [0,20],ForceField = Pert,ForceFieldSpan= FSpan,plot = True,Noise_Variance=Noise)
    XILQG,YILQG,_ = ILQG(0.6,1e5,1e3,1e-3,K = K,targets = [0,55],start = [0,20],plot = True)
    #plt.legend()
    ax.text(
        8,
        50,
        "LQG",
        color="green",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        X[40]+1,
        42,
        "Feedback\n Linearization",
        color="red",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        XILQG[50]-5,
        50,
        "ILQG",
        color="blue",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center",
    )
    if FSpan[0] != FSpan[1]:
        ax.text(
        -20,
        Y[int(FSpan[0]*1000)]+3,
        "Perturbation \n onset",
        color="black",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize = 8

        )
        plt.plot(np.linspace(-20,30,1000),np.ones(1000)*Y[int(FSpan[0]/0.001)],color = "black",alpha = .8)
        plt.xlim(-20,30)
    plt.savefig("img/Presentation_1.png",dpi = 300)
    plt.show()

    fig,ax = plt.subplots()
    plt.grid(alpha = .5)
    plt.xlabel("Time [sec]")
    plt.ylabel("Velocity of the Movement [cm/sec]")
    plt.plot(np.arange(0,0.598,0.001),(Y[1:]-Y[0:len(X)-1])/0.001)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig("img/Presentation_2.png",dpi = 300)
    plt.show()



def ExploreMovements(Pert,FSpan,Noise,LQG,Feedback_Linearization):
    fig,ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _,_ = LQG(0.6,1e6,1e6,1e6,1e6,1e-5,1e-5,targets = [0,55],starting_point = [0,20],ForceField = Pert,ForceFieldSpan= FSpan,plot = True,Noise_Variance=Noise)
    X,Y = Feedback_Linearization(0.6,1e6,1e6,1e5,1e5,1e-5,1e-5,targets = [0,55],starting_point = [0,20],ForceField = Pert,ForceFieldSpan= FSpan,plot = True,Noise_Variance=Noise)
    #plt.legend()
    ax.text(
        10,
        50,
        "LQG",
        color="green",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        X[50]-15,
        50,
        "Feedback\n Linearization",
        color="red",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center",
    )
    if FSpan[0] != FSpan[1]:
        ax.text(
        -20,
        Y[20]+3,
        "Perturbation \n onset",
        color="black",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize = 8

        )
        plt.plot(np.linspace(-20,30,1000),np.ones(1000)*Y[20],color = "black",alpha = .8)
        plt.xlim(-20,30)
        ax.text(
        -20,
        Y[40]+3,
        "Perturbation \n offset",
        color="black",
        horizontalalignment="left",
        verticalalignment="center",
        fontsize = 8

        )
        plt.plot(np.linspace(-20,30,1000),np.ones(1000)*Y[40],color = "black",alpha = .8)
        plt.xlim(-20,30)
    plt.savefig("img/Explore.png",dpi = 300)
    plt.show()

    fig,ax = plt.subplots()
    plt.grid(alpha = .5)
    plt.xlabel("Time [sec]")
    plt.ylabel("Velocity of the Movement [cm/sec]")
    plt.plot(np.arange(0,0.598,0.001),(Y[1:]-Y[0:len(X)-1])/0.001)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig("img/Explore_2.png",dpi = 300)
    plt.show()


def NonlinearityImpact(MovementArray,DurationArray,Func,ylabel,LQG,Feedback_Linearization):
    SIZE = len(DurationArray)
    COLORLQG = "#F89D36"
    COLORFL = "#8D99AE"
    fig,ax = plt.subplots(figsize = (10,10))

    for MovementLength in MovementArray:

        max_dev_FL = np.zeros(SIZE)
        max_dev_LQG = np.zeros(SIZE)

        for idx,Duration in enumerate(DurationArray):

            XLQG,YLQG = LQG(Duration,1e6,1e6,1e6,1e6,1e-5,1e-5,targets = [0,20+MovementLength],starting_point = [0,20],plot=False,Noise_Variance=1e-18)
            XFL,YFL = Feedback_Linearization(Duration,1e6,1e6,1e5,1e5,1e-5,1e-5,targets = [0,20+MovementLength],starting_point = [0,20],plot=False,Noise_Variance=1e-18)
            max_dev_FL[idx] = Func(XFL,YFL)
            max_dev_LQG[idx] = Func(XLQG,YLQG)

        if MovementLength == MovementArray[0]: 
            plt.plot(DurationArray,max_dev_FL,color = COLORFL,label = "Nonlinear Controller",alpha = .3, linestyle = "--")
            plt.plot(DurationArray,max_dev_LQG,color = COLORLQG,label = "Linear Controller")
        
        else : 
            plt.plot(DurationArray,max_dev_LQG,color = COLORLQG)
            plt.plot(DurationArray,max_dev_FL,color = COLORFL,alpha = .3, linestyle = "--")
        ax.text(
            -0.03,
            max_dev_LQG[0],
            str(MovementLength)+" cm",
            color=COLORLQG,
            fontsize = 8,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.text(
            -0.03,
            max_dev_FL[0],
            str(MovementLength)+" cm",
            color=COLORFL,
            fontsize = 8,
            horizontalalignment="left",
            verticalalignment="center",
            alpha = .3
        )
    plt.title(ylabel + " of hand trajectories in function\n of movement time")
    plt.xlabel("Movement Time [seconds]")
    plt.ylabel(ylabel)

    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


    plt.legend()

def PlotTraj(X,Y,EnvironmentDynamics,kdelay,dt,starting_point,targets):
        #plt.grid(linestyle='--')
        #if FF : 
            #plt.plot(np.linspace(-10,10,100),np.ones(100)*FFonset,linestyle = "--",alpha = .7,color = "grey")
        plt.axis("equal")
        

        color = "cyan"
        ls = "-"
        lw = 3
        if len(EnvironmentDynamics.keys()) >0 :
            if EnvironmentDynamics["Side"] == "Right": 
                ls = "--"
                lw = 1
                color = "red"
            if EnvironmentDynamics["FF"] == False:
                ls = "-"
                lw = 1
                color = "grey"
        if kdelay > 0 and len(EnvironmentDynamics.keys()) >0 : 
            if EnvironmentDynamics["FF"] ==False : plt.plot(X,Y,color = color,label = "FL("+str(int(kdelay*dt*1000))+ " ms delay)",linewidth = lw,linestyle = ls)
            if EnvironmentDynamics["FF"] == True : plt.plot(X,Y,color = color,label = "FL("+str(int(kdelay*dt*1000))+ " ms delay)\n "+ str(EnvironmentDynamics["Side"])+" FF",linewidth = lw,linestyle = ls)
        else : plt.plot(X,Y,color = "#48494B",label = "Feedback\nLinearization",linewidth = 1)
        plt.xlabel("X [cm]")
        plt.ylabel("Y [cm]")
        #plt.scatter([starting_point[0],targets[0]],[starting_point[1],targets[1]],color = "orange",marker = "s" , s = 600, alpha= .3)
        MultipleLabel()
        # Remove the right and top axes
        


def PlotReachinginAllDirections(Feedback_Linearization,ILQG,K=60,Duration = .6,L=20,start = [0,35],w1 = 1e7,w2 = 1e4,r1 = 1e-4,Noise = True):
    plt.style.use('seaborn-darkgrid')  # Other options: 'ggplot', 'fivethirtyeight', 'bmh', etc
    fig, ax = plt.subplots(figsize = (8,8))
    NUMTARG = 16
    TARG = []

    theta = np.linspace(0,2*pi,NUMTARG,endpoint=False)
    for i in range(NUMTARG):
        TARG.append(start+np.array([L*cos(theta[i]),L*sin(theta[i])]))

    for i in range(NUMTARG):
        plt.scatter(TARG[i][0],TARG[i][1],color = "black",s = 300)
        #plt.text(TARG[i][0]-.5,TARG[i][1]-.8,str(i+1),color = "red",size =  10)
                #plt.plot(np.linspace(start[0],TARG[i][0]),np.linspace(start[1],TARG[i][1]),label = str(i+1))

        targets = TARG[i]
        xILQG,yILQG,_,_ = ILQG(Duration,w1,w2,r1,targets,K,start,Noise = Noise,plot = False)
        X,Y = Feedback_Linearization(Duration,w1,w1,w2,w2,r1,r1,targets,start,Num_iter = K,Activate_Noise=Noise,Delay=0,plot = False)
        plt.plot(xILQG,yILQG,label = "ILQG",color = (0.44,0.91,0.86))
        plt.plot(X,Y,label = "FL",color = (0.51,0.25,0.7))
        plt.axis("equal")

        MultipleLabel()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)