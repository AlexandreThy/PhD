from Helpers import *
from Controller import *

def plotFL(Pert,FSpan,Noise):
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
    
def plotSimpleMovements(Pert,FSpan,Noise,K = 4000):
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



def ExploreMovements(Pert,FSpan,Noise):
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


def NonlinearityImpact(MovementArray,DurationArray,Func,ylabel):
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