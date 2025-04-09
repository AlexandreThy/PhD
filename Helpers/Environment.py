from Helpers.Helpers import *

def EnvironmentDynamics(dic,x,acc):
    if len(dic.keys()) == 0: return [0,0]
    else :    
        if (np.sin(x[0]+x[1])*33+np.sin(x[0])*30 > dic["FFonset"]) and (dic["FF"] == True) and (dic["Direction"] == "up"):

            F = Compute_f_new_version(x[0:2],x[2:4],acc,.3)
            if dic["Side"] == "Left": F*=-1

        else : 
            F = [0,0]
    return F


def activated_ff(onset = 30,direction = "up",side = "left") :
    return {"FF":True,"FFonset":onset,"Direction":direction, "Side": side }