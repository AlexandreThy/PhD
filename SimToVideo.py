import pyvista as pv
from pyvista import Cylinder, Sphere
import numpy as np
from IPython.display import Video, display

def Create_Arm_Mesh():
    upper_arm_length = 0.3  # Longueur du bras supérieur
    forearm_length = 0.33  # Longueur de l'avant-bras
    arm_radius = 0.05  # Rayon du bras
    joint_radius = 0.07

    # Créer le cylindre pour le bras supérieur
    upper_arm = Cylinder(center=(upper_arm_length/2, 0, 0), direction=(1, 0, 0), 
                        radius=arm_radius, height=upper_arm_length,capping=True)

    # Créer le cylindre pour l'avant-bras
    forearm = Cylinder(center=(upper_arm_length+forearm_length/2, 0, 0), direction=(1, 0, 0), 
                    radius=arm_radius, height=forearm_length,capping=True)

    # Créer une sphère pour l'épaule
    shoulder = Sphere(center=(0, 0, 0), radius=joint_radius)

    # Créer une sphère pour le coude
    elbow = Sphere(center=(upper_arm_length, 0, 0), radius=joint_radius)

    # Fusionner les géométries pour former un maillage unifié
    arm_mesh = upper_arm + forearm + shoulder + elbow
    
    return arm_mesh,[forearm,upper_arm,shoulder,elbow]

# Animation parameters

  # Elbow position

    

def Create_Plotter(filename = "test.mp4",frame_rate=60):
    plotter = pv.Plotter()
    plotter.open_movie(filename=filename,framerate = frame_rate)
    plotter.camera_position = [
    (-1,-1,1),  # Camera location (farther from the origin)
    (.3, 0, 0),  # Focal point (center of the scene)
    (0, 0, 1),  # View up vector
    ]
    plotter.show_axes_all()
    return plotter

def Angles_Diff(Angles,iterate):
    if iterate == 0:
        return Angles[0,0]-np.pi/2,Angles[1,0]
    else : return Angles[0,iterate]-Angles[0,iterate-1],Angles[1,iterate]-Angles[1,iterate-1]

def Rotate(rotation_center,diff_Angles):
    return [rotation_center[0]*np.cos(diff_Angles)-rotation_center[1]*np.sin(diff_Angles),rotation_center[0]*np.sin(diff_Angles)+rotation_center[1]*np.cos(diff_Angles),0]

def Cartesian_Point(thetas,thetae,index):
    return (.3*np.cos(thetas[index])+.33*np.cos(thetas[index]+thetae[index]),.3*np.sin(thetas[index])+.33*np.sin(thetas[index]+thetae[index]),0)

def Get_All_Cartesian(thetas,thetae,max_i = np.inf):
    if max_i > len(thetas) : max_i = len(thetas)
    if max_i == 0 : max_i = 1
    x,y,z = np.zeros(max_i),np.zeros(max_i),np.zeros(max_i)
    for i in range(max_i):
        x[i],y[i],z[i] = Cartesian_Point(thetas,thetae,i)
    return np.column_stack((x, y, z))

def Video_Creation(plotter,arm_mesh,parts,Angles):
    End_Pos = Cartesian_Point(Angles[0],Angles[1],-1)
    Start_Pos = Cartesian_Point(Angles[0],Angles[1],0)
    upper_arm_length = 0.3  # Longueur du bras supérieur
    rotation_center = [upper_arm_length, 0, 0]
    for i in range(len(Angles[0])):
        dThetas,dThetae = Angles_Diff(Angles,i)
        for j in range(len(parts)):
            parts[j] = parts[j].rotate_z(dThetas/2/np.pi*360,point = [0,0,0])
        rotation_center = Rotate(rotation_center,dThetas)
        parts[0] = parts[0].rotate_z(dThetae/2/np.pi*360, point=rotation_center)

        arm_mesh = pv.PolyData()
        for j in range(len(parts)):
            arm_mesh+=parts[j]
        cube = pv.Cube(center = End_Pos,
                          x_length=.1,y_length=.1,z_length=.1).rotate_z(-90,point = [0,0,0])
        plotter.clear() 
        points = Get_All_Cartesian(Angles[0],Angles[1],max_i=i)
        curve = pv.PolyData(points)
        curve.lines = np.array([len(points)] + list(range(len(points))))  # Define the connectivity
        curve = curve.rotate_z(-90,point = [0,0,0])
        
        plotter.add_mesh(arm_mesh, color="lightblue", show_edges=True)
        plotter.add_mesh(cube, color = "red", show_edges=True)
        plotter.add_mesh(curve, color = "green",line_width = 10)

        # Write current frame to the movie
        plotter.write_frame()
    plotter.close()

def Produce_ArmVideo(filename,Angles):  
    plotter= Create_Plotter(filename)
    arm_mesh,parts = Create_Arm_Mesh()
    Video_Creation(plotter,arm_mesh,parts,Angles)
    display(Video(filename=filename, html_attributes="controls muted autoplay",height=600,width=600))