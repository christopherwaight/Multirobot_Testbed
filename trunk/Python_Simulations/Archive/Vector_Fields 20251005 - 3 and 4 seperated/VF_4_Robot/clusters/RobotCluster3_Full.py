## Development on main algorithm continued, and now this file isn't compliant.
## Consider depricating


import numpy as np
import matplotlib.pyplot as plt
from env.VF_env import position_readings
#import control_primitives


class RobotCluster3_Full:

    def __init__(self):
        self.reset()

    def reset(self): # This configuration has these cluster variables: theta = 4pi/35 , p = q = 0.854
        
        # Initial Cluster Position and Orientation
        self.cluster_centre = [2.5, 1.7]
        phi_c = 0
        
        # Declare Desired Cluster Space Variables
        theta = 4*np.pi/35
        p = q = 0.854
        phi1 = phi_c
        phi2 = phi_c
        phi3 = phi_c
        
        # Initial Robot Positions - calculated using cluster space variables
        r = 0.5*np.sqrt(p**2 + q**2 - 2*p*q*np.cos(theta))
        s = np.sqrt(q**2 + r**2 - 2*q*r*np.cos(theta/2))
        self.robot_1_offset = [0, 2*s] 
        self.robot_2_offset = [q*np.sin(theta/2), -q*np.cos(theta/2)+2*s] 
        self.robot_3_offset = [-p*np.sin(theta/2), -p*np.cos(theta/2)+2*s] 
        #pos1, pos2, pos3 = self.pose()
        
        # Step Size for Movement
        self.step_size = 0.1                # I am not sure if this is the best place for this for now.
        self.angle = phi_c
         

    def pose(self): 
        pos1 = [self.cluster_centre[0] + self.robot_1_offset[0], self.cluster_centre[1] + self.robot_1_offset[1]]
        pos2 = [self.cluster_centre[0] + self.robot_2_offset[0], self.cluster_centre[1] + self.robot_2_offset[1]]
        pos3 = [self.cluster_centre[0] + self.robot_3_offset[0], self.cluster_centre[1] + self.robot_3_offset[1]]
        return pos1, pos2, pos3

    def plot(self):
        pos1, pos2, pos3 = self.pose()
        plt.scatter(pos1[0], pos1[1], marker='o', color='blue')
        plt.scatter(pos2[0], pos2[1], marker='o', color='yellow')
        plt.scatter(pos3[0], pos3[1], marker='o', color='green')
        #plt.show()
        
    def bot_readings(self):
        readings = []
        for pos in self.pose():
            readings.append(position_readings(pos[0], pos[1]))
        return readings
    
    def bot_compvec(self):
        mag = []
        angle = []
        for u, v in self.bot_readings():
            mag_bot = np.sqrt(u**2 + v**2)
            angle_bot = np.arctan2(v,u)
            mag.append(mag_bot)
            angle.append(angle_bot)
        return mag, angle
    
    def normal_angle(self):
        pos1, pos2, pos3 = self.pose()
        mag, angle = self.bot_compvec()

        poss1 = np.array([*pos1, mag[0]])
        poss2 = np.array([*pos2, mag[1]])
        poss3 = np.array([*pos3, mag[2]])

        R12 = poss2 - poss1
        R13 = poss3 - poss1
        Nx, Ny, _ = np.cross(-R12, R13)
        angle = np.arctan2(Ny, Nx)

        return Nx, Ny, angle
    
    def move(self, control_primitive):
        new_centre = control_primitive(self)
        self.cluster_centre = new_centre 
        self.pose()
        
    # Is there a way to dynamically reshape the cluster?
    # Is there a way to change how the pose variables are changed? incrementally?
    
    # def reshape(self):
    # # Novel idea, but It probably belongs under robot_cluster
    # pos1 = robot_1_start
    # [u1, v1] = position_readings(pos1[0],pos1[1])
    # pos2 = [pos1[0] + robot_offset, pos1[1] + robot_offset]
    # [u2, v2] = position_readings(pos2[0],pos2[1])
    # pos3 = [pos2[0] + robot_offset, pos2[1] + robot_offset]
    # [u3, v3] = position_readings(pos3[0],pos3[1])
    # [mag1, angle1] = composite_vector(u1,v1)
    # [mag2, angle2] = composite_vector(u2,v2)
    # [mag3, angle3] = composite_vector(u3,v3)
    # angle = (angle1+angle2+angle3)/3 +np.pi/2
    # pos1 = robot_1_start
    # pos2 = [pos1[0] + robot_offset*np.cos(angle),pos1[1] + robot_offset*np.sin(angle)]
    # pos3 = [pos2[0] + robot_offset*np.cos(angle),pos2[1] + robot_offset*np.sin(angle)]

    # return cluster.cluster_centre
