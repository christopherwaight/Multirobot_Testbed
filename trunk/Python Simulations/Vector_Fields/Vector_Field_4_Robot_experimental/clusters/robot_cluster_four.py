import numpy as np
import matplotlib.pyplot as plt
#from env.Fixed_Vortex import position_readings


class RobotCluster:

    def __init__(self, environment_function):
        self.environment_function = environment_function
        self.reset()

    def reset(self): 
        off_size = 1
        
        # Initialize cluster center as numpy array
        self.cluster_centre = np.random.rand(2)*.01 + 3.5
        

        self.robot_offsets = np.array([
            [0, off_size],                   # robot 1
            [- off_size*np.sqrt(1/3), 0],    # robot 2
            [ off_size*np.sqrt(1/3), 0],     # robot 3
            [0, -off_size]                    # robot 4
        ])
        
        # Calculate average offset
        average = np.mean(self.robot_offsets, axis=0)
        
        # Adjust positions
        self.cluster_centre += average
        self.robot_offsets -= average
        
        # Other parameters
        self.angle = np.pi 
        self.step_size = 0.05       
     

    def pose(self):
        positions = self.cluster_centre + self.robot_offsets
        return positions[0], positions[1], positions[2], positions[3]
    
    def move(self, control_primitive):
        new_centre = control_primitive(self)
        self.cluster_centre = new_centre 
        self.pose()

    def plot(self):
        pos1, pos2, pos3, pos4 = self.pose()
        plt.scatter(pos1[0], pos1[1], marker='o', color='blue')
        plt.scatter(pos2[0], pos2[1], marker='o', color='yellow')
        plt.scatter(pos3[0], pos3[1], marker='o', color='green')
        plt.scatter(pos4[0], pos4[1], marker='o', color='pink')
        #plt.show()

    def plot_centre(self):
        plt.plot([self.cluster_centre[0], self.cluster_centre[0]], [self.cluster_centre[1], self.cluster_centre[1]], marker='o', color='black')

        
    def bot_readings(self):
        positions = self.pose()
        return [self.environment_function(pos[0], pos[1]) for pos in positions]
        
    def bot_compvec(self):
        # Convert readings to numpy array
        readings = np.array(self.bot_readings())
        # Calculate magnitude and angle for all readings at once
        mag = np.sqrt(readings[:, 0]**2 + readings[:, 1]**2)
        angle = np.arctan2(readings[:, 1], readings[:, 0])
        return mag, angle
    
    def normal_angle(self):
        pos1, pos2, pos3, pos4 = self.pose()
        mag, angle = self.bot_compvec()

    
        poss1 = np.array([*pos1, mag[0]])
        poss2 = np.array([*pos2, mag[1]])
        poss3 = np.array([*pos3, mag[2]])

        R12 = poss2 - poss1
        R13 = poss3 - poss1
        Nx, Ny, _ = np.cross(-R12, R13)
        angle = np.arctan2(Ny, Nx)
        if angle < 0:
            angle += 2*np.pi

        return Nx, Ny, angle
