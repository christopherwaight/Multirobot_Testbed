import numpy as np
import matplotlib.pyplot as plt
from env.VF_env import position_readings
from itertools import combinations
#import random

class RobotClusterAll:

    def __init__(self, numbots,signal_noise):
        self.reset(numbots)
        self.signal_noise = signal_noise
        

    def reset(self,numbots): # This configuration has these cluster variables: theta = 4pi/35 , p = q = 0.854
        

        off_size = 0.4
        randompoint = np.array(np.random.uniform(0,5,2))
        print('Self cluster centre at initialization', randompoint)

        self.numbots = numbots
        self.robot_offsets = np.random.uniform(-1,1,(self.numbots,2)) # Can be imported seperately
        average = np.array(np.mean(self.robot_offsets, axis=0))
        print('Average offsets of all robots', average)

        self.cluster_centre = average + randompoint
        print('New cluster centre', self.cluster_centre)
        print()
        self.robot_offsets -= average #.reshape(2,1)

        self.step_size = 0.125                      
        self.angle = np.pi 

    def pose(self):

        self.positions = self.cluster_centre + self.robot_offsets
        return self.positions

    def plot(self):
        positions = self.pose()

        colors = ['blue', 'yellow', 'green', 'red']
        
        for i in range(self.numbots):
            plt.scatter(positions[i, 0], positions[i, 1], marker='o', color=colors[i % len(colors)])


    def plot_centre(self):
        plt.scatter(self.cluster_centre[0], self.cluster_centre[1], marker='o', color='black')

        
    def bot_readings(self):
        readings = [] # A list
        for i in range(self.numbots):
            readings.append( position_readings(self.positions[i,0], self.positions[i,1],self.signal_noise) )
        
        readings = np.array(readings).reshape(self.numbots,2) # Reshapes the list

        return readings # 2 columns, num_robot rows
    
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

        # Defining helper subfunctions
        def compute_normal_vector(p1, p2, p3):
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(-v1, v2)
            return normal
        
        # Function to calculate cosine similarity between two vectors
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            return dot_product / (norm_vec1 * norm_vec2)

        # Function to calculate the cross product in 2D
        def cross_product_2d(vec1, vec2):
            return vec1[0] * vec2[1] - vec1[1] * vec2[0]

        # Now unward with the logic
        positions = self.pose()
        mag, angle = self.bot_compvec()


        #print('positions ||', positions)
        #print('mag', mag)
        points = np.array([
            [positions[i, 0], positions[i, 1], mag[i]] for i in range(self.numbots)
        ])


    
        # List of combinations to use for cross products
        combinations_all = list(combinations(range(self.numbots), 3)) # saves as a list, then as an array
        combinations_all = np.array(combinations_all)
        #print('combinations all', combinations_all)


        # Initialize lists to store Nx and Ny values
        Nx_values = []
        Ny_values = []

        # Loop through each combination and compute the normal vectors
        for comb in combinations_all:
            i, j, k = comb

            array = np.array([points[i], points[j], points[k]])
            #print("This is my desired array", array)
            Xc, Yc, _ = np.mean(array, axis=0)
            #print("Xc and Yc", Xc, Yc)

            point_1_index = np.round(np.random.uniform(0,1,1)* 2)
            point_1 = array[int(point_1_index)]

            # Step 4 & 5: Calculate cosine similarity for each point with point 1
            cosine_similarities = []
            vector_from_center_to_point_1 = np.array([Xc, Yc]) - np.array([point_1[0], point_1[1]])

            for i, point in enumerate(array):
                vector_to_point_1 = point - point_1
                similarity = cosine_similarity(vector_to_point_1[:2], vector_from_center_to_point_1)
                direction = cross_product_2d(vector_from_center_to_point_1, vector_to_point_1)
                direction_str = "ccw" if direction > 0 else "cw" if direction < 0 else "col"
                cosine_similarities.append((i, point, similarity, direction_str))

            # Step 6: Separate points by direction
            collinear = [p for p in cosine_similarities if p[3] == "col"]
            ccw = [p for p in cosine_similarities if p[3] == "ccw"]
            cw = [p for p in cosine_similarities if p[3] == "cw"]

            # Sort ccw points in ascending order and cw points in descending order
            ccw.sort(key=lambda x: x[2], reverse=True)
            cw.sort(key=lambda x: x[2], reverse=False)

            # Combine sorted lists
            sorted_points = collinear + cw + ccw

            new_indices = {}
            new_index = 1
            for p in sorted_points:
                old_idx = p[0]
                if old_idx == point_1_index:
                    new_indices[old_idx] = 0
                else:
                    new_indices[old_idx] = new_index
                    new_index += 1

            # Step 7: Output the array of points with their cosine similarity scores and new indices
            output = np.array([( p[1][0], p[1][1]) for p in sorted_points])
            
            new_output = []

            # Iterate through each row in the output array
            for row in output:
                # Find the matching row in the original array
                matching_row = array[np.all(array[:, :2] == row, axis=1)]
                
                # Append the third element of the matching row to the current row
                if matching_row.size > 0:
                    new_row = np.append(row, matching_row[0, 2])
                    new_output.append(new_row)

            # Convert the list of new rows back to a NumPy array
            new_o = np.array(new_output)
            
            #print(array)
            #print(new_output)

            #normal = compute_normal_vector(points[i], points[j], points[k])
            normal = compute_normal_vector(new_o[0], new_o[1], new_o[2])
            
            Nx_values.append(normal[0])
            Ny_values.append(normal[1])

        Ny = np.average(Ny_values)
        Nx = np.average(Nx_values)
        angle = np.arctan2(Ny, Nx)
        if angle < 0:
            angle += 2*np.pi

        return Nx, Ny, angle


    def move(self, control_primitive):
        new_centre = control_primitive(self)
        self.cluster_centre = new_centre 
        self.pose()
        