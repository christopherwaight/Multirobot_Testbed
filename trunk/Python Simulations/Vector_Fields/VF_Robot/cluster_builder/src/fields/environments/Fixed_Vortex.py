import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def fixed_vortex(x,y): # Fill out the Vector Field Below
    
    #print("Input shapes to fixed_vortex:", np.shape(x), np.shape(y))

    x_source = 5
    y_source = 5


    x_centre = x - x_source
    y_centre = y - y_source
 
    r = np.sqrt(x_centre**2 + y_centre**2)  # Radial Distance from Centre
    r = np.where(r == 0, 1e-6, r)
    v_theta = r # Calculate the angular component of the velocity

    # Convert to cartesian coordinates
    u = -v_theta * y_centre/ r  
    v = v_theta * x_centre / r

    # # Normalize vectors to unit length
    # magnitude = np.sqrt(u**2 + v**2)
    # magnitude = np.where(magnitude == 0, 1e-6, magnitude)  # Avoid division by zero
    
    # u = u / -magnitude
    # v = v / -magnitude



    # Clip to prevent anything too large
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v


