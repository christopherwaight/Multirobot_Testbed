import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def sink(x, y):
    # Define sink center
    x_sink = 0
    y_sink = 0
    
    # Calculate distance from center
    x_centre = x - x_sink
    y_centre = y - y_sink
    
    # Calculate radial distance
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Create inward flow, strength increases as you get closer to center
    strength = -1.0  # Negative for sink (positive would make a source)
    v_r = strength * r  # Radial velocity
    
    # Convert to cartesian coordinates
    u = -v_r * x_centre / r
    v = -v_r * y_centre / r
    
    # Clip to prevent extreme values
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v

def sink3(x, y):

    center_x = 0
    center_y = 0
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u = - np.sin(theta) * r**2
    v =   np.cos(theta) * r**2

    # Calculating A sink
    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-5

    # Sink
    u = -0.15*x_centre / r2**2
    v = -0.15*y_centre / r2**2

    return u, v