import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def sink(x, y):
    # Define sink center
    x_sink = 5
    y_sink = 5
    
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

