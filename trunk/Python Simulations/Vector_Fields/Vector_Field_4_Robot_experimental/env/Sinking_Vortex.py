import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.



def sinking_vortex(x, y):
    # Define center
    x_center = 5
    y_center = 5
    
    # Calculate distance from center
    x_rel = x - x_center
    y_rel = y - y_center
    
    # Calculate radial distance
    r = np.sqrt(x_rel**2 + y_rel**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Rotational component (like vortex)
    v_theta = r  # Angular velocity increases with radius
    u_rot = +v_theta * y_rel / r
    v_rot = -v_theta * x_rel / r
    
    # Sink component (radial inflow)
    sink_strength = -1.0  # Negative for inward flow
    v_r = sink_strength * r  # Radial velocity
    u_sink = v_r * x_rel / r
    v_sink = v_r * y_rel / r
    
    # Combine components
    u = u_rot + u_sink
    v = v_rot + v_sink
    
    # Clip to prevent extreme values
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v