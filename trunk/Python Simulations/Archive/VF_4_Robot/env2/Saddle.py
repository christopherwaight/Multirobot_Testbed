import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def saddle(x, y):  
    
    x_center = 5.02
    y_center = 5.02
    
    # Calculate distance from center
    x_rel = x - x_center
    y_rel = y - y_center
    
    # Add uniform flow (stream) in positive x direction
    u_stream = .8
    v_stream = 0.0
    
    # Add source/spring component
    r = np.sqrt(x_rel**2 + y_rel**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    source_strength = 1
    u_source = source_strength * x_rel / r**2
    v_source = source_strength * y_rel / r**2
    
    # Combine all components
    u = u_stream + u_source
    v = v_stream + v_source
    
    # Clip to prevent extreme values
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v