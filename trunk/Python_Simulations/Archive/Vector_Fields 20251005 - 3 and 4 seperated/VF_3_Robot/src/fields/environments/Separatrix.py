import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def separatrix(x,y):
    # First source
    x_source = 5
    y_source = 5
    x_centre = x - x_source
    y_centre = y - y_source
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)
    
    # First spring source
    u = x_centre / r**2
    v = y_centre / r**2

    # Second source
    x_source2 = 5.1
    y_source2 = 5.1
    x_centre2 = x - x_source2
    y_centre2 = y - y_source2
    r2 = np.sqrt(x_centre2**2 + y_centre2**2)
    r2 = np.where(r2 == 0, 1e-6, r2)
    
    # Second spring source (sink)
    u2 = -x_centre2 / r2**2
    v2 = -y_centre2 / r2**2

    # Combine fields
    u_total = u + u2
    v_total = v + v2

    # Calculate magnitude for normalization
    magnitude = np.sqrt(u_total**2 + v_total**2)
    
    # Avoid division by zero
    magnitude = np.where(magnitude < 1e-6, 1e-6, magnitude)
    
    # Normalize
    u_normalized = u_total / magnitude
    v_normalized = v_total / magnitude
    
    return u_normalized, v_normalized


