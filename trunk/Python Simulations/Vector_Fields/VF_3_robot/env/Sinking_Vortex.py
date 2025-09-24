import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.



# def sinking_vortex(x, y):
#     # Define center
#     x_center = 0
#     y_center = 0
    
#     # Calculate distance from center
#     x_rel = x - x_center
#     y_rel = y - y_center
    
#     # Calculate radial distance
#     r = np.sqrt(x_rel**2 + y_rel**2)
#     r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
#     # Rotational component (like vortex)
#     v_theta = r  # Angular velocity increases with radius
#     u_rot = +v_theta * y_rel / r
#     v_rot = -v_theta * x_rel / r
    
#     # Sink component (radial inflow)
#     sink_strength = -1.0  # Negative for inward flow
#     v_r = sink_strength * r  # Radial velocity
#     u_sink = v_r * x_rel / r
#     v_sink = v_r * y_rel / r
    
#     # Combine components
#     u = u_rot + u_sink
#     v = v_rot + v_sink
    
#     # Clip to prevent extreme values
#     u = np.clip(u, -100, 100)
#     v = np.clip(v, -100, 100)
    
#     return u, v

def sinking_vortex(x, y):
    """
    Sinking vortex field with spinning plate effect.
    Combines rotational vortex with radial sink component.
    """
    # Define center (matching the original parameters)
    center_x = 0.01
    center_y = 0.01
    
    # Calculate radius and angle from center
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Vortex component: "spinning plate" effect
    # Field strength increases with radius squared
    u_vortex = -np.sin(theta) * r**2
    v_vortex = np.cos(theta) * r**2
    
    # Sink component: radial inward flow
    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3  # small offset to prevent singularity
    
    # Sink field (inverse square law)
    u_sink = -x_centre / r2**2
    v_sink = -y_centre / r2**2
    
    # Combine components with weights from original
    # 0.4 weight for vortex, 0.15 weight for sink
    u = 0.4 * u_vortex + 0.15 * u_sink
    v = 0.4 * v_vortex + 0.15 * v_sink
    
    # Clip to prevent extreme values (adjusted range for this field)
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v