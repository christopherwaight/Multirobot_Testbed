import numpy as np
import matplotlib.pyplot as plt

# 5. Vector Field Features

def uni_xflow(x, y): # uniform x flow
    u = np.ones_like(x)  
    v = np.zeros_like(y)
    return u, v

def boundary_layer(x, y): # uniform x flow but kinda warped.
    u = np.exp(-y*0.1*x)
    v = np.zeros_like(y)
    return u, v

def uni_yflow(x, y): # uniform y flow
    u = np.zeros_like(x)  
    v = np.ones_like(y)
    return u, v

def source(x, y, x_source, y_source):
    
    # Calculating A Source
    x_centre = x - x_source
    y_centre = y - y_source
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)

    # spring source
    u = x_centre / r**2
    v = y_centre / r**2

    return u, v

def sink(x, y, x_source, y_source):
    
    # Calculating A sink
    x_centre = x - x_source
    y_centre = y - y_source
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)

    # spring source
    u = -x_centre / r**2
    v = -y_centre / r**2

    return u, v

def free_vortex(x, y, x_source, y_source):
       
    x_centre = x - x_source
    y_centre = y - y_source
 
    r = np.sqrt(x_centre**2 + y_centre**2)  # Radial Distance from Centre
    r = np.where(r == 0, 1e-6, r)
    v_theta = 1 / r # Calculate the angular component of the velocity

    # Convert to cartesian coordinates
    u = +v_theta * y_centre / r  
    v = -v_theta * x_centre / r 

    return u, v

def fixed_vortex(x, y, x_source, y_source):
       
    x_centre = x - x_source
    y_centre = y - y_source
 
    r = np.sqrt(x_centre**2 + y_centre**2)  # Radial Distance from Centre
    r = np.where(r == 0, 1e-6, r)
    v_theta = r # Calculate the angular component of the velocity

    # Convert to cartesian coordinates
    u = +v_theta * y_centre / r  
    v = -v_theta * x_centre / r 

    return u, v

def doublet(x, y, x_source, y_source):
    
    # Sinkside
    x_centre = x - x_source
    y_centre = y - y_source
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)
    u = -x_centre / r**2
    v = -y_centre / r**2
    
    # Sourceside
    x_centre = x - x_source+0.0001
    y_centre = y - y_source+0.0001
#    x_centre = x - x_source+0.1
#    y_centre = y - y_source+0.1
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)
    u += x_centre / r**2
    v += y_centre / r**2

    return u, v


def rankine(x, y, x_source, y_source):
    # Convert inputs to arrays with np.atleast_1d to ensure the function works for both scalars and arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    x_centre = x - x_source
    y_centre = y - y_source
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, np.nan, r)  # Replace 0 with NaN to avoid division by zero

    eyerad = 2
    
    v_theta = eyerad*eyerad / r
    mask = r < eyerad
    v_theta[mask] = r[mask]

    u = v_theta * y_centre / r
    v = -v_theta * x_centre / r

    # Handle scalar input by reducing the output to scalar if the input was scalar
    if x.shape == (1,) and y.shape == (1,):
        return u[0], v[0]  # Return as scalars if input was scalar
    return u, v  # Return as arrays if input was arrays



def saddle(x, y, x_saddle, y_saddle): # I know, Saddle isn't technically a basis.
    
    # Calculating A Source
    x_centre = x - x_saddle
    y_centre = y - y_saddle
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)
    
    # Calculate the angular component of the velocity
    v_theta = 1 / r 

    # Convert to cartesian coordinates
    u = v_theta * y_centre / r
    v = v_theta * x_centre / r

    return u, v

def corner_flow(x,y,alpha): #Specialized geometric solution


    # Calculate r, theta
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Avoid division by zero at the center
    r[r==0] = 1e-6

    # Calculate velocity components in polar coordinates
    u_r = alpha * r**(alpha - 1) * np.cos(alpha * theta)
    u_theta = -alpha * r**(alpha - 1) * np.sin(alpha * theta)

    # Convert to cartesian coordinates
    u = -(u_r * np.cos(theta) - u_theta * np.sin(theta))
    v = -(u_r * np.sin(theta) + u_theta * np.cos(theta))

    return u, v

def field_noise(x,y): # Adding some static environmental noise

 
   
    np.random.seed(123)
    u = np.random.randn(*x.shape)
    
    np.random.seed(124)
    v = np.random.randn(*y.shape)


    return u, v