# env/grid_setup.py
import numpy as np

def make_environment_grid():
    x_start = -0.65
    y_start = -0.65
    x_end = 0.65
    y_end = 0.65
    quiver_step_size = .085
    
    x = np.arange(x_start, x_end, quiver_step_size)
    y = np.arange(y_start, y_end, quiver_step_size)
    X, Y = np.meshgrid(x, y)
    
    return X, Y