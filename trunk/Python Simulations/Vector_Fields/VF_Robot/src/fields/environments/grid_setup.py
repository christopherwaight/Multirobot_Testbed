# env/grid_setup.py
import numpy as np

def make_environment_grid(field_type=None):
    """
    Create a grid for field visualization.
    
    Args:
        field_type: 'scalar' for larger grid, None/'vector' for standard grid
    
    Returns:
        (X, Y): Meshgrid arrays
    """
    if field_type == 'scalar':
        x_start = -2.5
        y_start = -2.5
        x_end = 2.5
        y_end = 2.5
        grid_step = 0.25  # Smaller step for better contour resolution
    else:
        x_start = -0.65
        y_start = -0.65
        x_end = 0.65
        y_end = 0.65
        grid_step = 0.15  # Fewer arrows for cleaner visualization

    x = np.arange(x_start, x_end, grid_step)
    y = np.arange(y_start, y_end, grid_step)
    X, Y = np.meshgrid(x, y)

    return X, Y
