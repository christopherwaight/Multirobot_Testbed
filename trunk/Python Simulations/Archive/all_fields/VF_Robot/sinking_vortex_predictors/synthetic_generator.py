import numpy as np
import pandas as pd
from scipy import ndimage

# Define Helper Functions
def vortex_vector_field(x, y, center_x, center_y):
    """Calculate vortex vector field with sinking component"""
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Spinning plate effect
    u = -np.sin(theta) * r**2
    v = np.cos(theta) * r**2
    
    # Sink component
    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3
    
    u1 = -x_centre / r2**2
    v1 = -y_centre / r2**2
    
    # Combine components (matching original weights)
    u = 0.4*u + 0.15*u1
    v = 0.4*v + 0.15*v1
    
    return u, v

def calculate_magnitude_direction(u, v):
    """Calculate magnitude and direction from vector components"""
    magnitude = np.sqrt(u**2 + v**2)
    direction = (np.arctan2(v, u) + np.pi) / (2 * np.pi)  # Normalize to 0-1
    return magnitude, direction

def generate_sobel_points(magnitude_field, x_grid, y_grid, n_points=4000):
    """Generate sampling points using Sobel edge detection on magnitude field"""
    # Apply Sobel filter to find edges/gradients
    sobel_x = ndimage.sobel(magnitude_field, axis=1)
    sobel_y = ndimage.sobel(magnitude_field, axis=0)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize and create probability distribution
    sobel_prob = sobel_magnitude.flatten()
    sobel_prob = sobel_prob / np.sum(sobel_prob)
    
    # Add uniform sampling component for better coverage
    uniform_prob = np.ones_like(sobel_prob) / len(sobel_prob)
    
    # Weight more heavily toward Sobel edges
    combined_prob = 0.8 * sobel_prob + 0.2 * uniform_prob
    combined_prob = combined_prob / np.sum(combined_prob)
    
    # Sample indices based on probability
    flat_indices = np.random.choice(
        len(sobel_prob), 
        size=n_points, 
        replace=True, 
        p=combined_prob
    )
    
    # Convert to 2D indices
    rows = flat_indices // magnitude_field.shape[1]
    cols = flat_indices % magnitude_field.shape[1]
    
    # Get x, y coordinates from grid
    x_points = x_grid.flatten()[flat_indices]
    y_points = y_grid.flatten()[flat_indices]
    
    return x_points, y_points

# Set random seed for reproducibility
np.random.seed(42)

# Create grid matching original parameters
image_pts = 2550
edge_tick = 0.5
x_grid, y_grid = np.meshgrid(
    np.linspace(-edge_tick, edge_tick, image_pts),
    np.linspace(-edge_tick, edge_tick, image_pts)
)

# Calculate vector field on grid
center_x, center_y = 0.01, 0.01
u_grid, v_grid = vortex_vector_field(x_grid, y_grid, center_x, center_y)
magnitude_grid, direction_grid = calculate_magnitude_direction(u_grid, v_grid)

# Normalize magnitude with offset (matching original code)
offset = 0.3
mag_min = np.min(magnitude_grid)
mag_max = np.max(magnitude_grid)
magnitude_normalized = (magnitude_grid - mag_min) / (mag_max - mag_min)
magnitude_scaled_grid = magnitude_normalized * (1 - offset) + offset  # This is 0.3 to 1.0 range

# Generate Sobel-weighted sampling points
print("Generating Sobel-weighted sample points...")
x_samples, y_samples = generate_sobel_points(magnitude_grid, x_grid, y_grid, n_points=4000)

# Generate data for CSV
data = []
n_points = len(x_samples)

print(f"Generating {n_points} data points with 3 variants each...")

for i in range(n_points):
    row = {}
    
    # Generate 3 related points (could represent temporal evolution or nearby samples)
    for j in range(1, 4):
        # Add small perturbation for variation
        # Increase perturbation with j to create progression
        noise_scale = 0.001 * j
        dx = np.random.normal(0, noise_scale)
        dy = np.random.normal(0, noise_scale)
        
        # Perturbed coordinates
        x_point = x_samples[i] + dx
        y_point = y_samples[i] + dy
        
        # Ensure points stay within bounds
        x_point = np.clip(x_point, -edge_tick, edge_tick)
        y_point = np.clip(y_point, -edge_tick, edge_tick)
        
        # Calculate field values at this point
        u_point, v_point = vortex_vector_field(x_point, y_point, center_x, center_y)
        mag_point, dir_point = calculate_magnitude_direction(u_point, v_point)
        
        # Calculate theta (angle in radians)
        theta_point = np.arctan2(v_point, u_point)
        
        # Normalize magnitude with same scaling as grid
        mag_norm = (mag_point - mag_min) / (mag_max - mag_min + 1e-10)
        sat_value = mag_norm * (1 - offset) + offset  # Saturation in 0.3-1.0 range
        sat_value = np.clip(sat_value, 0, 1)  # Ensure 0-1 range
        
        # Hue is direction (already 0-1)
        hue_value = dir_point  # Already normalized to 0-1
        
        # Store in row
        row[f'x{j}'] = x_point
        row[f'y{j}'] = y_point
        row[f'theta{j}'] = theta_point  # Radians (can be negative)
        row[f'hue{j}'] = hue_value      # 0-1 range
        row[f'sat{j}'] = sat_value      # 0-1 range
    
    data.append(row)
    
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{n_points} points...")

# Create DataFrame
df = pd.DataFrame(data)

# Reorder columns to match original format
column_order = []
for j in range(1, 4):
    column_order.extend([f'x{j}', f'y{j}', f'theta{j}', f'hue{j}', f'sat{j}'])
df = df[column_order]

# Limit to 3492 rows to match original
df = df.iloc[:3492]

# Save to CSV
df.to_csv('sinking_vortex_data1.csv', index=False)

# Print summary statistics
print(f"\nGenerated sinking vortex data with {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
print("\nData ranges:")
for col in df.columns:
    print(f"  {col:8s}: [{df[col].min():.4f}, {df[col].max():.4f}]")

print("\nKey statistics:")
print(f"  x range: [{df[[f'x{j}' for j in range(1,4)]].min().min():.4f}, {df[[f'x{j}' for j in range(1,4)]].max().max():.4f}]")
print(f"  y range: [{df[[f'y{j}' for j in range(1,4)]].min().min():.4f}, {df[[f'y{j}' for j in range(1,4)]].max().max():.4f}]")
print(f"  hue range: [{df[[f'hue{j}' for j in range(1,4)]].min().min():.4f}, {df[[f'hue{j}' for j in range(1,4)]].max().max():.4f}]")
print(f"  sat range: [{df[[f'sat{j}' for j in range(1,4)]].min().min():.4f}, {df[[f'sat{j}' for j in range(1,4)]].max().max():.4f}]")

print("\nData saved to 'sinking_vortex_data1.csv'")