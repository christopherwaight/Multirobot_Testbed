"""
Vortex Field Visualization - 2x2 Comparison

Loads NN, RBF, and analytical models to create a side-by-side comparison:
- Top-left: Raw training data (quiver plot)
- Top-right: NN predicted field (hue+sat colorwheel)
- Bottom-left: RBF predicted field (hue+sat colorwheel)
- Bottom-right: Analytical field (hue+sat colorwheel)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fields.environments.Vortex import vortex1


# ==================== NEURAL NETWORK ARCHITECTURE ====================

class FlexibleNN(nn.Module):
    """Flexible neural network with configurable architecture"""
    def __init__(self, input_size, hidden_sizes, output_size, activation='tanh'):
        super(FlexibleNN, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==================== RBF INTERPOLATOR ====================

class HueSatRBFInterpolator:
    """Wrapper for dual RBF interpolators (hue and saturation)"""
    def __init__(self, hue_rbf, sat_rbf):
        self.hue_rbf = hue_rbf
        self.sat_rbf = sat_rbf

    def predict(self, points):
        """Predict hue and saturation for given points"""
        hue = self.hue_rbf(points)
        sat = self.sat_rbf(points)
        return hue, sat


# ==================== DATA LOADING ====================

def grid_based_uniform_sample(df, grid_size=25, max_per_cell=1):
    """
    Fast grid-based uniform sampling for cleaner visualization.

    Args:
        df: DataFrame with x, y columns
        grid_size: Number of grid cells per dimension
        max_per_cell: Maximum points to keep per grid cell

    Returns:
        Indices of selected points
    """
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    # Create grid bins
    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)

    # Assign each point to a grid cell
    x_indices = np.digitize(df['x'].values, x_bins) - 1
    y_indices = np.digitize(df['y'].values, y_bins) - 1

    # Clip to valid range
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)

    # Group points by grid cell
    selected = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Find points in this cell
            mask = (x_indices == i) & (y_indices == j)
            cell_indices = np.where(mask)[0]

            if len(cell_indices) > 0:
                # Randomly select up to max_per_cell points
                n_select = min(max_per_cell, len(cell_indices))
                selected.extend(np.random.choice(cell_indices, n_select, replace=False))

    return selected


def load_training_data(filename='all_vortex_data_1.csv', max_points=500):
    """
    Load and process training data for visualization.

    Returns:
        df: DataFrame with columns [x, y, hue, sat]
    """
    print(f"Loading training data from {filename}...")
    df = pd.read_csv(filename)

    # Stack the three measurement sets
    stacked_data = []

    for i in range(len(df)):
        # First measurement set (negate x,y to match simulation coordinates)
        stacked_data.append([
            -df.iloc[i, 0],  # x1 (negated)
            -df.iloc[i, 1],  # y1 (negated)
            df.iloc[i, 3],   # hue1
            df.iloc[i, 4]    # sat1
        ])
        # Second measurement set
        stacked_data.append([
            -df.iloc[i, 5],  # x2 (negated)
            -df.iloc[i, 6],  # y2 (negated)
            df.iloc[i, 8],   # hue2
            df.iloc[i, 9]    # sat2
        ])
        # Third measurement set
        stacked_data.append([
            -df.iloc[i, 10], # x3 (negated)
            -df.iloc[i, 11], # y3 (negated)
            df.iloc[i, 13],  # hue3
            df.iloc[i, 14]   # sat3
        ])

    # Create stacked dataframe
    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])
    print(f"Stacked {len(df)} rows → {len(stacked_df)} data points")

    # Filter to domain
    stacked_df = stacked_df[(stacked_df['x'] > -0.6) & (stacked_df['x'] < 0.6) &
                            (stacked_df['y'] > -0.6) & (stacked_df['y'] < 0.6)].copy()
    print(f"After filtering to [-0.6, 0.6]: {len(stacked_df)} points")

    # Grid-based uniform sampling for cleaner visualization
    if len(stacked_df) > max_points:
        print(f"Applying grid-based uniform sampling...")
        selected_indices = grid_based_uniform_sample(stacked_df, grid_size=25, max_per_cell=1)
        stacked_df = stacked_df.iloc[selected_indices].reset_index(drop=True)
        print(f"Selected {len(stacked_df)} uniformly distributed points")

    return stacked_df


# ==================== MODEL LOADING ====================

def load_nn_models():
    """Load saved NN models and architecture info"""
    print("\nLoading NN models...")

    # Load architecture info
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    print(f"Hue architecture: {model_info['hue_architecture']}")
    print(f"Sat architecture: {model_info['sat_architecture']}")

    # Create models
    hue_model = FlexibleNN(
        input_size=2,
        hidden_sizes=model_info['hue_architecture']['hidden_sizes'],
        output_size=model_info['hue_architecture']['output_neurons'],
        activation=model_info['hue_architecture']['activation']
    )

    sat_model = FlexibleNN(
        input_size=2,
        hidden_sizes=model_info['sat_architecture']['hidden_sizes'],
        output_size=model_info['sat_architecture']['output_neurons'],
        activation=model_info['sat_architecture']['activation']
    )

    # Load weights
    hue_model.load_state_dict(torch.load('hue_model.pth'))
    sat_model.load_state_dict(torch.load('sat_model.pth'))

    hue_model.eval()
    sat_model.eval()

    # Load scaling info
    with open('scaling_info.pkl', 'rb') as f:
        scaling_info = pickle.load(f)

    print("✓ NN models loaded")
    return hue_model, sat_model, scaling_info


def load_rbf_interpolator():
    """Load saved RBF interpolator"""
    print("\nLoading RBF interpolator...")

    with open('vortex_rbf_interpolator.pkl', 'rb') as f:
        rbf_data = pickle.load(f)

    rbf = HueSatRBFInterpolator(rbf_data['hue_rbf'], rbf_data['sat_rbf'])

    print("✓ RBF interpolator loaded")
    return rbf


# ==================== FIELD EVALUATION ====================

def evaluate_nn_field(hue_model, sat_model, scaling_info, X_grid, Y_grid):
    """Evaluate NN predictions on a grid"""
    print("\nEvaluating NN field...")

    points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

    with torch.no_grad():
        # Get predictions
        hue_pred = hue_model(torch.FloatTensor(points)).numpy()
        sat_pred = sat_model(torch.FloatTensor(points)).squeeze().numpy()

    # Reconstruct hue from sin/cos
    hue_sin, hue_cos = hue_pred[:, 0], hue_pred[:, 1]
    hue_raw = (np.arctan2(hue_sin, hue_cos) + np.pi) / (2 * np.pi)

    # Rotate 90° CCW: subtract 0.25 (90° / 360°) from hue
    hue = (hue_raw - 0.25) % 1.0
    hue = np.clip(hue, 0, 1)

    # Unscale saturation (was scaled from [0,1] to [-1,1] during training)
    sat = (sat_pred + 1) / 2  # From [-1,1] back to [0,1]
    sat = np.clip(sat, 0, 1)

    hue_grid = hue.reshape(X_grid.shape)
    sat_grid = sat.reshape(X_grid.shape)

    print("✓ NN evaluation complete (rotated 90° CCW)")
    return hue_grid, sat_grid


def evaluate_rbf_field(rbf, X_grid, Y_grid):
    """Evaluate RBF interpolator on a grid"""
    print("\nEvaluating RBF field...")

    points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    hue_sincos = rbf.predict(points)

    # RBF returns (hue_sincos, sat) from the predict method
    # But actually the rbf interpolators return sin/cos directly
    # Let me call them directly
    hue_pred = rbf.hue_rbf(points)  # Returns (N, 2) with sin, cos
    sat_pred = rbf.sat_rbf(points)  # Returns (N,) or (N, 1)

    # Reconstruct hue from sin/cos
    if hue_pred.ndim == 2:
        hue_sin, hue_cos = hue_pred[:, 0], hue_pred[:, 1]
    else:
        # If it's 1D, it's already the hue value
        hue = hue_pred

    if hue_pred.ndim == 2:
        hue_raw = (np.arctan2(hue_sin, hue_cos) + np.pi) / (2 * np.pi)
        # Rotate 90° CCW: subtract 0.25 from hue
        hue = (hue_raw - 0.25) % 1.0

    hue = np.clip(hue, 0, 1)

    # Process saturation
    if sat_pred.ndim == 2:
        sat = sat_pred[:, 0]
    else:
        sat = sat_pred

    sat = np.clip(sat, 0, 1)

    hue_grid = hue.reshape(X_grid.shape)
    sat_grid = sat.reshape(X_grid.shape)

    print("✓ RBF evaluation complete (rotated 90° CCW)")
    return hue_grid, sat_grid


def evaluate_analytical_field(X_grid, Y_grid):
    """Evaluate analytical vortex1 field and convert to hue/saturation"""
    print("\nEvaluating analytical field...")

    # Get vector field components
    U, V = vortex1(X_grid, Y_grid)

    # Rotate 180 degrees: (u, v) → (-u, -v)
    U_rot = -U
    V_rot = -V

    # Convert to hue (direction)
    hue = (np.arctan2(V_rot, U_rot) + np.pi) / (2 * np.pi)
    hue = np.clip(hue, 0, 1)

    # Convert to saturation (magnitude)
    magnitude = np.sqrt(U_rot**2 + V_rot**2)
    sat = magnitude / (np.max(magnitude) + 1e-10)  # Normalize to [0,1]
    sat = np.clip(sat, 0, 1)

    print("✓ Analytical evaluation complete (rotated 180°)")
    return hue, sat


# ==================== VISUALIZATION ====================

def create_2x2_visualization(df, hue_nn, sat_nn, hue_rbf, sat_rbf, hue_analytical, sat_analytical, X_grid, Y_grid):
    """Create 2x2 comparison figure"""
    print("\nCreating 2×2 visualization...")

    fig = plt.figure(figsize=(16, 16))

    # ========== PANEL 1: Raw Training Data (Quiver) ==========
    ax1 = plt.subplot(2, 2, 1)

    # Convert hue/sat to vectors for quiver plot
    angles = df['hue'].values * 2 * np.pi  # Convert hue to angle
    magnitudes = df['sat'].values

    U_data = magnitudes * np.cos(angles)
    V_data = magnitudes * np.sin(angles)

    # Color by hue
    colors_data = df['hue'].values

    quiv1 = ax1.quiver(df['x'].values, df['y'].values, U_data, V_data, colors_data,
                       cmap='hsv', scale=8, alpha=0.7, width=0.003)
    ax1.set_title('Raw Training Data', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.65, 0.65)
    ax1.set_ylim(-0.65, 0.65)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # ========== PANEL 2: NN Field (Hue+Sat Colorwheel) ==========
    ax2 = plt.subplot(2, 2, 2)

    # Convert hue/sat to RGB
    hsv_nn = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))
    hsv_nn[:, :, 0] = hue_nn
    hsv_nn[:, :, 1] = sat_nn
    hsv_nn[:, :, 2] = 1.0  # Full brightness
    rgb_nn = hsv_to_rgb(hsv_nn)

    ax2.imshow(rgb_nn, extent=[-0.65, 0.65, -0.65, 0.65], origin='lower', aspect='auto')
    ax2.set_title('Neural Network Field', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.65, 0.65)
    ax2.set_ylim(-0.65, 0.65)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # ========== PANEL 3: RBF Field (Hue+Sat Colorwheel) ==========
    ax3 = plt.subplot(2, 2, 3)

    # Convert hue/sat to RGB
    hsv_rbf = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))
    hsv_rbf[:, :, 0] = hue_rbf
    hsv_rbf[:, :, 1] = sat_rbf
    hsv_rbf[:, :, 2] = 1.0
    rgb_rbf = hsv_to_rgb(hsv_rbf)

    ax3.imshow(rgb_rbf, extent=[-0.65, 0.65, -0.65, 0.65], origin='lower', aspect='auto')
    ax3.set_title('RBF Interpolator Field', fontsize=14, fontweight='bold')
    ax3.set_xlim(-0.65, 0.65)
    ax3.set_ylim(-0.65, 0.65)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # ========== PANEL 4: Analytical Field (Hue+Sat Colorwheel) ==========
    ax4 = plt.subplot(2, 2, 4)

    # Convert hue/sat to RGB
    hsv_analytical = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))
    hsv_analytical[:, :, 0] = hue_analytical
    hsv_analytical[:, :, 1] = sat_analytical
    hsv_analytical[:, :, 2] = 1.0
    rgb_analytical = hsv_to_rgb(hsv_analytical)

    ax4.imshow(rgb_analytical, extent=[-0.65, 0.65, -0.65, 0.65], origin='lower', aspect='auto')
    ax4.set_title('Analytical Field', fontsize=14, fontweight='bold')
    ax4.set_xlim(-0.65, 0.65)
    ax4.set_ylim(-0.65, 0.65)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3, color='white', linewidth=0.5)

    plt.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.0)

    # Save figure
    output_path = 'field_comparison_2x2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.show()


# ==================== MAIN ====================

def main():
    print("=" * 80)
    print("VORTEX FIELD VISUALIZATION - 2×2 COMPARISON")
    print("=" * 80)

    # Change to script directory to find data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}\n")

    # Load training data
    df = load_training_data('all_vortex_data_1.csv', max_points=500)

    # Load models
    hue_model, sat_model, scaling_info = load_nn_models()
    rbf = load_rbf_interpolator()

    # Create evaluation grid
    print("\nCreating evaluation grid...")
    x_grid = np.linspace(-0.65, 0.65, 100)
    y_grid = np.linspace(-0.65, 0.65, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    print("✓ Grid created (100×100)")

    # Evaluate all fields
    hue_nn, sat_nn = evaluate_nn_field(hue_model, sat_model, scaling_info, X_grid, Y_grid)
    hue_rbf, sat_rbf = evaluate_rbf_field(rbf, X_grid, Y_grid)
    hue_analytical, sat_analytical = evaluate_analytical_field(X_grid, Y_grid)

    # Create visualization
    create_2x2_visualization(df, hue_nn, sat_nn, hue_rbf, sat_rbf,
                            hue_analytical, sat_analytical, X_grid, Y_grid)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
