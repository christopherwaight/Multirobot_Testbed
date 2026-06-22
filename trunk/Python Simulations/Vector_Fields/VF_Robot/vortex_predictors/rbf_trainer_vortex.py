"""
RBF (Radial Basis Function) Interpolation for Vector Field Approximation
This script processes vortex data and creates RBF interpolators for hue and saturation
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pickle
import os
from time import time

# ==================== RBF CONFIGURATION ====================
RBF_CONFIG = {
    'kernel': 'thin_plate_spline',  # Options: 'linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic', 'gaussian'
    'smoothing': 0.0,  # 0 for exact interpolation, increase for smoothing
    'degree': None,  # Polynomial degree (-1 for no polynomial)
    'neighbors': None,  # Use all points (None) or limit to N nearest neighbors
    'epsilon': None  # Shape parameter for some kernels (auto-determined if None)
}

# Alternative configurations you might want to try:
ALTERNATIVE_CONFIGS = {
    'smooth_rbf': {
        'kernel': 'gaussian',
        'smoothing': 0.001,
        'epsilon': 1.0
    },
    'local_rbf': {
        'kernel': 'multiquadric',
        'smoothing': 0.0,
        'neighbors': 50  # Only use 50 nearest neighbors
    }
}

# ==================== DATA PROCESSING (Reused from NN script) ====================

def load_and_stack_data(filename='all_vortex_data_possible.csv'):
    """Load CSV and stack the three measurement sets into single columns"""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    stacked_data = []
    
    for i in range(len(df)):
        # First measurement set
        # Negate x and y to match simulation coordinate system
        stacked_data.append([
            -df.iloc[i, 0],  # x1 (negated)
            -df.iloc[i, 1],  # y1 (negated)
            df.iloc[i, 3],  # hue1
            df.iloc[i, 4]   # sat1
        ])
        # Second measurement set
        stacked_data.append([
            -df.iloc[i, 5],  # x2 (negated)
            -df.iloc[i, 6],  # y2 (negated)
            df.iloc[i, 8],  # hue2
            df.iloc[i, 9]   # sat2
        ])
        # Third measurement set
        stacked_data.append([
            -df.iloc[i, 10], # x3 (negated)
            -df.iloc[i, 11], # y3 (negated)
            df.iloc[i, 13], # hue3
            df.iloc[i, 14]  # sat3
        ])
    
    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])
    print(f"Original rows: {len(df)}, Stacked rows: {len(stacked_df)}")
    
    return stacked_df

def filter_and_deduplicate(df, x_range=(-0.6, 0.6), y_range=(-0.6, 0.6), 
                          tolerance=0.01, target_size=None):
    """Filter data to specified range and remove near-duplicates"""
    print(f"Filtering data to range: x in {x_range}, y in {y_range}")
    
    # Filter to specified range
    df_filtered = df[(df['x'] > x_range[0]) & (df['x'] < x_range[1]) & 
                     (df['y'] > y_range[0]) & (df['y'] < y_range[1])].copy()
    print(f"After filtering: {len(df_filtered)} points (from {len(df)})")
    
    print(f"Removing near-duplicates with tolerance {tolerance}...")
    
    # Round coordinates to remove near-duplicates
    df_filtered['x_rounded'] = np.round(df_filtered['x'] / tolerance) * tolerance
    df_filtered['y_rounded'] = np.round(df_filtered['y'] / tolerance) * tolerance
    
    # Remove duplicates based on rounded coordinates, averaging the values
    df_dedup = df_filtered.groupby(['x_rounded', 'y_rounded']).agg({
        'x': 'mean',
        'y': 'mean',
        'hue': 'mean',
        'sat': 'mean'
    }).reset_index(drop=True)
    
    print(f"After deduplication: {len(df_dedup)} points")
    
    # If target_size specified and we have too many points, randomly sample
    if target_size and len(df_dedup) > target_size:
        df_dedup = df_dedup.sample(n=target_size, random_state=42)
        print(f"Randomly sampled to {target_size} points")
    
    return df_dedup

# ==================== RBF INTERPOLATION ====================

class VortexRBFInterpolator:
    """RBF interpolator for vortex field data"""
    
    def __init__(self, config=None):
        """Initialize with given configuration"""
        self.config = config or RBF_CONFIG
        self.hue_rbf = None
        self.sat_rbf = None
        self.training_points = None
        self.training_hue = None
        self.training_sat = None
        self.fit_time = {}
        
    def fit(self, X, hue_values, sat_values):
        """
        Fit RBF interpolators for hue and saturation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Training points (x, y coordinates)
        hue_values : array-like, shape (n_samples,)
            Hue values at training points [0, 1]
        sat_values : array-like, shape (n_samples,)
            Saturation values at training points [0, 1]
        """
        print("\nFitting RBF interpolators...")
        print(f"Configuration: {self.config}")
        print(f"Training on {len(X)} points")

        # Convert to float32 for compatibility with pythranized scipy
        self.training_points = np.array(X, dtype=np.float32)
        self.training_hue = np.array(hue_values, dtype=np.float32)
        self.training_sat = np.array(sat_values, dtype=np.float32)
        
        # Convert hue to sin/cos representation for circular continuity
        hue_angle = self.training_hue * 2 * np.pi
        hue_sin = np.sin(hue_angle)
        hue_cos = np.cos(hue_angle)
        
        # Fit hue RBF (using sin/cos representation)
        print("Fitting hue interpolator (sin/cos representation)...")
        start_time = time()
        
        # Stack sin and cos as two separate outputs
        hue_targets = np.column_stack([hue_sin, hue_cos])
        
        self.hue_rbf = RBFInterpolator(
            self.training_points,
            hue_targets,
            kernel=self.config.get('kernel', 'thin_plate_spline'),
            smoothing=self.config.get('smoothing', 0.0),
            degree=self.config.get('degree', None),
            neighbors=self.config.get('neighbors', None),
            epsilon=self.config.get('epsilon', None)
        )
        
        self.fit_time['hue'] = time() - start_time
        print(f"Hue fitting completed in {self.fit_time['hue']:.2f} seconds")
        
        # Fit saturation RBF
        print("Fitting saturation interpolator...")
        start_time = time()
        
        self.sat_rbf = RBFInterpolator(
            self.training_points,
            self.training_sat.reshape(-1, 1),  # Reshape for single output
            kernel=self.config.get('kernel', 'thin_plate_spline'),
            smoothing=self.config.get('smoothing', 0.0),
            degree=self.config.get('degree', None),
            neighbors=self.config.get('neighbors', None),
            epsilon=self.config.get('epsilon', None)
        )
        
        self.fit_time['sat'] = time() - start_time
        print(f"Saturation fitting completed in {self.fit_time['sat']:.2f} seconds")
        
    def predict(self, X):
        """
        Predict hue and saturation at given points
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Points to predict at (x, y coordinates)
            
        Returns:
        --------
        hue_pred : array, shape (n_samples,)
            Predicted hue values [0, 1]
        sat_pred : array, shape (n_samples,)
            Predicted saturation values [0, 1]
        """
        X = np.array(X)
        
        # Predict hue (sin/cos)
        hue_sincos = self.hue_rbf(X)
        
        # Convert back from sin/cos to angle
        hue_angle = np.arctan2(hue_sincos[:, 0], hue_sincos[:, 1])
        hue_pred = (hue_angle + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        
        # Predict saturation
        sat_pred = self.sat_rbf(X).flatten()
        
        # Clip to valid ranges
        sat_pred = np.clip(sat_pred, 0, 1)
        
        return hue_pred, sat_pred
    
    def evaluate_at_training_points(self):
        """Check how well we reproduce training data"""
        if self.training_points is None:
            raise ValueError("Model not fitted yet!")
            
        hue_pred, sat_pred = self.predict(self.training_points)
        
        # Calculate errors
        hue_error = np.abs(hue_pred - self.training_hue)
        # Handle circular nature of hue
        hue_error = np.minimum(hue_error, 1 - hue_error)
        
        sat_error = np.abs(sat_pred - self.training_sat)
        
        return {
            'hue_mae': np.mean(hue_error),
            'hue_max_error': np.max(hue_error),
            'sat_mae': np.mean(sat_error),
            'sat_max_error': np.max(sat_error)
        }
    
    def save(self, filename='rbf_interpolator.pkl'):
        """Save the interpolator to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'training_points': self.training_points,
                'training_hue': self.training_hue,
                'training_sat': self.training_sat,
                'hue_rbf': self.hue_rbf,
                'sat_rbf': self.sat_rbf,
                'fit_time': self.fit_time
            }, f)
        print(f"Saved RBF interpolator to {filename}")
    
    @classmethod
    def load(cls, filename='rbf_interpolator.pkl'):
        """Load a saved interpolator"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        interpolator = cls(config=data['config'])
        interpolator.training_points = data['training_points']
        interpolator.training_hue = data['training_hue']
        interpolator.training_sat = data['training_sat']
        interpolator.hue_rbf = data['hue_rbf']
        interpolator.sat_rbf = data['sat_rbf']
        interpolator.fit_time = data['fit_time']
        
        return interpolator

# ==================== VISUALIZATION ====================

# def visualize_rbf_results(interpolator, df, test_points=None):
#     """Create comprehensive visualization of RBF results"""
    
#     fig = plt.figure(figsize=(15, 10))
    
#     # 1. Training data distribution
#     ax1 = plt.subplot(2, 3, 1)
#     scatter1 = ax1.scatter(df['x'], df['y'], c=df['hue'], cmap='hsv', s=1, alpha=0.6)
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_title(f'Training Data - Hue ({len(df)} points)')
#     ax1.set_aspect('equal')
#     plt.colorbar(scatter1, ax=ax1, label='Hue')
    
#     # 2. Training data - Saturation
#     ax2 = plt.subplot(2, 3, 2)
#     scatter2 = ax2.scatter(df['x'], df['y'], c=df['sat'], cmap='viridis', s=1, alpha=0.6)
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
#     ax2.set_title('Training Data - Saturation')
#     ax2.set_aspect('equal')
#     plt.colorbar(scatter2, ax=ax2, label='Saturation')
    
#     # 3. Interpolated field - Hue
#     ax3 = plt.subplot(2, 3, 3)
#     x_grid = np.linspace(-0.65, 0.65, 100)
#     y_grid = np.linspace(-0.65, 0.65, 100)
#     X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
#     points_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
#     hue_grid, sat_grid = interpolator.predict(points_grid)
#     hue_grid = hue_grid.reshape(X_grid.shape)
    
#     im3 = ax3.pcolormesh(X_grid, Y_grid, hue_grid, cmap='hsv', shading='auto')
#     ax3.set_xlabel('X')
#     ax3.set_ylabel('Y')
#     ax3.set_title('RBF Interpolated - Hue')
#     ax3.set_aspect('equal')
#     plt.colorbar(im3, ax=ax3, label='Hue')
    
#     # 4. Interpolated field - Saturation
#     ax4 = plt.subplot(2, 3, 4)
#     sat_grid = sat_grid.reshape(X_grid.shape)
    
#     im4 = ax4.pcolormesh(X_grid, Y_grid, sat_grid, cmap='viridis', shading='auto')
#     ax4.set_xlabel('X')
#     ax4.set_ylabel('Y')
#     ax4.set_title('RBF Interpolated - Saturation')
#     ax4.set_aspect('equal')
#     plt.colorbar(im4, ax=ax4, label='Saturation')
    
#     # 5. Error at training points
#     ax5 = plt.subplot(2, 3, 5)
#     errors = interpolator.evaluate_at_training_points()
    
#     # Create bar plot of errors
#     error_types = ['Hue MAE', 'Hue Max', 'Sat MAE', 'Sat Max']
#     error_values = [errors['hue_mae'], errors['hue_max_error'], 
#                    errors['sat_mae'], errors['sat_max_error']]
    
#     bars = ax5.bar(error_types, error_values)
#     ax5.set_ylabel('Error')
#     ax5.set_title('Reproduction Error at Training Points')
#     ax5.set_ylim(0, max(error_values) * 1.2 if max(error_values) > 0 else 0.1)
    
#     # Add value labels on bars
#     for bar, val in zip(bars, error_values):
#         height = bar.get_height()
#         ax5.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{val:.2e}', ha='center', va='bottom')
    
#     # 6. Test points comparison (if provided)
#     ax6 = plt.subplot(2, 3, 6)
#     if test_points is not None:
#         X_test, hue_test, sat_test = test_points
#         hue_pred, sat_pred = interpolator.predict(X_test)
        
#         # Plot hue comparison
#         ax6.scatter(hue_test, hue_pred, alpha=0.5, s=1, label='Hue')
#         ax6.plot([0, 1], [0, 1], 'r--', alpha=0.5)
#         ax6.set_xlabel('Actual')
#         ax6.set_ylabel('Predicted')
#         ax6.set_title('Test Set Predictions')
#         ax6.legend()
#         ax6.grid(True, alpha=0.3)
#     else:
#         ax6.text(0.5, 0.5, 'No test set provided', 
#                 ha='center', va='center', transform=ax6.transAxes)
#         ax6.set_title('Test Set Comparison')
    
#     plt.tight_layout()
#     plt.show()

def visualize_rbf_results(interpolator, df, test_points=None):
    """Create comprehensive 3x3 visualization of RBF results"""
    
    fig = plt.figure(figsize=(18, 15))
    
    # Prepare grid for interpolation
    x_grid = np.linspace(-0.65, 0.65, 100)
    y_grid = np.linspace(-0.65, 0.65, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    points_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Get interpolated values
    hue_grid, sat_grid = interpolator.predict(points_grid)
    hue_grid_2d = hue_grid.reshape(X_grid.shape)
    sat_grid_2d = sat_grid.reshape(X_grid.shape)
    
    # Convert hue and saturation to RGB for combined visualization
    from matplotlib.colors import hsv_to_rgb
    hsv_combined = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))
    hsv_combined[:, :, 0] = hue_grid_2d  # Hue
    hsv_combined[:, :, 1] = sat_grid_2d  # Saturation
    hsv_combined[:, :, 2] = 1.0          # Full value/brightness
    rgb_combined = hsv_to_rgb(hsv_combined)
    
    # ========== ROW 1: Training Data ==========
    
    # 1. Training data - Hue
    ax1 = plt.subplot(3, 3, 1)
    scatter1 = ax1.scatter(df['x'], df['y'], c=df['hue'], cmap='hsv', s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Training Data - Hue ({len(df)} points)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Hue')
    
    # 2. Training data - Saturation
    ax2 = plt.subplot(3, 3, 2)
    scatter2 = ax2.scatter(df['x'], df['y'], c=df['sat'], cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Training Data - Saturation')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2, label='Saturation')
    
    # 3. Training data - Combined (Hue + Saturation as color)
    ax3 = plt.subplot(3, 3, 3)
    # Convert training data hue and sat to RGB
    hsv_train = np.zeros((len(df), 3))
    hsv_train[:, 0] = df['hue'].values
    hsv_train[:, 1] = df['sat'].values
    hsv_train[:, 2] = 1.0
    rgb_train = hsv_to_rgb(hsv_train.reshape(-1, 1, 3)).reshape(-1, 3)
    
    ax3.scatter(df['x'], df['y'], c=rgb_train, s=1, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Training Data - Hue+Sat Combined')
    ax3.set_aspect('equal')
    
    # ========== ROW 2: Interpolated Fields ==========
    
    # 4. Interpolated field - Hue
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.pcolormesh(X_grid, Y_grid, hue_grid_2d, cmap='hsv', shading='auto')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('RBF Interpolated - Hue')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='Hue')
    
    # 5. Interpolated field - Saturation
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.pcolormesh(X_grid, Y_grid, sat_grid_2d, cmap='viridis', shading='auto')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('RBF Interpolated - Saturation')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5, label='Saturation')
    
    # 6. Interpolated field - Combined (Hue + Saturation as RGB)
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(rgb_combined, extent=[-0.65, 0.65, -0.65, 0.65], origin='lower', aspect='auto')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('RBF Interpolated - Hue+Sat Combined')
    ax6.set_aspect('equal')
    
    # ========== ROW 3: Advanced Visualizations ==========
    
    # 7. 3D Surface plot - Saturation with Hue colors
    ax7 = plt.subplot(3, 3, 7, projection='3d')
    # Downsample for better 3D performance
    stride = 2
    surf = ax7.plot_surface(X_grid[::stride, ::stride], 
                           Y_grid[::stride, ::stride], 
                           sat_grid_2d[::stride, ::stride],
                           facecolors=rgb_combined[::stride, ::stride],
                           shade=False,
                           alpha=0.9)
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Saturation')
    ax7.set_title('3D Saturation Surface (Hue-colored)')
    ax7.view_init(elev=30, azim=45)
    
    # 8. Quiver plot - Vector field
    ax8 = plt.subplot(3, 3, 8)
    # Create a coarser grid for quiver plot
    x_quiver = np.linspace(-0.65, 0.65, 20)
    y_quiver = np.linspace(-0.65, 0.65, 20)
    X_quiver, Y_quiver = np.meshgrid(x_quiver, y_quiver)
    points_quiver = np.column_stack([X_quiver.ravel(), Y_quiver.ravel()])
    
    # Get predictions at quiver points
    hue_quiver, sat_quiver = interpolator.predict(points_quiver)
    
    # Convert hue to vector direction (angle) and saturation to magnitude
    angles = hue_quiver * 2 * np.pi + np.pi/2
    U = sat_quiver * np.cos(angles)
    V = sat_quiver * np.sin(angles)
    
    U_grid = U.reshape(X_quiver.shape)
    V_grid = V.reshape(X_quiver.shape)
    
    # Color by hue
    colors_quiver = hue_quiver.reshape(X_quiver.shape)
    
    quiv = ax8.quiver(X_quiver, Y_quiver, U_grid, V_grid, colors_quiver,
                      cmap='hsv', scale=10, alpha=0.8)
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_title('Vector Field (Direction=Hue, Magnitude=Sat)')
    ax8.set_aspect('equal')
    plt.colorbar(quiv, ax=ax8, label='Hue')
    
    # 9. Error analysis or test set comparison
    ax9 = plt.subplot(3, 3, 9)
    
    if test_points is not None:
        X_test, hue_test, sat_test = test_points
        hue_pred, sat_pred = interpolator.predict(X_test)
        
        # Calculate errors
        hue_error = np.abs(hue_pred - hue_test)
        hue_error = np.minimum(hue_error, 1 - hue_error)  # Circular
        sat_error = np.abs(sat_pred - sat_test)
        
        # Scatter plot with error magnitude as color
        total_error = np.sqrt(hue_error**2 + sat_error**2)
        scatter9 = ax9.scatter(X_test[:, 0], X_test[:, 1], 
                              c=total_error, cmap='hot_r', s=5, alpha=0.6)
        ax9.set_xlabel('X')
        ax9.set_ylabel('Y')
        ax9.set_title('Test Set Error Distribution')
        ax9.set_aspect('equal')
        plt.colorbar(scatter9, ax=ax9, label='Total Error')
        
        # Add error statistics as text
        textstr = f'Mean: {np.mean(total_error):.4f}\nMax: {np.max(total_error):.4f}'
        ax9.text(0.05, 0.95, textstr, transform=ax9.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    else:
        # Show training error reproduction
        errors = interpolator.evaluate_at_training_points()
        
        error_types = ['Hue\nMAE', 'Hue\nMax', 'Sat\nMAE', 'Sat\nMax']
        error_values = [errors['hue_mae'], errors['hue_max_error'], 
                       errors['sat_mae'], errors['sat_max_error']]
        
        bars = ax9.bar(error_types, error_values, color=['#ff6b6b', '#ff8787', '#4ecdc4', '#45b7aa'])
        ax9.set_ylabel('Error')
        ax9.set_title('Training Point Reproduction Error')
        ax9.set_ylim(0, max(error_values) * 1.2 if max(error_values) > 0 else 0.1)
        ax9.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, error_values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN EXECUTION ====================

def main():
    # Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working in directory: {script_dir}")
    csv_path = os.path.join(script_dir,'all_vortex_data_possible.csv')
    
    # Step 1: Load and process data
    stacked_df = load_and_stack_data(csv_path)
    
    # Step 2: Filter and deduplicate
    # For RBF, we might want to keep more points since it can handle them well
    processed_df = filter_and_deduplicate(
        stacked_df, 
        x_range=(-0.65, 0.65), 
        y_range=(-0.65, 0.65),
        tolerance=0.005,  # Slightly less aggressive deduplication
        target_size=10_000  # RBF can handle more points efficiently
    )
    
    # Step 3: Prepare data
    X = processed_df[['x', 'y']].values
    hue = processed_df['hue'].values
    sat = processed_df['sat'].values
    
    # Optional: Create a small test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, hue_train, hue_test, sat_train, sat_test = train_test_split(
        X, hue, sat, test_size=0.1, random_state=42
    )
    
    print(f"\nUsing {len(X_train)} points for RBF fitting")
    print(f"Reserved {len(X_test)} points for testing")
    
    # Step 4: Create and fit RBF interpolator
    rbf = VortexRBFInterpolator(config=RBF_CONFIG)
    rbf.fit(X_train, hue_train, sat_train)
    
    # Step 5: Evaluate
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Check training point reproduction
    train_errors = rbf.evaluate_at_training_points()
    print("\nTraining Point Reproduction:")
    print(f"  Hue MAE: {train_errors['hue_mae']:.6f}")
    print(f"  Hue Max Error: {train_errors['hue_max_error']:.6f}")
    print(f"  Sat MAE: {train_errors['sat_mae']:.6f}")
    print(f"  Sat Max Error: {train_errors['sat_max_error']:.6f}")
    
    # Evaluate on test set
    hue_pred_test, sat_pred_test = rbf.predict(X_test)
    
    hue_error_test = np.abs(hue_pred_test - hue_test)
    hue_error_test = np.minimum(hue_error_test, 1 - hue_error_test)  # Circular
    sat_error_test = np.abs(sat_pred_test - sat_test)
    
    print("\nTest Set Performance:")
    print(f"  Hue MAE: {np.mean(hue_error_test):.6f}")
    print(f"  Hue Max Error: {np.max(hue_error_test):.6f}")
    print(f"  Sat MAE: {np.mean(sat_error_test):.6f}")
    print(f"  Sat Max Error: {np.max(sat_error_test):.6f}")
    
    # Step 6: Save the interpolator
    rbf.save('vortex_rbf_interpolator.pkl')
    
    # Also save a compact version with just essential data
    compact_data = {
        'training_points': X_train,
        'training_hue': hue_train,
        'training_sat': sat_train,
        'config': RBF_CONFIG,
        'performance': {
            'train_errors': train_errors,
            'test_hue_mae': np.mean(hue_error_test),
            'test_sat_mae': np.mean(sat_error_test)
        }
    }
    
    with open('rbf_training_data.pkl', 'wb') as f:
        pickle.dump(compact_data, f)
    
    print("\nSaved files:")
    print("  - vortex_rbf_interpolator.pkl (full interpolator)")
    print("  - rbf_training_data.pkl (training data and config)")
    
    # Step 7: Visualize results
    print("\nCreating visualization...")
    visualize_rbf_results(rbf, processed_df, 
                         test_points=(X_test, hue_test, sat_test))
    
    # Step 8: Demonstrate exact reproduction at a training point
    print("\n" + "="*50)
    print("DEMONSTRATION: Exact reproduction at training points")
    print("="*50)
    
    # Pick a few random training points
    indices = np.random.choice(len(X_train), 5, replace=False)
    
    for idx in indices:
        point = X_train[idx]
        true_hue = hue_train[idx]
        true_sat = sat_train[idx]
        
        pred_hue, pred_sat = rbf.predict([point])
        
        print(f"\nPoint ({point[0]:.4f}, {point[1]:.4f}):")
        print(f"  True:      Hue={true_hue:.6f}, Sat={true_sat:.6f}")
        print(f"  Predicted: Hue={pred_hue[0]:.6f}, Sat={pred_sat[0]:.6f}")
        print(f"  Error:     Hue={abs(pred_hue[0]-true_hue):.2e}, Sat={abs(pred_sat[0]-true_sat):.2e}")
    
    print("\n" + "="*50)
    print("RBF interpolation complete!")
    print("="*50)

if __name__ == "__main__":
    main()