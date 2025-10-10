"""
Neural Network Training Pipeline for Vector Field Approximation
This script processes the vortex data and trains neural networks to predict hue and sat from (x,y)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

# ==================== GLOBAL ARCHITECTURE CONFIG ====================
# Change architecture here and it updates everywhere
HUE_ARCHITECTURE = {
    'hidden_sizes': [16, 8, 4],
    'activation': 'tanh',
    'output_neurons': 2  # For sin/cos representation
}

SAT_ARCHITECTURE = {
    'hidden_sizes': [15, 10, 5],
    'activation': 'tanh',
    'output_neurons': 1
}

# ==================== PART 1: DATA PROCESSING ====================

def load_and_stack_data(filename='sinking_vortex_data.csv'):
    """Load CSV and stack the three measurement sets into single columns"""
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # Stack the three sets of measurements
    stacked_data = []
    
    for i in range(len(df)):
        # First measurement set
        stacked_data.append([
            df.iloc[i, 0],  # x1
            df.iloc[i, 1],  # y1
            df.iloc[i, 3],  # hue1
            df.iloc[i, 4]   # sat1
        ])
        # Second measurement set
        stacked_data.append([
            df.iloc[i, 5],  # x2
            df.iloc[i, 6],  # y2
            df.iloc[i, 8],  # hue2
            df.iloc[i, 9]   # sat2
        ])
        # Third measurement set
        stacked_data.append([
            df.iloc[i, 10], # x3
            df.iloc[i, 11], # y3
            df.iloc[i, 13], # hue3
            df.iloc[i, 14]  # sat3
        ])
    
    # Create new dataframe with stacked data (excluding theta)
    stacked_df = pd.DataFrame(stacked_data, columns=['x', 'y', 'hue', 'sat'])
    print(f"Original rows: {len(df)}, Stacked rows: {len(stacked_df)}")
    
    return stacked_df

def filter_and_deduplicate(df, x_range=(-0.6, 0.6), y_range=(-0.6, 0.6), 
                          tolerance=0.01, target_size=3500):
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
    
    # If still too many points, randomly sample to get target size
    if len(df_dedup) > target_size:
        df_dedup = df_dedup.sample(n=target_size, random_state=42)
        print(f"Randomly sampled to {target_size} points")
    
    # Save the processed data in current directory
    #df_dedup.to_csv('processed_vortex_data.csv', index=False)
    #print("Saved processed data to 'processed_vortex_data.csv'")
    
    return df_dedup

# ==================== PART 2: NEURAL NETWORK DEFINITION ====================

class FieldPredictor(nn.Module):
    """Neural network for predicting field properties from (x,y) coordinates"""
    
    def __init__(self, architecture):
        super(FieldPredictor, self).__init__()
        
        hidden_sizes = architecture['hidden_sizes']
        activation = architecture['activation']
        output_neurons = architecture['output_neurons']
        
        layers = []
        input_size = 2  # x, y
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, output_neurons))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ==================== PART 3: TRAINING FUNCTIONS ====================

def prepare_data(df):
    """Prepare data for training with manual scaling"""
    # Extract features (already in good range)
    X = df[['x', 'y']].values
    
    # Convert hue from [0,1] to [0, 2π] then to sin/cos
    hue_angle = df['hue'].values * 2 * np.pi
    y_hue_sin = np.sin(hue_angle)
    y_hue_cos = np.cos(hue_angle)
    y_hue = np.column_stack([y_hue_sin, y_hue_cos])  # Shape: (n_samples, 2)
    
    # Scale saturation from [0,1] to [-1,1]
    y_sat = df['sat'].values * 2 - 1
    
    # Split into train/test
    X_train, X_test, y_hue_train, y_hue_test, y_sat_train, y_sat_test = train_test_split(
        X, y_hue, y_sat, test_size=0.2, random_state=42
    )
    
    # Save scaling info for later use
    scaling_info = {
        'input_range': (-0.65, 0.65),
        'hue_representation': 'circular_sin_cos',
        'sat_scaling': '[0,1] -> [-1,1]'
    }
    with open('scaling_info.pkl', 'wb') as f:
        pickle.dump(scaling_info, f)
    
    return (X_train, X_test, y_hue_train, y_hue_test, y_sat_train, y_sat_test)

# (a) function signature – add a couple opts
def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=200, batch_size=32, lr=0.001, model_name="model",
                use_scheduler=True, plateau_factor=0.5, plateau_patience=30, min_lr=1e-6):

    """Train a neural network model"""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (b) right after optimizer = ...
    scheduler = (optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=plateau_factor, patience=plateau_patience, min_lr=min_lr
    ) if use_scheduler else None)

    
    # Training loop
    train_losses = []
    test_losses = []
    
    print(f"\nTraining {model_name} model...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            if len(predictions.shape) == 1:
                predictions = predictions.unsqueeze(1)
            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Calculate test loss
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor)
            if len(test_pred.shape) == 1:
                test_pred = test_pred.unsqueeze(1)
            if len(y_test_tensor.shape) == 1:
                y_test_tensor_temp = y_test_tensor.unsqueeze(1)
            else:
                y_test_tensor_temp = y_test_tensor
            test_loss = criterion(test_pred, y_test_tensor_temp).item()
            test_losses.append(test_loss)
            if scheduler is not None:
                scheduler.step(test_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return model, train_losses, test_losses

def reconstruct_hue_from_predictions(sin_cos_pred):
    """Convert sin/cos predictions back to hue angle"""
    # sin_cos_pred shape: (n_samples, 2) where [:, 0] is sin, [:, 1] is cos
    angles = np.arctan2(sin_cos_pred[:, 0], sin_cos_pred[:, 1])
    # Convert from [-π, π] to [0, 2π]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    # Convert from [0, 2π] to [0, 1] for comparison with original
    return angles / (2 * np.pi)

# ==================== PART 4: MAIN EXECUTION ====================

# def main():
#     # Step 1: Load and stack data
#     # Adjust path as needed

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(script_dir)
#     print(f"Working in directory: {script_dir}")
#     csv_path = os.path.join(script_dir, 'all_vortex_data_possible.csv')
    
#     # Step 1: Load and stack data
#     stacked_df = load_and_stack_data(csv_path)
#     #stacked_df = load_and_stack_data('saddle_data.csv')
    
    
#     # Step 2: Filter and remove near-duplicates
#     processed_df = filter_and_deduplicate(stacked_df, 
#                                          x_range=(-0.65, 0.65), 
#                                          y_range=(-0.65, 0.65),
#                                          tolerance=0.005,  # Smaller tolerance to keep more points
#                                          target_size=15000)
    
#     # Step 3: Load processed data
#     print("\nLoading processed data...")
#     #df = pd.read_csv('processed_vortex_data.csv')
#     df = processed_df
#     print(f"Loaded {len(df)} data points")
    
#     # Step 4: Prepare data for training
#     X_train, X_test, y_hue_train, y_hue_test, y_sat_train, y_sat_test = prepare_data(df)
    
#     print(f"\nTraining set size: {len(X_train)}")
#     print(f"Test set size: {len(X_test)}")
    
#     # Step 5: Create and train models
#     # Hue model (2 outputs for sin/cos)
#     hue_model = FieldPredictor(HUE_ARCHITECTURE)
#     hue_model, hue_train_losses, hue_test_losses = train_model(
#         hue_model, X_train, y_hue_train, X_test, y_hue_test,
#         epochs=200, batch_size=64, lr=0.0004, model_name="Hue"
#     )
    
#     # Saturation model
#     sat_model = FieldPredictor(SAT_ARCHITECTURE)
#     sat_model, sat_train_losses, sat_test_losses = train_model(
#         sat_model, X_train, y_sat_train, X_test, y_sat_test,
#         epochs=200, batch_size=64, lr=0.0004, model_name="Saturation"
#     )
    
#     # Step 6: Create final visualization
#     print("\nCreating final visualization...")
#     fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
#     # Plot 1: Combined training histories
#     ax = axes[0, 0]
#     ax.plot(hue_train_losses, label='Hue Train', color='blue', alpha=0.7)
#     ax.plot(hue_test_losses, label='Hue Test', color='blue', linestyle='--')
#     ax.plot(sat_train_losses, label='Sat Train', color='orange', alpha=0.7)
#     ax.plot(sat_test_losses, label='Sat Test', color='orange', linestyle='--')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('MSE Loss')
#     ax.set_title('Training History')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
    
#     # Plot 2: Hue predictions
#     ax = axes[0, 1]
#     hue_model.eval()
#     with torch.no_grad():
#         hue_pred = hue_model(torch.FloatTensor(X_test)).numpy()
#     # Reconstruct hue from sin/cos
#     hue_actual = reconstruct_hue_from_predictions(y_hue_test)
#     hue_predicted = reconstruct_hue_from_predictions(hue_pred)
#     ax.scatter(hue_actual, hue_predicted, alpha=0.5, s=1)
#     ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
#     ax.set_xlabel('Actual Hue')
#     ax.set_ylabel('Predicted Hue')
#     ax.set_title('Hue Predictions')
#     ax.grid(True, alpha=0.3)
    
#     # Plot 3: Saturation predictions
#     ax = axes[1, 0]
#     sat_model.eval()
#     with torch.no_grad():
#         sat_pred = sat_model(torch.FloatTensor(X_test)).squeeze().numpy()
#     ax.scatter(y_sat_test, sat_pred, alpha=0.5, s=1)
#     ax.plot([y_sat_test.min(), y_sat_test.max()], 
#             [y_sat_test.min(), y_sat_test.max()], 'r--', alpha=0.5)
#     ax.set_xlabel('Actual Saturation (scaled)')
#     ax.set_ylabel('Predicted Saturation (scaled)')
#     ax.set_title('Saturation Predictions')
#     ax.grid(True, alpha=0.3)
    
#     # Plot 4: Data distribution
#     ax = axes[1, 1]
#     scatter = ax.scatter(df['x'], df['y'], c=df['hue'], cmap='hsv', s=1, alpha=0.6)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'Data Distribution ({len(df)} points)')
#     ax.set_xlim(-0.65, 0.65)
#     ax.set_ylim(-0.65, 0.65)
#     ax.set_aspect('equal')
#     plt.colorbar(scatter, ax=ax, label='Hue')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Step 7: Save models and architecture info
#     print("\nSaving models...")
#     torch.save(hue_model.state_dict(), 'hue_model.pth')
#     torch.save(sat_model.state_dict(), 'sat_model.pth')
    
#     # Save architecture info
#     model_info = {
#         'hue_architecture': HUE_ARCHITECTURE,
#         'sat_architecture': SAT_ARCHITECTURE
#     }
#     with open('model_info.pkl', 'wb') as f:
#         pickle.dump(model_info, f)
    
#     print("Models saved as 'hue_model.pth' and 'sat_model.pth'")
#     print("Architecture info saved as 'model_info.pkl'")
#     print("Scaling info saved as 'scaling_info.pkl'")
#     print("\nTraining complete!")
def main():
    # Step 1: Load and stack data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working in directory: {script_dir}")
    csv_path = os.path.join(script_dir, 'all_vortex_data_possible.csv')
    
    stacked_df = load_and_stack_data(csv_path)
    
    # Step 2: Filter and remove near-duplicates
    processed_df = filter_and_deduplicate(stacked_df, 
                                         x_range=(-0.65, 0.65), 
                                         y_range=(-0.65, 0.65),
                                         tolerance=0.005,
                                         target_size=15000)
    
    # Step 3: Use processed data
    print("\nUsing processed data...")
    df = processed_df
    print(f"Using {len(df)} data points")
    
    # Step 4: Prepare data for training
    X_train, X_test, y_hue_train, y_hue_test, y_sat_train, y_sat_test = prepare_data(df)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Step 5: Create and train models
    # Hue model (2 outputs for sin/cos)
    hue_model = FieldPredictor(HUE_ARCHITECTURE)
    hue_model, hue_train_losses, hue_test_losses = train_model(
        hue_model, X_train, y_hue_train, X_test, y_hue_test,
        epochs=200, batch_size=64, lr=0.0004, model_name="Hue"
    )
    
    # Saturation model
    sat_model = FieldPredictor(SAT_ARCHITECTURE)
    sat_model, sat_train_losses, sat_test_losses = train_model(
        sat_model, X_train, y_sat_train, X_test, y_sat_test,
        epochs=200, batch_size=64, lr=0.0004, model_name="Saturation"
    )
    
    # Step 6: Create grid predictions for visualization
    print("\nGenerating predictions on grid...")
    x_grid = np.linspace(-0.65, 0.65, 100)
    y_grid = np.linspace(-0.65, 0.65, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    points_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Get predictions
    hue_model.eval()
    sat_model.eval()
    with torch.no_grad():
        hue_pred_grid = hue_model(torch.FloatTensor(points_grid)).numpy()
        sat_pred_grid = sat_model(torch.FloatTensor(points_grid)).squeeze().numpy()
    
    # Reconstruct hue from sin/cos
    hue_pred_grid = reconstruct_hue_from_predictions(hue_pred_grid)
    hue_grid_2d = hue_pred_grid.reshape(X_grid.shape)
    
    # Convert saturation from [-1,1] back to [0,1]
    sat_pred_grid = (sat_pred_grid + 1) / 2
    sat_pred_grid = np.clip(sat_pred_grid, 0, 1)
    sat_grid_2d = sat_pred_grid.reshape(X_grid.shape)
    
    # Create RGB combined image
    from matplotlib.colors import hsv_to_rgb
    hsv_combined = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))
    hsv_combined[:, :, 0] = hue_grid_2d  # Hue
    hsv_combined[:, :, 1] = sat_grid_2d  # Saturation
    hsv_combined[:, :, 2] = 1.0          # Full value/brightness
    rgb_combined = hsv_to_rgb(hsv_combined)
    
    # Get test predictions for scatter plots
    with torch.no_grad():
        hue_pred_test = hue_model(torch.FloatTensor(X_test)).numpy()
        sat_pred_test = sat_model(torch.FloatTensor(X_test)).squeeze().numpy()
    
    hue_actual = reconstruct_hue_from_predictions(y_hue_test)
    hue_predicted = reconstruct_hue_from_predictions(hue_pred_test)
    
    # Step 7: Create 2x4 visualization
    print("\nCreating final visualization...")
    fig = plt.figure(figsize=(20, 10))
    
    # ========== ROW 1: Training & Predictions ==========
    
    # Plot 1: Combined training histories
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(hue_train_losses, label='Hue Train', color='blue', alpha=0.7)
    ax1.plot(hue_test_losses, label='Hue Test', color='blue', linestyle='--')
    ax1.plot(sat_train_losses, label='Sat Train', color='orange', alpha=0.7)
    ax1.plot(sat_test_losses, label='Sat Test', color='orange', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hue predictions scatter
    ax2 = plt.subplot(2, 4, 2)
    ax2.scatter(hue_actual, hue_predicted, alpha=0.5, s=1)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax2.set_xlabel('Actual Hue')
    ax2.set_ylabel('Predicted Hue')
    ax2.set_title('Hue Predictions')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Saturation predictions scatter
    ax3 = plt.subplot(2, 4, 3)
    ax3.scatter(y_sat_test, sat_pred_test, alpha=0.5, s=1)
    ax3.plot([y_sat_test.min(), y_sat_test.max()], 
            [y_sat_test.min(), y_sat_test.max()], 'r--', alpha=0.5)
    ax3.set_xlabel('Actual Saturation (scaled)')
    ax3.set_ylabel('Predicted Saturation (scaled)')
    ax3.set_title('Saturation Predictions')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data distribution (Hue)
    ax4 = plt.subplot(2, 4, 4)
    scatter4 = ax4.scatter(df['x'], df['y'], c=df['hue'], cmap='hsv', s=1, alpha=0.6)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title(f'Training Data - Hue ({len(df)} points)')
    ax4.set_xlim(-0.65, 0.65)
    ax4.set_ylim(-0.65, 0.65)
    ax4.set_aspect('equal')
    plt.colorbar(scatter4, ax=ax4, label='Hue')
    
    # ========== ROW 2: Advanced Visualizations ==========
    
    # Plot 5: 2D Saturation field
    ax5 = plt.subplot(2, 4, 5)
    im5 = ax5.pcolormesh(X_grid, Y_grid, sat_grid_2d, cmap='viridis', shading='auto')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('NN Predicted - Saturation Field')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5, label='Saturation')
    
    # Plot 6: Combined Hue+Saturation field
    ax6 = plt.subplot(2, 4, 6)
    ax6.imshow(rgb_combined, extent=[-0.65, 0.65, -0.65, 0.65], origin='lower', aspect='auto')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('NN Predicted - Hue+Sat Combined')
    ax6.set_aspect('equal')
    
    # Plot 7: 3D Surface - Saturation with Hue colors
    ax7 = plt.subplot(2, 4, 7, projection='3d')
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
    ax7.set_title('3D Saturation (Hue-colored)')
    ax7.view_init(elev=30, azim=45)
    
    # Plot 8: Quiver plot - Vector field
    ax8 = plt.subplot(2, 4, 8)
    # Create coarser grid for quiver
    x_quiver = np.linspace(-0.65, 0.65, 20)
    y_quiver = np.linspace(-0.65, 0.65, 20)
    X_quiver, Y_quiver = np.meshgrid(x_quiver, y_quiver)
    points_quiver = np.column_stack([X_quiver.ravel(), Y_quiver.ravel()])
    
    # Get predictions at quiver points
    with torch.no_grad():
        hue_quiver = hue_model(torch.FloatTensor(points_quiver)).numpy()
        sat_quiver = sat_model(torch.FloatTensor(points_quiver)).squeeze().numpy()
    
    hue_quiver = reconstruct_hue_from_predictions(hue_quiver)
    sat_quiver = (sat_quiver + 1) / 2  # Convert to [0,1]
    sat_quiver = np.clip(sat_quiver, 0, 1)
    
    # Convert hue to vector direction and saturation to magnitude
    angles = hue_quiver * 2 * np.pi + np.pi/2
    U = sat_quiver * np.cos(angles)
    V = sat_quiver * np.sin(angles)
    
    U_grid = U.reshape(X_quiver.shape)
    V_grid = V.reshape(X_quiver.shape)
    colors_quiver = hue_quiver.reshape(X_quiver.shape)
    
    quiv = ax8.quiver(X_quiver, Y_quiver, U_grid, V_grid, colors_quiver,
                      cmap='hsv', scale=10, alpha=0.8)
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_title('Vector Field (Dir=Hue, Mag=Sat)')
    ax8.set_aspect('equal')
    plt.colorbar(quiv, ax=ax8, label='Hue')
    
    plt.tight_layout()
    plt.show()
    
    # Step 8: Save models and architecture info
    print("\nSaving models...")
    torch.save(hue_model.state_dict(), 'hue_model.pth')
    torch.save(sat_model.state_dict(), 'sat_model.pth')
    
    # Save architecture info
    model_info = {
        'hue_architecture': HUE_ARCHITECTURE,
        'sat_architecture': SAT_ARCHITECTURE
    }
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("Models saved as 'hue_model.pth' and 'sat_model.pth'")
    print("Architecture info saved as 'model_info.pkl'")
    print("Scaling info saved as 'scaling_info.pkl'")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()