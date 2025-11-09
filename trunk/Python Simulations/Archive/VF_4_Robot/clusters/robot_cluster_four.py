# robot_cluster_four.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import os
import sys
from scipy.interpolate import RBFInterpolator

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define FieldPredictor class here to avoid import issues
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

class RobotCluster:
    def __init__(self, environment_function=None, use_nn=False, use_rbf=False,
                 use_blended=False, rbf_weight=0.9, predictor_dir='sinking_vortex_predictors'):
        """
        Initialize RobotCluster (4-robot version) with option to use neural network, RBF, or blended approach.

        Args:
            environment_function: Analytical function for vector field (used if none of the ML methods are True)
            use_nn: Boolean flag to use only neural network predictions
            use_rbf: Boolean flag to use only RBF predictions
            use_blended: Boolean flag to use blended RBF/NN predictions
            rbf_weight: Weight for RBF in blended mode (default: 0.9, meaning 90% RBF, 10% NN)
            predictor_dir: Name of the predictor directory (e.g., 'sinking_vortex_predictors', 'saddle_predictors', 'vortex_predictors')
        """
        # Store mode flags
        self.use_nn = use_nn
        self.use_rbf = use_rbf
        self.use_blended = use_blended
        self.rbf_weight = rbf_weight
        self.nn_weight = 1.0 - rbf_weight
        self.predictor_dir = predictor_dir

        # Determine which models to load
        load_nn = use_nn or use_blended
        load_rbf = use_rbf or use_blended

        # Initialize models based on mode
        if load_nn:
            print(f"Loading Neural Network models... (weight: {self.nn_weight if use_blended else 1.0})")
            self.load_nn_models()

        if load_rbf:
            print(f"Loading RBF interpolator... (weight: {self.rbf_weight if use_blended else 1.0})")
            self.load_rbf_models()

        if not (use_nn or use_rbf or use_blended):
            print("Initializing with analytical environment function...")
            self.environment_function = environment_function

        self.reset()

    def load_nn_models(self):
        """Load the trained neural network models and architecture info."""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to get to the Vector_Field_4_Robot directory
            main_dir = os.path.dirname(script_dir)
            # Go up one more level to get to Vector_Fields
            vector_fields_dir = os.path.dirname(main_dir)
            # Path to VF_3_robot predictors folder
            predictors_dir = os.path.join(vector_fields_dir, 'VF_3_robot', self.predictor_dir)

            # Load architecture info
            model_info_path = os.path.join(predictors_dir, 'model_info.pkl')
            print(f"Looking for model_info.pkl at: {model_info_path}")

            with open(model_info_path, 'rb') as f:
                model_info = pickle.load(f)

            # Initialize hue model (2 outputs for sin/cos)
            self.hue_model = FieldPredictor(model_info['hue_architecture'])
            hue_model_path = os.path.join(predictors_dir, 'hue_model.pth')
            self.hue_model.load_state_dict(torch.load(hue_model_path, map_location='cpu'))
            self.hue_model.eval()

            # Initialize saturation model (1 output)
            self.sat_model = FieldPredictor(model_info['sat_architecture'])
            sat_model_path = os.path.join(predictors_dir, 'sat_model.pth')
            self.sat_model.load_state_dict(torch.load(sat_model_path, map_location='cpu'))
            self.sat_model.eval()

            print("Successfully loaded neural network models!")
            print(f"  Hue architecture: {model_info['hue_architecture']['hidden_sizes']}")
            print(f"  Sat architecture: {model_info['sat_architecture']['hidden_sizes']}")

        except FileNotFoundError as e:
            print(f"Error: NN model files not found. Make sure you've trained the NN models first.")
            print(f"Missing file: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
            raise
        except Exception as e:
            print(f"Error loading NN models: {e}")
            raise

    def load_rbf_models(self):
        """Load the trained RBF interpolator."""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to get to the Vector_Field_4_Robot directory
            main_dir = os.path.dirname(script_dir)
            # Go up one more level to get to Vector_Fields
            vector_fields_dir = os.path.dirname(main_dir)
            # Path to VF_3_robot predictors folder
            predictors_dir = os.path.join(vector_fields_dir, 'VF_3_robot', self.predictor_dir)

            # Load the RBF interpolator
            rbf_path = os.path.join(predictors_dir, 'vortex_rbf_interpolator.pkl')
            print(f"Looking for RBF interpolator at: {rbf_path}")

            with open(rbf_path, 'rb') as f:
                rbf_data = pickle.load(f)

            # Store the RBF interpolators
            self.rbf_hue = rbf_data['hue_rbf']
            self.rbf_sat = rbf_data['sat_rbf']
            self.rbf_config = rbf_data['config']

            print("Successfully loaded RBF interpolator!")
            print(f"  RBF kernel: {self.rbf_config.get('kernel', 'unknown')}")
            print(f"  Training points: {rbf_data['training_points'].shape[0]}")

        except FileNotFoundError as e:
            print(f"Error: RBF interpolator file not found. Make sure you've trained the RBF model first.")
            print(f"Missing file: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
            raise
        except Exception as e:
            print(f"Error loading RBF models: {e}")
            raise

    def reset(self):
        off_size = 0.25

        # Initialize cluster center as numpy array
        self.cluster_centre = np.random.rand(2) * 0.5 + 0.01

        # Initialize robot offsets as numpy arrays (4-robot diamond formation)
        self.robot_offsets = np.array([
            [0, off_size],                   # robot 1 (top)
            [-off_size*np.sqrt(1/3), 0],    # robot 2 (left)
            [off_size*np.sqrt(1/3), 0],     # robot 3 (right)
            [0, -off_size]                   # robot 4 (bottom)
        ])

        # Calculate average offset
        average = np.mean(self.robot_offsets, axis=0)

        # Adjust positions
        self.cluster_centre += average
        self.robot_offsets -= average

        # Other parameters
        self.angle = np.pi
        self.step_size = 0.1

        # Momentum-based physics parameters
        self.velocity = np.array([0.0, 0.0])
        self.momentum_alpha = 0.7
        self.damping = 1       
     

    def pose(self):
        """Get positions of all four robots."""
        positions = self.cluster_centre + self.robot_offsets
        return positions[0], positions[1], positions[2], positions[3]

    def move(self, control_primitive):
        """Move cluster with momentum (second-order dynamics)."""
        desired_centre = control_primitive(self)
        desired_velocity = (desired_centre - self.cluster_centre) / self.step_size
        self.velocity = (self.momentum_alpha * self.velocity +
                        (1 - self.momentum_alpha) * desired_velocity)
        self.velocity *= self.damping
        self.cluster_centre = self.cluster_centre + self.velocity * self.step_size
        self.pose()

    def plot(self):
        """Plot robot positions."""
        pos1, pos2, pos3, pos4 = self.pose()
        plt.scatter(pos1[0], pos1[1], marker='o', color='blue')
        plt.scatter(pos2[0], pos2[1], marker='o', color='yellow')
        plt.scatter(pos3[0], pos3[1], marker='o', color='green')
        plt.scatter(pos4[0], pos4[1], marker='o', color='pink')

    def plot_centre(self):
        """Plot cluster center."""
        plt.plot([self.cluster_centre[0], self.cluster_centre[0]],
                [self.cluster_centre[1], self.cluster_centre[1]],
                marker='o', color='black')

    def get_nn_predictions(self, pos):
        """Get neural network predictions for a single position."""
        input_tensor = torch.FloatTensor([[pos[0], pos[1]]])

        with torch.no_grad():
            # Get hue prediction (2 outputs: sin and cos components)
            hue_pred = self.hue_model(input_tensor).numpy()[0]
            # Get saturation prediction (1 output, scaled to [-1, 1])
            sat_pred = self.sat_model(input_tensor).numpy()[0][0]

        # hue_pred contains [sin, cos] components
        return hue_pred, sat_pred

    def get_rbf_predictions(self, pos):
        """Get RBF predictions for a single position."""
        # RBF expects shape (n_samples, 2)
        # Convert to float32 to match the dtype used during RBF training
        pos_array = np.array([pos], dtype=np.float32)

        # Get predictions
        hue_sincos = self.rbf_hue(pos_array)
        sat_pred = self.rbf_sat(pos_array).flatten()[0]

        # hue_sincos contains [sin, cos] components
        return hue_sincos[0], sat_pred

    def bot_readings(self):
        """
        Get vector field readings at each robot position (4 robots).
        Supports NN-only, RBF-only, blended, or analytical function modes.

        Returns:
            List of [u, v] readings for each robot
        """
        positions = self.pose()

        if self.use_blended:
            # Blended mode: combine NN and RBF predictions
            readings = []

            for pos in positions:
                # Get NN predictions
                nn_hue_sincos, nn_sat_scaled = self.get_nn_predictions(pos)

                # Get RBF predictions
                rbf_hue_sincos, rbf_sat = self.get_rbf_predictions(pos)

                # Blend at the hue (sin/cos) and saturation level
                # Blend sin/cos components separately
                blended_sin = self.rbf_weight * rbf_hue_sincos[0] + self.nn_weight * nn_hue_sincos[0]
                blended_cos = self.rbf_weight * rbf_hue_sincos[1] + self.nn_weight * nn_hue_sincos[1]

                # NN saturation is in [-1, 1], RBF saturation is in [0, 1]
                # Convert NN saturation to [0, 1] for blending
                nn_sat = (nn_sat_scaled + 1) / 2
                blended_sat = self.rbf_weight * rbf_sat + self.nn_weight * nn_sat

                # Reconstruct hue from blended sin/cos
                hue_angle = np.arctan2(blended_sin, blended_cos)
                hue_angle = (hue_angle + 2 * np.pi) % (2 * np.pi)  # [0, 2π]
                hue = hue_angle / (2 * np.pi)  # [0, 1]

                # Convert saturation to magnitude
                magnitude = (blended_sat - 0.3) / 0.7

                # Convert hue to vector angle (add π/2)
                angle = hue * 2 * np.pi + np.pi/2
                u = magnitude * np.cos(angle) + np.random.randn() * 0.00
                v = magnitude * np.sin(angle) + np.random.randn() * 0.00

                readings.append([u, v])

            return readings

        elif self.use_rbf:
            # RBF-only mode
            readings = []

            for pos in positions:
                # Get RBF predictions
                hue_sincos, sat = self.get_rbf_predictions(pos)

                # Reconstruct hue from sin/cos
                hue_angle = np.arctan2(hue_sincos[0], hue_sincos[1])
                hue_angle = (hue_angle + 2 * np.pi) % (2 * np.pi)  # [0, 2π]
                hue = hue_angle / (2 * np.pi)  # [0, 1]

                # Convert saturation to magnitude
                magnitude = (sat - 0.3) / 0.7

                # Convert hue to vector angle (add π/2)
                angle = hue * 2 * np.pi + np.pi/2
                u = magnitude * np.cos(angle) + np.random.randn() * 0.00
                v = magnitude * np.sin(angle) + np.random.randn() * 0.00

                readings.append([u, v])

            return readings

        elif self.use_nn:
            # NN-only mode
            readings = []

            for pos in positions:
                # Get NN predictions
                hue_pred, sat_pred = self.get_nn_predictions(pos)

                # Reconstruct hue from sin/cos
                hue_angle = np.arctan2(hue_pred[0], hue_pred[1])
                hue_angle = (hue_angle + 2 * np.pi) % (2 * np.pi)  # [0, 2π]
                hue = hue_angle / (2 * np.pi)  # [0, 1]

                # Convert saturation from [-1, 1] back to [0, 1]
                sat = (sat_pred + 1) / 2

                # Saturation represents magnitude
                magnitude = (sat - 0.3) / 0.7

                # Convert hue to vector angle (add π/2)
                angle = hue * 2 * np.pi + np.pi/2
                u = magnitude * np.cos(angle) + np.random.randn() * 0.00
                v = magnitude * np.sin(angle) + np.random.randn() * 0.00

                readings.append([u, v])

            return readings
        else:
            # Use analytical environment function
            return [self.environment_function(pos[0], pos[1]) for pos in positions]
        
    def bot_compvec(self):
        # Convert readings to numpy array
        readings = np.array(self.bot_readings())
        # Calculate magnitude and angle for all readings at once
        mag = np.sqrt(readings[:, 0]**2 + readings[:, 1]**2)
        angle = np.arctan2(readings[:, 1], readings[:, 0])
        return mag, angle
    
    def normal_angle(self):
        pos1, pos2, pos3, pos4 = self.pose()
        mag, angle = self.bot_compvec()

    
        poss1 = np.array([*pos1, mag[0]])
        poss2 = np.array([*pos2, mag[1]])
        poss3 = np.array([*pos3, mag[2]])

        R12 = poss2 - poss1
        R13 = poss3 - poss1
        Nx, Ny, _ = np.cross(-R12, R13)
        angle = np.arctan2(Ny, Nx)
        if angle < 0:
            angle += 2*np.pi

        return Nx, Ny, angle
