"""
Field abstractions for vector and scalar field environments.

Field types provide a unified interface: get_value(x, y)
- Vector fields: get_value(x, y) -> (u, v)
- Scalar fields: get_value(x, y) -> phi
"""
import inspect
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

from .config_loader import load_field_config


def _build_extra_kwargs(func):
    """Return the set of optional kwargs beyond (x, y) that func accepts."""
    try:
        sig = inspect.signature(func)
        params = set(sig.parameters.keys()) - {"x", "y"}
        return params
    except (ValueError, TypeError):
        return set()


class Field:
    """Base class for all fields (scalar and vector)."""

    @property
    def field_type(self):
        """Return 'scalar' or 'vector'."""
        raise NotImplementedError("Subclasses must implement field_type property")

    def get_value(self, x, y):
        """
        Get field value at position (x, y).

        Args:
            x: x coordinate (scalar or array)
            y: y coordinate (scalar or array)

        Returns:
            For VectorField: (u, v) tuple
            For ScalarField: phi scalar
        """
        raise NotImplementedError("Subclasses must implement get_value()")


class VectorField(Field):
    """Base class for vector fields."""

    @property
    def field_type(self):
        """Return 'vector'."""
        return 'vector'

    def get_value(self, x, y):
        """
        Get vector field value at position (x, y).

        Args:
            x: x coordinate (scalar or array)
            y: y coordinate (scalar or array)

        Returns:
            (u, v): Vector field components
        """
        raise NotImplementedError("Subclasses must implement get_value()")


class AnalyticalField(VectorField):
    """
    Analytical vector field using a mathematical function.

    Optionally loads time-varying parameters from config/fields/<name>.yaml.
    Resolution order for the config name:
        1. config_name kwarg to __init__
        2. field_function.config_name attribute
        3. no config (empty dict; field uses its own defaults)

    The field carries an internal clock self.t advanced by step(dt). The
    simulation runner calls step(dt) once per iteration. Field functions
    that accept (x, y, t=..., config=...) will receive the current time
    and config; functions with the plain (x, y) signature are unaffected.
    """

    def __init__(self, field_function, config_name=None):
        """
        Args:
            field_function: Function with signature (x, y[, t, config]) -> (u, v).
            config_name: Override for the YAML basename. If None, falls back
                         to field_function.config_name, then to no config.
        """
        self.field_function = field_function
        if config_name is None:
            config_name = getattr(field_function, "config_name", None)
        self.config_name = config_name
        self.config = load_field_config(config_name) if config_name else {}
        self._extra_kwargs = _build_extra_kwargs(field_function)
        self.t = 0.0

    def step(self, dt):
        """Advance the field's internal clock by dt."""
        self.t += dt

    def reset_clock(self):
        """Reset the field clock to t=0."""
        self.t = 0.0

    def get_value(self, x, y):
        """Get field value using analytical function."""
        kwargs = {}
        if "t" in self._extra_kwargs:
            kwargs["t"] = self.t
        if "config" in self._extra_kwargs:
            kwargs["config"] = self.config
        return self.field_function(x, y, **kwargs)


class NNField(VectorField):
    """
    Vector field approximated by neural networks.
    Uses hue (direction) and saturation (magnitude) representation.
    """

    def __init__(self, predictor_dir='sinking_vortex_predictors'):
        """
        Args:
            predictor_dir: Directory containing trained NN models
        """
        self.predictor_dir = predictor_dir
        self._load_models()

    def _load_models(self):
        """Load trained neural network models."""
        # Get absolute path to predictor directory (navigate to project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')
        predictors_dir = os.path.join(project_root, self.predictor_dir)

        # Load architecture info
        model_info_path = os.path.join(predictors_dir, 'model_info.pkl')
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)

        # Initialize and load hue model
        self.hue_model = self._build_model(model_info['hue_architecture'])
        hue_model_path = os.path.join(predictors_dir, 'hue_model.pth')
        self.hue_model.load_state_dict(torch.load(hue_model_path, map_location='cpu'))
        self.hue_model.eval()

        # Initialize and load saturation model
        self.sat_model = self._build_model(model_info['sat_architecture'])
        sat_model_path = os.path.join(predictors_dir, 'sat_model.pth')
        self.sat_model.load_state_dict(torch.load(sat_model_path, map_location='cpu'))
        self.sat_model.eval()

    def _build_model(self, architecture):
        """Build neural network from architecture specification."""
        class FieldPredictor(nn.Module):
            def __init__(self, arch):
                super(FieldPredictor, self).__init__()

                hidden_sizes = arch['hidden_sizes']
                activation = arch['activation']
                output_neurons = arch['output_neurons']

                layers = []
                input_size = 2  # (x, y)

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

        return FieldPredictor(architecture)

    def get_value(self, x, y):
        """Get field value using neural network prediction."""
        # Prepare input
        input_tensor = torch.FloatTensor([[x, y]])

        with torch.no_grad():
            # Get hue prediction (sin, cos components)
            hue_pred = self.hue_model(input_tensor).numpy()[0]
            # Get saturation prediction (scaled to [-1, 1])
            sat_pred = self.sat_model(input_tensor).numpy()[0][0]

        # Reconstruct hue from sin/cos
        hue_angle = np.arctan2(hue_pred[0], hue_pred[1])
        hue_angle = (hue_angle + 2 * np.pi) % (2 * np.pi)  # [0, 2pi]
        hue = hue_angle / (2 * np.pi)  # [0, 1]

        # Convert saturation from [-1, 1] to [0, 1]
        sat = (sat_pred + 1) / 2

        # Convert to magnitude
        magnitude = (sat - 0.3) / 0.7

        # Convert hue to vector angle (add pi/2)
        angle = hue * 2 * np.pi + np.pi / 2
        u = magnitude * np.cos(angle)
        v = magnitude * np.sin(angle)

        return u, v


class RBFField(VectorField):
    """
    Vector field approximated by RBF interpolators.
    Uses hue (direction) and saturation (magnitude) representation.
    """

    def __init__(self, predictor_dir='sinking_vortex_predictors'):
        """
        Args:
            predictor_dir: Directory containing trained RBF models
        """
        self.predictor_dir = predictor_dir
        self._load_models()

    def _load_models(self):
        """Load trained RBF interpolators."""
        # Get absolute path to predictor directory (navigate to project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, '..', '..')
        predictors_dir = os.path.join(project_root, self.predictor_dir)

        # Load the RBF interpolator
        rbf_path = os.path.join(predictors_dir, 'vortex_rbf_interpolator.pkl')
        with open(rbf_path, 'rb') as f:
            rbf_data = pickle.load(f)

        self.rbf_hue = rbf_data['hue_rbf']
        self.rbf_sat = rbf_data['sat_rbf']

    def get_value(self, x, y):
        """Get field value using RBF interpolation."""
        # Prepare input (convert to float32 to match training)
        pos_array = np.array([[x, y]], dtype=np.float32)

        # Get predictions
        hue_sincos = self.rbf_hue(pos_array)[0]
        sat = self.rbf_sat(pos_array).flatten()[0]

        # Reconstruct hue from sin/cos
        hue_angle = np.arctan2(hue_sincos[0], hue_sincos[1])
        hue_angle = (hue_angle + 2 * np.pi) % (2 * np.pi)  # [0, 2pi]
        hue = hue_angle / (2 * np.pi)  # [0, 1]

        # Convert to magnitude
        magnitude = (sat - 0.3) / 0.7

        # Convert hue to vector angle (add pi/2)
        angle = hue * 2 * np.pi + np.pi / 2
        u = magnitude * np.cos(angle)
        v = magnitude * np.sin(angle)

        return u, v


class BlendedField(VectorField):
    """
    Vector field using a weighted blend of RBF and NN predictions.
    Blending happens at the hue/saturation level.
    """

    def __init__(self, predictor_dir='sinking_vortex_predictors', rbf_weight=0.9):
        """
        Args:
            predictor_dir: Directory containing trained models
            rbf_weight: Weight for RBF (0-1). NN weight = 1 - rbf_weight
        """
        self.predictor_dir = predictor_dir
        self.rbf_weight = rbf_weight
        self.nn_weight = 1.0 - rbf_weight

        # Create NN and RBF fields
        self.nn_field = NNField(predictor_dir)
        self.rbf_field = RBFField(predictor_dir)

    def get_value(self, x, y):
        """Get field value using blended prediction."""
        # Get NN predictions (need to access internal representations)
        input_tensor = torch.FloatTensor([[x, y]])
        with torch.no_grad():
            nn_hue_sincos = self.nn_field.hue_model(input_tensor).numpy()[0]
            nn_sat_scaled = self.nn_field.sat_model(input_tensor).numpy()[0][0]

        # Get RBF predictions
        pos_array = np.array([[x, y]], dtype=np.float32)
        rbf_hue_sincos = self.rbf_field.rbf_hue(pos_array)[0]
        rbf_sat = self.rbf_field.rbf_sat(pos_array).flatten()[0]

        # Blend sin/cos components
        blended_sin = self.rbf_weight * rbf_hue_sincos[0] + self.nn_weight * nn_hue_sincos[0]
        blended_cos = self.rbf_weight * rbf_hue_sincos[1] + self.nn_weight * nn_hue_sincos[1]

        # Convert NN saturation from [-1, 1] to [0, 1] for blending
        nn_sat = (nn_sat_scaled + 1) / 2
        blended_sat = self.rbf_weight * rbf_sat + self.nn_weight * nn_sat

        # Reconstruct hue from blended sin/cos
        hue_angle = np.arctan2(blended_sin, blended_cos)
        hue_angle = (hue_angle + 2 * np.pi) % (2 * np.pi)  # [0, 2pi]
        hue = hue_angle / (2 * np.pi)  # [0, 1]

        # Convert to magnitude
        magnitude = (blended_sat - 0.3) / 0.7

        # Convert hue to vector angle (add pi/2)
        angle = hue * 2 * np.pi + np.pi / 2
        u = magnitude * np.cos(angle)
        v = magnitude * np.sin(angle)

        return u, v


# ============================================================================
# Scalar Field Classes
# ============================================================================

class ScalarField(Field):
    """Base class for scalar fields."""

    @property
    def field_type(self):
        """Return 'scalar'."""
        return 'scalar'

    def get_value(self, x, y):
        """
        Get scalar field value at position (x, y).

        Args:
            x: x coordinate (scalar or array)
            y: y coordinate (scalar or array)

        Returns:
            phi: Scalar field value
        """
        raise NotImplementedError("Subclasses must implement get_value()")


class AnalyticalScalarField(ScalarField):
    """
    Analytical scalar field using a mathematical function.

    Mirrors AnalyticalField: carries an internal clock self.t and loads
    per-field YAML config. Field functions that accept (x, y, t, config)
    receive the current time and config on each call; plain (x, y) functions
    are unaffected.
    """

    def __init__(self, field_function, config_name=None):
        """
        Args:
            field_function: Function with signature (x, y[, t, config]) -> phi.
            config_name: Override for the YAML basename. If None, falls back
                         to field_function.config_name, then to no config.
        """
        self.field_function = field_function
        if config_name is None:
            config_name = getattr(field_function, "config_name", None)
        self.config_name = config_name
        self.config = load_field_config(config_name) if config_name else {}
        self._extra_kwargs = _build_extra_kwargs(field_function)
        self.t = 0.0

    def step(self, dt):
        """Advance the field's internal clock by dt."""
        self.t += dt

    def reset_clock(self):
        """Reset the field clock to t=0."""
        self.t = 0.0

    def get_value(self, x, y):
        """Get scalar field value using analytical function."""
        kwargs = {}
        if "t" in self._extra_kwargs:
            kwargs["t"] = self.t
        if "config" in self._extra_kwargs:
            kwargs["config"] = self.config
        return self.field_function(x, y, **kwargs)
