"""
OmniCluster - Coordinated control of 3 omnidirectional robots with formation control.
"""
import numpy as np
import yaml
import os
from .omnibot import Omnibot
from ..control.kinematics import (forward_kinematics, inverse_kinematics,
                       compute_inverse_jacobian, shape_velocities_to_robot_velocities)


class OmniCluster:
    """
    Manages a cluster of 3 omnidirectional robots with formation control.

    Architecture:
    1. Reads desired formation from YAML config
    2. Computes current formation using forward kinematics
    3. Computes formation error (desired - current)
    4. Generates shape velocities (proportional control)
    5. Converts to robot velocities using inverse Jacobian
    6. Commands individual robots
    """

    def __init__(self, formation_config_path, field, timestep=0.1, momentum_alpha=0.7):
        """
        Initialize OmniCluster.

        Args:
            formation_config_path: Path to YAML formation configuration file
            field: VectorField object for environment sensing
            timestep: Time step for robot integration
            momentum_alpha: Momentum coefficient for robots
        """
        self.field = field
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha

        # Load formation configuration
        self._load_formation_config(formation_config_path)

        # Initialize 3 robots in approximate desired formation
        self._initialize_robots()

        # For tracking trajectory
        self.center_history = []

        # For tracking individual robot trajectories
        self.robot_history = []

        # For tracking velocities
        self.velocity_history = []

        # For diagnostic tracking
        self.diagnostics = []

    def _load_formation_config(self, config_path):
        """Load desired formation parameters from YAML file."""
        # Resolve path relative to project root if it's not absolute
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, '..', '..')
            config_path = os.path.join(project_root, config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        formation = config['formation']
        self.desired_p = formation['p']
        self.desired_q = formation['q']
        self.desired_beta = np.radians(formation['beta_degrees'])
        self.position_gain = formation.get('position_gain', 1.0)
        self.angle_gain = formation.get('angle_gain', self.position_gain)  # Default to position_gain if not specified

        print(f"Loaded formation config: p={self.desired_p}, q={self.desired_q}, "
              f"beta={np.degrees(self.desired_beta):.1f}°")

    def _initialize_robots(self):
        """Initialize 3 robots in approximate desired formation."""
        # Start at fixed position (0.5, 0.5)
        # x_c_init = 0.5
        # y_c_init = 0.5
        # theta_c_init = np.pi  # Initial orientation

        # Random starting position (uncomment above and comment below to revert)
        x_c_init = np.random.uniform(-0.5, 0.5)
        y_c_init = np.random.uniform(-0.5, 0.5)
        theta_c_init = np.pi  # Initial orientation

        # Compute initial robot positions using inverse kinematics
        x1, y1, x2, y2, x3, y3 = inverse_kinematics(
            x_c_init, y_c_init, theta_c_init,
            self.desired_p, self.desired_beta, self.desired_q
        )

        # Create 3 robots
        self.robots = [
            Omnibot(x1, y1, self.timestep, self.momentum_alpha),
            Omnibot(x2, y2, self.timestep, self.momentum_alpha),
            Omnibot(x3, y3, self.timestep, self.momentum_alpha)
        ]

        print(f"Initialized 3 robots at centroid ({x_c_init:.3f}, {y_c_init:.3f})")

    def get_robot_positions(self):
        """Get current positions of all robots."""
        positions = [robot.get_position() for robot in self.robots]
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        x3, y3 = positions[2]
        return x1, y1, x2, y2, x3, y3

    def get_current_formation(self):
        """
        Compute current formation parameters using forward kinematics.

        Returns:
            dict with current formation state
        """
        x1, y1, x2, y2, x3, y3 = self.get_robot_positions()
        return forward_kinematics(x1, y1, x2, y2, x3, y3)

    def get_centroid(self):
        """Get current centroid position."""
        formation = self.get_current_formation()
        return np.array([formation['x_c'], formation['y_c']])

    def sample_field_at_robots(self):
        """
        Get vector field readings at each robot position.

        Returns:
            List of (u, v) tuples for each robot
        """
        return [robot.sample_field(self.field) for robot in self.robots]

    def move(self, control_primitive):
        """
        Move the cluster using a control primitive.

        Args:
            control_primitive: Function that takes OmniCluster and returns desired centroid velocity
        """
        # Get current formation state
        current_formation = self.get_current_formation()

        # Call control primitive to get desired centroid velocity
        vx_c_desired, vy_c_desired = control_primitive(self)

        # Compute formation errors (formation controller of the adaptive
        # navigation layer, paper Sec. III-C: proportional error feedback
        # on the SAS shape variables p, beta, q)
        error_p = self.desired_p - current_formation['p']
        error_q = self.desired_q - current_formation['q']

        # Wrap angle error to [-π, π] for shortest path.
        # The while-loop wrap is exact (adds/subtracts 2π only when out of
        # range), unlike atan2(sin, cos) which perturbs in-range values by
        # floating-point round-off.
        error_beta = self.desired_beta - current_formation['beta']
        while error_beta > np.pi:
            error_beta -= 2 * np.pi
        while error_beta < -np.pi:
            error_beta += 2 * np.pi

        # Compute shape velocities (proportional control)
        vp = self.position_gain * error_p
        vq = self.position_gain * error_q
        vbeta = self.angle_gain * error_beta  # Use separate angle gain for beta

        # Angular velocity (for now, keep current orientation stable)
        omega_c = 0.0

        # Compute inverse Jacobian at current configuration
        J_inv = compute_inverse_jacobian(
            current_formation['p'],
            current_formation['beta'],
            current_formation['q'],
            current_formation['theta_c']
        )

        # Convert shape velocities to robot velocities
        vx1, vy1, vx2, vy2, vx3, vy3 = shape_velocities_to_robot_velocities(
            J_inv, vx_c_desired, vy_c_desired, omega_c, vp, vbeta, vq
        )

        # Command individual robots
        self.robots[0].command_velocity(vx1, vy1)
        self.robots[1].command_velocity(vx2, vy2)
        self.robots[2].command_velocity(vx3, vy3)

        # Record centroid for trajectory tracking
        centroid = self.get_centroid()
        self.center_history.append(centroid.copy())

        # Collect diagnostics
        diagnostic_data = {
            'timestep': len(self.diagnostics),
            'x_c': centroid[0],
            'y_c': centroid[1],
            'radius': np.linalg.norm(centroid),
            'p_current': current_formation['p'],
            'q_current': current_formation['q'],
            'beta_current': np.degrees(current_formation['beta']),
            'p_error': error_p,
            'q_error': error_q,
            'beta_error': np.degrees(error_beta),
            'vp': vp,
            'vq': vq,
            'vbeta': vbeta,
            'vx_c': vx_c_desired,
            'vy_c': vy_c_desired,
            'jacobian_cond': np.linalg.cond(J_inv)
        }
        self.diagnostics.append(diagnostic_data)

        # Record individual robot positions for trajectory tracking
        x1, y1, x2, y2, x3, y3 = self.get_robot_positions()
        robot_positions = np.array([[x1, y1], [x2, y2], [x3, y3]])
        self.robot_history.append(robot_positions.copy())

        # Record robot velocities
        velocities = np.array([robot.get_velocity() for robot in self.robots])
        self.velocity_history.append(velocities.copy())

    def get_center_history(self):
        """Get history of centroid positions as numpy array."""
        return np.array(self.center_history)

    def get_velocity_history(self):
        """Get history of robot velocities as list of arrays."""
        return self.velocity_history

    def get_robot_history(self):
        """Get history of individual robot positions as numpy array.

        Returns:
            numpy array of shape (timesteps, num_robots, 2)
        """
        return np.array(self.robot_history)

    def get_diagnostics(self):
        """Get diagnostic data collected during simulation.

        Returns:
            list of diagnostic dictionaries
        """
        return self.diagnostics

    def reset(self, x_c=None, y_c=None):
        """
        Reset cluster to a new position.

        Args:
            x_c, y_c: New centroid position (defaults to (0.5, 0.5) if None)
        """
        if x_c is None:
            x_c = 0.5
        if y_c is None:
            y_c = 0.5

        theta_c = np.pi

        # Compute robot positions
        x1, y1, x2, y2, x3, y3 = inverse_kinematics(
            x_c, y_c, theta_c,
            self.desired_p, self.desired_beta, self.desired_q
        )

        # Reset robot positions
        self.robots[0].set_position(x1, y1)
        self.robots[1].set_position(x2, y2)
        self.robots[2].set_position(x3, y3)

        # Reset velocities
        for robot in self.robots:
            robot.velocity = np.array([0.0, 0.0])

        # Clear history.
        # NOTE: self.diagnostics is intentionally NOT cleared here, so
        # diagnostics accumulate across reset() calls (e.g. across Monte
        # Carlo trials) and their 'timestep' field keeps counting up.
        # Callers that need per-run diagnostics must clear the list manually.
        self.center_history = []
        self.robot_history = []
        self.velocity_history = []

    def plot(self, ax=None):
        """
        Plot robot positions.

        Args:
            ax: Matplotlib axis (uses current axis if None)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        positions = [robot.get_position() for robot in self.robots]
        colors = ['blue', 'yellow', 'green']

        for i, (pos, color) in enumerate(zip(positions, colors)):
            ax.scatter(pos[0], pos[1], marker='o', color=color, s=100, zorder=5)

    def plot_center(self, ax=None):
        """
        Plot cluster centroid.

        Args:
            ax: Matplotlib axis (uses current axis if None)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        centroid = self.get_centroid()
        ax.plot(centroid[0], centroid[1], marker='o', color='black', markersize=8, zorder=5)

    def __repr__(self):
        formation = self.get_current_formation()
        return (f"OmniCluster(centroid=({formation['x_c']:.3f}, {formation['y_c']:.3f}), "
                f"p={formation['p']:.3f}, q={formation['q']:.3f}, "
                f"beta={np.degrees(formation['beta']):.1f}°)")
