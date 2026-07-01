"""
QuadCluster - Coordinated control of 4 omnidirectional robots with formation control.
"""
import numpy as np
import yaml
import os
from .omnibot import Omnibot
from ..control.quad_kinematics import (forward_kinematics_quad, inverse_kinematics_quad,
                       compute_inverse_jacobian_quad, shape_velocities_to_robot_velocities_quad)


class QuadCluster:
    """
    Manages a cluster of 4 omnidirectional robots with formation control.

    Uses diagonal parameterization: d1, d2 (diagonal lengths), r1, r2 (intersection ratios),
    phi (angle between diagonals).

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
        Initialize QuadCluster.

        Args:
            formation_config_path: Path to YAML formation configuration file
            field: VectorField object for environment sensing
            timestep: Time step for robot integration
            momentum_alpha: Momentum coefficient for robots
        """
        self.field = field
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha

        # For angle unwrapping
        self.prev_theta_c = None

        # Load formation configuration
        self._load_formation_config(formation_config_path)

        # Initialize 4 robots in approximate desired formation
        self._initialize_robots()

        # For tracking trajectory
        self.center_history = []

        # For tracking individual robot trajectories
        self.robot_history = []

        # For tracking velocities
        self.velocity_history = []

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

        # Check if this is a quad formation config
        if formation.get('type') != 'quadrilateral':
            raise ValueError("Formation config must have type='quadrilateral' for QuadCluster")

        self.desired_d1 = formation['d1']
        self.desired_d2 = formation['d2']
        self.desired_r1 = formation['r1']
        self.desired_r2 = formation['r2']
        self.desired_phi = np.radians(formation['phi_degrees'])
        self.position_gain = formation.get('position_gain', 1.0)

        print(f"Loaded quad formation config: d1={self.desired_d1}, d2={self.desired_d2}, "
              f"r1={self.desired_r1}, r2={self.desired_r2}, phi={np.degrees(self.desired_phi):.1f}°")

    def _initialize_robots(self):
        """Initialize 4 robots in approximate desired formation."""
        # Start at fixed position (0.5, 0.5)
        x_c_init =  np.random.uniform(0.5, 1.0)# 0.5
        y_c_init = np.random.uniform(-2.2, -2.25)#0.5
        theta_c_init = np.random.uniform(0, np.pi)#np.pi/2  # Initial orientation

        # Store initial theta for unwrapping
        self.prev_theta_c = theta_c_init

        # Compute initial robot positions using inverse kinematics
        x1, y1, x2, y2, x3, y3, x4, y4 = inverse_kinematics_quad(
            x_c_init, y_c_init, theta_c_init,
            self.desired_d1, self.desired_d2,
            self.desired_r1, self.desired_r2,
            self.desired_phi
        )

        # Create 4 robots
        self.robots = [
            Omnibot(x1, y1, self.timestep, self.momentum_alpha),
            Omnibot(x2, y2, self.timestep, self.momentum_alpha),
            Omnibot(x3, y3, self.timestep, self.momentum_alpha),
            Omnibot(x4, y4, self.timestep, self.momentum_alpha)
        ]

        print(f"Initialized 4 robots at centroid ({x_c_init:.3f}, {y_c_init:.3f})")

    def get_robot_positions(self):
        """Get current positions of all robots."""
        positions = [robot.get_position() for robot in self.robots]
        x1, y1 = positions[0]
        x2, y2 = positions[1]
        x3, y3 = positions[2]
        x4, y4 = positions[3]
        return x1, y1, x2, y2, x3, y3, x4, y4

    def get_current_formation(self):
        """
        Compute current formation parameters using forward kinematics.

        Returns:
            dict with current formation state (includes angle unwrapping)
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self.get_robot_positions()
        formation = forward_kinematics_quad(x1, y1, x2, y2, x3, y3, x4, y4,
                                           prev_theta_c=self.prev_theta_c)

        # Update previous theta for next iteration
        self.prev_theta_c = formation['th_c']

        return formation

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
            control_primitive: Function that takes QuadCluster and returns desired centroid velocity
                              Can return (vx_c, vy_c) or (vx_c, vy_c, omega_c)
        """
        # Get current formation state
        current_formation = self.get_current_formation()

        # Call control primitive to get desired centroid velocity
        result = control_primitive(self)

        # Check if primitive returned omega_c
        if len(result) == 2:
            vx_c_desired, vy_c_desired = result
            omega_c = 0.0  # Default: no rotation
        elif len(result) == 3:
            vx_c_desired, vy_c_desired, omega_c = result
        else:
            raise ValueError(f"Control primitive must return (vx_c, vy_c) or (vx_c, vy_c, omega_c)")

        # Compute formation errors
        error_d1 = self.desired_d1 - current_formation['d1']
        error_d2 = self.desired_d2 - current_formation['d2']
        error_r1 = self.desired_r1 - current_formation['r1']
        error_r2 = self.desired_r2 - current_formation['r2']
        error_phi = self.desired_phi - current_formation['phi']

        # Wrap phi error to [-pi, pi]
        error_phi = np.mod(error_phi + np.pi, 2*np.pi) - np.pi

        # Compute shape velocities (proportional control)
        vd1 = self.position_gain * error_d1
        vd2 = self.position_gain * error_d2
        vr1 = self.position_gain * error_r1
        vr2 = self.position_gain * error_r2
        vphi = self.position_gain * error_phi

        # Compute inverse Jacobian at current configuration
        J_inv = compute_inverse_jacobian_quad(
            current_formation['d1'],
            current_formation['d2'],
            current_formation['r1'],
            current_formation['r2'],
            current_formation['phi'],
            current_formation['th_c']
        )

        # Convert shape velocities to robot velocities
        vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4 = shape_velocities_to_robot_velocities_quad(
            J_inv, vx_c_desired, vy_c_desired, omega_c, vd1, vd2, vr1, vr2, vphi
        )

        # Command individual robots
        self.robots[0].command_velocity(vx1, vy1)
        self.robots[1].command_velocity(vx2, vy2)
        self.robots[2].command_velocity(vx3, vy3)
        self.robots[3].command_velocity(vx4, vy4)

        # Record centroid for trajectory tracking
        centroid = self.get_centroid()
        self.center_history.append(centroid.copy())

        # Record individual robot positions for trajectory tracking
        x1, y1, x2, y2, x3, y3, x4, y4 = self.get_robot_positions()
        robot_positions = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
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
        self.prev_theta_c = theta_c

        # Compute robot positions
        x1, y1, x2, y2, x3, y3, x4, y4 = inverse_kinematics_quad(
            x_c, y_c, theta_c,
            self.desired_d1, self.desired_d2,
            self.desired_r1, self.desired_r2,
            self.desired_phi
        )

        # Reset robot positions
        self.robots[0].set_position(x1, y1)
        self.robots[1].set_position(x2, y2)
        self.robots[2].set_position(x3, y3)
        self.robots[3].set_position(x4, y4)

        # Reset velocities
        for robot in self.robots:
            robot.velocity = np.array([0.0, 0.0])

        # Clear history
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
        colors = ['blue', 'yellow', 'green', 'red']

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
        return (f"QuadCluster(centroid=({formation['x_c']:.3f}, {formation['y_c']:.3f}), "
                f"d1={formation['d1']:.3f}, d2={formation['d2']:.3f}, "
                f"phi={np.degrees(formation['phi']):.1f}°)")
