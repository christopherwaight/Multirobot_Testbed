"""
PentagonCluster - Coordinated control of 6 omnidirectional robots in a pentagon
cluster-of-clusters formation for scalar field saddle navigation.

Formation: 3 pairs of robots whose midpoints form a triangle (SAS parameterization).
The 6 robots provide exactly enough samples for a full quadratic fit of a scalar field
(6 unknowns: value, gradient (x,y), Hessian (xx,xy,yy)).

Control architecture (same two-tier structure as OmniCluster/QuadCluster):
  1. Control primitive returns desired centroid velocity (vx_c, vy_c).
  2. Cluster applies formation maintenance (proportional error correction on all
     shape parameters) plus the centroid command via inverse Jacobian.
  3. Each robot applies momentum dynamics and physical constraints.
"""
import numpy as np
import yaml
import os

from .omnibot import Omnibot
from ..control.pentagon_kinematics import (
    forward_kinematics,
    inverse_kinematics,
    compute_inverse_jacobian,
    shape_velocities_to_robot_velocities,
    STATE_VARS,
)


class PentagonCluster:
    """
    Manages a cluster of 6 omnidirectional robots in a pentagon formation.
    """

    def __init__(self, formation_config_path, field, timestep=0.1, momentum_alpha=0.7,
                 max_velocity=0.3, stiction_threshold=0.025):
        self.field = field
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha
        self.max_velocity = max_velocity
        self.stiction_threshold = stiction_threshold

        self._load_formation_config(formation_config_path)
        self._initialize_robots()

        self.center_history = []
        self.robot_history = []
        self.velocity_history = []
        self.diagnostics = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_formation_config(self, config_path):
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, '..', '..')
            config_path = os.path.join(project_root, config_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        f = config['formation']
        self.desired_L_2     = f['L_2']
        self.desired_theta_2 = f['theta_2']
        self.desired_L_3     = f['L_3']
        self.desired_theta_3 = f['theta_3']
        self.desired_L_4     = f['L_4']
        self.desired_theta_4 = f['theta_4']
        self.desired_p_1     = f['p_1']
        self.desired_beta_1  = f['beta_1']
        self.desired_q_1     = f['q_1']
        self.desired_theta_c = f['theta_c']
        self.position_gain   = f.get('position_gain', 1.0)
        self.angle_gain      = f.get('angle_gain', 0.1)

        print(f"Loaded pentagon formation: p_1={self.desired_p_1:.3f}, "
              f"q_1={self.desired_q_1:.3f}, L_2={self.desired_L_2:.3f}")

    def _initialize_robots(self):
        x_c_init = 0.5
        y_c_init = 0.5
        coords = inverse_kinematics(
            x_c_init, y_c_init, self.desired_theta_c,
            self.desired_p_1, self.desired_beta_1, self.desired_q_1,
            self.desired_L_2, self.desired_theta_2,
            self.desired_L_3, self.desired_theta_3,
            self.desired_L_4, self.desired_theta_4,
        )
        # coords = (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6)
        self.robots = [
            Omnibot(coords[2*i], coords[2*i + 1], self.timestep, self.momentum_alpha,
                     max_velocity=self.max_velocity, stiction_threshold=self.stiction_threshold)
            for i in range(6)
        ]
        print(f"Initialized 6 robots at centroid ({x_c_init:.3f}, {y_c_init:.3f})")

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_robot_positions(self):
        """Return flat tuple (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6)."""
        out = []
        for robot in self.robots:
            pos = robot.get_position()
            out.extend([pos[0], pos[1]])
        return tuple(out)

    def get_current_formation(self):
        coords = self.get_robot_positions()
        return forward_kinematics(*coords)

    def get_centroid(self):
        f = self.get_current_formation()
        return np.array([f['x_c'], f['y_c']])

    def sample_field_at_robots(self):
        """Return list of 6 scalar field readings, one per robot."""
        return [self.field.get_value(r.position[0], r.position[1])
                for r in self.robots]

    # ------------------------------------------------------------------
    # Control step
    # ------------------------------------------------------------------

    def move(self, control_primitive):
        """
        Advance the cluster one timestep.

        Args:
            control_primitive: callable(cluster) -> (vx_c, vy_c)
        """
        current = self.get_current_formation()
        vx_c, vy_c = control_primitive(self)

        # Formation errors (proportional control)
        desired = {
            'L_2':     self.desired_L_2,
            'theta_2': self.desired_theta_2,
            'L_3':     self.desired_L_3,
            'theta_3': self.desired_theta_3,
            'L_4':     self.desired_L_4,
            'theta_4': self.desired_theta_4,
            'p_1':     self.desired_p_1,
            'beta_1':  self.desired_beta_1,
            'q_1':     self.desired_q_1,
        }

        col = {v: i for i, v in enumerate(STATE_VARS)}
        q_dot = np.zeros(12)

        # Centroid velocity
        q_dot[col['x_c']] = vx_c
        q_dot[col['y_c']] = vy_c
        q_dot[col['theta_c']] = 0.0  # no spin

        # Shape corrections
        for var, des in desired.items():
            err = des - current[var]
            # Wrap angle errors to [-pi, pi]
            if 'theta' in var or 'beta' in var:
                while err >  np.pi: err -= 2 * np.pi
                while err < -np.pi: err += 2 * np.pi
                q_dot[col[var]] = self.angle_gain * err
            else:
                q_dot[col[var]] = self.position_gain * err

        # Robot velocities via inverse Jacobian
        J = compute_inverse_jacobian(current)
        r_dot = shape_velocities_to_robot_velocities(J, q_dot)

        for i, robot in enumerate(self.robots):
            robot.command_velocity(r_dot[2*i], r_dot[2*i + 1])

        # Record history
        centroid = self.get_centroid()
        self.center_history.append(centroid.copy())

        coords = self.get_robot_positions()
        robot_positions = np.array([[coords[2*i], coords[2*i+1]] for i in range(6)])
        self.robot_history.append(robot_positions.copy())

        velocities = np.array([robot.get_velocity() for robot in self.robots])
        self.velocity_history.append(velocities.copy())

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_center_history(self):
        return np.array(self.center_history)

    def get_robot_history(self):
        return np.array(self.robot_history)

    def get_velocity_history(self):
        return self.velocity_history

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, x_c=None, y_c=None):
        if x_c is None:
            x_c = 0.5
        if y_c is None:
            y_c = 0.5

        coords = inverse_kinematics(
            x_c, y_c, self.desired_theta_c,
            self.desired_p_1, self.desired_beta_1, self.desired_q_1,
            self.desired_L_2, self.desired_theta_2,
            self.desired_L_3, self.desired_theta_3,
            self.desired_L_4, self.desired_theta_4,
        )

        for i, robot in enumerate(self.robots):
            robot.set_position(coords[2*i], coords[2*i + 1])
            robot.velocity = np.array([0.0, 0.0])

        self.center_history = []
        self.robot_history = []
        self.velocity_history = []
        self.diagnostics = []

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        for i, robot in enumerate(self.robots):
            pos = robot.get_position()
            ax.scatter(pos[0], pos[1], color=colors[i], s=100, zorder=5)

    def __repr__(self):
        c = self.get_centroid()
        return f"PentagonCluster(centroid=({c[0]:.3f}, {c[1]:.3f}), N=6)"
