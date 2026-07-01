"""
Omnibot - Individual robot with momentum-based dynamics
"""
import numpy as np


class Omnibot:
    """
    Individual omnidirectional robot with momentum-based dynamics.

    State:
        - position: (x, y)
        - velocity: (vx, vy)

    Movement model:
        velocity = alpha * velocity_old + (1 - alpha) * velocity_commanded
        position += timestep * velocity
    """

    def __init__(self, x, y, timestep=0.1, momentum_alpha=0.7):
        """
        Initialize an Omnibot.

        Args:
            x: Initial x position
            y: Initial y position
            timestep: Time step for integration
            momentum_alpha: Momentum coefficient (0=no momentum, 1=full momentum)
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha

    def command_velocity(self, vx_cmd, vy_cmd):
        """
        Command a velocity and update the robot's state.

        Args:
            vx_cmd: Commanded x velocity
            vy_cmd: Commanded y velocity
        """
        # Apply momentum dynamics
        velocity_commanded = np.array([vx_cmd, vy_cmd], dtype=float)
        self.velocity = (self.momentum_alpha * self.velocity +
                        (1 - self.momentum_alpha) * velocity_commanded)

        # ========================================================================
        # PHYSICAL CONSTRAINTS
        # ========================================================================

        # 1. Maximum velocity constraint (motor/safety limits)
        # Clamp velocity magnitude to 0.3 m/s maximum
        velocity_magnitude = np.linalg.norm(self.velocity)
        MAX_VELOCITY = 0.3  # meters per second

        if velocity_magnitude > MAX_VELOCITY:
            # Scale velocity vector to max magnitude while preserving direction
            self.velocity = (self.velocity / velocity_magnitude) * MAX_VELOCITY

        # 2. Stiction (static friction) constraint
        # Below this threshold, friction overcomes motion and robot stops
        STICTION_THRESHOLD = 0.025  # meters per second

        if velocity_magnitude < STICTION_THRESHOLD:
            # Velocity too small to overcome static friction - robot stops
            self.velocity = np.array([0.0, 0.0], dtype=float)

        # ========================================================================

        # Update position
        self.position += self.timestep * self.velocity

    def sample_field(self, field):
        """
        Sample the vector field at the robot's current position.

        Args:
            field: Field object with get_value(x, y) method

        Returns:
            (u, v): Vector field reading at robot position
        """
        return field.get_value(self.position[0], self.position[1])

    def get_position(self):
        """Get current position as numpy array."""
        return self.position.copy()

    def get_velocity(self):
        """Get current velocity as numpy array."""
        return self.velocity.copy()

    def set_position(self, x, y):
        """Set robot position (for initialization)."""
        self.position = np.array([x, y], dtype=float)

    def __repr__(self):
        return f"Omnibot(pos={self.position}, vel={self.velocity})"
