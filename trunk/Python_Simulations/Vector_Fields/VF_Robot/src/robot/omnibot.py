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

    Movement model (paper Eqs. 12-13, first-order actuator lag):
        velocity = alpha * velocity_old + (1 - alpha) * velocity_commanded
        position += timestep * velocity

    Parameter note (audit 2026-07-03): the paper derives
    alpha = exp(-dt/tau) = exp(-0.1/0.3) ~= 0.717 for the measured Decabot
    time constant tau = 0.3 s. The default here is the fixed value 0.7,
    which corresponds to tau = -dt/ln(0.7) ~= 0.280 s and does NOT track dt.
    Closed-loop effect is small (Table II bias/precision reproduce with
    either value), but the two are not identical. Do not change without
    re-running the Monte Carlo baselines.
    """

    def __init__(self, x, y, timestep=0.1, momentum_alpha=0.7,
                 max_velocity=0.3, stiction_threshold=0.025, stiction_model="legacy"):
        """
        Initialize an Omnibot.

        Args:
            x: Initial x position
            y: Initial y position
            timestep: Time step for integration
            momentum_alpha: Momentum coefficient (0=no momentum, 1=full momentum)
            max_velocity: Maximum robot speed (m/s), default matches Decabot
                hardware (0.3 m/s).
            stiction_threshold: Minimum speed to overcome static friction
                (m/s), default matches Decabot hardware (0.025 m/s). Callers
                simulating a different vehicle (e.g. a boat, where "stiction"
                has little physical meaning) can lower this toward 0.
            stiction_model: how the threshold is applied (added 2026-10-29).
                "legacy"   zero the lagged velocity whenever its pre-cap
                           magnitude is below the threshold (original model).
                           A robot at rest then needs a command above
                           threshold / (1 - alpha), so the effective
                           threshold changes with the sample period.
                "deadzone" commands below the threshold are treated as zero
                           before the lag, v' = alpha v + (1 - alpha) DZ(u).
                           This is the exact zero-order-hold discretization of
                           tau dv/dt = -v + DZ(u), so it does not depend on
                           the sample period; a robot coasts to rest with
                           time constant tau once its command is dead.
                "scaled"   legacy test with threshold (1 - alpha) v_s; starts
                           match "deadzone" but a moving robot stops only
                           below (1 - alpha) v_s, which vanishes as dt -> 0.
                The default stays "legacy" so existing results reproduce.
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha
        self.max_velocity = max_velocity
        self.stiction_threshold = stiction_threshold
        if stiction_model not in ("legacy", "deadzone", "scaled"):
            raise ValueError(f"unknown stiction_model {stiction_model!r}")
        self.stiction_model = stiction_model

    def command_velocity(self, vx_cmd, vy_cmd):
        """
        Command a velocity and update the robot's state.

        Args:
            vx_cmd: Commanded x velocity
            vy_cmd: Commanded y velocity
        """
        # Apply momentum dynamics
        velocity_commanded = np.array([vx_cmd, vy_cmd], dtype=float)
        if (self.stiction_model == "deadzone"
                and np.linalg.norm(velocity_commanded) < self.stiction_threshold):
            velocity_commanded = np.zeros(2)
        self.velocity = (self.momentum_alpha * self.velocity +
                        (1 - self.momentum_alpha) * velocity_commanded)

        # ========================================================================
        # PHYSICAL CONSTRAINTS
        # ========================================================================

        # 1. Maximum velocity constraint (motor/safety limits)
        velocity_magnitude = np.linalg.norm(self.velocity)

        if velocity_magnitude > self.max_velocity:
            # Scale velocity vector to max magnitude while preserving direction
            self.velocity = (self.velocity / velocity_magnitude) * self.max_velocity

        # 2. Stiction (static friction) constraint
        # Below this threshold, friction overcomes motion and robot stops.
        # NOTE: this compares the PRE-clamp magnitude from step 1. Inert in
        # practice because stiction_threshold << max_velocity (a magnitude
        # cannot be both above max and below stiction), but the two
        # constraints intentionally share the same measured magnitude.
        if self.stiction_model == "legacy":
            threshold = self.stiction_threshold
        elif self.stiction_model == "scaled":
            threshold = (1 - self.momentum_alpha) * self.stiction_threshold
        else:
            threshold = 0.0      # deadzone acts on the command above
        if velocity_magnitude < threshold:
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
