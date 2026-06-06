"""Hub-and-spoke kinematics (Config 1)."""

import math
import sys
from typing import Dict

import numpy as np

from .backend import NumpyBackend, SympyBackend
from .errors import ClusterBuilderError


class HubAndSpokeKinematics:
    """
    N-robot hub-and-spoke formation.
    Hub = robot N (last index, 1-based).
    Spokes = robots 1 .. N-1.

    State vector q (length 2N, orientation=off):
        [x_h, y_h, theta_c, r_1, ..., r_{N-1}, gamma_2, ..., gamma_{N-1}]

    With orientation=on, appends [phi_1, ..., phi_N] for a total of 3N state vars.
    theta_ref for every robot is 'theta_c' (the hub frame is the root frame).
    phi_i = wrap(theta_i - theta_c) for each robot heading theta_i.

    Pointer gauge: theta_c = bearing from hub to spoke 1, gamma_1 ≡ 0.
    """

    def __init__(self, N: int, orientation: bool = False):
        if N < 2:
            raise ClusterBuilderError("Hub-and-spoke requires at least 2 robots.")
        self.N = N
        self.n_spokes = N - 1
        self.orientation = orientation
        self.state_vars = (
            ['x_h', 'y_h', 'theta_c'] +
            [f'r_{i}' for i in range(1, N)] +
            [f'gamma_{i}' for i in range(2, N)]
        )
        if not orientation:
            assert len(self.state_vars) == 2 * N
        if orientation:
            self.state_vars = self.state_vars + [f'phi_{i}' for i in range(1, N + 1)]
            assert len(self.state_vars) == 3 * N
        self.theta_ref_var = {i: 'theta_c' for i in range(1, N + 1)}

    def default_state(self) -> Dict:
        """Evenly spaced spokes at radius 0.3; phi_i=0 when orientation=on."""
        N = self.N
        state = {'x_h': 0.0, 'y_h': 0.0, 'theta_c': 0.0}
        for i in range(1, N):
            state[f'r_{i}'] = 0.3
        for i in range(2, N):
            raw = 2 * math.pi * (i - 1) / (N - 1)
            state[f'gamma_{i}'] = (raw + math.pi) % (2 * math.pi) - math.pi
        if self.orientation:
            for i in range(1, N + 1):
                state[f'phi_{i}'] = 0.0
        return state

    def forward_kinematics(self, positions: Dict) -> Dict:
        """
        positions: dict {robot_idx (1-based): (x, y)} when orientation=off,
                   or {robot_idx: (x, y, theta)} when orientation=on.
        Returns state dict.
        """
        if self.orientation:
            x_h, y_h = positions[self.N][0], positions[self.N][1]
        else:
            x_h, y_h = positions[self.N]
        betas = {}
        state = {'x_h': x_h, 'y_h': y_h}
        for i in range(1, self.N):
            if self.orientation:
                xi, yi = positions[i][0], positions[i][1]
            else:
                xi, yi = positions[i]
            dx, dy = xi - x_h, yi - y_h
            state[f'r_{i}'] = math.hypot(dx, dy)
            betas[i] = math.atan2(dy, dx)
        state['theta_c'] = betas[1]
        for i in range(2, self.N):
            gamma = betas[i] - betas[1]
            gamma = (gamma + math.pi) % (2 * math.pi) - math.pi
            state[f'gamma_{i}'] = gamma
        if self.orientation:
            for i in range(1, self.N + 1):
                theta_i = positions[i][2]
                theta_ref = state[self.theta_ref_var[i]]
                phi_i = (theta_i - theta_ref + math.pi) % (2 * math.pi) - math.pi
                state[f'phi_{i}'] = phi_i
        return state

    def inverse_kinematics(self, state: Dict) -> Dict:
        """Returns dict {robot_idx: (x, y)} when orientation=off,
        or {robot_idx: (x, y, theta)} when orientation=on."""
        x_h = state['x_h']
        y_h = state['y_h']
        theta_c = state['theta_c']
        if self.orientation:
            theta_hub = (theta_c + state['phi_{}'.format(self.N)] + math.pi) % (2 * math.pi) - math.pi
            positions = {self.N: (x_h, y_h, theta_hub)}
        else:
            positions = {self.N: (x_h, y_h)}
        for i in range(1, self.N):
            r_i = state[f'r_{i}']
            gamma_i = 0.0 if i == 1 else state[f'gamma_{i}']
            angle = theta_c + gamma_i
            xi = x_h + r_i * math.cos(angle)
            yi = y_h + r_i * math.sin(angle)
            if self.orientation:
                theta_i = (state[self.theta_ref_var[i]] + state[f'phi_{i}'] + math.pi) % (2 * math.pi) - math.pi
                positions[i] = (xi, yi, theta_i)
            else:
                positions[i] = (xi, yi)
        return positions

    def inverse_jacobian(self, state: Dict, backend=None) -> np.ndarray:
        """
        Analytical inverse Jacobian.
        orientation=off: (2N x 2N). Columns: [x_h, y_h, theta_c, r_1,...,r_{N-1}, gamma_2,...,gamma_{N-1}]
                         Rows: [x_1, y_1, ..., x_{N-1}, y_{N-1}, x_N, y_N]
        orientation=on:  (3N x 3N). Same position block, plus N heading rows:
                         [x_1,y_1,...,x_N,y_N, theta_1,...,theta_N]
                         phi columns are exactly zero in position rows.
                         For each robot i: d(theta_i)/d(phi_i)=1, d(theta_i)/d(theta_c)=1.
        backend: NumpyBackend (default) or SympyBackend for symbolic output.
        """
        if backend is None:
            backend = NumpyBackend
        N = self.N
        col_idx = {v: k for k, v in enumerate(self.state_vars)}
        n_rows = 3 * N if self.orientation else 2 * N
        J = backend.zeros((n_rows, len(self.state_vars)))
        theta_c = state['theta_c']

        for i in range(1, N):
            rx = 2 * (i - 1)
            ry = rx + 1
            r_i = state[f'r_{i}']
            g_i = 0 if i == 1 else state[f'gamma_{i}']
            alpha = theta_c + g_i
            ca, sa = backend.cos(alpha), backend.sin(alpha)

            J[rx, col_idx['x_h']] = 1
            J[ry, col_idx['y_h']] = 1
            J[rx, col_idx['theta_c']] = -r_i * sa
            J[ry, col_idx['theta_c']] =  r_i * ca
            J[rx, col_idx[f'r_{i}']] = ca
            J[ry, col_idx[f'r_{i}']] = sa
            if i >= 2:
                J[rx, col_idx[f'gamma_{i}']] = -r_i * sa
                J[ry, col_idx[f'gamma_{i}']] =  r_i * ca

        J[2 * (N - 1),     col_idx['x_h']] = 1
        J[2 * (N - 1) + 1, col_idx['y_h']] = 1

        if self.orientation:
            for i in range(1, N + 1):
                heading_row = 2 * N + (i - 1)
                J[heading_row, col_idx[f'phi_{i}']] = 1
                J[heading_row, col_idx[self.theta_ref_var[i]]] += 1

        return J

    def inverse_jacobian_symbolic(self, state: Dict = None):
        """
        Returns the symbolic inverse Jacobian as a sympy.Matrix.
        state: dict mapping state_var names to numeric values or sympy symbols.
               If None, uses fresh sympy.Symbol for every state variable.
        """
        try:
            import sympy
        except ImportError:
            sys.exit("sympy is required for symbolic Jacobian: pip install sympy")
        if state is None:
            state = {v: sympy.Symbol(v) for v in self.state_vars}
        return self.inverse_jacobian(state, backend=SympyBackend)

    def forward_jacobian(self, state: Dict) -> np.ndarray:
        try:
            return np.linalg.inv(self.inverse_jacobian(state))
        except np.linalg.LinAlgError:
            print("Warning: J_inv singular, using pseudo-inverse.")
            return np.linalg.pinv(self.inverse_jacobian(state))
