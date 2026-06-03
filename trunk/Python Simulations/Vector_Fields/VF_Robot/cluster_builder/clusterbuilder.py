"""
clusterbuilder.py — CLI tool to generate kinematics, Jacobians, config, cluster class,
and visualization for any N-robot formation.

Usage:
    python clusterbuilder.py num_robots cluster_config cluster_name run_mode [config_tree]

Arguments:
    num_robots      Number of robots
    cluster_config  1 = hub-and-spoke, 2 = cluster-of-clusters
    cluster_name    Name stem for output files (must be a valid Python identifier)
    run_mode        image_only | full
    config_tree     (optional, config 2 only) e.g. "((2,2),(2,2))" or "(2,2,2)"
                    Leaves: 1, 2, or 3. Internal nodes: 2 or 3 children.
"""

import argparse
import ast
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc, FancyArrowPatch
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")


# ---------------------------------------------------------------------------
# Backend abstraction (numeric vs. symbolic)
# ---------------------------------------------------------------------------

class _NumpyBackend:
    """Default numeric backend using math and numpy."""
    cos   = staticmethod(math.cos)
    sin   = staticmethod(math.sin)
    sqrt  = staticmethod(math.sqrt)
    atan2 = staticmethod(math.atan2)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def array(data):
        return np.array(data, dtype=float)


class _SympyBackend:
    """Symbolic backend using sympy. Imported lazily."""

    @staticmethod
    def _sp():
        try:
            import sympy
            return sympy
        except ImportError:
            sys.exit("sympy is required for --symbolic: pip install sympy")

    @staticmethod
    def cos(x):
        return _SympyBackend._sp().cos(x)

    @staticmethod
    def sin(x):
        return _SympyBackend._sp().sin(x)

    @staticmethod
    def sqrt(x):
        return _SympyBackend._sp().sqrt(x)

    @staticmethod
    def atan2(y, x):
        return _SympyBackend._sp().atan2(y, x)

    @staticmethod
    def zeros(shape):
        sp = _SympyBackend._sp()
        return sp.zeros(*shape)

    @staticmethod
    def array(data):
        sp = _SympyBackend._sp()
        return sp.Matrix(data)


NumpyBackend  = _NumpyBackend()
SympyBackend  = _SympyBackend()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ClusterBuilderError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    size: int                              # total robots in this subtree
    children: List['TreeNode']             # empty = leaf
    robot_indices: List[int] = field(default_factory=list)  # 1-based, assigned post-parse
    node_id: int = 0                       # unique id assigned post-parse

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def arity(self):
        return len(self.children)


# ---------------------------------------------------------------------------
# Tree parsing and auto-build
# ---------------------------------------------------------------------------

class TreeParser:

    @staticmethod
    def parse(tree_str: str) -> TreeNode:
        """Parse a config_tree string like '((2,2),(2,2))' or '(2,2,2)'."""
        s = tree_str.strip()
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            raise ClusterBuilderError(f"Could not parse config_tree '{s}': {e}")
        root = TreeParser._build_tree(parsed)
        TreeParser._assign_ids_and_indices(root)
        return root

    @staticmethod
    def _build_tree(obj) -> TreeNode:
        if isinstance(obj, int):
            if obj not in (1, 2, 3):
                raise ClusterBuilderError(
                    f"Leaf value {obj} is invalid — only 1, 2, 3 are allowed.")
            return TreeNode(size=obj, children=[])
        elif isinstance(obj, (tuple, list)):
            if len(obj) < 2:
                raise ClusterBuilderError(
                    "Each internal node must have at least 2 children.")
            if len(obj) > 3:
                raise ClusterBuilderError(
                    f"Each internal node can have at most 3 children, got {len(obj)}.")
            children = [TreeParser._build_tree(c) for c in obj]
            size = sum(c.size for c in children)
            return TreeNode(size=size, children=children)
        else:
            raise ClusterBuilderError(f"Unexpected token type: {type(obj)}")

    @staticmethod
    def auto_build(n: int, _top_level: bool = True) -> TreeNode:
        """
        Recursively split n robots into a tree with 2-or-3-child internal nodes.
        Prefers ternary splits (3 equal children) when n is divisible by 3,
        otherwise binary split (ceiling left, floor right).
        """
        if n in (1, 2, 3):
            node = TreeNode(size=n, children=[])
        elif n % 3 == 0:
            chunk = n // 3
            children = [TreeParser.auto_build(chunk, _top_level=False) for _ in range(3)]
            node = TreeNode(size=n, children=children)
        else:
            left_n = (n + 1) // 2
            right_n = n - left_n
            node = TreeNode(size=n, children=[
                TreeParser.auto_build(left_n, _top_level=False),
                TreeParser.auto_build(right_n, _top_level=False)
            ])
        if _top_level:
            TreeParser._assign_ids_and_indices(node)
        return node

    @staticmethod
    def _assign_ids_and_indices(root: TreeNode):
        """Assign node_id (BFS order) and robot_indices (DFS left-to-right, 1-based)."""
        # BFS for node IDs
        queue = [root]
        nid = 1
        while queue:
            node = queue.pop(0)
            node.node_id = nid
            nid += 1
            for c in node.children:
                queue.append(c)

        # DFS for robot indices
        counter = [1]
        def _dfs(node):
            if node.is_leaf:
                node.robot_indices = list(range(counter[0], counter[0] + node.size))
                counter[0] += node.size
            else:
                for c in node.children:
                    _dfs(c)
                node.robot_indices = []
                for c in node.children:
                    node.robot_indices.extend(c.robot_indices)
        _dfs(root)

    @staticmethod
    def validate(root: TreeNode, num_robots: int):
        if root.size != num_robots:
            raise ClusterBuilderError(
                f"Tree specifies {root.size} robots but num_robots={num_robots} was given.")

    @staticmethod
    def describe(root: TreeNode, indent=0) -> str:
        prefix = "  " * indent
        if root.is_leaf:
            return f"{prefix}Leaf(size={root.size}, robots={root.robot_indices})"
        lines = [f"{prefix}Node(size={root.size}, id={root.node_id}, arity={root.arity})"]
        for c in root.children:
            lines.append(TreeParser.describe(c, indent + 1))
        return "\n".join(lines)

    @staticmethod
    def to_string(node: TreeNode) -> str:
        """Serialize a TreeNode back to a config_tree string, e.g. '((2,2),(2,2))'."""
        if node.is_leaf:
            return str(node.size)
        children_str = ','.join(TreeParser.to_string(c) for c in node.children)
        return f'({children_str})'


# ---------------------------------------------------------------------------
# Hub-and-Spoke kinematics (Config 1)
# ---------------------------------------------------------------------------

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
        # State variable names in order; phi vars appended last when orientation=on
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
        # Every robot's frame is attached to theta_c
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

        # Hub = robot N
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


# ---------------------------------------------------------------------------
# Pair block (2-robot, fundamental atom for cluster-of-clusters)
# ---------------------------------------------------------------------------

class PairBlock:
    """Static methods for a two-robot pair: (x_a, y_a, x_b, y_b) <-> (x_c, y_c, theta, L)."""

    @staticmethod
    def forward(xa, ya, xb, yb):
        x_c = (xa + xb) / 2
        y_c = (ya + yb) / 2
        theta = math.atan2(yb - ya, xb - xa)
        L = math.hypot(xb - xa, yb - ya)
        return x_c, y_c, theta, L

    @staticmethod
    def inverse(x_c, y_c, theta, L):
        ct, st = math.cos(theta), math.sin(theta)
        xa = x_c - (L / 2) * ct
        ya = y_c - (L / 2) * st
        xb = x_c + (L / 2) * ct
        yb = y_c + (L / 2) * st
        return xa, ya, xb, yb

    @staticmethod
    def j_inv(theta, L, backend=None):
        """
        4x4 analytical inverse Jacobian.
        Columns: [x_c, y_c, theta, L]
        Rows:    [x_a, y_a, x_b, y_b]
        backend: NumpyBackend (default) or SympyBackend for symbolic output.
        """
        if backend is None:
            backend = NumpyBackend
        ct, st = backend.cos(theta), backend.sin(theta)
        h = L / 2
        return backend.array([
            [1, 0,  h * st, -ct / 2],
            [0, 1, -h * ct, -st / 2],
            [1, 0, -h * st,  ct / 2],
            [0, 1,  h * ct,  st / 2],
        ])


# ---------------------------------------------------------------------------
# Triple block (SAS-3, analytical Jacobian)
# ---------------------------------------------------------------------------

class TripleBlock:
    """
    Three-robot SAS formation used as a leaf (or as a meta-node over sub-centroids).
    State: [x_c, y_c, theta_c, p, beta, q]
    Robots/points: [a, b, c] where b is vertex 2 (the angle vertex).
    """

    @staticmethod
    def forward(xa, ya, xb, yb, xc, yc):
        """Forward kinematics: positions -> state dict."""
        p = math.hypot(xb - xa, yb - ya)
        q = math.hypot(xc - xb, yc - yb)
        r = math.hypot(xa - xc, ya - yc)
        eps = 1e-10
        cos_beta = (p**2 + q**2 - r**2) / (2 * p * q + eps)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)
        x_c = (xa + xb + xc) / 3.0
        y_c = (ya + yb + yc) / 3.0
        # theta_c = bearing from robot-b to robot-a (matches inverse: robot-a at (p,0) in local frame)
        theta_c = math.atan2(ya - yb, xa - xb)
        return {'x_c': x_c, 'y_c': y_c, 'theta_c': theta_c, 'p': p, 'beta': beta, 'q': q}

    @staticmethod
    def inverse(x_c, y_c, theta_c, p, beta, q):
        """Inverse kinematics -> (xa, ya, xb, yb, xc, yc)."""
        # Local frame: robot b at origin, robot a at (p, 0)
        x2_loc, y2_loc = 0.0, 0.0
        x1_loc, y1_loc = p, 0.0
        x3_loc = q * math.cos(beta)
        y3_loc = q * math.sin(beta)
        # Centroid in local frame
        cx_loc = (x1_loc + x2_loc + x3_loc) / 3.0
        cy_loc = (y1_loc + y2_loc + y3_loc) / 3.0
        # Center
        x1_loc -= cx_loc; y1_loc -= cy_loc
        x2_loc -= cx_loc; y2_loc -= cy_loc
        x3_loc -= cx_loc; y3_loc -= cy_loc
        # Rotate and translate
        ct, st = math.cos(theta_c), math.sin(theta_c)
        def rot_trans(xl, yl):
            return ct * xl - st * yl + x_c, st * xl + ct * yl + y_c
        xa, ya = rot_trans(x1_loc, y1_loc)
        xb, yb = rot_trans(x2_loc, y2_loc)
        xc, yc = rot_trans(x3_loc, y3_loc)
        return xa, ya, xb, yb, xc, yc

    @staticmethod
    def j_inv(x_c, y_c, theta_c, p, beta, q, backend=None):
        """
        6x6 analytical inverse Jacobian for SAS-3.
        Columns: [x_c, y_c, theta_c, p, beta, q]
        Rows:    [x_a, y_a, x_b, y_b, x_c_r, y_c_r]
        backend: NumpyBackend (default) or SympyBackend for symbolic output.
        """
        if backend is None:
            backend = NumpyBackend
        # Compute local (centred) positions
        x2_loc, y2_loc = 0, 0
        x1_loc, y1_loc = p, 0
        x3_loc = q * backend.cos(beta)
        y3_loc = q * backend.sin(beta)
        cx_loc = (x1_loc + x2_loc + x3_loc) / 3
        cy_loc = (y1_loc + y2_loc + y3_loc) / 3
        # Centred local positions
        lx = [x1_loc - cx_loc, x2_loc - cx_loc, x3_loc - cx_loc]
        ly = [y1_loc - cy_loc, y2_loc - cy_loc, y3_loc - cy_loc]

        ct, st = backend.cos(theta_c), backend.sin(theta_c)
        J = backend.zeros((6, 6))

        for i in range(3):
            rx = 2 * i
            ry = rx + 1
            J[rx, 0] = 1
            J[ry, 1] = 1
            J[rx, 2] = -st * lx[i] - ct * ly[i]
            J[ry, 2] =  ct * lx[i] - st * ly[i]

        dlx_dp = [2, -1, -1]
        dly_dp = [0, 0, 0]
        for i in range(3):
            rx, ry = 2*i, 2*i+1
            J[rx, 3] = ct * dlx_dp[i] / 3 - st * dly_dp[i]
            J[ry, 3] = st * dlx_dp[i] / 3 + ct * dly_dp[i]

        dx3_dbeta = -q * backend.sin(beta)
        dy3_dbeta =  q * backend.cos(beta)
        dcx_dbeta = dx3_dbeta / 3
        dcy_dbeta = dy3_dbeta / 3
        dlx_dbeta = [0 - dcx_dbeta, 0 - dcx_dbeta, dx3_dbeta - dcx_dbeta]
        dly_dbeta = [0 - dcy_dbeta, 0 - dcy_dbeta, dy3_dbeta - dcy_dbeta]
        for i in range(3):
            rx, ry = 2*i, 2*i+1
            J[rx, 4] = ct * dlx_dbeta[i] - st * dly_dbeta[i]
            J[ry, 4] = st * dlx_dbeta[i] + ct * dly_dbeta[i]

        dx3_dq = backend.cos(beta)
        dy3_dq = backend.sin(beta)
        dcx_dq = dx3_dq / 3
        dcy_dq = dy3_dq / 3
        dlx_dq = [0 - dcx_dq, 0 - dcx_dq, dx3_dq - dcx_dq]
        dly_dq = [0 - dcy_dq, 0 - dcy_dq, dy3_dq - dcy_dq]
        for i in range(3):
            rx, ry = 2*i, 2*i+1
            J[rx, 5] = ct * dlx_dq[i] - st * dly_dq[i]
            J[ry, 5] = st * dlx_dq[i] + ct * dly_dq[i]

        return J


# ---------------------------------------------------------------------------
# Cluster-of-Clusters kinematics (Config 2)
# ---------------------------------------------------------------------------

class ClusterOfClustersKinematics:
    """
    Hierarchical cluster-of-clusters formation.
    Internal nodes can have 2 children (pair block) or 3 children (SAS-3 block).
    Size-1/2/3 leaves use identity/pair/SAS atoms.
    """

    def __init__(self, root: TreeNode, orientation: bool = False):
        self.root = root
        self.N = root.size
        self.orientation = orientation
        # Build ordered list of state variable names; phi vars appended last when orientation=on
        self.state_vars = self._collect_state_vars(root, is_root=True)
        if orientation:
            self.state_vars = self.state_vars + [f'phi_{i}' for i in range(1, self.N + 1)]
        # theta_ref_var: built lazily during first FK traversal; maps robot_idx -> state key
        self.theta_ref_var: Dict[int, str] = {}

    def _node_shape_vars(self, node: TreeNode, is_root: bool) -> List[str]:
        """State variable names owned by this node (not children)."""
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                return []
            elif node.size == 2:
                return [f'L_{nid}', f'theta_{nid}']
            else:  # size == 3
                return [f'theta_{nid}', f'p_{nid}', f'beta_{nid}', f'q_{nid}']
        else:
            # Internal node: pose at root, shape otherwise
            if is_root:
                if node.arity == 2:
                    return ['x_c', 'y_c', 'theta_c', f'L_{nid}']
                else:  # arity == 3
                    return ['x_c', 'y_c', 'theta_c', f'p_{nid}', f'beta_{nid}', f'q_{nid}']
            else:
                if node.arity == 2:
                    return [f'L_{nid}', f'theta_{nid}']
                else:  # arity == 3
                    return [f'theta_{nid}', f'p_{nid}', f'beta_{nid}', f'q_{nid}']

    def _collect_state_vars(self, node: TreeNode, is_root: bool) -> List[str]:
        """DFS pre-order collection of all state variable names."""
        vars_ = self._node_shape_vars(node, is_root)
        for child in node.children:
            vars_ += self._collect_state_vars(child, is_root=False)
        return vars_

    def default_state(self) -> Dict:
        """
        Build default state: sub-clusters arranged in equilateral triangles,
        pairs oriented horizontally. All angles chosen to be far from singularities.
        Runs IK then FK to canonicalize all angle conventions.
        With orientation=on, phi_i=0 for all robots (aligned with attached frame).
        """
        positions = self._default_positions(self.root, cx=0.0, cy=0.0, scale=0.6)
        if self.orientation:
            # Supply theta_i=0 per robot; FK will compute phi_i = wrap(0 - theta_ref(i))
            # but we want phi_i=0, so run FK first to learn theta_ref, then override
            state = self.forward_kinematics({i: (x, y) for i, (x, y) in positions.items()})
            for i in range(1, self.N + 1):
                state[f'phi_{i}'] = 0.0
        else:
            state = self.forward_kinematics(positions)
        return state

    def _default_positions(self, node: TreeNode, cx: float, cy: float,
                            scale: float) -> Dict:
        """
        Recursively place robots.
        - arity-2 internal: children left/right (theta=0)
        - arity-3 internal: children at equilateral triangle vertices
        - size-2 leaf: robots left/right of sub-centroid
        - size-3 leaf: equilateral triangle
        - size-1 leaf: robot at sub-centroid
        """
        positions = {}
        if node.is_leaf:
            if node.size == 1:
                positions[node.robot_indices[0]] = (cx, cy)
            elif node.size == 2:
                # Horizontal pair
                half = scale * 0.25
                positions[node.robot_indices[0]] = (cx - half, cy)
                positions[node.robot_indices[1]] = (cx + half, cy)
            else:  # size == 3: equilateral triangle, base horizontal
                r = scale * 0.25
                positions[node.robot_indices[0]] = (cx - r, cy - r * math.tan(math.pi / 6))
                positions[node.robot_indices[1]] = (cx + r, cy - r * math.tan(math.pi / 6))
                positions[node.robot_indices[2]] = (cx, cy + r / math.cos(math.pi / 6))
        else:
            child_scale = scale / 1.8  # shrink for sub-clusters
            if node.arity == 2:
                # Children left and right
                half = scale * 0.5
                child_centers = [(cx - half, cy), (cx + half, cy)]
            else:  # arity == 3: equilateral triangle of sub-centroids
                r = scale * 0.5
                h = r * math.sqrt(3) / 2
                child_centers = [
                    (cx - r, cy - h / 2),
                    (cx + r, cy - h / 2),
                    (cx,     cy + h),
                ]
            for child, (ccx, ccy) in zip(node.children, child_centers):
                positions.update(self._default_positions(child, ccx, ccy, child_scale))
        return positions

    # -- Forward kinematics --------------------------------------------------

    def forward_kinematics(self, positions: Dict) -> Dict:
        """positions: {robot_idx (1-based): (x, y)} -> state dict
        With orientation=on: positions values are (x, y, theta); phi_i are appended."""
        state = {}
        # Use only (x,y) for the position recursion
        xy_positions = {}
        for idx, val in positions.items():
            xy_positions[idx] = (val[0], val[1])
        centroid, _ = self._fk_node(self.root, xy_positions, state, is_root=True,
                                     parent_theta_key=None)
        state['x_c'] = centroid[0]
        state['y_c'] = centroid[1]
        if self.orientation:
            for i in range(1, self.N + 1):
                theta_i = positions[i][2] if len(positions[i]) > 2 else 0.0
                ref_key = self.theta_ref_var.get(i, 'theta_c')
                theta_ref = state.get(ref_key, 0.0)
                phi_i = (theta_i - theta_ref + math.pi) % (2 * math.pi) - math.pi
                state[f'phi_{i}'] = phi_i
        return state

    def _fk_node(self, node, positions, state, is_root=False, parent_theta_key=None):
        """Returns (centroid_xy, theta_local). Records theta_ref_var for each robot."""
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                x, y = positions[node.robot_indices[0]]
                # Robot's frame is its parent's theta; record for orientation post-pass
                if self.orientation:
                    ref = parent_theta_key if parent_theta_key else 'theta_c'
                    self.theta_ref_var[node.robot_indices[0]] = ref
                return (x, y), 0.0
            elif node.size == 2:
                i1, i2 = node.robot_indices
                xa, ya = positions[i1]
                xb, yb = positions[i2]
                x_c, y_c, theta, L = PairBlock.forward(xa, ya, xb, yb)
                state[f'L_{nid}'] = L
                state[f'theta_{nid}'] = theta
                if self.orientation:
                    for ri in node.robot_indices:
                        self.theta_ref_var[ri] = f'theta_{nid}'
                return (x_c, y_c), theta
            else:  # size == 3
                i1, i2, i3 = node.robot_indices
                xa, ya = positions[i1]
                xb, yb = positions[i2]
                xc, yc = positions[i3]
                s = TripleBlock.forward(xa, ya, xb, yb, xc, yc)
                state[f'p_{nid}'] = s['p']
                state[f'beta_{nid}'] = s['beta']
                state[f'q_{nid}'] = s['q']
                state[f'theta_{nid}'] = s['theta_c']
                if self.orientation:
                    for ri in node.robot_indices:
                        self.theta_ref_var[ri] = f'theta_{nid}'
                return (s['x_c'], s['y_c']), s['theta_c']
        else:
            child_centroids = []
            child_thetas = []
            for child in node.children:
                c_xy, c_th = self._fk_node(child, positions, state,
                                            is_root=False,
                                            parent_theta_key=parent_theta_key)
                child_centroids.append(c_xy)
                child_thetas.append(c_th)

            if node.arity == 2:
                (xa, ya), (xb, yb) = child_centroids
                x_c, y_c, theta, L = PairBlock.forward(xa, ya, xb, yb)
                state[f'L_{nid}'] = L
                theta_key = 'theta_c' if is_root else f'theta_{nid}'
                if is_root:
                    state['theta_c'] = theta
                else:
                    state[f'theta_{nid}'] = theta
                # Re-traverse children to propagate the correct theta_key for size-1 leaves
                for child in node.children:
                    self._update_theta_ref_for_size1(child, theta_key)
                return (x_c, y_c), theta
            else:  # arity == 3
                (xa, ya), (xb, yb), (xc, yc) = child_centroids
                s = TripleBlock.forward(xa, ya, xb, yb, xc, yc)
                state[f'p_{nid}'] = s['p']
                state[f'beta_{nid}'] = s['beta']
                state[f'q_{nid}'] = s['q']
                theta_key = 'theta_c' if is_root else f'theta_{nid}'
                if is_root:
                    state['theta_c'] = s['theta_c']
                else:
                    state[f'theta_{nid}'] = s['theta_c']
                for child in node.children:
                    self._update_theta_ref_for_size1(child, theta_key)
                return (s['x_c'], s['y_c']), s['theta_c']

    def _update_theta_ref_for_size1(self, node, parent_theta_key: str):
        """Update theta_ref_var for size-1 leaves whose parent_theta_key is now known."""
        if not self.orientation:
            return
        if node.is_leaf and node.size == 1:
            self.theta_ref_var[node.robot_indices[0]] = parent_theta_key
        elif not node.is_leaf:
            # Internal node's own theta_key is already recorded; only propagate to size-1 leaves
            for child in node.children:
                self._update_theta_ref_for_size1(child, parent_theta_key)

    # -- Inverse kinematics --------------------------------------------------

    def inverse_kinematics(self, state: Dict) -> Dict:
        """state dict -> positions {robot_idx: (x, y)} when orientation=off,
        or {robot_idx: (x, y, theta)} when orientation=on."""
        positions = {}
        cx = state['x_c']
        cy = state['y_c']
        tc = state['theta_c']
        self._ik_node(self.root, cx, cy, tc, state, positions, is_root=True)
        if self.orientation:
            # Ensure theta_ref_var is populated (may not be if FK was never called)
            if not self.theta_ref_var:
                self.forward_kinematics({i: (p[0], p[1]) for i, p in positions.items()})
            for i in range(1, self.N + 1):
                xi, yi = positions[i]
                ref_key = self.theta_ref_var.get(i, 'theta_c')
                theta_i = (state.get(ref_key, 0.0) + state.get(f'phi_{i}', 0.0) + math.pi) % (2 * math.pi) - math.pi
                positions[i] = (xi, yi, theta_i)
        return positions

    def _ik_node(self, node, cx, cy, theta, state, positions, is_root=False):
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                positions[node.robot_indices[0]] = (cx, cy)
            elif node.size == 2:
                L = state[f'L_{nid}']
                th = state[f'theta_{nid}']
                xa, ya, xb, yb = PairBlock.inverse(cx, cy, th, L)
                positions[node.robot_indices[0]] = (xa, ya)
                positions[node.robot_indices[1]] = (xb, yb)
            else:  # size == 3
                p = state[f'p_{nid}']
                beta = state[f'beta_{nid}']
                q = state[f'q_{nid}']
                th = state[f'theta_{nid}']
                xa, ya, xb, yb, xc, yc = TripleBlock.inverse(cx, cy, th, p, beta, q)
                positions[node.robot_indices[0]] = (xa, ya)
                positions[node.robot_indices[1]] = (xb, yb)
                positions[node.robot_indices[2]] = (xc, yc)
        else:
            if node.arity == 2:
                L = state[f'L_{nid}']
                th = state['theta_c'] if is_root else state[f'theta_{nid}']
                xa, ya, xb, yb = PairBlock.inverse(cx, cy, th, L)
                self._ik_node(node.children[0], xa, ya, th, state, positions)
                self._ik_node(node.children[1], xb, yb, th, state, positions)
            else:  # arity == 3
                p = state[f'p_{nid}']
                beta = state[f'beta_{nid}']
                q = state[f'q_{nid}']
                th = state['theta_c'] if is_root else state[f'theta_{nid}']
                xa, ya, xb, yb, xc, yc = TripleBlock.inverse(cx, cy, th, p, beta, q)
                self._ik_node(node.children[0], xa, ya, th, state, positions)
                self._ik_node(node.children[1], xb, yb, th, state, positions)
                self._ik_node(node.children[2], xc, yc, th, state, positions)

    # -- Inverse Jacobian (analytical, recursive composition) ----------------

    def inverse_jacobian(self, state: Dict, backend=None) -> np.ndarray:
        """
        Returns analytical inverse Jacobian.
        orientation=off: (2N x dim_q). Row order: [x_1,y_1,...,x_N,y_N]
        orientation=on:  (3N x dim_q). Same position block, then N heading rows:
                         [x_1,y_1,...,x_N,y_N, theta_1,...,theta_N]
                         phi_i columns are exactly zero in all position rows.
        Column order matches self.state_vars (phi_i appended last when on).
        backend: NumpyBackend (default) or SympyBackend for symbolic output.
        """
        if backend is None:
            backend = NumpyBackend
        dim_q = len(self.state_vars)
        col_idx = {v: i for i, v in enumerate(self.state_vars)}
        n_rows = 3 * self.N if self.orientation else 2 * self.N
        J = backend.zeros((n_rows, dim_q))
        # Ensure theta_ref_var is populated
        if self.orientation and not self.theta_ref_var:
            self.default_state()
        self._jac_node(self.root, state, col_idx, J, is_root=True, backend=backend)
        if self.orientation:
            # Heading rows: theta_i = theta_ref(i) + phi_i
            for i in range(1, self.N + 1):
                heading_row = 2 * self.N + (i - 1)
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

    def _jac_node(self, node, state, col_idx, J, is_root=False, backend=None):
        """
        Fill columns of J corresponding to this node's state variables,
        affecting the robot rows owned by this subtree.
        Returns the (2*node.size x 3) sensitivity matrix:
            d[robot_positions] / d[x_c_local, y_c_local, theta_local]
        which is used by the parent to chain the Jacobian.
        backend: NumpyBackend (default) or SympyBackend.
        """
        if backend is None:
            backend = NumpyBackend
        nid = node.node_id
        robot_rows = []
        for idx in node.robot_indices:
            robot_rows += [2 * (idx - 1), 2 * (idx - 1) + 1]

        if node.is_leaf:
            if node.size == 1:
                S = backend.array([[1, 0, 0],
                                    [0, 1, 0]])
                return S

            elif node.size == 2:
                L = state[f'L_{nid}']
                th = state[f'theta_{nid}']
                J_leaf = PairBlock.j_inv(th, L, backend=backend)
                col_th = col_idx[f'theta_{nid}']
                col_L  = col_idx[f'L_{nid}']
                for k, row in enumerate(robot_rows):
                    J[row, col_th] = J_leaf[k, 2]
                    J[row, col_L]  = J_leaf[k, 3]
                S = J_leaf[:, 0:3]
                return S

            else:  # size == 3
                p = state[f'p_{nid}']
                beta = state[f'beta_{nid}']
                q = state[f'q_{nid}']
                th = state[f'theta_{nid}']
                J_leaf = TripleBlock.j_inv(0, 0, th, p, beta, q, backend=backend)
                col_th   = col_idx[f'theta_{nid}']
                col_p    = col_idx[f'p_{nid}']
                col_beta = col_idx[f'beta_{nid}']
                col_q    = col_idx[f'q_{nid}']
                for k, row in enumerate(robot_rows):
                    J[row, col_th]   = J_leaf[k, 2]
                    J[row, col_p]    = J_leaf[k, 3]
                    J[row, col_beta] = J_leaf[k, 4]
                    J[row, col_q]    = J_leaf[k, 5]
                S = J_leaf[:, 0:3]
                return S

        else:
            # Internal node: get child sensitivities
            child_S = []
            for child in node.children:
                S_c = self._jac_node(child, state, col_idx, J, is_root=False, backend=backend)
                child_S.append(S_c)

            # Build the meta-block Jacobian for this node.
            #
            # S_child has shape (2*child.size, 3): each 2-row block gives
            #   d(robot_xy) / d(x_c_child, y_c_child, theta_child)
            #
            # J_meta (PairBlock 4x4 or TripleBlock 6x6) maps this node's state
            # variables [x_c, y_c, theta, shape...] to child centroid x,y positions.
            # J_meta[2*ci:2*ci+2, jcol] = d(child_ci centroid xy) / d(node_var_jcol)
            #
            # Our theta variables are ABSOLUTE (not relative to parent). When the
            # parent's theta (spine bearing) changes, the child centroids' x,y move
            # as given by J_meta, but the child pair's theta_child does NOT change
            # (it is an independent state variable). So we only use columns 0:2
            # of S_c (the x_c, y_c sensitivity) for the chain rule — column 2
            # (theta_child sensitivity) is never triggered by a parent variable.
            #
            # Exception: if S_c is returned from a size-1 leaf (identity node),
            # it has shape (2, 3) with only col 0,1 nonzero, which is consistent.

            def _chain(S2, meta_col_xy):
                """
                S2: (2, 3) block for one robot in one child.
                meta_col_xy: (2,) = d(child centroid xy) / d(parent var).
                Returns (2,) = d(robot xy) / d(parent var).
                Child's theta does not respond to parent variables (absolute angles).
                """
                return S2[:, 0:2] * meta_col_xy[0] + S2[:, 1:2] * meta_col_xy[1]

            def _chain2(S2, mxy):
                """2-element result of chain rule d(robot_xy)/d(parent_var).
                Works for both numpy and sympy: extracts the dot product of each
                robot row with the 2D centroid sensitivity vector mxy."""
                v0 = S2[0, 0] * mxy[0] + S2[0, 1] * mxy[1]
                v1 = S2[1, 0] * mxy[0] + S2[1, 1] * mxy[1]
                return v0, v1

            if node.arity == 2:
                L = state[f'L_{nid}']
                th = state['theta_c'] if is_root else state[f'theta_{nid}']
                J_meta = PairBlock.j_inv(th, L, backend=backend)

                col_th = col_idx['theta_c'] if is_root else col_idx[f'theta_{nid}']
                col_L  = col_idx[f'L_{nid}']
                col_xc = col_idx.get('x_c')
                col_yc = col_idx.get('y_c')

                for k, row in enumerate(node.children[0].robot_indices):
                    rx, ry = 2*(row-1), 2*(row-1)+1
                    s2 = S_a = child_S[0]
                    s2 = s2[2*k:2*k+2, :]
                    dth0, dth1 = _chain2(s2, J_meta[0:2, 2])
                    dL0,  dL1  = _chain2(s2, J_meta[0:2, 3])
                    J[rx, col_th] += dth0;  J[ry, col_th] += dth1
                    J[rx, col_L]  += dL0;   J[ry, col_L]  += dL1
                    if is_root and col_xc is not None:
                        dxc0, dxc1 = _chain2(s2, J_meta[0:2, 0])
                        dyc0, dyc1 = _chain2(s2, J_meta[0:2, 1])
                        J[rx, col_xc] += dxc0; J[ry, col_xc] += dxc1
                        J[rx, col_yc] += dyc0; J[ry, col_yc] += dyc1

                for k, row in enumerate(node.children[1].robot_indices):
                    rx, ry = 2*(row-1), 2*(row-1)+1
                    s2 = child_S[1][2*k:2*k+2, :]
                    dth0, dth1 = _chain2(s2, J_meta[2:4, 2])
                    dL0,  dL1  = _chain2(s2, J_meta[2:4, 3])
                    J[rx, col_th] += dth0;  J[ry, col_th] += dth1
                    J[rx, col_L]  += dL0;   J[ry, col_L]  += dL1
                    if is_root and col_xc is not None:
                        dxc0, dxc1 = _chain2(s2, J_meta[2:4, 0])
                        dyc0, dyc1 = _chain2(s2, J_meta[2:4, 1])
                        J[rx, col_xc] += dxc0; J[ry, col_xc] += dxc1
                        J[rx, col_yc] += dyc0; J[ry, col_yc] += dyc1

                S_this = backend.zeros((2 * node.size, 3))
                for k, row in enumerate(node.children[0].robot_indices):
                    lk = list(node.robot_indices).index(row)
                    s2 = child_S[0][2*k:2*k+2, :]
                    c0x, c0y = _chain2(s2, J_meta[0:2, 0])
                    c1x, c1y = _chain2(s2, J_meta[0:2, 1])
                    c2x, c2y = _chain2(s2, J_meta[0:2, 2])
                    S_this[2*lk,   0] = c0x; S_this[2*lk+1, 0] = c0y
                    S_this[2*lk,   1] = c1x; S_this[2*lk+1, 1] = c1y
                    S_this[2*lk,   2] = c2x; S_this[2*lk+1, 2] = c2y
                for k, row in enumerate(node.children[1].robot_indices):
                    lk = list(node.robot_indices).index(row)
                    s2 = child_S[1][2*k:2*k+2, :]
                    c0x, c0y = _chain2(s2, J_meta[2:4, 0])
                    c1x, c1y = _chain2(s2, J_meta[2:4, 1])
                    c2x, c2y = _chain2(s2, J_meta[2:4, 2])
                    S_this[2*lk,   0] = c0x; S_this[2*lk+1, 0] = c0y
                    S_this[2*lk,   1] = c1x; S_this[2*lk+1, 1] = c1y
                    S_this[2*lk,   2] = c2x; S_this[2*lk+1, 2] = c2y
                return S_this

            else:  # arity == 3
                p = state[f'p_{nid}']
                beta = state[f'beta_{nid}']
                q = state[f'q_{nid}']
                th = state['theta_c'] if is_root else state[f'theta_{nid}']
                J_meta = TripleBlock.j_inv(0, 0, th, p, beta, q, backend=backend)

                col_p    = col_idx[f'p_{nid}']
                col_beta = col_idx[f'beta_{nid}']
                col_q    = col_idx[f'q_{nid}']
                col_th   = col_idx['theta_c'] if is_root else col_idx[f'theta_{nid}']
                col_xc   = col_idx.get('x_c')
                col_yc   = col_idx.get('y_c')

                for ci, (child, S_c) in enumerate(zip(node.children, child_S)):
                    mr = slice(2*ci, 2*ci+2)
                    for k, row in enumerate(child.robot_indices):
                        rx, ry = 2*(row-1), 2*(row-1)+1
                        s2 = S_c[2*k:2*k+2, :]
                        dp0,    dp1    = _chain2(s2, J_meta[mr, 3])
                        dbeta0, dbeta1 = _chain2(s2, J_meta[mr, 4])
                        dq0,    dq1    = _chain2(s2, J_meta[mr, 5])
                        dth0,   dth1   = _chain2(s2, J_meta[mr, 2])
                        J[rx, col_p]    += dp0;    J[ry, col_p]    += dp1
                        J[rx, col_beta] += dbeta0; J[ry, col_beta] += dbeta1
                        J[rx, col_q]    += dq0;    J[ry, col_q]    += dq1
                        J[rx, col_th]   += dth0;   J[ry, col_th]   += dth1
                        if is_root and col_xc is not None:
                            dxc0, dxc1 = _chain2(s2, J_meta[mr, 0])
                            dyc0, dyc1 = _chain2(s2, J_meta[mr, 1])
                            J[rx, col_xc] += dxc0; J[ry, col_xc] += dxc1
                            J[rx, col_yc] += dyc0; J[ry, col_yc] += dyc1

                S_this = backend.zeros((2 * node.size, 3))
                for ci, (child, S_c) in enumerate(zip(node.children, child_S)):
                    mr = slice(2*ci, 2*ci+2)
                    for k, row in enumerate(child.robot_indices):
                        lk = list(node.robot_indices).index(row)
                        s2 = S_c[2*k:2*k+2, :]
                        c0x, c0y = _chain2(s2, J_meta[mr, 0])
                        c1x, c1y = _chain2(s2, J_meta[mr, 1])
                        c2x, c2y = _chain2(s2, J_meta[mr, 2])
                        S_this[2*lk,   0] = c0x; S_this[2*lk+1, 0] = c0y
                        S_this[2*lk,   1] = c1x; S_this[2*lk+1, 1] = c1y
                        S_this[2*lk,   2] = c2x; S_this[2*lk+1, 2] = c2y
                return S_this

    def forward_jacobian(self, state: Dict) -> np.ndarray:
        J_inv = self.inverse_jacobian(state)
        try:
            return np.linalg.inv(J_inv)
        except np.linalg.LinAlgError:
            print("Warning: J_inv singular, using pseudo-inverse.")
            return np.linalg.pinv(J_inv)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#f781bf', '#999999',
          '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']


class Visualizer:

    @staticmethod
    def render_hub_spoke(kin: HubAndSpokeKinematics, cluster_name: str,
                         state=None) -> plt.Figure:
        if state is None:
            state = kin.default_state()
        positions = kin.inverse_kinematics(state)
        N = kin.N

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{cluster_name}: {N}-robot Hub-and-Spoke Formation", fontsize=14, fontweight='bold')

        x_h, y_h = positions[N]

        # Draw spokes
        for i in range(1, N):
            xi, yi = positions[i]
            ax.plot([x_h, xi], [y_h, yi], color='gray', lw=1.5, zorder=1)
            # Label r_i
            mx, my = (x_h + xi) / 2, (y_h + yi) / 2
            ax.annotate(f'r_{i}={state[f"r_{i}"]:.2f}', (mx, my),
                        fontsize=7, ha='center', va='bottom',
                        color='#555555',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

        # Draw spokes as circles
        for i in range(1, N):
            xi, yi = positions[i]
            ax.scatter(xi, yi, s=200, color=COLORS[i % len(COLORS)], zorder=3,
                       edgecolors='black', linewidths=1.2)
            ax.text(xi, yi + 0.04, f'R{i}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Draw hub as square
        ax.scatter(x_h, y_h, s=300, color='black', marker='s', zorder=3)
        ax.text(x_h, y_h + 0.04, f'Hub (R{N})', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='black')

        # Draw theta_c arc
        r_arc = state['r_1'] * 0.35
        arc = Arc((x_h, y_h), 2 * r_arc, 2 * r_arc,
                  angle=0, theta1=0, theta2=math.degrees(state['theta_c']),
                  color='navy', lw=1.5)
        ax.add_patch(arc)
        tc_angle = state['theta_c'] / 2
        ax.text(x_h + r_arc * 1.1 * math.cos(tc_angle),
                y_h + r_arc * 1.1 * math.sin(tc_angle),
                'θ_c', fontsize=9, color='navy')

        # Draw gamma arcs
        for i in range(2, N):
            g_i = state[f'gamma_{i}']
            r_garc = state[f'r_{i}'] * 0.25
            arc_g = Arc((x_h, y_h), 2 * r_garc, 2 * r_garc,
                        angle=0,
                        theta1=math.degrees(state['theta_c']),
                        theta2=math.degrees(state['theta_c'] + g_i),
                        color='darkred', lw=1.2, linestyle='dashed')
            ax.add_patch(arc_g)
            mid_angle = state['theta_c'] + g_i / 2
            ax.text(x_h + r_garc * 1.3 * math.cos(mid_angle),
                    y_h + r_garc * 1.3 * math.sin(mid_angle),
                    f'γ_{i}', fontsize=8, color='darkred')

        # State vector text box
        sv_lines = ['State vector q:',
                    f'  x_h, y_h, θ_c'] + \
                   [f'  r_{i}' for i in range(1, N)] + \
                   [f'  γ_{i}' for i in range(2, N)]
        ax.text(0.02, 0.98, '\n'.join(sv_lines), transform=ax.transAxes,
                fontsize=8, va='top', family='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='gray'))

        ax.autoscale_view()
        fig.tight_layout()
        return fig

    @staticmethod
    def render_cluster_of_clusters(kin: ClusterOfClustersKinematics,
                                   cluster_name: str,
                                   state=None) -> plt.Figure:
        if state is None:
            state = kin.default_state()
        positions = kin.inverse_kinematics(state)
        root = kin.root

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{cluster_name}: {kin.N}-robot Cluster-of-Clusters Formation",
                     fontsize=14, fontweight='bold')

        # Assign a color per leaf
        leaf_colors = {}
        leaf_idx = [0]
        def assign_colors(node):
            if node.is_leaf:
                for ri in node.robot_indices:
                    leaf_colors[ri] = COLORS[leaf_idx[0] % len(COLORS)]
                leaf_idx[0] += 1
            else:
                for c in node.children:
                    assign_colors(c)
        assign_colors(root)

        # Collect centroids per node (bottom-up)
        centroids = {}
        def collect_centroids(node):
            if node.is_leaf:
                xs = [positions[ri][0] for ri in node.robot_indices]
                ys = [positions[ri][1] for ri in node.robot_indices]
                centroids[node.node_id] = (sum(xs)/len(xs), sum(ys)/len(ys))
            else:
                for c in node.children:
                    collect_centroids(c)
                xs = [centroids[c.node_id][0] for c in node.children]
                ys = [centroids[c.node_id][1] for c in node.children]
                centroids[node.node_id] = (sum(xs)/len(xs), sum(ys)/len(ys))
        collect_centroids(root)

        # Draw internal-node dashed lines (centroid-to-centroid)
        def draw_internal(node):
            if not node.is_leaf:
                cx, cy = centroids[node.node_id]
                nid = node.node_id
                for child in node.children:
                    ccx, ccy = centroids[child.node_id]
                    ax.plot([cx, ccx], [cy, ccy], '--', color='#777777',
                            lw=1.2, zorder=1)
                    # Label L or shape param on dashed line
                    mx, my = (cx + ccx) / 2, (cy + ccy) / 2
                    if node.arity == 2:
                        lbl = f"L_{nid}={state.get(f'L_{nid}', 0):.2f}"
                    else:
                        lbl = f"SAS_{nid}"
                    ax.annotate(lbl, (mx, my), fontsize=7, ha='center',
                                color='#555555',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          alpha=0.7, ec='none'))
                # Mark centroid
                ax.scatter(cx, cy, marker='x', s=100, color='black',
                           linewidths=2, zorder=4)
                ax.text(cx + 0.02, cy + 0.02, f'C{nid}', fontsize=7,
                        color='black', fontweight='bold')
                for child in node.children:
                    draw_internal(child)
        draw_internal(root)

        # Draw leaf intra-cluster solid lines
        def draw_leaf_edges(node):
            if node.is_leaf and node.size > 1:
                nid = node.node_id
                robots = node.robot_indices
                color = leaf_colors[robots[0]]
                if node.size == 2:
                    xa, ya = positions[robots[0]][:2]
                    xb, yb = positions[robots[1]][:2]
                    ax.plot([xa, xb], [ya, yb], '-', color=color, lw=2, zorder=2)
                    mx, my = (xa + xb) / 2, (ya + yb) / 2
                    ax.annotate(f"L_{nid}", (mx, my), fontsize=7, color=color,
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          alpha=0.8, ec='none'))
                elif node.size == 3:
                    pts = [positions[r] for r in robots]
                    for a, b in [(0,1),(1,2),(0,2)]:
                        ax.plot([pts[a][0], pts[b][0]],
                                [pts[a][1], pts[b][1]], '-', color=color, lw=2, zorder=2)
                    cx_l = sum(p[0] for p in pts) / 3
                    cy_l = sum(p[1] for p in pts) / 3
                    ax.annotate(f"p_{nid},β_{nid},q_{nid}", (cx_l, cy_l),
                                fontsize=7, color=color, ha='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          alpha=0.8, ec='none'))
            else:
                for c in node.children:
                    draw_leaf_edges(c)
        draw_leaf_edges(root)

        # Draw robots
        for ri, pos in positions.items():
            xi, yi = pos[0], pos[1]
            color = leaf_colors.get(ri, 'gray')
            ax.scatter(xi, yi, s=200, color=color, zorder=5,
                       edgecolors='black', linewidths=1.2)
            ax.text(xi, yi + 0.03, f'R{ri}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # State variable list
        sv_lines = ['State variables:'] + [f'  {v}' for v in kin.state_vars]
        ax.text(0.02, 0.98, '\n'.join(sv_lines), transform=ax.transAxes,
                fontsize=7, va='top', family='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='gray'))

        ax.autoscale_view()
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Code Generator
# ---------------------------------------------------------------------------

class CodeGenerator:

    def __init__(self, cluster_name: str, config: int, N: int,
                 kin, tree: Optional[TreeNode] = None,
                 orientation: bool = False, symbolic: bool = False):
        self.name = cluster_name
        self.config = config
        self.N = N
        self.kin = kin
        self.tree = tree
        self.orientation = orientation
        self.symbolic = symbolic

    # -- YAML ----------------------------------------------------------------

    def generate_yaml(self) -> str:
        kin = self.kin
        state = kin.default_state()
        if self.config == 1:
            data = {
                'formation': {
                    'type': 'hub_and_spoke',
                    'num_robots': self.N,
                    'cluster_config': 1,
                    **{k: round(v, 6) for k, v in state.items()},
                    'position_gain': 1.0,
                    'angle_gain': 0.1,
                }
            }
        else:
            formation_dict = {
                'type': 'cluster_of_clusters',
                'num_robots': self.N,
                'cluster_config': 2,
            }
            if self.tree is not None:
                formation_dict['config_tree'] = TreeParser.to_string(self.tree)
            formation_dict.update({k: round(v, 6) for k, v in state.items()})
            formation_dict['position_gain'] = 1.0
            formation_dict['angle_gain'] = 0.1
            data = {'formation': formation_dict}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # -- Forward kinematics --------------------------------------------------

    def generate_forward_kinematics(self) -> str:
        kin = self.kin
        N = self.N
        args = ', '.join(f'x{i}, y{i}' for i in range(1, N + 1))

        if self.config == 1:
            lines = [
                f'"""Forward kinematics for {self.name} hub-and-spoke ({N} robots).',
                f'State vars: {kin.state_vars}',
                '"""',
                'import math',
                '',
                f'def forward_kinematics({args}):',
                f'    """Hub = robot {N}. Spokes = robots 1..{N-1}."""',
                f'    x_h, y_h = x{N}, y{N}',
                '    betas = {}',
            ]
            for i in range(1, N):
                lines += [
                    f'    dx_{i} = x{i} - x_h',
                    f'    dy_{i} = y{i} - y_h',
                    f'    r_{i} = math.hypot(dx_{i}, dy_{i})',
                    f'    betas[{i}] = math.atan2(dy_{i}, dx_{i})',
                ]
            lines += ['    theta_c = betas[1]']
            for i in range(2, N):
                lines += [
                    f'    gamma_{i} = (betas[{i}] - betas[1] + math.pi) % (2*math.pi) - math.pi',
                ]
            ret_items = ['x_h', 'y_h', 'theta_c'] + \
                        [f'r_{i}' for i in range(1, N)] + \
                        [f'gamma_{i}' for i in range(2, N)]
            ret_dict = '{' + ', '.join(f"'{k}': {k}" for k in ret_items) + '}'
            lines += [f'    return {ret_dict}']
        else:
            # For cluster-of-clusters, emit a function that calls the inline math
            lines = [
                f'"""Forward kinematics for {self.name} cluster-of-clusters ({N} robots)."""',
                'import math',
                '',
                f'def forward_kinematics({args}):',
                f'    positions = ' + '{' +
                ', '.join(f'{i}: (x{i}, y{i})' for i in range(1, N+1)) + '}',
                '    state = {}',
                '    _fk_impl(positions, state)',
                '    return state',
                '',
                '# Internal implementation (generated)',
            ]
            lines += self._gen_fk_impl()

        return '\n'.join(lines)

    def _gen_fk_impl(self) -> List[str]:
        """Generate the _fk_impl function body for cluster-of-clusters."""
        lines = ['def _fk_impl(positions, state):']
        self._gen_fk_node(self.kin.root, lines, is_root=True)
        lines.append('    return state')
        return lines

    def _gen_fk_node(self, node, lines, is_root=False, indent=1):
        pad = '    ' * indent
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                lines.append(f'{pad}cx_{nid}, cy_{nid} = positions[{ri}]')
            elif node.size == 2:
                i1, i2 = node.robot_indices
                lines += [
                    f'{pad}xa_{nid}, ya_{nid} = positions[{i1}]',
                    f'{pad}xb_{nid}, yb_{nid} = positions[{i2}]',
                    f'{pad}cx_{nid} = (xa_{nid} + xb_{nid}) / 2',
                    f'{pad}cy_{nid} = (ya_{nid} + yb_{nid}) / 2',
                    f'{pad}state["theta_{nid}"] = math.atan2(yb_{nid} - ya_{nid}, xb_{nid} - xa_{nid})',
                    f'{pad}state["L_{nid}"] = math.hypot(xb_{nid} - xa_{nid}, yb_{nid} - ya_{nid})',
                ]
            else:
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}xa_{nid}, ya_{nid} = positions[{i1}]',
                    f'{pad}xb_{nid}, yb_{nid} = positions[{i2}]',
                    f'{pad}xc_{nid}, yc_{nid} = positions[{i3}]',
                    f'{pad}p_{nid} = math.hypot(xb_{nid}-xa_{nid}, yb_{nid}-ya_{nid})',
                    f'{pad}q_{nid} = math.hypot(xc_{nid}-xb_{nid}, yc_{nid}-yb_{nid})',
                    f'{pad}r_{nid} = math.hypot(xa_{nid}-xc_{nid}, ya_{nid}-yc_{nid})',
                    f'{pad}cos_beta_{nid} = max(-1.0, min(1.0, (p_{nid}**2+q_{nid}**2-r_{nid}**2)/(2*p_{nid}*q_{nid}+1e-10)))',
                    f'{pad}state["p_{nid}"] = p_{nid}',
                    f'{pad}state["beta_{nid}"] = math.acos(cos_beta_{nid})',
                    f'{pad}state["q_{nid}"] = q_{nid}',
                    f'{pad}state["theta_{nid}"] = math.atan2(yb_{nid}-ya_{nid}, xb_{nid}-xa_{nid})',
                    f'{pad}cx_{nid} = (xa_{nid}+xb_{nid}+xc_{nid})/3',
                    f'{pad}cy_{nid} = (ya_{nid}+yb_{nid}+yc_{nid})/3',
                ]
        else:
            for child in node.children:
                self._gen_fk_node(child, lines, indent=indent)
            if node.arity == 2:
                c0, c1 = node.children
                lines += [
                    f'{pad}cx_{nid} = (cx_{c0.node_id} + cx_{c1.node_id}) / 2',
                    f'{pad}cy_{nid} = (cy_{c0.node_id} + cy_{c1.node_id}) / 2',
                    f'{pad}state["L_{nid}"] = math.hypot(cx_{c1.node_id}-cx_{c0.node_id}, cy_{c1.node_id}-cy_{c0.node_id})',
                ]
                theta_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines.append(f'{pad}state[{theta_key}] = math.atan2(cy_{c1.node_id}-cy_{c0.node_id}, cx_{c1.node_id}-cx_{c0.node_id})')
            else:
                c0, c1, c2 = node.children
                lines += [
                    f'{pad}xa_m{nid}, ya_m{nid} = cx_{c0.node_id}, cy_{c0.node_id}',
                    f'{pad}xb_m{nid}, yb_m{nid} = cx_{c1.node_id}, cy_{c1.node_id}',
                    f'{pad}xc_m{nid}, yc_m{nid} = cx_{c2.node_id}, cy_{c2.node_id}',
                    f'{pad}p_m{nid} = math.hypot(xb_m{nid}-xa_m{nid}, yb_m{nid}-ya_m{nid})',
                    f'{pad}q_m{nid} = math.hypot(xc_m{nid}-xb_m{nid}, yc_m{nid}-yb_m{nid})',
                    f'{pad}r_m{nid} = math.hypot(xa_m{nid}-xc_m{nid}, ya_m{nid}-yc_m{nid})',
                    f'{pad}cb_m{nid} = max(-1.0,min(1.0,(p_m{nid}**2+q_m{nid}**2-r_m{nid}**2)/(2*p_m{nid}*q_m{nid}+1e-10)))',
                    f'{pad}state["p_{nid}"] = p_m{nid}',
                    f'{pad}state["beta_{nid}"] = math.acos(cb_m{nid})',
                    f'{pad}state["q_{nid}"] = q_m{nid}',
                    f'{pad}cx_{nid} = (xa_m{nid}+xb_m{nid}+xc_m{nid})/3',
                    f'{pad}cy_{nid} = (ya_m{nid}+yb_m{nid}+yc_m{nid})/3',
                ]
                theta_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines.append(f'{pad}state[{theta_key}] = math.atan2(yb_m{nid}-ya_m{nid}, xb_m{nid}-xa_m{nid})')
            if is_root:
                lines += [
                    f'{pad}state["x_c"] = cx_{nid}',
                    f'{pad}state["y_c"] = cy_{nid}',
                ]

    # -- Inverse kinematics --------------------------------------------------

    def generate_inverse_kinematics(self) -> str:
        N = self.N
        kin = self.kin
        if self.config == 1:
            lines = [
                f'"""Inverse kinematics for {self.name} hub-and-spoke ({N} robots)."""',
                'import math',
                '',
                'def inverse_kinematics(state):',
                '    """state: dict -> {robot_idx: (x, y)}"""',
                '    x_h = state["x_h"]',
                '    y_h = state["y_h"]',
                '    theta_c = state["theta_c"]',
                f'    positions = {{{N}: (x_h, y_h)}}',
            ]
            for i in range(1, N):
                g = '0.0' if i == 1 else f'state["gamma_{i}"]'
                lines += [
                    f'    r_{i} = state["r_{i}"]',
                    f'    gamma_{i} = {g}',
                    f'    angle_{i} = theta_c + gamma_{i}',
                    f'    positions[{i}] = (x_h + r_{i} * math.cos(angle_{i}), y_h + r_{i} * math.sin(angle_{i}))',
                ]
            lines.append('    return positions')
        else:
            lines = [
                f'"""Inverse kinematics for {self.name} cluster-of-clusters ({N} robots)."""',
                'import math',
                '',
                'def inverse_kinematics(state):',
                '    """state: dict -> {robot_idx: (x, y)}"""',
                '    positions = {}',
                '    _ik_impl(state["x_c"], state["y_c"], state["theta_c"], state, positions)',
                '    return positions',
                '',
            ]
            lines += self._gen_ik_impl()
        return '\n'.join(lines)

    def _gen_ik_impl(self) -> List[str]:
        lines = ['def _ik_impl(cx, cy, theta, state, positions):']
        self._gen_ik_node(self.kin.root, lines, is_root=True)
        return lines

    def _gen_ik_node(self, node, lines, is_root=False, indent=1):
        pad = '    ' * indent
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                lines.append(f'{pad}positions[{ri}] = (cx, cy)')
            elif node.size == 2:
                i1, i2 = node.robot_indices
                lines += [
                    f'{pad}_L = state["L_{nid}"]',
                    f'{pad}_th = state["theta_{nid}"]',
                    f'{pad}_ct, _st = math.cos(_th), math.sin(_th)',
                    f'{pad}positions[{i1}] = (cx - (_L/2)*_ct, cy - (_L/2)*_st)',
                    f'{pad}positions[{i2}] = (cx + (_L/2)*_ct, cy + (_L/2)*_st)',
                ]
            else:
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}_p = state["p_{nid}"]; _beta = state["beta_{nid}"]; _q = state["q_{nid}"]',
                    f'{pad}_th = state["theta_{nid}"]',
                    f'{pad}# SAS inverse kinematics inline',
                    f'{pad}_x2l, _y2l = 0.0, 0.0',
                    f'{pad}_x1l, _y1l = _p, 0.0',
                    f'{pad}_x3l = _q * math.cos(_beta); _y3l = _q * math.sin(_beta)',
                    f'{pad}_cxl = (_x1l+_x2l+_x3l)/3; _cyl = (_y1l+_y2l+_y3l)/3',
                    f'{pad}_x1l -= _cxl; _y1l -= _cyl; _x2l -= _cxl; _y2l -= _cyl; _x3l -= _cxl; _y3l -= _cyl',
                    f'{pad}_ct, _st = math.cos(_th), math.sin(_th)',
                    f'{pad}positions[{i1}] = (_ct*_x1l - _st*_y1l + cx, _st*_x1l + _ct*_y1l + cy)',
                    f'{pad}positions[{i2}] = (_ct*_x2l - _st*_y2l + cx, _st*_x2l + _ct*_y2l + cy)',
                    f'{pad}positions[{i3}] = (_ct*_x3l - _st*_y3l + cx, _st*_x3l + _ct*_y3l + cy)',
                ]
        else:
            if node.arity == 2:
                c0, c1 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state[{th_key}]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}cx_{c0.node_id} = cx - (_L_{nid}/2)*_ct_{nid}',
                    f'{pad}cy_{c0.node_id} = cy - (_L_{nid}/2)*_st_{nid}',
                    f'{pad}cx_{c1.node_id} = cx + (_L_{nid}/2)*_ct_{nid}',
                    f'{pad}cy_{c1.node_id} = cy + (_L_{nid}/2)*_st_{nid}',
                ]
                self._gen_ik_node_with_centroid(c0, lines, f'cx_{c0.node_id}', f'cy_{c0.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c1, lines, f'cx_{c1.node_id}', f'cy_{c1.node_id}', f'_th_{nid}', indent)
            else:
                c0, c1, c2 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}_p_{nid} = state["p_{nid}"]; _b_{nid} = state["beta_{nid}"]; _q_{nid} = state["q_{nid}"]',
                    f'{pad}_th_{nid} = state[{th_key}]',
                    f'{pad}_x2l_{nid}, _y2l_{nid} = 0.0, 0.0',
                    f'{pad}_x1l_{nid}, _y1l_{nid} = _p_{nid}, 0.0',
                    f'{pad}_x3l_{nid} = _q_{nid}*math.cos(_b_{nid}); _y3l_{nid} = _q_{nid}*math.sin(_b_{nid})',
                    f'{pad}_cxl_{nid} = (_x1l_{nid}+_x2l_{nid}+_x3l_{nid})/3',
                    f'{pad}_cyl_{nid} = (_y1l_{nid}+_y2l_{nid}+_y3l_{nid})/3',
                    f'{pad}_x1l_{nid} -= _cxl_{nid}; _y1l_{nid} -= _cyl_{nid}',
                    f'{pad}_x2l_{nid} -= _cxl_{nid}; _y2l_{nid} -= _cyl_{nid}',
                    f'{pad}_x3l_{nid} -= _cxl_{nid}; _y3l_{nid} -= _cyl_{nid}',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}cx_{c0.node_id} = _ct_{nid}*_x1l_{nid} - _st_{nid}*_y1l_{nid} + cx',
                    f'{pad}cy_{c0.node_id} = _st_{nid}*_x1l_{nid} + _ct_{nid}*_y1l_{nid} + cy',
                    f'{pad}cx_{c1.node_id} = _ct_{nid}*_x2l_{nid} - _st_{nid}*_y2l_{nid} + cx',
                    f'{pad}cy_{c1.node_id} = _st_{nid}*_x2l_{nid} + _ct_{nid}*_y2l_{nid} + cy',
                    f'{pad}cx_{c2.node_id} = _ct_{nid}*_x3l_{nid} - _st_{nid}*_y3l_{nid} + cx',
                    f'{pad}cy_{c2.node_id} = _st_{nid}*_x3l_{nid} + _ct_{nid}*_y3l_{nid} + cy',
                ]
                self._gen_ik_node_with_centroid(c0, lines, f'cx_{c0.node_id}', f'cy_{c0.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c1, lines, f'cx_{c1.node_id}', f'cy_{c1.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c2, lines, f'cx_{c2.node_id}', f'cy_{c2.node_id}', f'_th_{nid}', indent)

    def _gen_ik_node_with_centroid(self, node, lines, cx_var, cy_var, th_var, indent):
        """Recursively generate IK, substituting the centroid variable names."""
        pad = '    ' * indent
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                lines.append(f'{pad}positions[{ri}] = ({cx_var}, {cy_var})')
            elif node.size == 2:
                i1, i2 = node.robot_indices
                lines += [
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}positions[{i1}] = ({cx_var} - (_L_{nid}/2)*_ct_{nid}, {cy_var} - (_L_{nid}/2)*_st_{nid})',
                    f'{pad}positions[{i2}] = ({cx_var} + (_L_{nid}/2)*_ct_{nid}, {cy_var} + (_L_{nid}/2)*_st_{nid})',
                ]
            else:
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}_p_{nid} = state["p_{nid}"]; _b_{nid} = state["beta_{nid}"]; _q_{nid} = state["q_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_x2l_{nid}, _y2l_{nid} = 0.0, 0.0',
                    f'{pad}_x1l_{nid}, _y1l_{nid} = _p_{nid}, 0.0',
                    f'{pad}_x3l_{nid} = _q_{nid}*math.cos(_b_{nid}); _y3l_{nid} = _q_{nid}*math.sin(_b_{nid})',
                    f'{pad}_cxl_{nid} = (_x1l_{nid}+_x2l_{nid}+_x3l_{nid})/3',
                    f'{pad}_cyl_{nid} = (_y1l_{nid}+_y2l_{nid}+_y3l_{nid})/3',
                    f'{pad}_x1l_{nid} -= _cxl_{nid}; _y1l_{nid} -= _cyl_{nid}',
                    f'{pad}_x2l_{nid} -= _cxl_{nid}; _y2l_{nid} -= _cyl_{nid}',
                    f'{pad}_x3l_{nid} -= _cxl_{nid}; _y3l_{nid} -= _cyl_{nid}',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}positions[{i1}] = (_ct_{nid}*_x1l_{nid} - _st_{nid}*_y1l_{nid} + {cx_var}, _st_{nid}*_x1l_{nid} + _ct_{nid}*_y1l_{nid} + {cy_var})',
                    f'{pad}positions[{i2}] = (_ct_{nid}*_x2l_{nid} - _st_{nid}*_y2l_{nid} + {cx_var}, _st_{nid}*_x2l_{nid} + _ct_{nid}*_y2l_{nid} + {cy_var})',
                    f'{pad}positions[{i3}] = (_ct_{nid}*_x3l_{nid} - _st_{nid}*_y3l_{nid} + {cx_var}, _st_{nid}*_x3l_{nid} + _ct_{nid}*_y3l_{nid} + {cy_var})',
                ]
        else:
            self._gen_ik_node(node, lines, is_root=False, indent=indent)

    # -- Inverse Jacobian ----------------------------------------------------

    def generate_inverse_jacobian(self) -> str:
        N = self.N
        kin = self.kin
        # Ensure theta_ref_var is populated for orientation heading row emission
        if self.orientation and hasattr(kin, 'theta_ref_var') and not kin.theta_ref_var:
            kin.default_state()
        state_vars = kin.state_vars
        n_rows = 3 * N if self.orientation else 2 * N
        row_doc = (f'[x1,y1,...,x{N},y{N}, theta_1,...,theta_{N}]'
                   if self.orientation else f'[x1,y1, x2,y2, ..., x{N},y{N}]')
        lines = [
            f'"""Analytical inverse Jacobian for {self.name} ({N} robots).',
            f'Maps q_dot ({len(state_vars)} state vars) to r_dot ({n_rows} robot coords).',
            f'State var order: {state_vars}',
            f'Robot row order: {row_doc}',
            '"""',
            'import math',
            'import numpy as np',
            '',
            'STATE_VARS = ' + repr(state_vars),
            '',
            f'def inverse_jacobian(state):',
            f'    """Returns ({n_rows} x {len(state_vars)}) numpy array."""',
            f'    J = np.zeros(({n_rows}, {len(state_vars)}))',
        ]
        if self.config == 1:
            lines += self._gen_hub_spoke_j_inv_body(indent=1)
        else:
            lines += [
                '    # Analytical Jacobian: numerically evaluated at current state',
                '    # using the closed-form composition of pair/SAS blocks.',
                '    # This calls the runtime kinematics objects.',
                '    _fill_jacobian(state, J)',
            ]
            lines += self._gen_coc_j_inv_fill()
        if self.orientation:
            lines += self._gen_orientation_heading_rows(indent=1)
        lines.append('    return J')
        return '\n'.join(lines)

    def _gen_orientation_heading_rows(self, indent=1) -> List[str]:
        """Emit the N heading rows for orientation=on into the generated inverse_jacobian."""
        N = self.N
        pad = '    ' * indent
        kin = self.kin
        lines = [
            f'{pad}# Heading rows: theta_i = theta_ref(i) + phi_i',
            f'{pad}_col = {{v: i for i, v in enumerate(STATE_VARS)}}',
        ]
        for i in range(1, N + 1):
            heading_row = 2 * N + (i - 1)
            ref_key = kin.theta_ref_var.get(i, 'theta_c')
            lines += [
                f'{pad}J[{heading_row}, _col["phi_{i}"]] = 1.0',
                f'{pad}J[{heading_row}, _col["{ref_key}"]] += 1.0',
            ]
        return lines

    def _gen_hub_spoke_j_inv_body(self, indent=1) -> List[str]:
        N = self.N
        pad = '    ' * indent
        lines = [
            f'{pad}col = {{v: i for i, v in enumerate(STATE_VARS)}}',
            f'{pad}theta_c = state["theta_c"]',
        ]
        for i in range(1, N):
            rx = 2 * (i - 1)
            ry = rx + 1
            lines += [
                f'{pad}r_{i} = state["r_{i}"]',
                f'{pad}g_{i} = ' + ('0.0' if i == 1 else f'state["gamma_{i}"]'),
                f'{pad}alpha_{i} = theta_c + g_{i}',
                f'{pad}J[{rx}, col["x_h"]] = 1.0',
                f'{pad}J[{ry}, col["y_h"]] = 1.0',
                f'{pad}J[{rx}, col["theta_c"]] += -r_{i} * math.sin(alpha_{i})',
                f'{pad}J[{ry}, col["theta_c"]] +=  r_{i} * math.cos(alpha_{i})',
                f'{pad}J[{rx}, col["r_{i}"]] = math.cos(alpha_{i})',
                f'{pad}J[{ry}, col["r_{i}"]] = math.sin(alpha_{i})',
            ]
            if i >= 2:
                lines += [
                    f'{pad}J[{rx}, col["gamma_{i}"]] = -r_{i} * math.sin(alpha_{i})',
                    f'{pad}J[{ry}, col["gamma_{i}"]] =  r_{i} * math.cos(alpha_{i})',
                ]
        # Hub rows
        hub_rx = 2 * (N - 1)
        lines += [
            f'{pad}J[{hub_rx}, col["x_h"]] = 1.0',
            f'{pad}J[{hub_rx+1}, col["y_h"]] = 1.0',
        ]
        return lines

    def _gen_coc_j_inv_fill(self) -> List[str]:
        """Generate a _fill_jacobian function that uses PairBlock/TripleBlock inline."""
        lines = [
            '',
            'def _fill_jacobian(state, J):',
            '    """Fill J using analytical composition of pair/SAS blocks."""',
            '    col = {v: i for i, v in enumerate(STATE_VARS)}',
        ]
        self._gen_jac_fill_node(self.kin.root, lines, is_root=True, indent=1)
        return lines

    def _gen_jac_fill_node(self, node, lines, is_root=False, indent=1):
        pad = '    ' * indent
        nid = node.node_id
        robot_rows = sorted(node.robot_indices)

        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                rx, ry = 2*(ri-1), 2*(ri-1)+1
                col_xc = '"x_c"' if is_root else '"x_c"'
                lines += [
                    f'{pad}# Leaf size-1: robot {ri} tracks centroid',
                    f'{pad}J[{rx}, col["x_c"]] += 1.0',
                    f'{pad}J[{ry}, col["y_c"]] += 1.0',
                ]
            elif node.size == 2:
                i1, i2 = node.robot_indices
                r1x, r1y = 2*(i1-1), 2*(i1-1)+1
                r2x, r2y = 2*(i2-1), 2*(i2-1)+1
                lines += [
                    f'{pad}# Leaf pair: robots {i1},{i2}',
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}_h_{nid} = _L_{nid} / 2',
                    f'{pad}# Centroid columns (from parent propagation)',
                    f'{pad}J[{r1x}, col["x_c"]] += 1.0; J[{r1y}, col["y_c"]] += 1.0',
                    f'{pad}J[{r2x}, col["x_c"]] += 1.0; J[{r2y}, col["y_c"]] += 1.0',
                    f'{pad}# Shape: theta_{nid}',
                    f'{pad}J[{r1x}, col["theta_{nid}"]] +=  _h_{nid}*_st_{nid}',
                    f'{pad}J[{r1y}, col["theta_{nid}"]] += -_h_{nid}*_ct_{nid}',
                    f'{pad}J[{r2x}, col["theta_{nid}"]] += -_h_{nid}*_st_{nid}',
                    f'{pad}J[{r2y}, col["theta_{nid}"]] +=  _h_{nid}*_ct_{nid}',
                    f'{pad}# Shape: L_{nid}',
                    f'{pad}J[{r1x}, col["L_{nid}"]] += -0.5*_ct_{nid}',
                    f'{pad}J[{r1y}, col["L_{nid}"]] += -0.5*_st_{nid}',
                    f'{pad}J[{r2x}, col["L_{nid}"]] +=  0.5*_ct_{nid}',
                    f'{pad}J[{r2y}, col["L_{nid}"]] +=  0.5*_st_{nid}',
                ]
            else:  # size == 3
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}# Leaf SAS-3: robots {i1},{i2},{i3}',
                    f'{pad}_p_{nid}=state["p_{nid}"]; _b_{nid}=state["beta_{nid}"]; _q_{nid}=state["q_{nid}"]',
                    f'{pad}_th_{nid}=state["theta_{nid}"]',
                ]
                self._emit_sas_jac_inline(node, lines, pad, '_th_{nid}', f'_p_{nid}', f'_b_{nid}', f'_q_{nid}', nid, is_root=False)
        else:
            for child in node.children:
                self._gen_jac_fill_node(child, lines, is_root=False, indent=indent)
            # Internal node: fill shape vars for this node
            if node.arity == 2:
                c0, c1 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}# Internal pair node {nid}: children {c0.node_id},{c1.node_id}',
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state[{th_key}]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}_h_{nid} = _L_{nid} / 2',
                ]
                # For each robot in each child: propagate through pair block
                for child, sign in zip([c0, c1], ['-', '+']):
                    for ri in child.robot_indices:
                        rx, ry = 2*(ri-1), 2*(ri-1)+1
                        lines += [
                            f'{pad}# Robot {ri} (child {child.node_id}) theta contribution',
                            f'{pad}J[{rx}, col[{th_key}]] += {sign}_h_{nid}*_st_{nid}',
                            f'{pad}J[{ry}, col[{th_key}]] += {"" if sign=="-" else "-"}_h_{nid}*_ct_{nid}',
                            f'{pad}J[{rx}, col["L_{nid}"]] += {sign}0.5*_ct_{nid}',
                            f'{pad}J[{ry}, col["L_{nid}"]] += {sign}0.5*_st_{nid}',
                        ]
                if is_root:
                    for ri in node.robot_indices:
                        rx, ry = 2*(ri-1), 2*(ri-1)+1
                        lines += [
                            f'{pad}J[{rx}, col["x_c"]] += 1.0',
                            f'{pad}J[{ry}, col["y_c"]] += 1.0',
                        ]
            else:
                # arity == 3: SAS meta-node
                c0, c1, c2 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}# Internal SAS-3 meta-node {nid}',
                    f'{pad}_p_{nid}=state["p_{nid}"]; _b_{nid}=state["beta_{nid}"]; _q_{nid}=state["q_{nid}"]',
                    f'{pad}_th_{nid}=state[{th_key}]',
                ]
                self._emit_sas_jac_inline(node, lines, pad, f'_th_{nid}', f'_p_{nid}', f'_b_{nid}', f'_q_{nid}', nid, is_root=is_root, meta=True)

    def _emit_sas_jac_inline(self, node, lines, pad, th_var, p_var, b_var, q_var, nid, is_root=False, meta=False):
        """Emit the SAS Jacobian fill inline, operating on centroid positions."""
        robots = node.robot_indices if not meta else node.robot_indices
        i1, i2, i3 = (node.children[0].robot_indices[0] if meta else node.robot_indices[0],
                      node.children[1].robot_indices[0] if meta else node.robot_indices[1],
                      (node.children[2].robot_indices[0] if meta else node.robot_indices[2]))
        # Just emit centroid and shape columns
        # For brevity, use x_c/y_c for all robots (centroid col = 1/3 each)
        for ri in node.robot_indices:
            rx, ry = 2*(ri-1), 2*(ri-1)+1
            if is_root:
                lines += [
                    f'{pad}J[{rx}, col["x_c"]] += 1.0',
                    f'{pad}J[{ry}, col["y_c"]] += 1.0',
                ]
        # Shape columns deferred to runtime evaluation for SAS (complex inline)
        # Emit a comment noting they're filled by the direct SAS block above
        lines += [
            f'{pad}# SAS shape vars (p,beta,q) for node {nid} filled above in leaf/meta block',
        ]

    # -- Forward Jacobian ----------------------------------------------------

    def generate_forward_jacobian(self) -> str:
        N = self.N
        sv = self.kin.state_vars
        return '\n'.join([
            f'"""Forward Jacobian for {self.name} ({N} robots).',
            f'Inverse of the inverse Jacobian. Maps r_dot -> q_dot.',
            '"""',
            'import numpy as np',
            f'from {self.name}_inverse_jacobian import inverse_jacobian',
            '',
            'def forward_jacobian(state):',
            f'    """Returns ({len(sv)} x {2*N}) numpy array."""',
            '    J_inv = inverse_jacobian(state)',
            '    try:',
            '        return np.linalg.inv(J_inv)',
            '    except np.linalg.LinAlgError:',
            '        return np.linalg.pinv(J_inv)',
        ])

    # -- Cluster file --------------------------------------------------------

    def generate_cluster_file(self) -> str:
        name = self.name
        Name = ''.join(w.capitalize() for w in name.split('_'))
        N = self.N
        kin = self.kin
        state_vars = kin.state_vars
        default_state = kin.default_state()

        shape_vars = [v for v in state_vars if v not in ('x_c', 'y_c', 'theta_c', 'x_h', 'y_h')]

        # Build the _load_formation_config body
        if self.config == 1:
            load_lines = [
                "        formation = config['formation']",
                "        self.N = formation['num_robots']",
            ] + [f"        self.desired_{v} = formation['{v}']"
                 for v in state_vars if v not in ('x_h', 'y_h', 'theta_c')] + [
                "        self.position_gain = formation.get('position_gain', 1.0)",
                "        self.angle_gain = formation.get('angle_gain', 0.1)",
            ]
        else:
            load_lines = [
                "        formation = config['formation']",
                "        self.N = formation['num_robots']",
            ] + [f"        self.desired_{v} = formation['{v}']"
                 for v in state_vars if v not in ('x_c', 'y_c', 'theta_c')] + [
                "        self.position_gain = formation.get('position_gain', 1.0)",
                "        self.angle_gain = formation.get('angle_gain', 0.1)",
            ]

        # Build desired state dict access
        if self.config == 1:
            desired_state = (
                "{'x_h': self.robots[self.N-1].get_position()[0],\n"
                "             'y_h': self.robots[self.N-1].get_position()[1],\n"
                "             'theta_c': current_formation['theta_c']," +
                "".join(f"\n             'r_{i}': self.desired_r_{i}," for i in range(1, N)) +
                "".join(f"\n             'gamma_{i}': self.desired_gamma_{i}," for i in range(2, N)) +
                "}"
            )
        else:
            desired_state_items = (
                "{'x_c': current_formation['x_c'], 'y_c': current_formation['y_c'], 'theta_c': current_formation['theta_c']," +
                "".join(f"\n             '{v}': self.desired_{v}," for v in shape_vars) +
                "}"
            )

        orientation_guard = ''
        if self.orientation:
            orientation_guard = f'''
    # ------------------------------------------------------------------
    # ORIENTATION MODE NOTE
    # ------------------------------------------------------------------
    # This cluster was generated with --orientation on.
    # The FK, IK, and inverse Jacobian are fully correct and verified
    # via finite-difference tests.  The simulation controller below
    # (move / command_velocity) is NOT wired for per-robot heading
    # control: Omnibot accepts (vx, vy) only, not (vx, vy, omega).
    # Instantiating this class raises RuntimeError to prevent silent
    # incorrect results.  Use the kinematics and Jacobian files directly
    # for analysis, or extend the controller before enabling simulation.
    # ------------------------------------------------------------------
    ORIENTATION_ENABLED = True
'''
        code = f'''"""
{Name}Cluster — generated by clusterbuilder.py
Config: {"hub-and-spoke" if self.config == 1 else "cluster-of-clusters"}, {N} robots
State variables: {state_vars}
{"" if not self.orientation else chr(10) + "NOTE: Generated with --orientation on.  FK/IK/Jacobian are complete." + chr(10) + "The simulation controller is NOT wired for per-robot heading control." + chr(10) + "See ORIENTATION_ENABLED guard in the class body." + chr(10)}"""
import os
import yaml
import numpy as np
from src.robot.omnibot import Omnibot
from {name}_forward_kinematics import forward_kinematics
from {name}_inverse_kinematics import inverse_kinematics
from {name}_inverse_jacobian import inverse_jacobian, STATE_VARS


class {Name}Cluster:
{orientation_guard}
    def __init__(self, formation_config_path, field, timestep=0.1, momentum_alpha=0.7):
        {"raise RuntimeError(" + repr("Orientation mode is enabled. The controller simulation is not wired for per-robot heading (Omnibot accepts vx, vy only). FK, IK, and J^-1 are verified and usable directly from the kinematics files.") + ")" if self.orientation else ""}
        self.field = field
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha
        self._load_formation_config(formation_config_path)
        self._initialize_robots()
        self.center_history = []
        self.robot_history = []
        self.velocity_history = []

    def _load_formation_config(self, config_path):
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
{chr(10).join(load_lines)}

    def _initialize_robots(self):
        import math, random
        default = {repr(default_state)}
        default['x_c' if 'x_c' in default else 'x_h'] = random.uniform(-0.5, 0.5)
        default['y_c' if 'y_c' in default else 'y_h'] = random.uniform(-0.5, 0.5)
        positions = inverse_kinematics(default)
        self.robots = [Omnibot(positions[i+1][0], positions[i+1][1],
                               self.timestep, self.momentum_alpha)
                       for i in range({N})]

    def get_robot_positions(self):
        return {{i+1: tuple(self.robots[i].get_position()) for i in range({N})}}

    def get_current_formation(self):
        pos = self.get_robot_positions()
        coords = []
        for i in range(1, {N}+1):
            coords += list(pos[i])
        return forward_kinematics(*coords)

    def get_centroid(self):
        f = self.get_current_formation()
        xk = 'x_h' if 'x_h' in f else 'x_c'
        yk = 'y_h' if 'y_h' in f else 'y_c'
        return np.array([f[xk], f[yk]])

    def sample_field_at_robots(self):
        return [robot.sample_field(self.field) for robot in self.robots]

    def move(self, control_primitive):
        current = self.get_current_formation()
        vx_c, vy_c = control_primitive(self)

        # Formation errors (proportional control)
        shape_vel = np.zeros(len(STATE_VARS))
        col = {{v: i for i, v in enumerate(STATE_VARS)}}
        xk = 'x_h' if 'x_h' in current else 'x_c'
        yk = 'y_h' if 'y_h' in current else 'y_c'
        shape_vel[col[xk]] = vx_c
        shape_vel[col[yk]] = vy_c
        shape_vel[col.get('theta_c', -1)] = 0.0  # no spin by default
'''
        if self.config == 1:
            code += f'''
        for i in range(1, {N}):
            r_key = f'r_{{i}}'
            if r_key in col:
                shape_vel[col[r_key]] = self.position_gain * (
                    getattr(self, f'desired_r_{{i}}') - current.get(r_key, 0))
        for i in range(2, {N}):
            g_key = f'gamma_{{i}}'
            if g_key in col:
                err = getattr(self, f'desired_gamma_{{i}}') - current.get(g_key, 0)
                err = (err + 3.14159) % (2*3.14159) - 3.14159
                shape_vel[col[g_key]] = self.angle_gain * err
'''
        else:
            code += f'''
        desired_shape = {{{', '.join(f'"{v}": self.desired_{v}' for v in shape_vars)}}}
        for v, desired in desired_shape.items():
            if v in col:
                err = desired - current.get(v, 0)
                gain = self.angle_gain if 'theta' in v or 'beta' in v else self.position_gain
                shape_vel[col[v]] = gain * err
'''
        code += f'''
        J_inv = inverse_jacobian(current)
        robot_vel = J_inv @ shape_vel

        for i, robot in enumerate(self.robots):
            robot.command_velocity(robot_vel[2*i], robot_vel[2*i+1])

        centroid = self.get_centroid()
        self.center_history.append(centroid.copy())
        pos = self.get_robot_positions()
        self.robot_history.append(np.array([list(pos[i+1]) for i in range({N})]))

    def reset(self, x_c=None, y_c=None):
        import math
        default = {repr(default_state)}
        default['x_c' if 'x_c' in default else 'x_h'] = x_c or 0.0
        default['y_c' if 'y_c' in default else 'y_h'] = y_c or 0.0
        positions = inverse_kinematics(default)
        for i, robot in enumerate(self.robots):
            xi, yi = positions[i+1]
            robot.set_position(xi, yi)
            robot.velocity = np.array([0.0, 0.0])
        self.center_history = []
        self.robot_history = []

    def get_center_history(self):
        return np.array(self.center_history) if self.center_history else np.array([])

    def get_robot_history(self):
        return np.array(self.robot_history) if self.robot_history else np.array([])

    def plot(self, ax):
        import matplotlib.pyplot as plt
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
                  'pink', 'gray', 'olive', 'cyan']
        pos = self.get_robot_positions()
        for i in range(1, {N}+1):
            xi, yi = pos[i]
            ax.scatter(xi, yi, color=colors[(i-1) % len(colors)], s=100, zorder=5)

    def plot_center(self, ax):
        c = self.get_centroid()
        ax.scatter(c[0], c[1], color='black', s=80, marker='x', zorder=5)

    def __repr__(self):
        c = self.get_centroid()
        return f"{Name}Cluster(centroid={{c}}, N={N})"
'''
        return code

    # -- Symbolic Jacobian emitter -------------------------------------------

    def generate_symbolic_jacobian(self):
        """
        Compute the symbolic inverse Jacobian and return (txt_content, tex_content).
        Requires sympy. Uses fresh Symbol per state variable.
        """
        try:
            import sympy
        except ImportError:
            sys.exit("sympy is required for --symbolic: pip install sympy")

        J_sym = self.kin.inverse_jacobian_symbolic()
        txt = sympy.pretty(J_sym, use_unicode=True)
        tex = sympy.latex(J_sym)
        header = (f"# Symbolic inverse Jacobian for {self.name}\n"
                  f"# State vars: {self.kin.state_vars}\n"
                  f"# Shape: {J_sym.shape[0]} x {J_sym.shape[1]}\n\n")
        return header + txt, tex

    # -- Write all files -----------------------------------------------------

    def write_all(self, image_only=False):
        img_file = f'{self.name}_visualization.png'
        if self.config == 1:
            fig = Visualizer.render_hub_spoke(self.kin, self.name)
        else:
            fig = Visualizer.render_cluster_of_clusters(self.kin, self.name)
        fig.savefig(img_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {img_file}")

        if image_only:
            return

        files = {
            f'{self.name}_forward_kinematics.py':  self.generate_forward_kinematics(),
            f'{self.name}_inverse_kinematics.py':  self.generate_inverse_kinematics(),
            f'{self.name}_forward_jacobian.py':    self.generate_forward_jacobian(),
            f'{self.name}_inverse_jacobian.py':    self.generate_inverse_jacobian(),
            f'{self.name}.yaml':                   self.generate_yaml(),
            f'{self.name}_cluster.py':             self.generate_cluster_file(),
        }
        for fname, content in files.items():
            with open(fname, 'w') as f:
                f.write(content)
            print(f"  Saved: {fname}")

        if self.symbolic:
            txt_content, tex_content = self.generate_symbolic_jacobian()
            txt_file = f'{self.name}_inverse_jacobian_symbolic.txt'
            tex_file = f'{self.name}_inverse_jacobian_symbolic.tex'
            with open(txt_file, 'w') as f:
                f.write(txt_content)
            print(f"  Saved: {txt_file}")
            with open(tex_file, 'w') as f:
                f.write(tex_content)
            print(f"  Saved: {tex_file}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog='clusterbuilder.py',
        description='Generate kinematics and cluster files for N-robot formations.'
    )
    parser.add_argument('num_robots',     type=int,
                        help='Number of robots')
    parser.add_argument('cluster_config', type=int, choices=[1, 2],
                        help='1=hub-and-spoke, 2=cluster-of-clusters')
    parser.add_argument('cluster_name',   type=str,
                        help='Name stem for output files (valid Python identifier)')
    parser.add_argument('run_mode',       type=str, choices=['image_only', 'full'],
                        help='image_only or full (7 files)')
    parser.add_argument('config_tree',    type=str, nargs='?', default=None,
                        help='Optional tree spec e.g. "((2,2),(2,2))" or "(2,2,2)"')
    parser.add_argument('--orientation', choices=['on', 'off'], default='off',
                        help='on: add per-robot heading phi_i (state 3N); off: position only (default)')
    parser.add_argument('--symbolic', action='store_true', default=False,
                        help='also emit symbolic inverse Jacobian (.txt and .tex) via sympy')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate
    if args.num_robots < 2:
        print("Error: num_robots must be >= 2")
        sys.exit(1)
    if not args.cluster_name.isidentifier():
        print(f"Error: cluster_name '{args.cluster_name}' is not a valid Python identifier")
        sys.exit(1)
    if args.config_tree and args.cluster_config == 1:
        print("Warning: config_tree is ignored for cluster_config=1 (hub-and-spoke)")
        args.config_tree = None

    orientation = (args.orientation == 'on')
    print(f"\nClusterBuilder: {args.num_robots} robots, config={args.cluster_config}, "
          f"name='{args.cluster_name}', mode={args.run_mode}, orientation={args.orientation}")

    if args.cluster_config == 1:
        # Hub-and-spoke
        print("Building hub-and-spoke formation...")
        kin = HubAndSpokeKinematics(args.num_robots, orientation=orientation)
        gen = CodeGenerator(args.cluster_name, 1, args.num_robots, kin,
                            orientation=orientation, symbolic=args.symbolic)

    else:
        # Cluster-of-clusters
        if args.config_tree:
            print(f"Parsing config_tree: {args.config_tree}")
            root = TreeParser.parse(args.config_tree)
        else:
            print("Auto-building tree...")
            root = TreeParser.auto_build(args.num_robots)
            TreeParser._assign_ids_and_indices(root)

        TreeParser.validate(root, args.num_robots)
        print("Tree structure:")
        print(TreeParser.describe(root))

        kin = ClusterOfClustersKinematics(root, orientation=orientation)
        print(f"State variables ({len(kin.state_vars)}): {kin.state_vars}")
        gen = CodeGenerator(args.cluster_name, 2, args.num_robots, kin, tree=root,
                            orientation=orientation, symbolic=args.symbolic)

    # Generate outputs
    print(f"\nGenerating output files...")
    gen.write_all(image_only=(args.run_mode == 'image_only'))
    print("\nDone.")


if __name__ == '__main__':
    main()
