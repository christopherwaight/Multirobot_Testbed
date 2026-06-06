"""Cluster-of-clusters kinematics (Config 2)."""

import math
import sys
from typing import Dict, List

import numpy as np

from .backend import NumpyBackend, SympyBackend
from .tree import TreeNode
from .leaf_blocks import PairBlock, TripleBlock


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
        self.state_vars = self._collect_state_vars(root, is_root=True)
        if orientation:
            self.state_vars = self.state_vars + [f'phi_{i}' for i in range(1, self.N + 1)]
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
                half = scale * 0.25
                positions[node.robot_indices[0]] = (cx - half, cy)
                positions[node.robot_indices[1]] = (cx + half, cy)
            else:  # size == 3: equilateral triangle, base horizontal
                r = scale * 0.25
                positions[node.robot_indices[0]] = (cx - r, cy - r * math.tan(math.pi / 6))
                positions[node.robot_indices[1]] = (cx + r, cy - r * math.tan(math.pi / 6))
                positions[node.robot_indices[2]] = (cx, cy + r / math.cos(math.pi / 6))
        else:
            child_scale = scale / 1.8
            if node.arity == 2:
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
        if self.orientation and not self.theta_ref_var:
            self.default_state()
        self._jac_node(self.root, state, col_idx, J, is_root=True, backend=backend)
        if self.orientation:
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
            child_S = []
            for child in node.children:
                S_c = self._jac_node(child, state, col_idx, J, is_root=False, backend=backend)
                child_S.append(S_c)

            def _chain2(S2, mxy):
                """2-element result of chain rule d(robot_xy)/d(parent_var)."""
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
                    s2 = child_S[0][2*k:2*k+2, :]
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
