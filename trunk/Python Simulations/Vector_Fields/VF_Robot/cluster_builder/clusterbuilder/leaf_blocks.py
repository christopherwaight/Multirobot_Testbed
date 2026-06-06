"""Pair and Triple (SAS-3) leaf blocks: closed-form FK / IK / inverse Jacobian."""

import math

from .backend import NumpyBackend


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
        # Known limitation: beta uses acos and loses sign near collinear.
        cos_beta = (p**2 + q**2 - r**2) / (2 * p * q + eps)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)
        x_c = (xa + xb + xc) / 3.0
        y_c = (ya + yb + yc) / 3.0
        theta_c = math.atan2(ya - yb, xa - xb)
        return {'x_c': x_c, 'y_c': y_c, 'theta_c': theta_c, 'p': p, 'beta': beta, 'q': q}

    @staticmethod
    def inverse(x_c, y_c, theta_c, p, beta, q):
        """Inverse kinematics -> (xa, ya, xb, yb, xc, yc)."""
        x2_loc, y2_loc = 0.0, 0.0
        x1_loc, y1_loc = p, 0.0
        x3_loc = q * math.cos(beta)
        y3_loc = q * math.sin(beta)
        cx_loc = (x1_loc + x2_loc + x3_loc) / 3.0
        cy_loc = (y1_loc + y2_loc + y3_loc) / 3.0
        x1_loc -= cx_loc; y1_loc -= cy_loc
        x2_loc -= cx_loc; y2_loc -= cy_loc
        x3_loc -= cx_loc; y3_loc -= cy_loc
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
        x2_loc, y2_loc = 0, 0
        x1_loc, y1_loc = p, 0
        x3_loc = q * backend.cos(beta)
        y3_loc = q * backend.sin(beta)
        cx_loc = (x1_loc + x2_loc + x3_loc) / 3
        cy_loc = (y1_loc + y2_loc + y3_loc) / 3
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
