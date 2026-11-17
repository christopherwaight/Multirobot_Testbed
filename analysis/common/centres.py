"""Where the critical point p* of a printed map sits, fitted from the robots' own readings.

The zero of the field: decode each reading to a vector, fit a linear field v = J p + h over a ring
around the current centre (0.08 to 0.35 m, the scale of the formation), and move the centre to
p* = -J^-1 h; repeat until it stops moving. Any fixed rotation or reflection of the decoded vectors
leaves the zero unchanged, so the decode convention does not matter here.
"""
import numpy as np


def decode(hue, sat):
    """NNField's decode: direction 2 pi h + pi/2, magnitude (sat - 0.3) / 0.7."""
    ang = 2 * np.pi * np.asarray(hue) + np.pi / 2
    m = (np.asarray(sat) - 0.3) / 0.7
    return np.stack([m * np.cos(ang), m * np.sin(ang)], -1)


def field_zero(P, F, c0, ring=(0.08, 0.35), iters=8):
    """P (n, 2) positions, F (n, 2) decoded vectors. Returns (p*, J, readings used)."""
    c = np.asarray(c0, float)
    for _ in range(iters):
        r = np.hypot(*(P - c).T)
        b = (r > ring[0]) & (r < ring[1])
        coef = np.linalg.lstsq(np.c_[P[b], np.ones(b.sum())], F[b], rcond=None)[0]
        J, h0 = coef[:2].T, coef[2]
        c = -np.linalg.solve(J, h0)
    return c, J, int(b.sum())
