"""Vectorized simulation of a rigid formation of any size (3R triangle, 4R square), for the paper
protocol and the hardware-matched runs.

Why rigid is exact here: in simulation every robot has the same dynamics and nothing disturbs the
formation, so while the formation is exact the cluster-space controller sends every robot the
centroid command and the formation stays exact (vecsim's 3R max side error is 0). The cluster is then
a rigid shape translating with the centroid velocity, theta_c fixed (not controlled, as on hardware).
validate_rigidsim.py checks this against vecsim (3R) and QuadCluster (4R).

Per step (dt = 0.1 s), for every trial at once:
    estimate   plane fit to (u, v) over the k robots about the centroid; p_hat = c - J^-1 v(c)
    command    attract: v = k (p_hat - c), speed capped at vcmd_max, each axis clipped at vaxis
               orbit:   v = turn k_t t + clip(k_r (r - r_d), +-rclamp) r_hat   (Eq. 11)
               singular J: vector-sum fallback, as vecsim
    robot      the 2026-11-16 model (vecsim DEFAULTS): delay, gain, lag, deadband, cap
    position   p += dt v
"""
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import paths  # noqa: E402,F401
import vecsim as vs  # noqa: E402

PI = np.pi


def triangle(theta, side):
    """(n, 3, 2) offsets of the equilateral triangle about its centroid, as vecsim.equilateral."""
    return vs.equilateral(np.zeros((len(theta), 2)), theta, side)


def square(theta, diag=0.30):
    """(n, 4, 2) offsets of the 2026 4R square: diagonals 1-3 and 2-4 of length diag, crossing at their
    midpoints at 90 deg (d1 = d2 = 0.30, r1 = r2 = 0.5, phi = pi/2); robots 1-2-3-4 go around it."""
    h = diag / 2
    a1 = np.stack([np.cos(theta), np.sin(theta)], -1)                     # direction 1 -> 3
    a2 = np.stack([np.cos(theta - PI / 2), np.sin(theta - PI / 2)], -1)   # direction 2 -> 4
    return np.stack([-h * a1, -h * a2, h * a1, h * a2], 1)


def simulate(field, task, method, P, T, offsets, starts, dt=0.1, seed=0, stiction_model="deadzone"):
    """Returns c (n, steps+1, 2) centroid path and phat (n, steps+1, 2) estimate (NaN at the last step)."""
    rng = np.random.default_rng(seed)
    n, k = offsets.shape[:2]
    c = np.asarray(starts, float).copy()
    vel = np.zeros((n, 2))
    al, vsn, vmax, gain = P["alpha"], P["vs"], P["vmax"], P["gain"]
    delay = float(P["delay"][0])
    steps = delay / dt
    ddel = int(np.floor(steps + 1e-9))
    fdel = steps - ddel if steps - ddel >= 1e-9 else 0.0
    hist = deque([np.zeros((n, 2))] * (ddel + 2), maxlen=ddel + 2)
    n_steps = int(round(T / dt))
    C = np.full((n, n_steps + 1, 2), np.nan)
    PH = np.full((n, n_steps + 1, 2), np.nan)
    C[:, 0] = c
    tgt = field.target
    for j in range(n_steps):
        pos = c[:, None, :] + offsets
        with np.errstate(all="ignore"):
            U, V = field.fn(pos[..., 0], pos[..., 1])
        U, V = np.asarray(U, float), np.asarray(V, float)
        if np.any(P["sigma_n"] > 0):
            U = U + P["sigma_n"][:, None] * rng.standard_normal(U.shape)
            V = V + P["sigma_n"][:, None] * rng.standard_normal(V.shape)
        if method == "oracle":
            phat = np.broadcast_to(tgt, (n, 2)).copy()
            fb = np.zeros(n, bool)
        else:
            gux, guy, ub = vs._fit(pos, c, U)
            gvx, gvy, vb = vs._fit(pos, c, V)
            det = gux * gvy - guy * gvx
            with np.errstate(all="ignore"):
                phat = c - np.stack([(gvy * ub - guy * vb) / det, (-gvx * ub + gux * vb) / det], 1)
            fb = ~np.isfinite(phat).all(1)
        PH[:, j] = phat
        w = phat - c
        dist = np.hypot(w[:, 0], w[:, 1])
        with np.errstate(all="ignore"):
            wh = w / dist[:, None]
        if task == "attract":
            spd = np.minimum(P["k"] * dist, P["vcmd_max"])
            vc = np.clip(wh * spd[:, None], -P["vaxis"][:, None], P["vaxis"][:, None])
            vc[dist < 1e-6] = 0.0
        else:
            tang = np.stack([wh[:, 1], -wh[:, 0]], 1) * P["turn"][:, None]
            vr = np.clip(P["kr"] * (dist - P["rd"]), -P["rclamp"], P["rclamp"])
            vc = P["kt"][:, None] * tang + vr[:, None] * wh
            small = dist < 1e-6
            if small.any():
                a = rng.uniform(0, 2 * PI, small.sum())
                vc[small] = np.stack([np.cos(a), np.sin(a)], 1)
        if fb.any():
            vc[fb] = np.stack([U[fb].sum(1), V[fb].sum(1)], 1)
        vc[~np.isfinite(vc).all(1)] = 0.0
        # robot model (identical robots, so one velocity per trial)
        hist.append(vc)
        ud = (1 - fdel) * hist[-1 - ddel] + fdel * hist[-2 - ddel]
        um = np.hypot(ud[:, 0], ud[:, 1])
        if stiction_model == "deadzone":
            ud = np.where((um < vsn)[:, None], 0.0, ud)
            new = al[:, None] * vel + (1 - al[:, None]) * gain[:, None] * ud
        else:
            raise ValueError("rigidsim implements the deadzone model only")
        mag = np.hypot(new[:, 0], new[:, 1])
        with np.errstate(all="ignore"):
            new = new * np.where(mag > vmax, vmax / mag, 1.0)[:, None]
        vel = new
        c = c + dt * vel
        C[:, j + 1] = c
    return C, PH
