"""Seed-condition (eq:seed_cond) instantiation at the objectivity start.

Double gyre, A=0.1, x_f=x+1, y_f=y+0.5, Omega=0.2 rad/s, seed (0.05,0.40).
Everything closed form from Draft_6a eq:dg_u and eq:s1_analytic_main.
"""
import numpy as np

A = 0.1
x, y = 0.05, 0.40
Om = 0.2
xf, yf = x + 1.0, y + 0.5

# flow (eq:dg_u)
u = -np.pi * A * np.sin(np.pi * xf) * np.cos(np.pi * yf)
v = np.pi * A * np.cos(np.pi * xf) * np.sin(np.pi * yf)
v0 = np.array([u, v])

# s1 and grad s1 (eq:s1_analytic_main), shear strain identically zero so the
# eigenvectors are the coordinate axes.
c = np.cos(np.pi * xf) * np.cos(np.pi * yf)
sgn = np.sign(c)
s1 = -np.pi**2 * A * abs(c)
ds1dx = -np.pi**2 * A * sgn * (-np.pi * np.sin(np.pi * xf) * np.cos(np.pi * yf))
ds1dy = -np.pi**2 * A * sgn * (-np.pi * np.cos(np.pi * xf) * np.sin(np.pi * yf))
g = np.array([ds1dx, ds1dy])

ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
d1, d2 = abs(g @ ex), abs(g @ ey)
t_raw = ex if d1 > d2 else ey

p = np.array([x, y])
transport = Om * np.array([-y, x])          # QdotQ^T p

print(f"s1        = {s1:.6f}")
print(f"grad s1   = ({g[0]:.4f}, {g[1]:.4f})   |g.ex|={d1:.4f}  |g.ey|={d2:.4f}"
      f"  -> t_raw = {'ex' if d1 > d2 else 'ey'}")
print(f"v0        = ({v0[0]:.6f}, {v0[1]:.6f})   |v0| = {np.linalg.norm(v0):.6f}")
print(f"|v0.t_raw|              = {abs(v0 @ t_raw):.6f}")
print(f"Omega*|p_seed|          = {Om*np.linalg.norm(p):.6f}")
print(f"|transport . t_raw|     = {abs(transport @ t_raw):.6f}")
print()
print(f"magnitude form  margin  = {abs(v0 @ t_raw)/(Om*np.linalg.norm(p)):.4f}x"
      f"   Omega_crit = {abs(v0 @ t_raw)/np.linalg.norm(p):.4f}"
      f"   ({100*(abs(v0 @ t_raw)/np.linalg.norm(p)/Om - 1):.0f}% faster)")
den = abs(np.array([-y, x]) @ t_raw)
print(f"projected form  margin  = {abs(v0 @ t_raw)/abs(transport @ t_raw):.4f}x"
      f"   Omega_crit = {abs(v0 @ t_raw)/den:.4f}")
print()
print(f"(paper's current |v0| against Omega|p| ratio = "
      f"{np.linalg.norm(v0)/(Om*np.linalg.norm(p)):.4f}x, "
      f"Omega_crit = {np.linalg.norm(v0)/np.linalg.norm(p):.4f})")
