import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = sys.argv[1]; out = sys.argv[2]
sys.path.insert(0, root)
from src.fields.field_types import load_field_config
A = load_field_config("double_gyre").get("A", 0.1)
L = 0.6
x = np.linspace(-L, L, 401); X, Y = np.meshgrid(x, x)
U = np.pi*A*np.cos(np.pi*X)*np.sin(np.pi*Y)
V = -np.pi*A*np.sin(np.pi*X)*np.cos(np.pi*Y)
M = np.hypot(U, V)
fig, ax = plt.subplots(figsize=(6, 5.2))
im = ax.pcolormesh(X, Y, M, cmap="viridis", shading="auto")
fig.colorbar(im, ax=ax, label="|v| (m/s)")
ax.contour(X, Y, M, levels=8, colors="w", linewidths=0.4, alpha=0.6)
d = np.array([[0.5,0],[0,0.5],[-0.5,0],[0,-0.5],[0.5,0]])
ax.plot(d[:,0], d[:,1], "r--", lw=1, label=r"$\det J=0$")
ax.plot([0], [0], "wo", ms=5, mec="k", label="center")
sx = [0.5,0.5,-0.5,-0.5]; sy = [0.5,-0.5,0.5,-0.5]
ax.plot(sx, sy, "w^", ms=6, mec="k", label="saddles")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Single gyre speed |v|, A = {A}")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=150)
print("A", A, "max |v| in view", M.max(), "center", M[200,200])
