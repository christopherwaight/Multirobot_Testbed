import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/christopherwaight/Desktop/Multirobot_Testbed/trunk/Python_Simulations/Vector_Fields/VF_Robot")
from src.fields.environments.Double_Gyre import double_gyre_static, SADDLE_TOP, SADDLE_BOTTOM

CFG = {"A": 0.1, "eps": 0.0, "omega": 0.628318530717958}
T = 0.0
N = 301
x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-0.5, 0.5, N)
X, Y = np.meshgrid(x, y)

U = np.zeros_like(X); V = np.zeros_like(X)
for i in range(N):
    for j in range(N):
        U[i,j], V[i,j] = double_gyre_static(X[i,j], Y[i,j], T, CFG)

dx = x[1]-x[0]; dy = y[1]-y[0]
dUdy, dUdx = np.gradient(U, dy, dx)
dVdy, dVdx = np.gradient(V, dy, dx)
curl = dVdx - dUdy
div  = dUdx + dVdy

# analytic check
A = CFG["A"]; xf = X + 1.0; yf = Y + 0.5
curl_a = -2*np.pi**2*A*np.sin(np.pi*xf)*np.sin(np.pi*yf)
print("max |div| numeric      :", np.abs(div).max())
print("max |curl| numeric     :", np.abs(curl).max())
print("max |curl - analytic|  :", np.abs(curl-curl_a).max())
print("curl at gyre center (-0.5,0):", curl_a[N//2, np.argmin(abs(x+0.5))])
print("curl at gyre center ( 0.5,0):", curl_a[N//2, np.argmin(abs(x-0.5))])
print("max |curl| on x=0 line :", np.abs(curl_a[:, np.argmin(abs(x))]).max())

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, F, name in ((axes[0], curl, "Curl  " + r"$\partial v/\partial x - \partial u/\partial y$"),
                    (axes[1], div,  "Divergence  " + r"$\partial u/\partial x + \partial v/\partial y$")):
    lim = max(np.abs(curl).max(), 1e-12)
    im = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax.contour(X, Y, F, levels=[0.0], colors="k", linewidths=1.0)
    sk = 14
    ax.quiver(X[::sk,::sk], Y[::sk,::sk], U[::sk,::sk], V[::sk,::sk],
              color="0.25", scale=2.2, width=0.0032)
    ax.axvline(0.0, color="k", ls="--", lw=1.1)
    ax.plot([SADDLE_BOTTOM[0], SADDLE_TOP[0]], [SADDLE_BOTTOM[1], SADDLE_TOP[1]],
            "ko", ms=6, mfc="w", mew=1.4)
    ax.plot([-0.5, 0.5], [0, 0], "k+", ms=10, mew=1.6)
    ax.set_title(name); ax.set_xlabel("x (world)"); ax.set_ylabel("y (world)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
fig.suptitle("Double gyre (A=0.1, eps=0, t=0), both panels on the same color scale", y=0.99)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "double_gyre_curl_div.png")
fig.savefig(out, dpi=150)
print("wrote", out)
