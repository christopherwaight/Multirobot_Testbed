"""Follow-up checks: fine delay scan, speed dependence, low-speed behaviour, per-run gain, rotation."""
import pickle
import sys

import numpy as np

sys.path.insert(0, sys.path[0])
from fit import R_ALL, ratio, pack, simulate, onestep, rmse, fit, tau, delayed  # noqa: E402

healthy = [r for r in R_ALL if ratio(r) >= 0.75]
U, V = pack(healthy)
m = np.isfinite(V).all(axis=2)

print("[A] Fine delay scan, free-run fit, vk free and vk = 0 (D = total delay in steps)")
best = None
for D in np.arange(0.40, 1.21, 0.05):
    d, f = int(np.floor(D + 1e-9)), D - np.floor(D + 1e-9)
    x, c = fit(U, V, "oe", d, f, x0=(0.05, 0.88, 0.005))
    # vk = 0 and a = 0: model is linear, v = g uu, so g is closed form
    UU = delayed(U, d, f)
    g0 = (UU[m] * V[m]).sum() / (UU[m] ** 2).sum()
    c0 = rmse(g0 * UU, V)
    print(f"  D {D:.2f} ({D*100:.0f} ms): a {x[0]:.3f} g {x[1]:.3f} vk {x[2]:.4f} free-run {c:.4f}"
          f" | pure delay+gain: g {g0:.3f} free-run {c0:.4f}")
    if best is None or c0 < best[1]:
        best = (D, c0, g0)
print(f"  best pure delay+gain: D {best[0]:.2f}, g {best[2]:.3f}, RMSE {best[1]:.4f}")

Dbest = best[0]
d, f = int(np.floor(Dbest + 1e-9)), Dbest - np.floor(Dbest + 1e-9)
UU = delayed(U, d, f)

print("\n[B] Steady-state speed ratio vs command speed (command constant within 2 cm/s for 0.5 s)")
um = np.linalg.norm(U, axis=2)
vm = np.linalg.norm(V, axis=2)
steady = np.zeros_like(m)
for k in range(5, U.shape[1]):
    w = U[:, k - 5:k + 1]
    steady[:, k] = (np.linalg.norm(w - U[:, k:k + 1], axis=2).max(axis=1) < 0.02) & m[:, k]
# speed along the command direction
along = (V * U).sum(axis=2) / np.maximum(um, 1e-9)
edges = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.45]
for lo, hi in zip(edges[:-1], edges[1:]):
    s = steady & (um >= lo) & (um < hi)
    if s.sum() >= 5:
        print(f"  |u| {lo:.3f}-{hi:.3f}: n {s.sum():5d}  median |v| {np.median(vm[s]):.3f}  "
              f"along/|u| {np.median(along[s] / um[s]):.3f}  (p25 {np.percentile(along[s]/um[s],25):.3f}, "
              f"p75 {np.percentile(along[s]/um[s],75):.3f})")
print(f"  share of all samples with |u| > 0.3: {(um[m] > 0.3).mean()*100:.2f}%, > 0.25: {(um[m] > 0.25).mean()*100:.2f}%"
      f", < 0.05: {(um[m] < 0.05).mean()*100:.2f}%")

print("\n[C] Robots that were moving and then got a small command (|u| < 0.05 for 3+ steps)")
rows = []
for i in range(U.shape[0]):
    for k in range(4, U.shape[1]):
        if m[i, k] and um[i, k - 3:k + 1].max() < 0.05 and um[i, k - 6:k - 3].min() > 0.05:
            rows.append((um[i, k], vm[i, k]))
rows = np.array(rows)
if len(rows):
    print(f"  n {len(rows)}; median |u| {np.median(rows[:,0]):.3f}, median |v| {np.median(rows[:,1]):.3f}, "
          f"share with |v| < 0.01: {(rows[:,1] < 0.01).mean():.2f}")
else:
    print("  none in the data")

print("\n[D] Per-run gain with the pure delay model (D fixed), across all 760 robot-runs")
Ua, Va = pack(R_ALL)
ma = np.isfinite(Va).all(axis=2)
UUa = delayed(Ua, d, f)
gs = np.array([(UUa[i][ma[i]] * Va[i][ma[i]]).sum() / max((UUa[i][ma[i]] ** 2).sum(), 1e-12)
               for i in range(len(R_ALL))])
hist, be = np.histogram(gs, bins=np.arange(0.2, 1.11, 0.05))
for h, b0 in zip(hist, be[:-1]):
    print(f"  g {b0:.2f}-{b0+0.05:.2f}: {'#' * h} {h}")
for thr in (0.6, 0.7, 0.75, 0.8):
    sel = gs >= thr
    Us, Vs = pack([r for r, s in zip(R_ALL, sel) if s])
    ms = np.isfinite(Vs).all(axis=2)
    UUs = delayed(Us, d, f)
    gp = (UUs[ms] * Vs[ms]).sum() / (UUs[ms] ** 2).sum()
    print(f"  screen per-run g >= {thr}: keeps {sel.sum()} robot-runs, pooled g {gp:.3f}")
print("  per robot median per-run g (all runs): " + ", ".join(
    f"{nm} {np.median([g for g, r in zip(gs, R_ALL) if r['robot'] == nm]):.3f}"
    for nm in ("celeste", "tidal", "pacific-blue", "redwood")))
print("  per condition median per-run g: ")
conds = sorted({(r['field'], r['nr'], r['rd']) for r in R_ALL})
for c in conds:
    gg = [g for g, r in zip(gs, R_ALL) if (r['field'], r['nr'], r['rd']) == c]
    print(f"    {c[0]:<6} {c[1]}R r_d {c[2]:<5}: median {np.median(gg):.3f}  min {np.min(gg):.3f}")

print("\n[E] Rotation and anisotropy (complex gain, pure delay model)")
z_u = UU[..., 0] + 1j * UU[..., 1]
z_v = V[..., 0] + 1j * V[..., 1]
G = (np.conj(z_u[m]) * z_v[m]).sum() / (np.abs(z_u[m]) ** 2).sum()
print(f"  pooled complex gain |G| {abs(G):.3f}, angle {np.degrees(np.angle(G)):+.2f} deg")
for nm in ("celeste", "tidal", "pacific-blue", "redwood"):
    sel = np.array([r["robot"] == nm for r in healthy])
    mm = m & sel[:, None]
    G = (np.conj(z_u[mm]) * z_v[mm]).sum() / (np.abs(z_u[mm]) ** 2).sum()
    gx = (UU[..., 0][mm] * V[..., 0][mm]).sum() / (UU[..., 0][mm] ** 2).sum()
    gz = (UU[..., 1][mm] * V[..., 1][mm]).sum() / (UU[..., 1][mm] ** 2).sum()
    print(f"  {nm:<13} |G| {abs(G):.3f} angle {np.degrees(np.angle(G)):+.2f} deg | g_x {gx:.3f} g_z {gz:.3f}")

print("\n[F] Residual whiteness of the pure delay model: lag-1 autocorrelation of free-run residual")
E = g0 * UU - V
e = E[..., 0]
mk = m[:, 1:] & m[:, :-1]
print(f"  lag-1 autocorr (x): {np.corrcoef(e[:, 1:][mk], e[:, :-1][mk])[0,1]:+.3f}")
