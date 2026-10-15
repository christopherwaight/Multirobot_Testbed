"""Extract the main FTLE ridge (main_ridge.npy) by least-cost path through the 24-h FTLE cache. Run: venv/bin/python3 experiments/ocean_small_rho/extract_main_ridge.py $PWD experiments/ocean_small_rho/main_ridge.npy 34.0 -120.25"""
import sys, os, numpy as np
VF = sys.argv[1]; OUT = sys.argv[2]
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
z = np.load(os.path.join(VF, 'experiments/outputs/oecs/ftle_cache_24h_u4.npz'), allow_pickle=True)
lat, lon, f, land = z['lat'], z['lon'], z['ftle'], z['land']
KLA = 111.32; KLO = 111.32*np.cos(np.radians(34.2))
fn = np.where(np.isfinite(f) & ~land, f, 0) / np.nanpercentile(f[~land], 99)
cost = 1.0/np.clip(fn, 0.02, None)**4      # strongly prefer the crest
ny, nx = f.shape
idx = np.arange(ny*nx).reshape(ny, nx)
rows, cols, w = [], [], []
for di, dj in [(0,1),(1,0),(1,1),(1,-1)]:
    a = idx[max(0,-di):ny-max(0,di), max(0,-dj):nx-max(0,dj)]
    b = idx[max(0,di):ny+min(0,di) or None, max(0,dj):nx+min(0,dj) or None]
    a=a.ravel(); b=b.ravel()
    ok = (~land.ravel()[a]) & (~land.ravel()[b])
    a, b = a[ok], b[ok]
    L = np.hypot(di*0.493, dj*0.472)
    c = 0.5*(cost.ravel()[a]+cost.ravel()[b])*L
    rows += [a, b]; cols += [b, a]; w += [c, c]
G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(ny*nx, ny*nx)).tocsr()
def node(la, lo): return np.argmin(abs(lat-la))*nx + np.argmin(abs(lon-lo))
def path(p, q):
    d, pred = dijkstra(G, indices=node(*p), return_predecessors=True)
    n = node(*q); out=[]
    while n >= 0: out.append(n); n = pred[n]
    out = out[::-1]
    return np.array([[lat[k//nx], lon[k%nx]] for k in out])
# optional 5th/6th args: an upstream head to prepend (main_ridge_ne.npy used 34.448 -120.279)
waypts = [(34.425,-120.37), (34.14,-120.41)]
end_se = (float(sys.argv[3]), float(sys.argv[4]))
r = np.vstack([path(waypts[0], waypts[1]), path(waypts[1], end_se)[1:]])
if len(sys.argv) > 6:
    r = np.vstack([path((float(sys.argv[5]), float(sys.argv[6])), waypts[0])[:-1], r])
np.save(OUT, r)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7,7))
ax.pcolormesh(lon, lat, np.ma.array(f, mask=land), cmap='jet', vmin=np.percentile(f[~land],34), vmax=np.percentile(f[~land],98.8))
ax.plot(r[:,1], r[:,0], 'w-', lw=1)
ax.set_xlim(-120.65,-119.95); ax.set_ylim(33.92,34.52); ax.set_aspect(1/np.cos(np.radians(34.2)))
fig.savefig(OUT.replace('.npy','.png'), dpi=110)
print(len(r), 'pts, length km', np.sum(np.hypot(np.diff(r[:,0])*KLA, np.diff(r[:,1])*KLO)))
