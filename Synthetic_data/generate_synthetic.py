"""
Build the Synthetic_data folder: real copies plus made-up stand-ins for runs not yet taken.

EVERYTHING under synthetic/ IS FABRICATED. Do not quote a number from it. Replace it with
real logs and keep the same file and variable names so the pipeline does not change.

Real (copied unmodified from trunk/robots_3, three-robot logs only, decision 2026-10-30):
  real_copy/attraction_3robot/{vortex,saddle}/runN.mat     80 and 78 runs
  real_copy/orbit_3robot/{vortex,saddle}/radiusNN.mat      the orbit logs that exist

Synthetic:
  synthetic/orbit_3robot/{vortex,saddle}/rdXXX_trialKK.mat   8 radii x 10 trials x 2 fields
  synthetic/attraction_4robot/{vortex,saddle}/runN.mat       each real 3-robot run + noise profile
  synthetic/orbit_4robot/{vortex,saddle}/rdXXX_trialKK.mat   same grid, four-robot square

Positions are OptiTrack metres, critical point at the origin, 10 Hz, as in the real logs.
Run from anywhere: python3 generate_synthetic.py   (needs numpy and scipy; the VF_Robot venv has both)
"""
import csv
import os
import re
import shutil

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROBOTS3 = os.path.join(REPO, "trunk", "robots_3")
SRC = {"vortex": os.path.join(ROBOTS3, "vortex_tests", "3-robot-tests"),
       "saddle": os.path.join(ROBOTS3, "saddle_tests")}

SEED = 20261101
DT = 0.1
ORBIT_RADII = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_TRIALS = 10
N_ORBIT = 601
# Hardware tangential speed k_t [m/s]: omega * r in the real orbit logs (vortex 0.14, saddle 0.20).
# Recomputed from the real logs below; these are only the fallback.
KT_FALLBACK = {"vortex": 0.14, "saddle": 0.20}
SQUARE_SIDE_4 = 0.24

rng = np.random.default_rng(SEED)


def loadmat(p):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sio.loadmat(p)


def rd_of(name):
    s = re.fullmatch(r"radius(\d+)", name).group(1)
    return 0.01 if s == "001" else int(s) / 10


def rd_tag(rd):
    return f"rd{int(round(rd * 1000)):03d}"


# ----------------------------------------------------------------------------- real copies
def copy_real():
    manifest = []
    for field, folder in SRC.items():
        for name in sorted(os.listdir(folder), key=lambda s: (len(s), s)):
            m = re.fullmatch(r"(run\d+|radius\d+)\.mat", name)
            if not m:
                continue
            kind = "attraction_3robot" if name.startswith("run") else "orbit_3robot"
            dst = os.path.join(HERE, "real_copy", kind, field)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(os.path.join(folder, name), os.path.join(dst, name))
            manifest.append(dict(origin="real", kind=kind, field=field, n_robots=3,
                                 file=os.path.relpath(os.path.join(dst, name), HERE),
                                 rd="" if name.startswith("run") else rd_of(name[:-4]),
                                 trial="", source=os.path.relpath(os.path.join(folder, name), REPO)))
    return manifest


# ----------------------------------------------------------------------------- calibration
def ar1(n, std, tau, size=1):
    """Stationary AR(1) with marginal std and correlation time tau [s]."""
    a = np.exp(-DT / tau)
    e = rng.normal(0, std * np.sqrt(1 - a * a), (n, size))
    x = np.zeros((n, size))
    x[0] = rng.normal(0, std, size)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def calibrate_orbit():
    """Per field: radii, radial bias, radial std, estimate scatter, k_t, and the body-frame
    formation offsets, all read from the real three-robot orbit logs (after a 10 s transient)."""
    cal = {}
    for field, folder in SRC.items():
        rows, kts, bodies, pqb = [], [], [], []
        for name in sorted(os.listdir(folder)):
            if not re.fullmatch(r"radius\d+\.mat", name):
                continue
            m = loadmat(os.path.join(folder, name))
            rd = rd_of(name[:-4])
            c = m["cluster_pose"]
            t = m["tout"].ravel()
            ph = m["detected_critical_point"]
            r = np.linalg.norm(c[:, :2], axis=1)[100:]
            mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
            th = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
            w = np.polyfit(t[100:], th[100:], 1)[0]
            rows.append((rd, r.mean() - rd, r.std(), 0.5 * (mad(ph[100:, 0]) + mad(ph[100:, 1]))))
            kts.append(abs(w) * r.mean())
            R = np.stack([m[f"robot{k}_pose"][:, :2] for k in (1, 2, 3)], 1)
            off = R - c[:, None, :2]
            h = c[:, 2]
            rot = np.stack([[np.cos(-h), -np.sin(-h)], [np.sin(-h), np.cos(-h)]]).transpose(2, 0, 1)
            bodies.append(np.median(np.einsum("nij,nkj->nki", rot, off), 0))
            pqb.append(np.median(c[:, 3:6], 0))
        rows = np.array(sorted(rows))
        cal[field] = dict(rd=rows[:, 0], bias=rows[:, 1], std=rows[:, 2], est=rows[:, 3],
                          kt=float(np.median(kts)) if kts else KT_FALLBACK[field],
                          body3=np.median(bodies, 0), pqb=np.median(pqb, 0))
    return cal


# ----------------------------------------------------------------------------- formations
def body_offsets(n_robots, cal_field):
    if n_robots == 3:
        return cal_field["body3"]
    half = SQUARE_SIDE_4 / 2
    return np.array([[half, half], [-half, half], [-half, -half], [half, -half]])


def place_robots(c, heading, n_robots, cal_field, pos_noise=0.001, shape_noise=0.004):
    """Robots at body offsets rotated by the heading, plus OptiTrack noise and slow shape error.
    The reported centroid is left as given (real 4-robot logs report the robot mean, so the
    caller sets that for four robots)."""
    n = len(c)
    body = body_offsets(n_robots, cal_field)
    rot = np.stack([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]).transpose(2, 0, 1)
    R = c[:, None, :] + np.einsum("nij,kj->nki", rot, body)
    R = R + ar1(n, shape_noise, 3.0, n_robots * 2).reshape(n, n_robots, 2)
    R = R + rng.normal(0, pos_noise, R.shape)
    hd = heading[:, None] + rng.normal(0, 0.01, (n, n_robots)) + rng.uniform(-0.05, 0.05, n_robots)
    return np.concatenate([R, hd[:, :, None]], axis=2)


def write_log(path, t, c_xy, phat, vdes, robots, heading, cal_field, n_robots):
    n = len(t)
    d = {"tout": t.reshape(-1, 1), "detected_critical_point": phat}
    for k in range(n_robots):
        d[f"robot{k + 1}_pose"] = robots[:, k, :]
    if n_robots == 3:
        v = np.gradient(c_xy, DT, axis=0)
        p, q, b = cal_field["pqb"]
        d["cluster_pose"] = np.column_stack([c_xy, heading, np.full(n, p), np.full(n, q), np.full(n, b),
                                            v, np.gradient(heading, DT)])
        d["desired_velocities"] = np.column_stack([vdes, np.zeros(n)])
    else:
        d["cluster_position"] = c_xy
        d["detected_critical_point_avg"] = np.cumsum(phat, 0) / np.arange(1, n + 1)[:, None]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sio.savemat(path, d)


# ----------------------------------------------------------------------------- synthetic orbit
def synth_orbit(field, rd, n_robots, cal):
    cf = cal[field]
    scale = 1.0 if n_robots == 3 else 0.8          # four robots: smaller bias and scatter (assumed)
    bias_nom = np.interp(rd, cf["rd"], cf["bias"]) * scale
    std_nom = np.interp(rd, cf["rd"], cf["std"]) * scale
    est_nom = np.interp(rd, cf["rd"], cf["est"]) * scale
    n = N_ORBIT
    t = np.arange(n) * DT
    bias = bias_nom + rng.normal(0, 0.15 * abs(bias_nom) + 0.005)
    # settle from a start near the commanded radius, error decaying with k_r ~ 1/ (3 s)
    r0 = rd + rng.normal(0, 0.03) + 0.1 * rd
    trans = (r0 - rd - bias) * np.exp(-t / 3.0)
    s_trial = std_nom * rng.lognormal(0, 0.15)
    th = np.zeros(n)
    th0 = rng.uniform(0, 2 * np.pi)
    kt = cf["kt"] * (1 + rng.normal(0, 0.02))
    ar = ar1(n, s_trial * np.sqrt(0.5), 2.0)[:, 0]
    rho = np.full(n, rd)
    phi = rng.uniform(0, 2 * np.pi)
    th[0] = th0
    for i in range(n):
        if i > 0:
            th[i] = th[i - 1] - kt / max(rho[i - 1], 0.05) * DT * (1 + rng.normal(0, 0.01))
        once = s_trial * np.sqrt(0.5) * np.sqrt(2) * np.sin(th[i] - th0 + phi)   # once-per-rev field-fit error
        rho[i] = max(rd + bias + trans[i] + once + ar[i], 0.003)
    centre = rng.normal(0, 0.02 + 0.03 * rd, 2)                                   # per-trial estimate offset
    c_true = centre + rho[:, None] * np.column_stack([np.cos(th), np.sin(th)])
    heading = rng.normal(-0.02, 0.15) + ar1(n, 0.035, 5.0)[:, 0]
    robots = place_robots(c_true, heading, n_robots, cf)
    if n_robots == 4:
        c_rep = robots[:, :, :2].mean(1)
    else:
        c_rep = c_true + rng.normal(0, 0.001, (n, 2))
    est_std = max(est_nom, 0.01)
    phat = centre + ar1(n, est_std, 1.5, 2) + rng.normal(0, 0.003, (n, 2))
    vdes = np.gradient(c_true, DT, axis=0) * 1.1
    return t, c_rep, phat, vdes, robots, heading, rd, bias


# ----------------------------------------------------------------------------- synthetic 4-robot attraction
def synth_attraction_4(field, src_path, cal):
    m = loadmat(src_path)
    t0 = m["tout"].ravel()
    c0 = m["cluster_pose"][:, :2]
    ph0 = m["detected_critical_point"]
    h0 = m["cluster_pose"][:, 2]
    n = len(t0)
    # different dynamics with four robots: stretch time by ~5 percent, start jitter, residual stiction offset
    stretch = rng.normal(1.0, 0.05)
    tq = t0 * stretch
    c = np.column_stack([np.interp(t0, tq, c0[:, 0]), np.interp(t0, tq, c0[:, 1])])
    c = c + rng.normal(0, 0.02, 2) * (1 - np.arange(n) / (n - 1))[:, None]               # start jitter, fades out
    reach = np.clip(1 - np.linalg.norm(c, axis=1) / max(np.linalg.norm(c[0]), 1e-6), 0, 1)
    c = c + reach[:, None] * rng.normal(0, 0.008, 2)                                      # parked-offset noise
    c = c + ar1(n, 0.003, 1.5, 2)
    heading = np.interp(t0, tq, h0) + rng.normal(0, 0.1)
    robots = place_robots(c, heading, 4, cal[field])
    c_rep = robots[:, :, :2].mean(1)
    ph = ph0 * rng.normal(1.0, 0.05) * 0.85 + ar1(n, 0.01, 1.0, 2)
    return t0, c_rep, ph, None, robots, heading


def main():
    manifest = copy_real()
    cal = calibrate_orbit()
    for field in cal:
        cf = cal[field]
        print(f"calibration {field}: k_t={cf['kt']:.3f} m/s, "
              + ", ".join(f"rd {r:.2f}: bias {b:+.3f} std {s:.3f}" for r, b, s in zip(cf["rd"], cf["bias"], cf["std"])))

    for n_robots in (3, 4):
        kind = f"orbit_{n_robots}robot"
        for field in ("vortex", "saddle"):
            for rd in ORBIT_RADII:
                for k in range(1, N_TRIALS + 1):
                    t, c, ph, vd, R, h, rd_, bias = synth_orbit(field, rd, n_robots, cal)
                    fn = f"{rd_tag(rd)}_trial{k:02d}.mat"
                    path = os.path.join(HERE, "synthetic", kind, field, fn)
                    write_log(path, t, c, ph, vd, R, h, cal[field], n_robots)
                    manifest.append(dict(origin="SYNTHETIC", kind=kind, field=field, n_robots=n_robots,
                                         file=os.path.relpath(path, HERE), rd=rd, trial=k, source=""))

    for field, folder in SRC.items():
        for name in sorted(os.listdir(folder), key=lambda s: (len(s), s)):
            if not re.fullmatch(r"run\d+\.mat", name):
                continue
            t, c, ph, vd, R, h = synth_attraction_4(field, os.path.join(folder, name), cal)
            path = os.path.join(HERE, "synthetic", "attraction_4robot", field, name)
            write_log(path, t, c, ph, vd, R, h, cal[field], 4)
            manifest.append(dict(origin="SYNTHETIC", kind="attraction_4robot", field=field, n_robots=4,
                                 file=os.path.relpath(path, HERE), rd="", trial="",
                                 source=os.path.relpath(os.path.join(folder, name), REPO)))

    with open(os.path.join(HERE, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["origin", "kind", "field", "n_robots", "file", "rd", "trial", "source"])
        w.writeheader()
        w.writerows(manifest)
    n_syn = sum(r["origin"] == "SYNTHETIC" for r in manifest)
    print(f"{len(manifest) - n_syn} real copies, {n_syn} synthetic files, manifest.csv written")


if __name__ == "__main__":
    main()
