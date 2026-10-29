"""
Cell list for the hyperparameter study. A cell is one configuration run as a
batch of N trials with random starts and headings. Factors not named in a cell
take the hardware baseline (vecsim.DEFAULTS):
    side 0.33 m, alpha 0.7, stiction 0.025 m/s, v_max 0.3 m/s, k 1 /s
    (command cap 0.5 m/s), k_t 0.2 m/s, k_r 0.5 /s, dt 0.1 s, no sensor noise.
IDEAL turns the robot non-idealities off (alpha 0, stiction 0, no speed cap).

E1  baseline per field x estimator, attraction and orbit at the paper's radii
E2  achievable orbit radius: r_d x {alpha, k_t, k_r, dt, stiction}
E3  attraction dynamics: k x alpha, stiction x alpha, dt x k
E4  formation size x field x estimator
E5  curvature eps x family x side x r_d, plus basins
E6  sensor noise x side x field (temporal noise; reconstructed fields carry fixed distortion)
E7  random global design, every parameter drawn per trial
"""
import zlib

import numpy as np

import vecsim as vs

HW = {}
IDEAL = dict(alpha=0.0, vs=0.0, vmax=1e9)
TAU = 0.28                                     # Decabot time constant, alpha = exp(-dt / tau)

SIDES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.33, 0.5, 0.75, 1.0]
ALPHAS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
STICTIONS = [0.0, 0.005, 0.01, 0.025, 0.05, 0.1]
KS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
KTS = [0.05, 0.1, 0.2, 0.3, 0.5]
KRS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
RDS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
PAPER_RDS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
SIGMAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
DTS = [0.02, 0.05, 0.1, 0.2]
EPSS = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
METHODS = ["fit", "oracle", "magnitude"]

T_ATTRACT = 60.0


def orbit_timing(rd, kt=0.2, kr=0.5, alpha=0.7, dt=0.1):
    """Transient exclusion and window: the window starts after 5 radial time
    constants (1/k_r), 10 lag time constants, and at least 20 s, and lasts 1.5
    revolutions clipped to 40..300 s."""
    tau = -dt / np.log(alpha) if alpha > 0 else 0.0
    w0 = max(20.0, 5.0 / kr, 10 * tau)
    W = float(np.clip(1.5 * 2 * np.pi * (rd + 0.2) / max(kt, 1e-3), 40.0, 300.0))
    return w0 + W, w0


def _seed(s):
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def _adder(C, stiction="legacy", scale=None):
    scale = scale or {}

    def add(exp, task, method, field, N, eps=1.0, family="potential", dt=0.1, T=None, hw=None,
            st=None, **p):
        st = st or stiction
        cid = f"{exp}|{task}|{method}|{field}|{family}{eps:+g}|dt{dt:g}|" + \
              ",".join(f"{k}={v:g}" for k, v in sorted(p.items()))
        if st != "legacy":
            cid += f"|st={st}"
        if hw:
            cid += f"|hw={hw}"
        if task == "attract":
            T, w0 = (T or T_ATTRACT), 0.0
        else:
            T, w0 = orbit_timing(p.get("rd", 0.3), p.get("kt", 0.2), p.get("kr", 0.5),
                                 p.get("alpha", 0.7), dt)
        mult = next((v for k, v in scale.items() if exp.startswith(k)), 1)
        C.append(dict(id=cid, exp=exp, task=task, method=method, field=field, eps=eps,
                      family=family, N=int(N * mult), dt=dt, T=T, window=w0, params=p,
                      seed=_seed(cid), stiction=st, hw=hw))
    return add


def build(stiction="legacy", scale=None):
    C = []
    add = _adder(C, stiction, scale)

    def max_rd(field):
        return 10.0 if field in vs.LINEAR else (1.0 if field == "gyre" else 0.5)

    # ---------------- E1 baseline
    for field in vs.FIELDS:
        for m in METHODS:
            for tag, base in (("hw", HW), ("ideal", IDEAL)):
                add("E1" + tag, "attract", m, field, 10000, **base)
                for rd in PAPER_RDS:
                    add("E1" + tag, "orbit", m, field, 1000, rd=rd, **base)

    # ---------------- E2 achievable radius
    for field in ("vortex", "saddle", "gyre", "vortex_nn"):
        for m in ("fit", "oracle"):
            for rd in RDS:
                if rd > max_rd(field):
                    continue
                for a in ALPHAS:
                    add("E2alpha", "orbit", m, field, 200, rd=rd, alpha=a)
    for m in ("fit", "oracle"):
        for rd in RDS:
            for kt in KTS:
                add("E2kt", "orbit", m, "vortex", 200, rd=rd, kt=kt)
            for kr in KRS:
                add("E2kr", "orbit", m, "vortex", 200, rd=rd, kr=kr)
            for dt in DTS:
                add("E2dt", "orbit", m, "vortex", 200, dt=dt, rd=rd, alpha=float(np.exp(-dt / TAU)))
            for v in STICTIONS:
                for a in (0.7, 0.9):
                    add("E2vs", "orbit", m, "vortex", 200, rd=rd, vs=v, alpha=a)

    # ---------------- E3 attraction dynamics
    for field in ("vortex", "saddle", "gyre", "saddle_nn"):
        for a in ALPHAS:
            for k in KS:
                add("E3k", "attract", "fit", field, 1000, k=k, alpha=a)
            for v in STICTIONS:
                add("E3vs", "attract", "fit", field, 1000, vs=v, alpha=a)
    for a in ALPHAS:
        for k in KS:
            add("E3k", "attract", "oracle", "vortex", 1000, k=k, alpha=a)
    for dt in DTS:
        for k in KS:
            add("E3dt", "attract", "fit", "vortex", 1000, dt=dt, k=k, alpha=float(np.exp(-dt / TAU)))

    # ---------------- E4 formation size
    size_fields = [("vortex", 1.0, "potential"), ("gyre", 1.0, "potential"),
                   ("curved_saddle", 1.0, "potential"), ("curved_saddle", 1.0, "cellular"),
                   ("vortex_nn", 1.0, "potential"), ("saddle_nn", 1.0, "potential")]
    for field, eps, fam in size_fields:
        for s in SIDES:
            for m in ("fit", "magnitude"):
                for tag, base in (("hw", HW), ("ideal", IDEAL)):
                    add("E4" + tag, "attract", m, field, 1000, eps, fam, side=s, **base)
                    for rd in (0.1, 0.2, 0.3):
                        add("E4" + tag, "orbit", m, field, 300, eps, fam, side=s, rd=rd, **base)

    # ---------------- E5 curvature
    for fam in ("potential", "cellular"):
        for eps in EPSS:
            for s in (0.1, 0.33):
                for tag, base in (("hw", HW), ("ideal", IDEAL)):
                    for rd in (0.1, 0.2, 0.3, 0.4):
                        add("E5" + tag, "orbit", "fit", "curved_saddle", 300, eps, fam, side=s, rd=rd, **base)
                add("E5basin", "attract", "fit", "curved_saddle", 2000, eps, fam, side=s)
    for s in (0.1, 0.2, 0.33):
        add("E5basin", "attract", "fit", "gyre", 2000, side=s)

    # ---------------- E6 noise
    for field in vs.LINEAR:
        for sg in SIGMAS:
            for s in (0.1, 0.33, 1.0):
                for tag, base in (("hw", HW), ("ideal", IDEAL)):
                    add("E6" + tag, "attract", "fit", field, 1000, side=s, sigma_n=sg, **base)
            for rd in (0.2, 0.5):
                add("E6hw", "orbit", "fit", field, 500, rd=rd, sigma_n=sg)
                add("E6ideal", "orbit", "fit", field, 500, rd=rd, sigma_n=sg, **IDEAL)
    for field in vs.RECONSTRUCTED:
        for sg in SIGMAS:
            add("E6hw", "attract", "fit", field, 1000, sigma_n=sg)
            add("E6ideal", "attract", "fit", field, 1000, sigma_n=sg, **IDEAL)
    return C


def build_extra(stiction="deadzone"):
    """Version 2 additions.
    E8a  attraction overshoot / undershoot: tau x dt x k x caps x stiction (vortex, plane fit)
    E8o  orbit radial transient from r0 = 0.05 or 0.6 to r_d = 0.3: tau x dt x k_r x caps
    E9a  attraction with the robot speed cap raised: v_max x alpha x k (no command cap)
    E9o  orbit with raised caps: v_max x k_t x r_d x radial clamp
    E10  stiction model comparison: model x dt (tau fixed) x v_s, attraction and orbit
    E10hw  hardware start positions, 10 s runs, each model, linear and reconstructed fields"""
    C = []
    add = _adder(C, stiction)
    OFF = dict(vmax=1e9, vcmd_max=1e9)
    for tau in (0.0, 0.1, 0.28, 0.5, 1.0):
        for dt in (0.01, 0.02, 0.05, 0.1, 0.2):
            a = float(np.exp(-dt / tau)) if tau > 0 else 0.0
            for k in (0.3, 1.0, 3.0, 10.0, 30.0):
                for caps, base in (("hw", {}), ("off", OFF)):
                    for v in (0.0, 0.025):
                        add("E8a" + caps, "attract", "fit", "vortex", 500, dt=dt, alpha=a, k=k, vs=v, **base)
            if dt in (0.02, 0.1, 0.2):
                for kr in (0.25, 0.5, 1.0, 2.0, 5.0):
                    for r0 in (0.05, 0.6):
                        for caps, base in (("hw", {}), ("off", dict(vmax=1e9, rclamp=1e9))):
                            add("E8o" + caps, "orbit", "fit", "vortex", 200, dt=dt, alpha=a, kr=kr, rd=0.3,
                                r0=r0, **base)
    for vm in (0.1, 0.3, 1.0, 3.0, 1e9):
        for a in (0.0, 0.5, 0.7, 0.9):
            for k in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
                add("E9a", "attract", "fit", "vortex", 1000, alpha=a, k=k, vmax=vm, vcmd_max=1e9)
    for vm in (0.3, 1.0, 3.0, 1e9):
        for kt in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
            for rd in (0.1, 0.3, 1.0, 3.0):
                for rc in (0.3, 1e9):
                    add("E9o", "orbit", "fit", "vortex", 200, kt=kt, rd=rd, vmax=vm, rclamp=rc)
    for model in ("legacy", "scaled", "deadzone"):
        for dt in (0.01, 0.02, 0.05, 0.1, 0.2):
            a = float(np.exp(-dt / TAU))
            for v in (0.01, 0.025, 0.05):
                for k in (1.0, 10.0):
                    add("E10", "attract", "fit", "vortex", 1000, dt=dt, st=model, alpha=a, vs=v, k=k)
                for kt in (0.05, 0.2):
                    add("E10", "orbit", "fit", "vortex", 200, dt=dt, st=model, alpha=a, vs=v, kt=kt, rd=0.3)
        for field, hw in (("vortex", "vortex"), ("vortex_nn", "vortex"), ("saddle", "saddle"), ("saddle_nn", "saddle")):
            add("E10hw", "attract", "fit", field, 50, T=10.0, st=model, hw=hw)
    return C


def random_batches(n_batches=1000, per_batch=1000, seed=7, stiction="legacy"):
    """E7: each batch fixes (field, method, task, dt); every other parameter is
    drawn per trial in run_study.random_params from the batch seed."""
    rng = np.random.default_rng(seed)
    B = []
    fields = vs.FIELDS
    for b in range(n_batches):
        field = fields[b % len(fields)]
        eps, fam = 1.0, "potential"
        if field == "curved_saddle":
            eps = float(rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
            fam = str(rng.choice(["potential", "cellular"]))
        task = "attract" if (b // len(fields)) % 2 == 0 else "orbit"
        method = METHODS[(b // (2 * len(fields))) % 3]
        dt = float(rng.choice([0.05, 0.1, 0.2]))
        T = T_ATTRACT if task == "attract" else 180.0
        cid = f"E7|{b:04d}|{task}|{method}|{field}|{fam}{eps:+g}|dt{dt:g}"
        if stiction != "legacy":
            cid += f"|st={stiction}"
        B.append(dict(id=cid, exp="E7", task=task, method=method, field=field, eps=eps,
                      family=fam, N=per_batch, dt=dt, T=T, window=60.0 if task == "orbit" else 0.0,
                      params={}, seed=_seed(cid), random=True, stiction=stiction, hw=None))
    return B


V2_SCALE = {"E1": 10, "E2": 5, "E3": 5, "E4": 5, "E5": 5, "E6": 5}


def build_v2():
    return build("deadzone", V2_SCALE) + build_extra("deadzone") + \
        random_batches(per_batch=10000, stiction="deadzone")


if __name__ == "__main__":
    import sys
    if "--v2" in sys.argv:
        C = [c for c in build_v2() if not c.get("random")]
        R = [c for c in build_v2() if c.get("random")]
    else:
        C = build()
        R = random_batches()
    from collections import Counter
    cnt = Counter(c["exp"] for c in C)
    trials = sum(c["N"] for c in C) + sum(b["N"] for b in R)
    steps = sum(c["N"] * c["T"] / c["dt"] for c in C + R)
    print(dict(cnt), "E7 batches", len(R))
    print(f"{len(C)} cells + {len(R)} random batches, {trials:,} trials, {steps:.2e} trial-steps")
