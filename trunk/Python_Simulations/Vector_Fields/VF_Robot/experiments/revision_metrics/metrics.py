"""
Per-trial metric definitions for the IEEE Systems revision (Reviewers 1 and 3).
The same functions score hardware logs and simulation, so every number in the
revised tables is computed one way.

Conventions
    Samples are j = 0..T at spacing dt; c[j] is the cluster centroid, phat[j]
    the critical point estimate formed from the positions at sample j.
    target is the true critical point (the origin of the printed map).

ATTRACTION, per trial
    e[j] = |c[j] - p*|, e_final = e[T]
    est_final = |phat[last] - p*|, est_ss = mean |phat[j] - p*| over the last 20 % of samples
    for each eps: t_hit = first j dt with e[j] <= eps
                  t_settle = first j dt with e[m] <= eps for every m >= j (NaN if e[T] > eps)
                  fail = centroid leaves the workspace  or  e_final > eps
    effort, realized: L_path = sum |c[j+1] - c[j]|, E_u = sum |c[j+1] - c[j]|^2 / dt
            commanded (where logged): L_cmd = sum |v_des| dt, E_cmd = sum |v_des|^2 dt
    path efficiency = |c[0] - c[T]| / L_path

CLASSIFICATION (Table I), per cycle from the estimated Jacobian J = [a b; d e]
    tr = a + e, det = ae - bd, disc = tr^2 - 4 det
    det < 0: saddle. det > 0: center if |tr| <= eps sqrt(det); node if disc >= 0 or
    sqrt(-disc) <= eps sqrt(det); spiral otherwise; stable or unstable by the sign of tr.
    The two bands are the same test on the eigenvalue angle: within asin(eps / 2)
    (about 6 degrees for eps = 0.2) of the imaginary axis is a center, of the
    real axis a node. Without them the canonical center (tr = 0) and the star
    nodes J = -I, +I (disc = 0) sit exactly on class boundaries.
    Coarse class: saddle, attracting (sink, stable spiral), repelling (source,
    unstable spiral), center. The true type uses the same rule on the true J.
    Scores: fraction of valid cycles correct (strict and coarse), over a window,
    time of the first correct label and time from which the label stays correct.

ORBIT, per run, window W1 = samples j > 100 (10 s transient excluded), W2 = j >= 1
    r*[j] = |c[j] - p*|, r_est[j] = |c[j] - phat[j]|
    bias = mean(r* - r_d), sd_within = std(r*), MAE, RMSE, ctrl_bias = mean(r_est - r_d)
    Estimates outside the printed map (near-singular fits) are counted (n_est_outside) and
    left out of ctrl_bias and of every localization error.
    laps: the unwrapped angle of c about p* over W1, cut at every 2 pi of
          progress in the direction of travel; partial laps are dropped
"""
import numpy as np

DT = 0.1
EPS_CONV = (0.03, 0.04, 0.05)
MAP_SIDE = 1.6764                 # printed map side, OptiTrack metres
HALF_WORKSPACE = MAP_SIDE / 2
TRANSIENT_STEPS = 100             # 10 s at 10 Hz


def _norm(a):
    return np.hypot(a[..., 0], a[..., 1])


def _plausible_est_err(phat, tgt):
    """|phat - p*| with estimates outside the printed map set to NaN, and the count of those."""
    pe = _norm(np.asarray(phat, float) - tgt)
    bad = ~(pe <= HALF_WORKSPACE)
    pe[bad] = np.nan
    return pe, int(bad.sum())


def settle_times(err, eps, dt=DT):
    """First-hit and permanent-settling times of err[j] <= eps (NaN when absent)."""
    inside = np.asarray(err) <= eps
    t_hit = np.argmax(inside) * dt if inside.any() else np.nan
    if not inside[-1]:
        return t_hit, np.nan
    outside = np.flatnonzero(~inside)
    t_settle = 0.0 if outside.size == 0 else (outside[-1] + 1) * dt
    return t_hit, t_settle


def left_workspace(c, half=HALF_WORKSPACE):
    c = np.asarray(c)
    ok = np.isfinite(c).all(-1)
    return bool((~ok).any() or (np.abs(c[ok]) > half).any())


def attraction_trial(c, phat, target=(0.0, 0.0), dt=DT, eps_list=EPS_CONV,
                     vcmd=None, robot_cmd_speed=None, robot_vel_speed=None, robots=None):
    """Scalars for one attraction trial.
    c: (T+1, 2). phat: (m, 2) estimates (m = T or T+1). vcmd: (m, 2) centroid
    command or None. robot_cmd_speed / robot_vel_speed: (T, n_r) speeds, or
    robots: (T+1, n_r, 2) positions from which realized robot speeds are taken."""
    tgt = np.asarray(target, float)
    c = np.asarray(c, float)
    e = _norm(c - tgt)
    out = dict(final_x=c[-1, 0] - tgt[0], final_y=c[-1, 1] - tgt[1], e_final=e[-1],
               start_dist=e[0], min_dist=np.nanmin(e))
    pe, out["n_est_outside"] = _plausible_est_err(phat, tgt)
    fin = np.isfinite(pe)
    out["est_final"] = pe[fin][-1] if fin.any() else np.nan
    tail = pe[int(np.floor(0.8 * len(pe))):]
    out["est_ss"] = np.nanmean(tail) if np.isfinite(tail).any() else np.nan
    out["left_ws"] = left_workspace(c)
    for eps in eps_list:
        tag = f"{round(eps * 100):d}cm"
        out[f"t_hit_{tag}"], out[f"t_settle_{tag}"] = settle_times(e, eps, dt)
        out[f"fail_{tag}"] = bool(out["left_ws"] or not (e[-1] <= eps))
    step = _norm(np.diff(c, axis=0))
    out["L_path"] = np.nansum(step)
    out["E_real"] = np.nansum(step ** 2) / dt
    out["path_eff"] = _norm(c[-1] - c[0]) / out["L_path"] if out["L_path"] > 0 else np.nan
    if vcmd is not None:
        s = _norm(np.asarray(vcmd, float))
        out["L_cmd"] = np.nansum(s) * dt
        out["E_cmd"] = np.nansum(s ** 2) * dt
    if robots is not None:
        robot_vel_speed = _norm(np.diff(np.asarray(robots, float), axis=0)) / dt
    if robot_vel_speed is not None:
        s = np.asarray(robot_vel_speed, float)
        out["L_robot_real"] = np.nanmean(np.nansum(s, 0)) * dt
        out["E_robot_real"] = np.nanmean(np.nansum(s ** 2, 0)) * dt
    if robot_cmd_speed is not None:
        s = np.asarray(robot_cmd_speed, float)
        out["L_robot_cmd"] = np.nanmean(np.nansum(s, 0)) * dt
        out["E_robot_cmd"] = np.nanmean(np.nansum(s ** 2, 0)) * dt
    return out


def segment_laps(c, target=(0.0, 0.0)):
    """Lap boundaries for a trajectory: sample indices b[0] < b[1] < ... at the
    first sample where the angle about target has advanced 2 pi m in the
    direction of travel, and the same crossings as fractional indices
    (linear interpolation) for lap periods.
    Returns (boundaries, fractional boundaries, direction, total revolutions)."""
    d = np.asarray(c, float) - np.asarray(target, float)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    prog = th - th[0]
    sgn = 1.0 if prog[-1] >= 0 else -1.0
    prog = sgn * prog
    n_laps = int(np.floor(np.nanmax(prog) / (2 * np.pi) + 1e-9))
    b, bf = [0], [0.0]
    for m in range(1, n_laps + 1):
        goal = 2 * np.pi * m
        i = int(np.argmax(prog >= goal - 1e-9))
        b.append(i)
        bf.append(i - 1 + (goal - prog[i - 1]) / (prog[i] - prog[i - 1]) if i > 0 else 0.0)
    return np.array(b), np.array(bf), sgn, np.nanmax(prog) / (2 * np.pi)


def _radial_stats(r, r_est, rd):
    e = r - rd
    return dict(bias=np.mean(e), sd_within=np.std(e), mae=np.mean(np.abs(e)),
                rmse=np.sqrt(np.mean(e ** 2)), mean_r=np.mean(r),
                ctrl_bias=np.nanmean(r_est - rd) if r_est is not None else np.nan)


def orbit_run(c, phat, rd, target=(0.0, 0.0), dt=DT, transient=TRANSIENT_STEPS,
              vcmd=None, robots=None, robot_vel_speed=None):
    """Scalars for one orbit run, plus a list of per-lap dicts.
    c: (T+1, 2); phat: (m, 2), aligned so phat[j] is the estimate at c[j]."""
    tgt = np.asarray(target, float)
    c = np.asarray(c, float)
    ph = np.asarray(phat, float)
    m = min(len(c), len(ph))
    r = _norm(c - tgt)
    pe, n_out = _plausible_est_err(ph[:m], tgt)
    r_est = np.full(len(c), np.nan)
    r_est[:m] = np.where(np.isfinite(pe), _norm(c[:m] - ph[:m]), np.nan)
    out = dict(rd=rd, n_samples=len(c), max_r=np.nanmax(r), left_ws=left_workspace(c), n_est_outside=n_out)
    for tag, sl in (("w1", slice(transient + 1, None)), ("w2", slice(1, None))):
        for k, v in _radial_stats(r[sl], r_est[sl], rd).items():
            out[f"{k}_{tag}"] = v
    w1 = slice(transient + 1, None)
    step = _norm(np.diff(c, axis=0))
    out["speed_w1"] = np.nanmean(step[transient:]) / dt
    out["est_err_w1"] = np.nanmean(pe[transient + 1:]) if m > transient + 1 else np.nan
    b, bf, sgn, revs = segment_laps(c[w1], tgt)
    out["revs_w1"] = revs
    out["revs_w2"] = segment_laps(c[1:], tgt)[3]
    out["direction"] = sgn
    laps = []
    off = transient + 1
    for i in range(len(b) - 1):
        lo, hi = off + b[i], off + b[i + 1]
        s = _radial_stats(r[lo:hi], r_est[lo:hi], rd)
        s.update(lap=i + 1, period=(bf[i + 1] - bf[i]) * dt, E_real=np.nansum(step[lo:hi] ** 2) / dt,
                 L_path=np.nansum(step[lo:hi]))
        laps.append(s)
    out["n_laps"] = len(laps)
    if laps:
        mr = np.array([L["mean_r"] for L in laps])
        out["lap_mean_r_mean"] = mr.mean()
        out["lap_mean_r_sd"] = mr.std(ddof=1) if len(laps) > 1 else np.nan
        out["lap_rmse_mean"] = np.mean([L["rmse"] for L in laps])
        out["lap_period_mean"] = np.mean([L["period"] for L in laps])
        out["lap_period_sd"] = np.std([L["period"] for L in laps], ddof=1) if len(laps) > 1 else np.nan
        out["lap_within_sd"] = np.mean([L["sd_within"] for L in laps])
        out["E_real_per_orbit"] = np.mean([L["E_real"] for L in laps])
    if vcmd is not None:
        s = _norm(np.asarray(vcmd, float))[transient + 1:]
        out["E_cmd_w1"] = np.nansum(s ** 2) * dt
    if robots is not None:
        robot_vel_speed = _norm(np.diff(np.asarray(robots, float), axis=0)) / dt
    if robot_vel_speed is not None:
        s = np.asarray(robot_vel_speed, float)[transient:]
        out["E_robot_real_w1"] = np.nanmean(np.nansum(s ** 2, 0)) * dt
    return out, laps


TYPE_NAMES = {0: "invalid", 1: "saddle", 2: "sink", 3: "source", 4: "stable spiral",
              5: "unstable spiral", 6: "center"}
COARSE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 4}
COARSE_NAMES = {0: "invalid", 1: "saddle", 2: "attracting", 3: "repelling", 4: "center"}
EPS_CENTER = 0.2


def classify(J, eps_c=EPS_CENTER):
    """Type codes (TYPE_NAMES) for Jacobians given as (..., 4) [a, b, d, e] or (..., 2, 2)."""
    J = np.asarray(J, float)
    if J.shape[-2:] == (2, 2):
        J = J.reshape(J.shape[:-2] + (4,))
    a, b, d, e = J[..., 0], J[..., 1], J[..., 2], J[..., 3]
    tr, det = a + e, a * e - b * d
    disc = tr * tr - 4 * det
    code = np.zeros(np.shape(tr), int)
    with np.errstate(invalid="ignore"):
        pos = det > 0
        band = eps_c * np.sqrt(np.where(pos, det, 0.0))
        center = pos & (np.abs(tr) <= band)
        node = pos & ~center & ((disc >= 0) | (np.sqrt(np.maximum(-disc, 0.0)) <= band))
        spiral = pos & ~center & ~node
        code[det < 0] = 1
        code[node & (tr < 0)] = 2
        code[node & (tr > 0)] = 3
        code[spiral & (tr < 0)] = 4
        code[spiral & (tr > 0)] = 5
        code[center] = 6
    code[~np.isfinite(det)] = 0
    return code


def classification_scores(J, true_code, dt=DT, start=0, eps_c=EPS_CENTER):
    """Scores for one trial. J: (m, 4) estimates at cycles 0..m-1. Accuracies use
    valid cycles at or after `start`; times are measured from cycle 0."""
    code = classify(J, eps_c)
    coarse = np.vectorize(COARSE.get)(code) if len(code) else code
    ok = code[start:] > 0
    strict = code[start:][ok] == true_code
    crs = coarse[start:][ok] == COARSE[true_code]
    right = code == true_code
    wrong = np.flatnonzero(~right)
    return dict(cls_valid_frac=ok.mean() if len(ok) else np.nan,
                cls_acc=strict.mean() if strict.size else np.nan,
                cls_acc_coarse=crs.mean() if crs.size else np.nan,
                cls_t_first=np.argmax(right) * dt if right.any() else np.nan,
                cls_t_stable=(0.0 if wrong.size == 0 else (wrong[-1] + 1) * dt) if right[-1] else np.nan,
                cls_final=int(code[-1]))


def orbit_r_task_formula(rd, kt=0.2, kr=0.5, tau=0.28, dt=DT):
    """The small-angle lag formula as stated in the revision request."""
    rd = np.asarray(rd, float)
    return (rd + np.sqrt(rd ** 2 + 4 * (kt ** 2 / kr) * (tau + dt / 2))) / 2


def summarize_attraction(rows, eps_list=EPS_CONV):
    """Across-trial statistics for a list of attraction_trial dicts."""
    R = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in rows[0]
         if isinstance(rows[0][k], (bool, int, float, np.number, np.bool_))}
    e = R["e_final"]
    fx, fy = R["final_x"], R["final_y"]
    mx, my = np.mean(fx), np.mean(fy)
    S = dict(n=len(rows), rest_x=mx, rest_y=my, bias=np.hypot(mx, my),
             scatter=np.sqrt(np.mean((fx - mx) ** 2 + (fy - my) ** 2)),
             e_mean=e.mean(), e_sd=e.std(ddof=1), e_med=np.median(e),
             e_q1=np.percentile(e, 25), e_q3=np.percentile(e, 75), e_p95=np.percentile(e, 95),
             e_max=e.max(), est_final_mean=np.nanmean(R["est_final"]),
             est_ss_mean=np.nanmean(R["est_ss"]), est_ss_med=np.nanmedian(R["est_ss"]),
             left_ws=int(np.sum(R["left_ws"])), L_path_mean=np.mean(R["L_path"]),
             E_real_mean=np.mean(R["E_real"]), path_eff_med=np.nanmedian(R["path_eff"]))
    for k in ("L_cmd", "E_cmd", "L_robot_real", "E_robot_real", "L_robot_cmd", "E_robot_cmd"):
        if k in R:
            S[k + "_mean"] = np.nanmean(R[k])
    for eps in eps_list:
        tag = f"{round(eps * 100):d}cm"
        fail = R[f"fail_{tag}"].astype(bool)
        S[f"success_{tag}"] = 100.0 * (1 - fail.mean())
        S[f"n_fail_{tag}"] = int(fail.sum())
        S[f"t_hit_med_{tag}"] = np.nanmedian(R[f"t_hit_{tag}"])
        S[f"t_settle_med_{tag}"] = np.nanmedian(R[f"t_settle_{tag}"])
        S[f"t_settle_p95_{tag}"] = np.nanpercentile(R[f"t_settle_{tag}"], 95) \
            if np.isfinite(R[f"t_settle_{tag}"]).any() else np.nan
    return S
