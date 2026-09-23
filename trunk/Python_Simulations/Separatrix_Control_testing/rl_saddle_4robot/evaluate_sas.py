"""Score free-shape policies and the analytic baseline on one common protocol.

Why this exists alongside evaluate.py
-------------------------------------
evaluate.py scores 4-channel policies on square starts drawn from the default
annulus.  A 'sas_full' policy emits 7 channels, and the runs it comes from use
randomized start geometry, so the two are not directly comparable.  Rather
than change evaluate.py (its numbers back the existing ablation table), this
module re-scores EVERY controller, the analytic baseline included, on whatever
distribution is passed in.  A comparison is only meaningful if the baseline is
measured on the same episodes, not quoted from a previous protocol.

The analytic teacher is 4-channel.  Embedding it in the 7-channel space needs
care.  Zero-padding looked neutral and is not: zero is the MIDPOINT of each
bound, so it commands d1 = d2 = 0.83 m, i.e. a ring radius of 0.415 m against
the 0.1 m the law was tuned for.  Measured, that drops the baseline from 76%
to 0% on identical episodes, because k_rot/k_trans and max_omega = 0.3/r0 are
all calibrated to the small formation.  The padding therefore reproduces the
teacher's own geometry (d1 = d2 = 2*r0, phi = 90 deg, ratio = 0.5), which is
the faithful embedding: same commanded shape, same size, just expressed in
the wider action space.

Also reports the ideal-Newton ceiling: the fraction of the SAME episodes whose
exact Newton flow reaches the saddle at all, with no dynamics and no
estimator.  A controller cannot be blamed for episodes that are unsolvable in
principle, and success against that ceiling is the honest score.
"""

import argparse
import json
import os

import numpy as np

import saddle_fields as sf
from baselines import rotating_hessian, run_episode
from evaluate import load_policy
from quad_saddle_env import QuadSaddleEnv

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
EVAL_SEED0 = 500_000


def _inv_aff(value, lo, hi):
    """Action channel in [-1, 1] that maps to `value` under the env's map."""
    return float(np.clip(2.0 * (value - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def padded_controller(ctl, n_act, r0=0.1):
    """Embed a 4-channel controller in an n_act-channel action space.

    The shape channels are set to the teacher's OWN geometry, a square of
    ring radius r0, rather than to zero.  Zero is the midpoint of each bound
    (d = 0.83 m), which silently hands the law a formation four times the size
    it was tuned for and takes it from 76% to 0%.
    """
    from quad_saddle_env import (D_MIN, D_MAX, PHI_MIN, PHI_MAX,
                                 RATIO_MIN, RATIO_MAX)
    pad = np.array([
        _inv_aff(2.0 * r0, D_MIN, D_MAX),          # d1
        _inv_aff(2.0 * r0, D_MIN, D_MAX),          # d2
        _inv_aff(np.pi / 2.0, PHI_MIN, PHI_MAX),   # phi = 90 deg
        _inv_aff(0.5, RATIO_MIN, RATIO_MAX),       # ratio = 0.5
    ], dtype=float)

    def wrapped(env):
        a = np.asarray(ctl(env), dtype=float).ravel()
        if len(a) >= n_act:
            return a[:n_act]
        out = np.zeros(n_act, dtype=float)
        out[:len(a)] = a[:min(len(a), n_act)]
        k = min(n_act - 4, len(pad))
        if k > 0:
            out[4:4 + k] = pad[:k]
        # a[3] is the 4-channel size command; in the 7-channel space the two
        # diagonal channels carry it instead.
        if n_act == 7:
            out[3] = pad[0]
            out[4] = pad[1]
            out[5] = pad[2]
            out[6] = pad[3]
        return out
    wrapped.label = getattr(ctl, "label", "analytic")
    return wrapped


def policy_controller_n(model, mean, var, clip, n_act):
    def controller(env):
        obs = env.current_obs()
        if mean is not None:
            obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = model.predict(obs.astype(np.float32), deterministic=True)
        a = np.asarray(a, dtype=float).ravel()
        if len(a) < n_act:
            out = np.zeros(n_act)
            out[:len(a)] = a
            return out
        return a[:n_act]
    controller.label = "PPO"
    return controller


def newton_converges(fld, start, n=300, h=0.02):
    """Does exact Newton flow from `start` reach the saddle? Topology only."""
    p = np.array(start, float)
    for _ in range(n):
        g = sf.fd_gradient(fld.phi, p[0], p[1])
        H = sf.fd_hessian(fld.phi, p[0], p[1])
        if abs(np.linalg.det(H)) < 1e-12:
            return False
        s = -np.linalg.solve(H, g)
        nn = np.linalg.norm(s)
        if nn > h:
            s = s / nn * h
        p = p + s
        if not np.all(np.isfinite(p)):
            return False
        if np.linalg.norm(p - fld.saddle) < 0.05:
            return True
        if np.linalg.norm(p - fld.saddle) > 8.0:
            return False
    return bool(np.linalg.norm(p - fld.saddle) < 0.15)


def score(controller, n_eval, env_kwargs, seed0=EVAL_SEED0, ceiling=False):
    env = QuadSaddleEnv(**env_kwargs)
    rows = []
    for i in range(n_eval):
        r = run_episode(env, controller, seed0 + i)
        if ceiling:
            c0 = env.episode_log["centroid"][0]
            r["reachable"] = newton_converges(env.fld, c0)
        rows.append(r)
    env.close()
    return rows


def summarize(rows):
    succ = np.array([r["success"] for r in rows], dtype=float)
    out = dict(
        n=len(rows),
        success_rate=float(succ.mean()),
        time_in_tol=float(np.mean([r["time_in_tol"] for r in rows])),
        e_final_median=float(np.median([r["e_final"] for r in rows])),
        e_min_median=float(np.median([r["e_min"] for r in rows])),
    )
    if "reachable" in rows[0]:
        reach = np.array([r["reachable"] for r in rows], dtype=float)
        out["reachable_frac"] = float(reach.mean())
        if reach.sum() > 0:
            out["success_given_reachable"] = float(succ[reach > 0].mean())
        if (1 - reach).sum() > 0:
            out["success_given_unreachable"] = float(succ[reach == 0].mean())
    per = {}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        per[fam] = dict(n=len(sub),
                        success_rate=float(np.mean([r["success"] for r in sub])),
                        e_final_median=float(np.median([r["e_final"] for r in sub])))
    return out, per


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=[],
                   help="tags under outputs/, e.g. sas_a1_best")
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--action-mode", default="sas_full")
    p.add_argument("--reward-mode", default="free_shape")
    p.add_argument("--start-shape", default="random")
    p.add_argument("--obs-mode", default="raw+est")
    p.add_argument("--r-lo", type=float, default=1.0)
    p.add_argument("--r-hi", type=float, default=2.5)
    p.add_argument("--out", default="eval_sas.json")
    p.add_argument("--no-ceiling", action="store_true")
    args = p.parse_args()

    env_kwargs = dict(obs_mode=args.obs_mode, action_mode=args.action_mode,
                      reward_mode=args.reward_mode,
                      start_shape=args.start_shape,
                      start_r_range=(args.r_lo, args.r_hi))
    n_act = 7 if args.action_mode == "sas_full" else 4
    results = {}

    print(f"protocol: n={args.n_eval}  action={args.action_mode}  "
          f"start={args.start_shape}  r in [{args.r_lo}, {args.r_hi}]\n")

    base = padded_controller(
        rotating_hessian(mode="single", k_rot=0.5, k_trans=1.5, r0=0.1,
                         max_omega=2.0), n_act)
    rows = score(base, args.n_eval, env_kwargs, ceiling=not args.no_ceiling)
    s, per = summarize(rows)
    results["analytic_baseline"] = dict(summary=s, per_family=per)
    print(f"  {'analytic (rot-Hessian, best gains)':<40s} "
          f"succ={s['success_rate']:.3f}  tol={s['time_in_tol']:.3f}  "
          f"e_med={s['e_final_median']:.4f}")
    if "reachable_frac" in s:
        print(f"  {'  ideal-Newton ceiling on these episodes':<40s} "
              f"{s['reachable_frac']:.3f}   "
              f"(succ|reachable {s.get('success_given_reachable', float('nan')):.3f})")

    for tag in args.models:
        mp_ = os.path.join(OUT_DIR, f"{tag}.zip")
        vp = os.path.join(OUT_DIR, f"{tag}_vecnormalize.pkl")
        if not os.path.exists(mp_):
            print(f"  [skip] {tag}: no {mp_}")
            continue
        model, mean, var, clip = load_policy(mp_, vp)
        ctl = policy_controller_n(model, mean, var, clip, n_act)
        rows = score(ctl, args.n_eval, env_kwargs, ceiling=not args.no_ceiling)
        s, per = summarize(rows)
        results[tag] = dict(summary=s, per_family=per)
        line = (f"  {tag:<40s} succ={s['success_rate']:.3f}  "
                f"tol={s['time_in_tol']:.3f}  e_med={s['e_final_median']:.4f}")
        if "success_given_reachable" in s:
            line += f"  succ|reach={s['success_given_reachable']:.3f}"
        print(line)

    with open(os.path.join(OUT_DIR, args.out), "w") as f:
        json.dump(dict(protocol=vars(args), results=results), f, indent=2)
    print(f"\n  wrote {OUT_DIR}/{args.out}")


if __name__ == "__main__":
    main()
