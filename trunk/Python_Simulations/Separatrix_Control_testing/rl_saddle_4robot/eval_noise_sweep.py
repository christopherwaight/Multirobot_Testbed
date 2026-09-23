"""Score controllers across reading-noise levels on identical episodes.

The overnight portfolio trained and evaluated at sigma_z = 0, where shrinking
the formation is free: the estimator is a finite-difference stencil, so a
small stencil costs nothing without noise.  That is why every earlier policy
pinned its size channel to the floor.  `sas_a6_noisy` was trained at
sigma_z = 0.01 specifically to test whether a policy discovers that a larger
stencil beats noise, but the final evaluation scored it at sigma_z = 0 like
everything else, i.e. in the one regime where its training signal is absent.

This sweep closes that gap.  Every controller sees the same fields and the
same start poses at each noise level, and the commanded formation size is
logged alongside success, so "did it grow the stencil" is answered directly
rather than inferred.

Prediction from the forced-size sweep (analytic law, fixed r0, distance-only
gate): success at r0 = 0.06 falls 0.74 -> 0.03 from sigma_z 0 to 0.02, while
r0 = 0.35 holds 0.68 -> 0.72.  A controller that adapts its size should
therefore degrade far more gracefully than one that does not.
"""
import argparse
import json
import os

import numpy as np

from baselines import rotating_hessian, run_episode
from evaluate import load_policy
from evaluate_sas import padded_controller, policy_controller_n
from quad_saddle_env import QuadSaddleEnv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def ring_radii(env):
    xy = np.asarray(env.episode_log["robots"])
    c = xy.mean(axis=1, keepdims=True)
    return np.linalg.norm(xy - c, axis=2).mean(axis=1)


def score(ctl, n, sigma_z, action_mode, start_shape, seed0=500_000):
    env = QuadSaddleEnv(obs_mode="raw+est", action_mode=action_mode,
                        reward_mode="free_shape", start_shape=start_shape,
                        start_r_range=(1.0, 2.5), sigma_z=sigma_z)
    succ, tol, R_med, R_fin = [], [], [], []
    for i in range(n):
        r = run_episode(env, ctl, seed0 + i)
        succ.append(r["success"])
        tol.append(r["time_in_tol"])
        R = ring_radii(env)
        R_med.append(float(np.median(R)))
        R_fin.append(float(R[-1]))
    env.close()
    return dict(success_rate=float(np.mean(succ)),
                time_in_tol=float(np.mean(tol)),
                R_median=float(np.median(R_med)),
                R_final=float(np.median(R_fin)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=150)
    p.add_argument("--sigmas", type=float, nargs="*",
                   default=[0.0, 0.005, 0.01, 0.02])
    p.add_argument("--out", default="eval_noise_sweep.json")
    args = p.parse_args()

    controllers = []
    base = rotating_hessian(mode="single", k_rot=0.5, k_trans=1.5, r0=0.1,
                            max_omega=2.0)
    controllers.append(("analytic (r0=0.10, fixed)",
                        padded_controller(base, 7), "sas_full", "random"))
    base_big = rotating_hessian(mode="single", k_rot=0.5, k_trans=1.5, r0=0.35,
                                max_omega=2.0)
    controllers.append(("analytic (r0=0.35, fixed)",
                        padded_controller(base_big, 7, r0=0.35),
                        "sas_full", "random"))
    for tag in ["sas_a5_e5ft_best", "sas_a6_noisy_best"]:
        mp_ = os.path.join(OUT, f"{tag}.zip")
        vp = os.path.join(OUT, f"{tag}_vecnormalize.pkl")
        if not os.path.exists(mp_):
            print(f"  [skip] {tag}")
            continue
        m, mean, var, clip = load_policy(mp_, vp)
        controllers.append((tag, policy_controller_n(m, mean, var, clip, 7),
                            "sas_full", "random"))

    results = {}
    print(f"n={args.n_eval} per cell, identical seeds across every cell\n")
    hdr = "  " + "controller".ljust(30) + "".join(
        f"s={s:<7}" for s in args.sigmas)
    print(hdr)
    for label, ctl, am, ss in controllers:
        cells = {}
        line = "  " + label.ljust(30)
        for s in args.sigmas:
            r = score(ctl, args.n_eval, s, am, ss)
            cells[str(s)] = r
            line += f"{r['success_rate']:.2f}     "
        results[label] = cells
        print(line, flush=True)

    print("\n  median formation radius R (does it grow the stencil?)")
    print(hdr)
    for label in results:
        line = "  " + label.ljust(30)
        for s in args.sigmas:
            line += f"{results[label][str(s)]['R_median']:.3f}    "
        print(line)

    with open(os.path.join(OUT, args.out), "w") as f:
        json.dump(dict(n_eval=args.n_eval, sigmas=args.sigmas,
                       results=results), f, indent=2)
    print(f"\n  wrote {OUT}/{args.out}")


if __name__ == "__main__":
    main()
