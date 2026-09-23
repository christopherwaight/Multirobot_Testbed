"""Optimal fixed formation radius as a function of reading noise.

The noise sweep showed that the analytic law at r0 = 0.35 beats every learned
policy under noise while the default r0 = 0.10 collapses (0.63 -> 0.17 from
sigma_z 0 to 0.02).  Size, not learning, is the dominant variable.  This maps
the full surface so the right radius can be READ OFF for a given sensor noise
level rather than guessed, which is what a hardware run needs.

Two competing effects set the optimum: a larger stencil resists reading noise
(better SNR on the finite differences), while a smaller one has less
truncation error (the field is only locally quadratic).
"""
import argparse, json, os
import numpy as np
from baselines import rotating_hessian, run_episode
from evaluate_sas import padded_controller
from quad_saddle_env import QuadSaddleEnv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--radii", type=float, nargs="*",
                   default=[0.10, 0.20, 0.30, 0.40, 0.55, 0.75])
    p.add_argument("--sigmas", type=float, nargs="*",
                   default=[0.0, 0.01, 0.02, 0.04])
    p.add_argument("--out", default="sweep_r0_noise.json")
    args = p.parse_args()

    print(f"n={args.n_eval}/cell, identical seeds. Analytic law, fixed r0.\n")
    print("  r0     " + "".join(f"s={s:<7}" for s in args.sigmas))
    res = {}
    for r0 in args.radii:
        ctl = padded_controller(
            rotating_hessian(mode="single", k_rot=0.5, k_trans=1.5, r0=r0,
                             max_omega=2.0), 7, r0=r0)
        line = f"  {r0:<6.2f} "
        row = {}
        for s in args.sigmas:
            env = QuadSaddleEnv(obs_mode="raw+est", action_mode="sas_full",
                                reward_mode="free_shape", start_shape="random",
                                start_r_range=(1.0, 2.5), sigma_z=s)
            v = float(np.mean([run_episode(env, ctl, 500_000 + i)["success"]
                               for i in range(args.n_eval)]))
            env.close()
            row[str(s)] = v
            line += f"{v:.2f}     "
        res[str(r0)] = row
        print(line, flush=True)

    print("\n  best r0 per noise level:")
    for s in args.sigmas:
        best = max(res, key=lambda r: res[r][str(s)])
        print(f"    sigma_z={s:<6} -> r0={best}  ({res[best][str(s)]:.2f})")
    with open(os.path.join(OUT, args.out), "w") as f:
        json.dump(dict(n_eval=args.n_eval, radii=args.radii,
                       sigmas=args.sigmas, results=res), f, indent=2)
    print(f"\n  wrote {OUT}/{args.out}")


if __name__ == "__main__":
    main()
