"""
Weight interpolation between the behaviour-cloned policy and its fine-tuned
descendant ("model soup" / WiSE-FT).

Why this is the cheap fix for the easy-family regression.  Measured on 1000
held-out fields, the two endpoints fail in opposite directions:

    family              BC clone   E5 (fine-tuned)
    quadratic              98.5%       94.0%
    cubic_perturbed        95.2%       94.4%
    log_sum_exp            39.2%       69.2%
    overall                66.4%       68.7%

Fine-tuning bought +30 points on log_sum_exp and gave back 4-6 on the families
the clone already had. Because the fine-tuned weights are a short optimisation
path away from the clone, the two sit in the same loss basin, and the linear
interpolant between them is usually a valid model rather than nonsense. That is
the WiSE-FT observation, and it costs no training: just average the weights and
evaluate.

    theta(a) = (1 - a) * theta_BC + a * theta_finetuned

a = 0 recovers the clone, a = 1 the fine-tune. If the regression is a mild pull
away from the clone rather than a genuine tradeoff, some intermediate a keeps
most of the log_sum_exp gain while restoring quadratic and cubic_perturbed.

Both checkpoints must share an architecture, which they do by construction:
`bc_pretrain.bc_warm_start` saves `{tag}_bconly` from the same model object it
then fine-tunes.
"""
import argparse
import copy
import os

import numpy as np
import torch

from stable_baselines3 import PPO

from baselines import run_episode, summarize
from evaluate import policy_controller, load_policy
from quad_saddle_env import QuadSaddleEnv
import saddle_fields as sf

OUT_DIR = "outputs"


def interpolate(path_a, path_b, alpha):
    """Return a PPO model whose policy weights are (1-a)*A + a*B."""
    ma = PPO.load(path_a, device="cpu")
    mb = PPO.load(path_b, device="cpu")
    sa, sb = ma.policy.state_dict(), mb.policy.state_dict()
    if set(sa) != set(sb):
        raise ValueError("checkpoints have different architectures")
    merged = {}
    for k in sa:
        if sa[k].shape != sb[k].shape:
            raise ValueError(f"shape mismatch on {k}: {sa[k].shape} vs {sb[k].shape}")
        if sa[k].is_floating_point():
            merged[k] = (1.0 - alpha) * sa[k] + alpha * sb[k]
        else:
            merged[k] = sb[k]
    out = copy.deepcopy(mb)
    out.policy.load_state_dict(merged)
    out.policy.eval()
    return out


def score(model, mean, var, clip, n=200, seed0=900_000, obs_mode="raw+est"):
    ctl = policy_controller(model, mean, var, clip)
    env = QuadSaddleEnv(obs_mode=obs_mode, reward_mode="metric", seed=seed0)
    rows = [run_episode(env, ctl, seed0 + i) for i in range(n)]
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc", default=os.path.join(OUT_DIR, "ppo_e5_bconly.zip"))
    p.add_argument("--ft", default=os.path.join(OUT_DIR, "ppo_e5_best.zip"))
    p.add_argument("--vec", default=os.path.join(OUT_DIR,
                                                 "ppo_e5_best_vecnormalize.pkl"))
    p.add_argument("--alphas", type=float, nargs="*",
                   default=[0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    p.add_argument("--n", type=int, default=200,
                   help="validation fields; seeds 900k+, disjoint from eval")
    p.add_argument("--save-best", action="store_true")
    args = p.parse_args()

    # The observation normalizer must be interpolated ALONGSIDE the weights.
    # A first version held it fixed at the fine-tuned run's statistics on the
    # assumption they had barely drifted from the clone's.  They had: scoring
    # the cloned weights under the fine-tuned normalizer gave 28.0% against the
    # 66.4% that same clone scores under its own, so every low-alpha point was
    # measuring a normalizer mismatch rather than the weights. Interpolating
    # both keeps each point a coherent policy.
    _, mean_b, var_b, clip = load_policy(args.ft, args.vec)
    bc_vec = args.bc.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(bc_vec):
        _, mean_a, var_a, _ = load_policy(args.bc, bc_vec)
    else:
        mean_a, var_a = mean_b, var_b

    print("=" * 92)
    print(f"WEIGHT INTERPOLATION  {os.path.basename(args.bc)} -> "
          f"{os.path.basename(args.ft)}")
    print(f"  {args.n} validation fields (seeds 900000+), deterministic policy")
    print("=" * 92)
    hard = ["log_sum_exp", "streamfunction_quad", "gaussian_pair"]
    easy = ["quadratic", "cubic_perturbed", "rational_envelope"]
    print(f"  {'alpha':>6s}{'success':>9s}{'in_tol':>8s}{'e_final':>9s}"
          f"{'easy 3':>9s}{'hard 3':>9s}")
    print("  " + "-" * 52)

    best, results = None, []
    for a in args.alphas:
        model = interpolate(args.bc, args.ft, a)
        mean = (1.0 - a) * mean_a + a * mean_b
        var = (1.0 - a) * var_a + a * var_b
        rows = score(model, mean, var, clip, n=args.n)
        s = summarize(rows)
        fam = np.array([r["family"] for r in rows])
        ok = np.array([r["success"] for r in rows], float)
        e_m = np.mean([ok[fam == f].mean() for f in easy if (fam == f).any()])
        h_m = np.mean([ok[fam == f].mean() for f in hard if (fam == f).any()])
        results.append((a, s, e_m, h_m))
        print(f"  {a:6.2f}{s['success_rate']:9.1%}{s['time_in_tol']:8.1%}"
              f"{s['e_final_median']:9.3f}{e_m:9.1%}{h_m:9.1%}")
        if best is None or s["success_rate"] > best[1]["success_rate"]:
            best = (a, s, model)

    print()
    print(f"  best alpha = {best[0]:.2f} at {best[1]['success_rate']:.1%}")
    print("  (alpha=0 is the clone, alpha=1 the fine-tune; an interior optimum")
    print("   means the regression was a pull away from the clone, not a tradeoff)")

    if args.save_best:
        out = os.path.join(OUT_DIR, "ppo_soup_best")
        best[2].save(out)
        import shutil
        shutil.copy(args.vec, out + "_vecnormalize.pkl")
        print(f"\n  saved {out}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
