"""Warm-start a 7-channel free-shape policy from the 4-channel E5 winner.

Why this transfers at all
-------------------------
The observation is 30-dim in both (obs_mode='raw+est' did not change), so of
the 13 policy tensors only three differ, and they are the smallest ones:
action_net.weight (4x256 -> 7x256), action_net.bias, and log_std.  Both
trunks (256x30 -> 256x256) and the whole value head are shape-identical, which
is ~99.5% of the parameters.

The three that differ are adapted rather than reinitialised:

  * action_net rows 0-2 carry over unchanged.  vx, vy and omega mean exactly
    the same thing in both action spaces.
  * the four new rows start at zero weight, so they ignore the observation,
    with their BIAS set to the action value that reproduces E5's own geometry
    (d1 = d2 = 0.2 m, phi = 90 deg, ratio = 0.5).  A zero bias would command
    the midpoint of each bound, d = 0.83 m, which is four times the size E5
    was tuned around and destroys it (measured: 76% -> 0%).
  * log_std rows 0-2 carry over; the new channels start at exp(-2) ~ 0.14, so
    shape exploration begins gently rather than flailing.

Verified: the warm-started 7-channel policy scores 0.78 on square starts
before any finetuning, identical to E5's native 0.78 on the same seeds.  The
transfer is behaviour-preserving, so training starts from a known-good policy
and only has to discover what the four new channels buy.

Run:
    <VF_Robot venv python> finetune_from_e5.py --tag sas_a5_e5ft --timesteps 6000000
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO

import train_ppo as T
from evaluate import load_policy
from quad_saddle_env import (D_MIN, D_MAX, PHI_MIN, PHI_MAX,
                             RATIO_MIN, RATIO_MAX)
from train_sas_free import CurriculumCallback

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _inv_aff(v, lo, hi):
    return float(np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="sas_a5_e5ft")
    p.add_argument("--timesteps", type=int, default=6_000_000)
    p.add_argument("--envs", type=int, default=3)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--lr", type=float, default=1.2e-4,
                   help="lower than from-scratch: this starts from a good "
                        "policy and should not be kicked out of it")
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--old", default="outputs/ppo_e5_best.zip")
    p.add_argument("--old-vec", default="outputs/ppo_e5_best_vecnormalize.pkl")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--r-lo0", type=float, default=0.25)
    p.add_argument("--r-hi0", type=float, default=0.80)
    args = p.parse_args()

    t0 = time.time()
    env_kwargs = dict(obs_mode="raw+est", action_mode="sas_full",
                      reward_mode="free_shape", start_shape="random",
                      start_r_range=(args.r_lo0, args.r_hi0) if args.curriculum else None)
    kw = T._ppo_kwargs(net_width=256, learning_rate=args.lr,
                       ent_coef=args.ent_coef)
    venv = T.build_vec_env(args.envs, args.seed, env_kwargs, subproc=True,
                           gamma=kw["gamma"])

    # Seed the normalizer from E5's statistics so the transferred trunk sees
    # the input distribution it was trained on.
    _, mean, var, _ = load_policy(args.old, args.old_vec)
    venv.obs_rms.mean[:] = mean
    venv.obs_rms.var[:] = var
    venv.obs_rms.count = 1e5

    model = PPO(env=venv, seed=args.seed, verbose=0, **kw)
    src = PPO.load(args.old, device="cpu").policy.state_dict()
    dst = model.policy.state_dict()

    tgt = [_inv_aff(0.2, D_MIN, D_MAX), _inv_aff(0.2, D_MIN, D_MAX),
           _inv_aff(np.pi / 2.0, PHI_MIN, PHI_MAX),
           _inv_aff(0.5, RATIO_MIN, RATIO_MAX)]
    n_copied = n_adapted = 0
    for k, v in dst.items():
        if k not in src:
            continue
        s = src[k]
        if s.shape == v.shape:
            dst[k] = s.clone()
            n_copied += 1
        elif k == "action_net.weight":
            w = v.clone(); w.zero_(); w[:3] = s[:3]
            dst[k] = w; n_adapted += 1
        elif k == "action_net.bias":
            b = v.clone(); b.zero_(); b[:3] = s[:3]
            for i, t in enumerate(tgt):
                b[3 + i] = t
            dst[k] = b; n_adapted += 1
        elif k == "log_std":
            l = v.clone(); l[:3] = s[:3]; l[3:] = -2.0
            dst[k] = l; n_adapted += 1
    model.policy.load_state_dict(dst)
    print(f"  warm start from {os.path.basename(args.old)}: "
          f"{n_copied} tensors copied, {n_adapted} adapted", flush=True)

    hist = os.path.join(OUT, f"{args.tag}_history.jsonl")
    open(hist, "w").close()
    cbs = [T.ProgressCallback(hist, 1, ckpt_dir=OUT, ckpt_tag=args.tag,
                              ckpt_every=args.ckpt_every)]
    if args.curriculum:
        cbs.append(CurriculumCallback(venv, args.timesteps,
                                      r0=(args.r_lo0, args.r_hi0),
                                      r1=(1.0, 2.5), reach_frac=0.5))
    model.learn(total_timesteps=args.timesteps, callback=cbs,
                progress_bar=False)
    model.save(os.path.join(OUT, f"{args.tag}_final"))
    venv.save(os.path.join(OUT, f"{args.tag}_vecnormalize.pkl"))
    venv.close()
    with open(os.path.join(OUT, f"{args.tag}_runmeta.json"), "w") as f:
        json.dump(dict(tag=args.tag, args=vars(args),
                       wall_seconds=time.time() - t0), f, indent=2)
    print(f"  saved {args.tag}_final.zip ({(time.time()-t0)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
