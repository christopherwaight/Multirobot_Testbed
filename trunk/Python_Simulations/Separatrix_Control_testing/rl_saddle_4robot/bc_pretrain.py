"""
Behaviour-cloning warm start from the tuned analytic controller.

Why this exists.  The gain-swept rotating-Hessian law is a strong solution that
lives inside the policy class PPO is already searching, so making PPO rediscover
it from scratch spends most of the training budget re-deriving something we
already have in closed form.  Cloning it first turns the question from "can PPO
find this" into "how far past it can PPO get", which is the question actually
worth answering.

What this does and does not change.  It sets the policy network's INITIAL
weights, nothing else.  The observation, the action space, the reward, and the
environment are untouched, and PPO is free to diverge from the analytic law from
the first gradient step onward.  It is a starting point, not a constraint.

Two details that matter for correctness:

  * The VecNormalize observation statistics are fitted on the cloning data and
    handed to the training environment.  Skipping this trains the policy against
    one normalizer and then runs it under a different one, which silently
    destroys the cloned behaviour on the first PPO rollout.

  * When the policy squashes its output (squash_output=True, the default in
    train_ppo.py), the network's action mean passes through a tanh.  Regressing
    on the raw target would ask the pre-tanh activation to reach +/-1, i.e.
    infinity, so targets are clipped to +/-TANH_CLIP first.
"""
import argparse
import os

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from quad_saddle_env import QuadSaddleEnv
from baselines import rotating_hessian
import train_ppo as T

OUT_DIR = "outputs"
TANH_CLIP = 0.999


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

def collect(n_episodes=300, seed0=700_000, env_kwargs=None, gains=None,
            max_steps=None, verbose=True):
    """Roll out the analytic controller, recording (observation, action).

    Seeds start at 700_000, disjoint from both the training range and
    evaluate.py's EVAL_SEED0 = 500_000, so cloning never sees a held-out field.
    """
    env_kwargs = dict(env_kwargs or {})
    gains = gains or dict(mode="single", k_rot=0.5, k_trans=1.5, r0=0.1)
    ctl = rotating_hessian(max_omega=min(2.0, 0.3 / gains.get("r0", 0.1)), **gains)

    env = QuadSaddleEnv(**env_kwargs)
    obs_buf, act_buf = [], []
    for i in range(n_episodes):
        env.reset(seed=seed0 + i)
        n = 0
        while True:
            o = env.current_obs()
            a = ctl(env)
            obs_buf.append(np.asarray(o, dtype=np.float32))
            act_buf.append(np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0))
            _, _, term, trunc, _ = env.step(a)
            n += 1
            if term or trunc or (max_steps and n >= max_steps):
                break
        if verbose and (i + 1) % 50 == 0:
            print(f"    collected {i+1}/{n_episodes} episodes, "
                  f"{len(obs_buf):,} transitions")
    env.close()
    return np.asarray(obs_buf), np.asarray(act_buf)


# --------------------------------------------------------------------------
# Supervised fit of the policy mean
# --------------------------------------------------------------------------

def clone(model, obs, act, epochs=30, batch_size=512, lr=1e-3, verbose=True):
    """Regress the policy's deterministic action mean onto the demonstrations.

    Operates on the same normalized observations the policy will see at run
    time, so `obs` must already be normalized.
    """
    policy = model.policy
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    dev = policy.device

    X = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.clip(act, -TANH_CLIP, TANH_CLIP),
                        dtype=torch.float32, device=dev)
    n = len(X)

    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for k in range(0, n, batch_size):
            idx = perm[k:k + batch_size]
            xb, yb = X[idx], Y[idx]

            feats = policy.extract_features(xb)
            if policy.share_features_extractor:
                latent_pi = policy.mlp_extractor.forward_actor(feats)
            else:
                latent_pi = policy.mlp_extractor.forward_actor(feats[0])
            mean_actions = policy.action_net(latent_pi)
            if policy.squash_output:
                mean_actions = torch.tanh(mean_actions)

            loss = torch.nn.functional.mse_loss(mean_actions, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"    epoch {ep+1:3d}/{epochs}  mse {tot/n:.5f}")
    policy.eval()
    return model


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def bc_warm_start(tag="ppo_bc", n_episodes=300, epochs=30, timesteps=5_000_000,
                  n_envs=12, seed=0, env_kwargs=None, ppo_kwargs=None,
                  ckpt_every=1_000_000):
    os.makedirs(OUT_DIR, exist_ok=True)
    env_kwargs = dict(env_kwargs or {})
    ppo_kwargs = ppo_kwargs or T._ppo_kwargs()

    print("=" * 78)
    print(f"BEHAVIOUR CLONING WARM START  tag={tag}")
    print("=" * 78)

    print(f"\n  1. collecting {n_episodes} analytic-controller episodes")
    obs, act = collect(n_episodes, env_kwargs=env_kwargs)
    print(f"     {len(obs):,} transitions, obs dim {obs.shape[1]}")
    print(f"     action mean {act.mean(0).round(3)}  std {act.std(0).round(3)}")

    print("\n  2. building the vectorized env and fitting obs statistics")
    venv = T.build_vec_env(n_envs, seed, env_kwargs, subproc=True,
                           gamma=ppo_kwargs["gamma"])
    # Fit the normalizer on the demonstration distribution, then freeze those
    # statistics into the training env so the cloned policy and PPO agree.
    venv.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
    venv.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    obs_n = np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + 1e-8),
                    -venv.clip_obs, venv.clip_obs).astype(np.float32)

    print("\n  3. cloning the policy mean")
    model = PPO(env=venv, seed=seed, verbose=0, **ppo_kwargs)
    clone(model, obs_n, act, epochs=epochs)

    bc_path = os.path.join(OUT_DIR, f"{tag}_bconly")
    model.save(bc_path)
    venv.save(os.path.join(OUT_DIR, f"{tag}_bconly_vecnormalize.pkl"))
    print(f"     saved {bc_path}.zip  (pre-PPO, for the ablation table)")

    print(f"\n  4. PPO fine-tuning for {timesteps:,} steps")
    hist = os.path.join(OUT_DIR, f"{tag}_history.jsonl")
    open(hist, "w").close()
    model.learn(total_timesteps=timesteps,
                callback=T.ProgressCallback(hist, 1, ckpt_dir=OUT_DIR,
                                            ckpt_tag=tag,
                                            ckpt_every=ckpt_every),
                progress_bar=False)

    model.save(os.path.join(OUT_DIR, f"{tag}_final"))
    venv.save(os.path.join(OUT_DIR, f"{tag}_vecnormalize.pkl"))
    venv.close()
    print(f"\n  saved {OUT_DIR}/{tag}_final.zip")
    return os.path.join(OUT_DIR, f"{tag}_final.zip")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="ppo_bc")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-mode", choices=["raw", "raw+est"], default="raw+est")
    p.add_argument("--reward-mode", choices=["shaped", "pure", "metric"],
                   default="metric")
    p.add_argument("--bc-only", action="store_true",
                   help="clone and stop, skip PPO fine-tuning")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="PPO learning rate. Lower values preserve more of the "
                        "cloned behaviour during fine-tuning; the 5M run at "
                        "3e-4 regressed on log_sum_exp (0.032 -> 1.235) "
                        "relative to the same config trained from scratch, "
                        "i.e. fine-tuning was unlearning part of the clone.")
    p.add_argument("--net-width", type=int, default=128)
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    args = p.parse_args()

    env_kwargs = dict(obs_mode=args.obs_mode, reward_mode=args.reward_mode)
    ppo_kwargs = T._ppo_kwargs(learning_rate=args.lr, net_width=args.net_width)
    bc_warm_start(tag=args.tag, n_episodes=args.episodes, epochs=args.epochs,
                  timesteps=0 if args.bc_only else args.timesteps,
                  n_envs=args.envs, seed=args.seed, env_kwargs=env_kwargs,
                  ppo_kwargs=ppo_kwargs, ckpt_every=args.ckpt_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
