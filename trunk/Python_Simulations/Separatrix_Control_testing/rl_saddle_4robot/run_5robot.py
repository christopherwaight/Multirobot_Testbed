"""Warm-start the 33-dim (5-robot) policy from the 30-dim (4-robot) winner.

The extra three observation channels are appended, so every weight except the
first layer transfers unchanged; the first layer's new input columns are
zero-initialised, which makes the warm-started policy exactly equivalent to its
4-robot parent at step 0 and lets training discover what the new channels buy.
"""
import numpy as np, torch, os, shutil
from stable_baselines3 import PPO
from evaluate import load_policy
import train_ppo as T

def main():
    OLD = "outputs/ppo_e5_best.zip"
    OLD_VEC = "outputs/ppo_e5_best_vecnormalize.pkl"
    TAG = "ppo_5r"

    kw = T._ppo_kwargs(learning_rate=1.2e-4, net_width=256)
    venv = T.build_vec_env(12, 0, dict(obs_mode="raw+est+c", reward_mode="metric"),
                           subproc=True, gamma=kw["gamma"])
    _, mean, var, _ = load_policy(OLD, OLD_VEC)
    n_old = len(mean)
    venv.obs_rms.mean[:n_old] = mean
    venv.obs_rms.var[:n_old] = var
    venv.obs_rms.count = 1e5

    model = PPO(env=venv, seed=0, verbose=0, **kw)
    src = PPO.load(OLD, device="cpu").policy.state_dict()
    dst = model.policy.state_dict()
    n_copied = n_padded = 0
    for k, v in dst.items():
        if k not in src:
            continue
        s = src[k]
        if s.shape == v.shape:
            dst[k] = s.clone(); n_copied += 1
        elif s.dim() == 2 and v.dim() == 2 and s.shape[0] == v.shape[0] \
                and s.shape[1] < v.shape[1]:
            w = v.clone(); w.zero_(); w[:, :s.shape[1]] = s      # zero-pad new inputs
            dst[k] = w; n_padded += 1
    model.policy.load_state_dict(dst)
    print(f"  warm start: {n_copied} tensors copied, {n_padded} zero-padded", flush=True)

    hist = f"outputs/{TAG}_history.jsonl"; open(hist, "w").close()
    model.learn(total_timesteps=2_200_000,
                callback=T.ProgressCallback(hist, 1, ckpt_dir="outputs",
                                            ckpt_tag=TAG, ckpt_every=550_000,
                                            log_every=1000),
                progress_bar=False)
    model.save(f"outputs/{TAG}_final"); venv.save(f"outputs/{TAG}_vecnormalize.pkl")
    venv.close()
    print("TRAINDONE", flush=True)

    from train_ppo import pick_best_checkpoint
    pick_best_checkpoint(TAG, n_eval=150, seed0=900000,
                         env_kwargs=dict(obs_mode="raw+est+c", reward_mode="metric"))
    print("ALLDONE", flush=True)

if __name__ == "__main__":
    main()
