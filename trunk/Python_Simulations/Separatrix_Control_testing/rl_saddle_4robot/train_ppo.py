"""
PPO driver for the 4-robot saddle-seeking environment.

Uses stable-baselines3.  The environment does the interesting work; this file
is mostly plumbing.

History, because it changed the defaults below.  A first 3M-step run used a
3-stage curriculum (quadratic only, then three families, then all eight) and
plateaued at ~38% held-out success, with the SAME failure on the SAME four
field families whether the reward was the full shaped one or literally
reward=-e alone (see README).  That rules the reward out as the cause.  It
does not rule out the curriculum: stage 1 trains exclusively on `quadratic`,
which has no critical point besides the saddle, so nothing there ever
penalizes walking confidently toward a gradient-zero point that isn't the
saddle.  Stages 2-3 introduce the families that DO have such points (extrema,
plateaus) on a shrinking fraction of an already-committed budget.  A policy
that never has to unlearn "chase the strongest signal" may simply never
un-learn it.

So the default here is now NO curriculum: every episode from step zero draws
from the full family mix, together with heavier, temporally-correlated
exploration (gSDE rather than i.i.d. per-step Gaussian noise) on the theory
that escaping a false attractor needs a sustained multi-step excursion, not
one-step jitter, plus a bigger network and a much longer budget.  Pass
--curriculum to run the old 3-stage schedule for comparison.

The algorithm, in the register used elsewhere in this repo:

    Initialize:
        policy pi(a | s; w)        2 hidden layers of 128, tanh
        value  V(s; v)             same shape
        action distribution        diagonal Gaussian, gSDE (state-dependent
                                    exploration, resampled every sde_freq
                                    steps rather than every step)
        observation normalizer     running mean and variance
        N_env = 12 parallel environments
        T_roll = 1024 steps per environment per update
        gamma = 0.99, lambda = 0.95, clip = 0.2

    Loop for each update:
        Loop for each environment and each step t = 1..T_roll:
            sample a_t ~ pi(. | s_t)      (gSDE: same noise sample reused
                                            across sde_freq consecutive steps)
            step the environment, observe r_t, s_{t+1}, done
            store (s_t, a_t, r_t, V(s_t), log pi(a_t | s_t))

        Compute advantages backward over the rollout by GAE(lambda):
            delta_t  = r_t + gamma V(s_{t+1}) (1 - done) - V(s_t)
            A_t      = delta_t + gamma lambda (1 - done) A_{t+1}
            Return_t = A_t + V(s_t)
        Normalize A over the batch.

        Loop for each of 10 epochs, for each minibatch of 512:
            ratio   = exp(log pi_new(a) - log pi_old(a))
            L_clip  = -min(ratio A, clip(ratio, 1-eps, 1+eps) A)
            L_value = (V(s) - Return)^2
            L_ent   = -entropy(pi)
            step Adam on L_clip + 0.5 L_value + ent_coef L_ent
"""
import argparse
import json
import os
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)

from quad_saddle_env import QuadSaddleEnv

OUT_DIR = "outputs"

STAGES = [
    (["quadratic"], 0.25),
    (["quadratic", "log_sum_exp", "gaussian_pair"], 0.30),
    (None, 0.45),          # None means all six families
]

def _linear_schedule(initial_value, final_ratio=0.1):
    """SB3 learning-rate schedule: linear decay to `final_ratio` * initial.

    progress_remaining runs 1.0 (start) -> 0.0 (end), SB3's own convention.
    Long, high-entropy PPO runs are prone to late-training collapse once the
    policy has found a good region and a constant learning rate keeps taking
    full-size steps there; annealing is the standard fix. See train()'s
    docstring note on the 15M-step run that collapsed from ~40% to ~15%
    success between steps 9M and 15M with a constant LR.
    """
    def schedule(progress_remaining):
        return initial_value * (final_ratio + (1.0 - final_ratio) * progress_remaining)
    return schedule


def _ppo_kwargs(ent_coef=0.005, use_sde=True, sde_freq=8, net_width=128,
               n_steps=1024, batch_size=512, lr_anneal=True,
               learning_rate=3e-4, squash=True):
    """PPO hyperparameters.

    Two defaults changed after measuring the v3 run's action statistics.  Its
    rotation command sat above 1.9 of a 2.0 cap for 91% of all steps, a
    constant bang-bang spin that burned 0.17 m/s of the shared 0.3 m/s
    per-robot budget while the tuned analytic baseline spent 0.045.

    Cause: SB3's MlpPolicy uses an UNSQUASHED diagonal Gaussian.  Actions are
    clipped into range on the way to the environment, but log_prob and entropy
    are computed on the unclipped Gaussian, so pushing the mean past +/-1 is
    behaviourally free and nothing pulls it back.  ent_coef = 0.02 (SB3's
    default is 0.0) then actively pays for a large log_std, which under
    clipping is the same as paying for bang-bang.

    Fix: squash_output=True puts a tanh on the output so the distribution and
    the environment agree on the action bounds (supported with gSDE), and
    ent_coef drops to 0.005.
    """
    return dict(
        policy="MlpPolicy",
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        learning_rate=(_linear_schedule(learning_rate) if lr_anneal
                       else learning_rate),
        max_grad_norm=0.5,
        use_sde=use_sde,
        sde_sample_freq=sde_freq if use_sde else -1,
        policy_kwargs=dict(net_arch=dict(pi=[net_width, net_width],
                                         vf=[net_width, net_width]),
                           squash_output=(squash and use_sde)),
    )


# --------------------------------------------------------------------------
# Environment construction
# --------------------------------------------------------------------------

def make_env_fn(rank, seed, env_kwargs):
    def _init():
        env = QuadSaddleEnv(seed=seed + 1000 * rank, **env_kwargs)
        return Monitor(env, info_keywords=("family", "is_success",
                                           "time_in_tol"))
    return _init


def build_vec_env(n_envs, seed, env_kwargs, subproc=True, gamma=0.99):
    fns = [make_env_fn(i, seed, env_kwargs) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, gamma=gamma)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    """Per-rollout console line, a JSONL history for the figures, and
    periodic checkpoints (model + matching VecNormalize stats).

    Checkpointing exists because a 15M-step run without it collapsed late
    (success ~40% at 9M steps, ~15% by 15M) and there was no way to recover
    the mid-run policy that was actually good -- only the degraded final one
    got saved. `pick_best_checkpoint` in this file does a quick screen over
    whatever got saved here so that failure mode can't happen again.
    """

    def __init__(self, path, stage, ckpt_dir=None, ckpt_tag="ppo",
                ckpt_every=1_000_000, log_every=2):
        super().__init__()
        self.path = path
        self.stage = stage
        self.ckpt_dir = ckpt_dir
        self.ckpt_tag = ckpt_tag
        self.ckpt_every = ckpt_every
        self.log_every = log_every
        self.n_roll = 0
        self.t0 = time.time()
        self._next_ckpt = ckpt_every

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.n_roll += 1
        buf = self.model.ep_info_buffer
        if not buf:
            return
        rew = float(np.mean([e["r"] for e in buf]))
        ln = float(np.mean([e["l"] for e in buf]))
        succ = [e.get("is_success") for e in buf if e.get("is_success") is not None]
        tit = [e.get("time_in_tol") for e in buf if e.get("time_in_tol") is not None]
        sr = float(np.mean(succ)) if succ else float("nan")
        ti = float(np.mean(tit)) if tit else float("nan")

        row = dict(stage=self.stage, rollout=self.n_roll,
                   timesteps=int(self.num_timesteps), ep_rew_mean=rew,
                   ep_len_mean=ln, success_rate=sr, time_in_tol=ti,
                   wall=round(time.time() - self.t0, 1))
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if self.n_roll % self.log_every == 0:
            print(f"    stage {self.stage}  steps {self.num_timesteps:>9,}  "
                  f"rew {rew:8.2f}  len {ln:6.1f}  succ {sr:6.1%}  "
                  f"in_tol {ti:6.1%}  {row['wall']:7.0f}s")

        if self.ckpt_dir is not None and self.num_timesteps >= self._next_ckpt:
            base = os.path.join(self.ckpt_dir,
                                f"{self.ckpt_tag}_ckpt_{self.num_timesteps}")
            self.model.save(base)
            self.training_env.save(base + "_vecnormalize.pkl")
            print(f"    checkpoint -> {base}.zip")
            self._next_ckpt += self.ckpt_every


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(total_timesteps, n_envs, seed, env_kwargs, curriculum=False,
          tag="ppo", subproc=True, ppo_kwargs=None, ckpt_every=1_000_000):
    os.makedirs(OUT_DIR, exist_ok=True)
    hist_path = os.path.join(OUT_DIR, f"{tag}_history.jsonl")
    open(hist_path, "w").close()

    ppo_kwargs = ppo_kwargs or _ppo_kwargs()
    stages = STAGES if curriculum else [(env_kwargs.get("families"), 1.0)]
    model = None
    venv = None

    for i, (families, frac) in enumerate(stages, start=1):
        budget = int(total_timesteps * frac)
        kw = dict(env_kwargs)
        kw["families"] = families
        label = "all eight" if families is None else ", ".join(families)
        print(f"\n  stage {i}/{len(stages)}: {label}  ({budget:,} steps)")

        new_venv = build_vec_env(n_envs, seed + 7 * i, kw, subproc,
                                 gamma=ppo_kwargs["gamma"])
        if model is None:
            model = PPO(env=new_venv, seed=seed, verbose=0, **ppo_kwargs)
        else:
            # Carry the running observation statistics across the stage
            # boundary. Resetting them would throw away the normalizer the
            # policy was trained against and undo much of the prior stage.
            new_venv.obs_rms = venv.obs_rms
            new_venv.ret_rms = venv.ret_rms
            venv.close()
            model.set_env(new_venv)
        venv = new_venv

        model.learn(total_timesteps=budget, reset_num_timesteps=False,
                    callback=ProgressCallback(hist_path, i, ckpt_dir=OUT_DIR,
                                              ckpt_tag=tag,
                                              ckpt_every=ckpt_every),
                    progress_bar=False)

    model_path = os.path.join(OUT_DIR, f"{tag}_final")
    vec_path = os.path.join(OUT_DIR, f"{tag}_vecnormalize.pkl")
    model.save(model_path)
    venv.save(vec_path)
    venv.close()
    print(f"\n  saved {model_path}.zip")
    print(f"  saved {vec_path}")
    print(f"  saved {hist_path}")
    return model_path + ".zip", vec_path


# --------------------------------------------------------------------------
# Checkpoint selection
# --------------------------------------------------------------------------

def pick_best_checkpoint(tag, n_eval=40, seed0=900_000):
    """Quick screen over every checkpoint saved during training, plus the
    final model, and copy the winner to '{tag}_best.{zip,vecnormalize.pkl}'.

    Exists because the final checkpoint of a long PPO run is not guaranteed
    to be the best one; see ProgressCallback's docstring for the run that
    motivated this. n_eval is small (40, not the 200 held-out evaluate.py
    uses for the final report) because this only has to RANK checkpoints
    relative to each other, not report a publishable number.
    """
    import glob
    import shutil
    from evaluate import load_policy, policy_controller
    from baselines import run_episode, summarize
    from quad_saddle_env import QuadSaddleEnv

    ckpts = sorted(glob.glob(os.path.join(OUT_DIR, f"{tag}_ckpt_*.zip")))
    final = os.path.join(OUT_DIR, f"{tag}_final.zip")
    if os.path.exists(final):
        ckpts.append(final)
    if not ckpts:
        print("  no checkpoints found, nothing to select")
        return None

    print(f"\n  screening {len(ckpts)} checkpoints on {n_eval} fields each")
    print(f"  {'checkpoint':38s} {'success':>8s} {'in_tol':>8s} "
          f"{'e_final_med':>12s}")
    best, best_key = None, None
    for ckpt in ckpts:
        vec = ckpt.replace(".zip", "_vecnormalize.pkl")
        if not os.path.exists(vec):
            continue
        model, mean, var, clip = load_policy(ckpt, vec)
        ctl = policy_controller(model, mean, var, clip)
        env = QuadSaddleEnv(seed=seed0)
        rows = [run_episode(env, ctl, seed0 + i) for i in range(n_eval)]
        env.close()
        s = summarize(rows)
        name = os.path.basename(ckpt)
        print(f"  {name:38s} {s['success_rate']:8.1%} "
              f"{s['time_in_tol']:8.1%} {s['e_final_median']:12.3f}")
        key = (s["success_rate"], -s["e_final_median"])
        if best_key is None or key > best_key:
            best_key, best, best_vec = key, ckpt, vec

    print(f"\n  best: {os.path.basename(best)}")
    out_model = os.path.join(OUT_DIR, f"{tag}_best.zip")
    out_vec = os.path.join(OUT_DIR, f"{tag}_best_vecnormalize.pkl")
    shutil.copy(best, out_model)
    shutil.copy(best_vec, out_vec)
    print(f"  copied to {out_model}")
    print(f"  copied to {out_vec}")
    return out_model, out_vec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="ppo")
    p.add_argument("--curriculum", action="store_true",
                   help="opt into the old 3-stage schedule (default: off, "
                        "full family mix from step 0)")
    p.add_argument("--no-subproc", action="store_true",
                   help="single process, easier to debug")
    # exploration / capacity, see module docstring for the reasoning
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--no-squash", action="store_true",
                   help="disable tanh-squashed actions (see _ppo_kwargs)")
    p.add_argument("--no-sde", action="store_true",
                   help="disable gSDE, fall back to i.i.d. per-step noise")
    p.add_argument("--sde-freq", type=int, default=8)
    p.add_argument("--net-width", type=int, default=128)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--no-lr-anneal", action="store_true",
                   help="disable linear LR decay (on by default; see the "
                        "15M-step collapse this was added to fix)")
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--select-best", action="store_true",
                   help="after training, quick-screen all checkpoints and "
                        "copy the winner to {tag}_best.zip")
    # ablation switches, passed straight through to the environment
    p.add_argument("--obs-mode", choices=["raw", "raw+est"], default="raw")
    p.add_argument("--action-mode",
                   choices=["full", "no_rot", "fixed_size"], default="full")
    p.add_argument("--ideal-plant", action="store_true")
    p.add_argument("--sigma-z", type=float, default=0.0)
    p.add_argument("--rot-penalty", type=float, default=0.0)
    p.add_argument("--reward-mode", choices=["shaped", "pure", "metric"],
                   default="shaped")
    p.add_argument("--families", nargs="*", default=None)
    args = p.parse_args()

    env_kwargs = dict(obs_mode=args.obs_mode, action_mode=args.action_mode,
                      ideal_plant=args.ideal_plant, sigma_z=args.sigma_z,
                      rot_penalty=args.rot_penalty, reward_mode=args.reward_mode,
                      families=args.families)
    ppo_kwargs = _ppo_kwargs(ent_coef=args.ent_coef, use_sde=not args.no_sde,
                             sde_freq=args.sde_freq, net_width=args.net_width,
                             n_steps=args.n_steps, batch_size=args.batch_size,
                             lr_anneal=not args.no_lr_anneal,
                             squash=not args.no_squash)

    print("=" * 78)
    print(f"PPO  tag={args.tag}  {args.timesteps:,} steps  {args.envs} envs")
    print("=" * 78)
    for k, v in env_kwargs.items():
        print(f"  {k:14s} {v}")
    print(f"  {'curriculum':14s} {args.curriculum}")
    print(f"  {'lr_anneal':14s} {not args.no_lr_anneal}")
    print(f"  {'ckpt_every':14s} {args.ckpt_every:,}")
    for k in ("ent_coef", "use_sde", "sde_sample_freq", "n_steps", "batch_size"):
        print(f"  {k:14s} {ppo_kwargs[k]}")
    print(f"  {'net_width':14s} {args.net_width}")

    t0 = time.time()
    train(args.timesteps, args.envs, args.seed, env_kwargs,
          curriculum=args.curriculum, tag=args.tag,
          subproc=not args.no_subproc, ppo_kwargs=ppo_kwargs,
          ckpt_every=args.ckpt_every)
    dt = time.time() - t0
    print(f"\n  wall time {dt/60:.1f} min  "
          f"({args.timesteps/max(dt,1e-9):.0f} steps/s)")

    if args.select_best:
        pick_best_checkpoint(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
