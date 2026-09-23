"""Free-shape SAS training: one arm of the overnight portfolio.

What this run is testing
------------------------
The historical action space commands one scalar of formation shape, a common
diagonal length, with r1 = r2 = 0.5 and phi = 90 deg pinned at reset.  The
formation is therefore a square of varying size for the whole episode, and
measured on ten E5 rollouts the size channel was not even being used: the
policy drove it to the R_MIN floor within a few steps and held it there on
99.4% of steps.  The demonstrations explain why.  The analytic teacher
`rotating_hessian` computes its size command once, outside its own loop, and
its docstring says plainly that the notebook law has no size control; BC's
fourth action channel had standard deviation exactly 0.000.

`action_mode='sas_full'` opens the other four shape parameters, so the policy
can command rectangles, kites and slivers.  `reward_mode='free_shape'` removes
every term that paid for shrinking, and drops the R < R_TARGET clause from the
success gate, so the question becomes whether the centroid can find and hold
the saddle rather than whether it can also be small.

The start-radius curriculum
---------------------------
Measured separately (ideal Newton flow, exact gradient and Hessian, no
dynamics and no noise), convergence from the default start annulus is:

    r = 0.30    100.0%        r = 1.25     57.5%
    r = 0.50    100.0%        r = 1.50     60.0%
    r = 0.75     82.5%        r = 2.00     55.0%
    r = 1.00     72.5%        r = 2.50     42.5%

and pooled over the annulus r in [1.0, 2.5] it is 52.1%.  Four of the eight
families are far worse than that pooled figure: gaussian_pair converges from
6.7% of annulus starts, streamfunction_quad from 5.0%.  Those are statements
about field topology, not about control: no controller reaches the saddle
from a start whose Newton flow leads somewhere else.  Training on a fixed far
annulus therefore spends most of its samples on episodes that cannot be won.

The curriculum starts inside the basin and anneals outward, so early learning
happens on solvable episodes and the final distribution still matches the
evaluation protocol.

Run:
    <VF_Robot venv python> train_sas_free.py --tag sas_a1 --timesteps 4000000
"""

import argparse
import json
import os
import time

import numpy as np

import train_ppo as T
from quad_saddle_env import QuadSaddleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


class CurriculumCallback(BaseCallback):
    """Anneal the start-radius annulus outward as training proceeds.

    `frac` runs 0 -> 1 over the run.  The annulus interpolates from
    (r_lo0, r_hi0) to (r_lo1, r_hi1), reaching the final (evaluation) range at
    `reach_frac` so the last portion of training is on-distribution.
    """

    def __init__(self, venv, total, r0=(0.25, 0.8), r1=(1.0, 2.5),
                 reach_frac=0.6, every=20_000, log_path=None):
        super().__init__()
        self.venv = venv
        self.total = int(total)
        self.r0 = r0
        self.r1 = r1
        self.reach_frac = float(reach_frac)
        self.every = int(every)
        self.log_path = log_path
        self._last = -1

    def _current_range(self, frac):
        u = min(1.0, frac / self.reach_frac) if self.reach_frac > 0 else 1.0
        lo = self.r0[0] + u * (self.r1[0] - self.r0[0])
        hi = self.r0[1] + u * (self.r1[1] - self.r0[1])
        return (float(lo), float(hi))

    def _on_step(self):
        n = self.num_timesteps
        if n - self._last < self.every:
            return True
        self._last = n
        rng = self._current_range(n / max(1, self.total))
        try:
            self.venv.env_method("set_start_r_range", rng)
        except Exception:
            pass
        return True


def build(tag, args):
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
        start_shape=args.start_shape,
        start_r_range=(args.r_lo0, args.r_hi0) if args.curriculum else None,
        sigma_z=args.sigma_z,
    )
    ppo_kwargs = T._ppo_kwargs(
        ent_coef=args.ent_coef,
        net_width=args.net_width,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_anneal=not args.no_lr_anneal,
    )
    venv = T.build_vec_env(args.envs, args.seed, env_kwargs,
                           subproc=not args.no_subproc,
                           gamma=ppo_kwargs["gamma"])
    return venv, ppo_kwargs, env_kwargs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="sas_free")
    p.add_argument("--timesteps", type=int, default=4_000_000)
    p.add_argument("--envs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-mode", default="raw+est")
    p.add_argument("--action-mode", default="sas_full")
    p.add_argument("--reward-mode", default="free_shape")
    p.add_argument("--start-shape", default="random")
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--net-width", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-lr-anneal", action="store_true")
    p.add_argument("--no-subproc", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--sigma-z", type=float, default=0.0,
                   help="reading noise. Size only matters when this is > 0.")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--r-lo0", type=float, default=0.25)
    p.add_argument("--r-hi0", type=float, default=0.80)
    p.add_argument("--r-lo1", type=float, default=1.00)
    p.add_argument("--r-hi1", type=float, default=2.50)
    p.add_argument("--reach-frac", type=float, default=0.6)
    p.add_argument("--bc-warm", action="store_true",
                   help="clone the analytic teacher first (translation and "
                        "rotation only; it has nothing to say about the four "
                        "new shape channels)")
    p.add_argument("--bc-episodes", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=20)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    print("=" * 78)
    print(f"SAS FREE-SHAPE RUN  tag={args.tag}")
    print(f"  action_mode={args.action_mode}  reward_mode={args.reward_mode}")
    print(f"  start_shape={args.start_shape}  curriculum={args.curriculum}")
    print(f"  ent_coef={args.ent_coef}  net={args.net_width}  envs={args.envs}")
    print(f"  timesteps={args.timesteps:,}  bc_warm={args.bc_warm}")
    print("=" * 78, flush=True)

    venv, ppo_kwargs, env_kwargs = build(args.tag, args)
    model = PPO(env=venv, seed=args.seed, verbose=0, **ppo_kwargs)

    if args.bc_warm:
        # The teacher emits 4 channels; this action space has 7.  The extra
        # shape channels are cloned toward 0 (the midpoint of each bound),
        # which is a neutral prior, not a demonstration: the analytic law has
        # no opinion about them.
        import bc_pretrain as B
        print(f"\n  BC warm start: {args.bc_episodes} teacher episodes")
        bc_kwargs = dict(env_kwargs)
        bc_kwargs["action_mode"] = "full"        # teacher speaks 4-channel
        obs, act4 = B.collect(args.bc_episodes, env_kwargs=bc_kwargs)
        act7 = np.zeros((len(act4), 7), dtype=np.float32)
        act7[:, :3] = act4[:, :3]                # vx, vy, omega carry over
        print(f"     {len(obs):,} transitions, obs dim {obs.shape[1]}")
        venv.obs_rms.mean = obs.mean(axis=0).astype(np.float64)
        venv.obs_rms.var = obs.var(axis=0).astype(np.float64) + 1e-8
        venv.obs_rms.count = float(len(obs))
        obs_n = np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + 1e-8),
                        -venv.clip_obs, venv.clip_obs).astype(np.float32)
        B.clone(model, obs_n, act7, epochs=args.bc_epochs)
        model.save(os.path.join(OUT_DIR, f"{args.tag}_bconly"))
        venv.save(os.path.join(OUT_DIR, f"{args.tag}_bconly_vecnormalize.pkl"))

    hist = os.path.join(OUT_DIR, f"{args.tag}_history.jsonl")
    open(hist, "w").close()
    cbs = [T.ProgressCallback(hist, 1, ckpt_dir=OUT_DIR, ckpt_tag=args.tag,
                              ckpt_every=args.ckpt_every)]
    if args.curriculum:
        cbs.append(CurriculumCallback(
            venv, args.timesteps,
            r0=(args.r_lo0, args.r_hi0), r1=(args.r_lo1, args.r_hi1),
            reach_frac=args.reach_frac))

    model.learn(total_timesteps=args.timesteps, callback=cbs,
                progress_bar=False)

    model.save(os.path.join(OUT_DIR, f"{args.tag}_final"))
    venv.save(os.path.join(OUT_DIR, f"{args.tag}_vecnormalize.pkl"))
    venv.close()

    meta = dict(tag=args.tag, args=vars(args), env_kwargs={
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in env_kwargs.items()},
        wall_seconds=time.time() - t_start)
    with open(os.path.join(OUT_DIR, f"{args.tag}_runmeta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  saved {OUT_DIR}/{args.tag}_final.zip "
          f"({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
