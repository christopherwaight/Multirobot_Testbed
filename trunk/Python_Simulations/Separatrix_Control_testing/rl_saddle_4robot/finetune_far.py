"""Finetune the best policy on FAR starts only.

Diagnosis this addresses.  On the final evaluation `sas_a5_e5ft_best` reaches
0.878 success on reachable episodes against the analytic law's 0.927, i.e. it
throws away winnable episodes.  Breaking those losses down, 100% of them are
episodes where it never got close (e_min > 0.5), not episodes where it arrived
and drifted off, and their start distance is systematically larger: median
e0 = 1.89 on losses versus 1.48 on wins.  They also cluster in `quadratic` and
`cubic_perturbed`, the two families the analytic law solves at 0.98, so these
are easy fields failed from far starts.

Cause: the curriculum anneals the start annulus from [0.25, 0.80] out to
[1.0, 2.5] and only reaches the evaluation range at 60% of training, so the
far tail of the distribution gets a fraction of the samples. This run
continues from the screened best checkpoint with the annulus pinned at the
far end for the whole run.
"""
import argparse, json, os, time
import numpy as np
from stable_baselines3 import PPO
import train_ppo as T
from evaluate import load_policy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="sas_a7_far")
    p.add_argument("--from-tag", default="sas_a5_e5ft_best")
    p.add_argument("--timesteps", type=int, default=2_500_000)
    p.add_argument("--envs", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--r-lo", type=float, default=1.4)
    p.add_argument("--r-hi", type=float, default=2.5)
    p.add_argument("--ckpt-every", type=int, default=500_000)
    args = p.parse_args()

    t0 = time.time()
    env_kwargs = dict(obs_mode="raw+est", action_mode="sas_full",
                      reward_mode="free_shape", start_shape="random",
                      start_r_range=(args.r_lo, args.r_hi))
    kw = T._ppo_kwargs(net_width=256, learning_rate=args.lr,
                       ent_coef=args.ent_coef)
    venv = T.build_vec_env(args.envs, args.seed, env_kwargs, subproc=True,
                           gamma=kw["gamma"])
    src_zip = os.path.join(OUT, f"{args.from_tag}.zip")
    src_vec = os.path.join(OUT, f"{args.from_tag}_vecnormalize.pkl")
    _, mean, var, _ = load_policy(src_zip, src_vec)
    venv.obs_rms.mean[:] = mean
    venv.obs_rms.var[:] = var
    venv.obs_rms.count = 1e5

    model = PPO(env=venv, seed=args.seed, verbose=0, **kw)
    model.policy.load_state_dict(PPO.load(src_zip, device="cpu").policy.state_dict())
    print(f"  continued from {args.from_tag}, starts pinned to "
          f"[{args.r_lo}, {args.r_hi}]", flush=True)

    hist = os.path.join(OUT, f"{args.tag}_history.jsonl"); open(hist, "w").close()
    model.learn(total_timesteps=args.timesteps,
                callback=T.ProgressCallback(hist, 1, ckpt_dir=OUT,
                                            ckpt_tag=args.tag,
                                            ckpt_every=args.ckpt_every),
                progress_bar=False)
    model.save(os.path.join(OUT, f"{args.tag}_final"))
    venv.save(os.path.join(OUT, f"{args.tag}_vecnormalize.pkl"))
    venv.close()
    print(f"  saved {args.tag}_final.zip ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
