"""
Fine-tuning hyperparameter sweep from a fixed behaviour-cloned start.

Covers two questions at once:

  (3) Can the easy-family regression be recovered?  E5 gives back 4-6 points on
      `quadratic` and `cubic_perturbed`, the families the clone already had at
      98.5% and 95.2%.  Lower learning rates should forget less, so the sweep
      spans an order of magnitude in lr.

  (4) Fairness.  The analytic law's gains came from a 48-cell sweep selected on
      the reported statistic.  PPO has only ever had hand-picked configs, so any
      comparison so far has been selection-asymmetric.

Scope, stated honestly: this sweeps FINE-TUNING from a shared clone, not
training from scratch.  A from-scratch sweep at a meaningful budget is hours,
not the ~30 minutes this takes.  Every cell starts from the same
`{bc_tag}_bconly` checkpoint, so differences are the hyperparameters.

Selection uses validation seeds 900k+, disjoint from evaluate.py's 500k
held-out range, so the winner can still be scored honestly afterwards.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from stable_baselines3 import PPO

from baselines import run_episode, summarize
from evaluate import load_policy, policy_controller
from quad_saddle_env import QuadSaddleEnv
import train_ppo as T

OUT_DIR = "outputs"

# (label, learning_rate, n_epochs, ent_coef)
# Labels must not contain '.': they become filename stems, and SB3's save/load
# path handling mis-parses a dot as an extension boundary, which silently
# produced 'name.zip.zip' lookups. lr is encoded in units of 1e-6.
CELLS = [
    ("lr50_ep10",  5e-5, 10, 0.005),
    ("lr120_ep10", 1.2e-4, 10, 0.005),
    ("lr300_ep10", 3e-4, 10, 0.005),
    ("lr120_ep20", 1.2e-4, 20, 0.005),
    ("lr120_ent0", 1.2e-4, 10, 0.0),
]


def screen(model_path, vec_path, n=150, seed0=900_000):
    model, mean, var, clip = load_policy(model_path, vec_path)
    ctl = policy_controller(model, mean, var, clip)
    env = QuadSaddleEnv(obs_mode="raw+est", reward_mode="metric", seed=seed0)
    rows = [run_episode(env, ctl, seed0 + i) for i in range(n)]
    env.close()
    s = summarize(rows)
    fam = np.array([r["family"] for r in rows])
    ok = np.array([r["success"] for r in rows], float)
    easy = ["quadratic", "cubic_perturbed", "rational_envelope"]
    hard = ["log_sum_exp", "streamfunction_quad", "gaussian_pair"]
    s["easy3"] = float(np.mean([ok[fam == f].mean() for f in easy if (fam == f).any()]))
    s["hard3"] = float(np.mean([ok[fam == f].mean() for f in hard if (fam == f).any()]))
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc", default=os.path.join(OUT_DIR, "ppo_e5_bconly.zip"))
    p.add_argument("--bc-vec",
                   default=os.path.join(OUT_DIR, "ppo_e5_bconly_vecnormalize.pkl"))
    p.add_argument("--steps", type=int, default=800_000)
    p.add_argument("--ckpt-every", type=int, default=200_000)
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--screen-n", type=int, default=150)
    p.add_argument("--net-width", type=int, default=256)
    args = p.parse_args()

    print("=" * 100)
    print(f"FINE-TUNE SWEEP from {os.path.basename(args.bc)}")
    print(f"  {len(CELLS)} cells x {args.steps:,} steps, "
          f"checkpoint every {args.ckpt_every:,}")
    print(f"  screened on {args.screen_n} validation fields (seeds 900000+)")
    print("=" * 100)

    t0 = time.time()
    all_rows = []
    for label, lr, ep, ent in CELLS:
        tag = f"sw_{label}"
        print(f"\n--- {label}: lr={lr:g} epochs={ep} ent_coef={ent:g} ---",
              flush=True)
        kw = T._ppo_kwargs(learning_rate=lr, net_width=args.net_width,
                           ent_coef=ent)
        kw["n_epochs"] = ep
        venv = T.build_vec_env(args.envs, 0,
                               dict(obs_mode="raw+est", reward_mode="metric"),
                               subproc=True, gamma=kw["gamma"])
        # start every cell from the identical cloned weights and statistics
        src = PPO.load(args.bc, device="cpu")
        _, mean, var, _ = load_policy(args.bc, args.bc_vec)
        venv.obs_rms.mean, venv.obs_rms.var = mean.copy(), var.copy()
        venv.obs_rms.count = 1e5
        model = PPO(env=venv, seed=0, verbose=0, **kw)
        model.policy.load_state_dict(src.policy.state_dict())

        hist = os.path.join(OUT_DIR, f"{tag}_history.jsonl")
        open(hist, "w").close()
        model.learn(total_timesteps=args.steps,
                    callback=T.ProgressCallback(hist, 1, ckpt_dir=OUT_DIR,
                                                ckpt_tag=tag,
                                                ckpt_every=args.ckpt_every,
                                                log_every=1000),
                    progress_bar=False)
        model.save(os.path.join(OUT_DIR, f"{tag}_final"))
        venv.save(os.path.join(OUT_DIR, f"{tag}_vecnormalize.pkl"))
        venv.close()

        import glob
        cks = sorted(glob.glob(os.path.join(OUT_DIR, f"{tag}_ckpt_*.zip"))) \
            + [os.path.join(OUT_DIR, f"{tag}_final.zip")]
        for ck in cks:
            v = ck.replace(".zip", "_vecnormalize.pkl")
            if not os.path.exists(v):
                v = os.path.join(OUT_DIR, f"{tag}_vecnormalize.pkl")
            if not os.path.exists(v):
                continue
            try:
                s = screen(ck, v, n=args.screen_n)
            except Exception as ex:
                print(f"    {os.path.basename(ck):40s} FAILED {ex}", flush=True)
                continue
            all_rows.append(dict(cell=label, ckpt=os.path.basename(ck),
                                 path=ck, vec=v, **{k: s[k] for k in
                                 ("success_rate", "time_in_tol",
                                  "e_final_median", "easy3", "hard3")}))
            print(f"    {os.path.basename(ck):40s} succ={s['success_rate']:6.1%} "
                  f"easy3={s['easy3']:6.1%} hard3={s['hard3']:6.1%} "
                  f"e={s['e_final_median']:.3f}", flush=True)

    print()
    print("=" * 100)
    print(f"SWEEP COMPLETE in {(time.time()-t0)/60:.1f} min")
    print("=" * 100)
    all_rows.sort(key=lambda r: -r["success_rate"])
    print(f"  {'cell':18s}{'checkpoint':34s}{'succ':>8s}{'easy3':>8s}{'hard3':>8s}")
    for r in all_rows[:12]:
        print(f"  {r['cell']:18s}{r['ckpt']:34s}{r['success_rate']:8.1%}"
              f"{r['easy3']:8.1%}{r['hard3']:8.1%}")

    with open(os.path.join(OUT_DIR, "sweep_finetune.json"), "w") as f:
        json.dump(all_rows, f, indent=2)

    if all_rows:
        import shutil
        b = all_rows[0]
        shutil.copy(b["path"], os.path.join(OUT_DIR, "ppo_sweep_best.zip"))
        shutil.copy(b["vec"],
                    os.path.join(OUT_DIR, "ppo_sweep_best_vecnormalize.pkl"))
        print(f"\n  best: {b['cell']} / {b['ckpt']} at {b['success_rate']:.1%}")
        print("  copied to outputs/ppo_sweep_best.zip")
    print("SWEEPDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
