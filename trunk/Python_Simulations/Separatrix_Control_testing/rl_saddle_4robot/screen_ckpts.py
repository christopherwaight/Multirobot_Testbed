"""Screen an arm's checkpoints and copy the winner to <tag>_best.zip.

A late-training collapse is a documented failure mode in this workstream (a
15M-step run went 40% success at 9M to 15% by 15M, and only the degraded final
policy had been saved).  Screening every checkpoint on a common set of
episodes makes that unrecoverable case impossible.
"""
import glob
import os
import shutil
import sys

import numpy as np

from baselines import run_episode
from evaluate import load_policy
from evaluate_sas import policy_controller_n
from quad_saddle_env import QuadSaddleEnv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def screen(tag, action_mode, start_shape, n=120, seed0=600_000):
    n_act = 7 if action_mode == "sas_full" else 4
    cks = sorted(glob.glob(os.path.join(OUT, f"{tag}_ckpt_*.zip")))
    fin = os.path.join(OUT, f"{tag}_final.zip")
    if os.path.exists(fin):
        cks.append(fin)
    if not cks:
        print(f"  {tag}: no checkpoints found")
        return None
    env_kwargs = dict(obs_mode="raw+est", action_mode=action_mode,
                      reward_mode="free_shape", start_shape=start_shape,
                      start_r_range=(1.0, 2.5))
    print(f"\n== screening {tag}: {len(cks)} checkpoints, {n} episodes each ==",
          flush=True)
    best, best_s = None, -1.0
    for c in cks:
        vp = c.replace(".zip", "_vecnormalize.pkl")
        if not os.path.exists(vp):
            vp = os.path.join(OUT, f"{tag}_vecnormalize.pkl")
        try:
            m, mean, var, clip = load_policy(c, vp)
        except Exception as exc:
            print(f"   {os.path.basename(c):<46s} load failed: {exc}")
            continue
        ctl = policy_controller_n(m, mean, var, clip, n_act)
        env = QuadSaddleEnv(**env_kwargs)
        rows = [run_episode(env, ctl, seed0 + i) for i in range(n)]
        env.close()
        s = float(np.mean([r["success"] for r in rows]))
        t = float(np.mean([r["time_in_tol"] for r in rows]))
        print(f"   {os.path.basename(c):<46s} succ {s:6.1%}  in_tol {t:6.1%}",
              flush=True)
        if s > best_s:
            best_s, best = s, c
    if best is not None:
        shutil.copy(best, os.path.join(OUT, f"{tag}_best.zip"))
        vp = best.replace(".zip", "_vecnormalize.pkl")
        if not os.path.exists(vp):
            vp = os.path.join(OUT, f"{tag}_vecnormalize.pkl")
        if os.path.exists(vp):
            shutil.copy(vp, os.path.join(OUT, f"{tag}_best_vecnormalize.pkl"))
        print(f"   best: {os.path.basename(best)} -> {tag}_best.zip "
              f"({best_s:.1%})", flush=True)
    return best


if __name__ == "__main__":
    tag = sys.argv[1]
    am = sys.argv[2] if len(sys.argv) > 2 else "sas_full"
    ss = sys.argv[3] if len(sys.argv) > 3 else "random"
    screen(tag, am, ss)
