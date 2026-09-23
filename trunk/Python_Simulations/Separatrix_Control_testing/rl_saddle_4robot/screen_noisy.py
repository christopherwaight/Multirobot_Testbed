"""Screen checkpoints under reading noise (screen_ckpts.py assumes sigma_z=0)."""
import glob, os, shutil, sys
import numpy as np
from baselines import run_episode
from evaluate import load_policy
from evaluate_sas import policy_controller_n
from quad_saddle_env import QuadSaddleEnv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
tag = sys.argv[1]; sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
cks = sorted(glob.glob(os.path.join(OUT, f"{tag}_ckpt_*.zip")))
fin = os.path.join(OUT, f"{tag}_final.zip")
if os.path.exists(fin): cks.append(fin)
best, best_s = None, -1.0
print(f"== screening {tag} at sigma_z={sigma} ==", flush=True)
for c in cks:
    vp = c.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vp): vp = os.path.join(OUT, f"{tag}_vecnormalize.pkl")
    try: m, mean, var, clip = load_policy(c, vp)
    except Exception as e: print("  load fail", e); continue
    ctl = policy_controller_n(m, mean, var, clip, 7)
    env = QuadSaddleEnv(obs_mode="raw+est", action_mode="sas_full",
                        reward_mode="free_shape", start_shape="random",
                        start_r_range=(1.0, 2.5), sigma_z=sigma)
    rows = [run_episode(env, ctl, 600_000 + i) for i in range(120)]
    env.close()
    s = float(np.mean([r["success"] for r in rows]))
    print(f"   {os.path.basename(c):<44s} {s:6.1%}", flush=True)
    if s > best_s: best_s, best = s, c
if best:
    shutil.copy(best, os.path.join(OUT, f"{tag}_best.zip"))
    vp = best.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vp): vp = os.path.join(OUT, f"{tag}_vecnormalize.pkl")
    if os.path.exists(vp): shutil.copy(vp, os.path.join(OUT, f"{tag}_best_vecnormalize.pkl"))
    print(f"   best: {os.path.basename(best)} ({best_s:.1%})", flush=True)
