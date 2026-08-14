"""
Helpers for the side-by-side comparison notebook.

Kept out of the notebook itself so the notebook stays short enough to read in
one screen.  Everything here is a thin wrapper over code that already exists:
`baselines.rotating_hessian` for the analytic law, `evaluate.load_policy` /
`policy_controller` for a trained checkpoint, and `QuadSaddleEnv` for the plant.

The one thing worth stating plainly: `run_one` builds a FRESH environment per
call and seeds it, so two controllers given the same seed see byte-identical
fields, start positions, headings, and formation radii.  Any difference in the
resulting trajectories is the controller, not the draw.
"""
import glob
import os

import numpy as np

from baselines import rotating_hessian, run_episode
from evaluate import load_policy, policy_controller
from quad_saddle_env import QuadSaddleEnv
import saddle_fields as sf

OUT_DIR = "outputs"

# Gains from the 48-cell sweep in baselines.py; the best cell on the fixed env.
ANALYTIC_GAINS = dict(mode="single", k_rot=0.5, k_trans=1.5, r0=0.1)


def list_policies():
    """Available checkpoints, newest-looking first, with a usable label."""
    out = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "*.zip"))):
        vec = p.replace(".zip", "_vecnormalize.pkl")
        stem = os.path.basename(p)[:-4]
        # every checkpoint has its own normalizer; the *_final ones share the
        # run-level pkl instead
        if not os.path.exists(vec):
            alt = os.path.join(OUT_DIR,
                               stem.split("_final")[0] + "_vecnormalize.pkl")
            vec = alt if os.path.exists(alt) else None
        if vec:
            out.append((stem, p, vec))
    return out


def make_analytic(**overrides):
    g = dict(ANALYTIC_GAINS)
    g.update(overrides)
    ctl = rotating_hessian(max_omega=min(2.0, 0.3 / g.get("r0", 0.1)), **g)
    ctl.label = "analytic (tuned rotating-Hessian)"
    return ctl


def make_policy(stem):
    """Load a checkpoint by stem, e.g. 'ppo_e4_ckpt_1511424'."""
    hits = [t for t in list_policies() if t[0] == stem]
    if not hits:
        raise FileNotFoundError(
            f"no checkpoint {stem!r}. Available:\n  "
            + "\n  ".join(t[0] for t in list_policies()))
    _, model_path, vec_path = hits[0]
    model, mean, var, clip = load_policy(model_path, vec_path)
    ctl = policy_controller(model, mean, var, clip)
    ctl.label = f"PPO ({stem})"
    # obs width the checkpoint expects, so the caller can build a matching env
    ctl.n_obs = int(model.observation_space.shape[0])
    return ctl


def env_for(ctl, family=None, reward_mode="metric"):
    """Environment whose observation width matches what `ctl` expects."""
    n = getattr(ctl, "n_obs", 24)
    obs_mode = "raw+est" if n >= 30 else "raw"
    fams = [family] if isinstance(family, str) else family
    return QuadSaddleEnv(families=fams, obs_mode=obs_mode,
                         reward_mode=reward_mode, seed=0)


def run_one(ctl, seed, family=None):
    """One episode. Returns (metrics dict, per-step log dict, SaddleField)."""
    env = env_for(ctl, family)
    row = run_episode(env, ctl, seed)
    log = {k: np.asarray(v) for k, v in env.episode_log.items()}
    fld = env.fld
    env.close()
    return row, log, fld


def run_pair(policy_ctl, analytic_ctl, seed, family=None):
    """Both controllers on the identical field. Returns a dict of results."""
    r_p, l_p, fld = run_one(policy_ctl, seed, family)
    r_a, l_a, _ = run_one(analytic_ctl, seed, family)
    return dict(seed=seed, field=fld,
                policy=dict(row=r_p, log=l_p, label=policy_ctl.label),
                analytic=dict(row=r_a, log=l_a, label=analytic_ctl.label))


def batch(policy_ctl, analytic_ctl, n=40, seed0=500_000, family=None):
    """N fields, both controllers, paired. Returns a list of row pairs."""
    out = []
    env_p = env_for(policy_ctl, family)
    env_a = env_for(analytic_ctl, family)
    for i in range(n):
        s = seed0 + i
        rp = run_episode(env_p, policy_ctl, s)
        ra = run_episode(env_a, analytic_ctl, s)
        out.append(dict(seed=s, family=rp["family"],
                        p_succ=rp["success"], a_succ=ra["success"],
                        p_e=rp["e_final"], a_e=ra["e_final"]))
    env_p.close()
    env_a.close()
    return out


def batch_table(rows):
    """Print a paired win/loss summary of `batch` output."""
    p = np.array([r["p_succ"] for r in rows], bool)
    a = np.array([r["a_succ"] for r in rows], bool)
    fam = np.array([r["family"] for r in rows])
    print(f"  {len(rows)} paired fields")
    print(f"    analytic success : {a.mean():6.1%}")
    print(f"    policy   success : {p.mean():6.1%}")
    print(f"    both             : {(p & a).mean():6.1%}")
    print(f"    policy only      : {(p & ~a).mean():6.1%}")
    print(f"    analytic only    : {(~p & a).mean():6.1%}")
    print(f"    neither          : {(~p & ~a).mean():6.1%}")
    print(f"    either (oracle)  : {(p | a).mean():6.1%}")
    print()
    print(f"    {'family':22s}{'n':>4s}{'analytic':>10s}{'policy':>9s}")
    for f in sf.FAMILY_NAMES:
        m = fam == f
        if m.sum():
            print(f"    {f:22s}{m.sum():4d}{a[m].mean():10.1%}{p[m].mean():9.1%}")
