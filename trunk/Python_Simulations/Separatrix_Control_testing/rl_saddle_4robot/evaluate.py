"""
Evaluate a trained policy against the hand-derived baselines.

Every controller, learned or hand-written, is driven through the identical
QuadSaddleEnv with the identical reward, termination rule, and plant.  The only
thing that differs is which function produces the action.

Held-out fields.  Training draws fields from seeds 0 + 1000*rank upward;
evaluation uses seeds from EVAL_SEED0 = 500_000.  The field parameters are
continuous, so an exact repeat has probability zero regardless, but keeping the
seed ranges disjoint makes that explicit rather than probabilistic.

Outputs a JSON blob under outputs/ that visualize.py turns into figures, plus
a console table.
"""
import argparse
import json
import os

import numpy as np

from baselines import (rotating_hessian, zero_controller, run_episode,
                       summarize, _print_summary)
from quad_saddle_env import QuadSaddleEnv
import saddle_fields as sf

EVAL_SEED0 = 500_000
OUT_DIR = "outputs"


# --------------------------------------------------------------------------
# Policy as a controller
# --------------------------------------------------------------------------

def load_policy(model_path, vec_path=None):
    """Load a saved PPO model and its observation normalizer.

    The normalizer matters.  A policy trained under VecNormalize sees
    standardized observations, so replaying it on raw observations produces
    nonsense.  Rather than rebuild a VecNormalize wrapper (which would hide the
    per-step episode log the figures need), the running statistics are pulled
    out and applied by hand, exactly as VecNormalize.normalize_obs does.
    """
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")

    mean = var = None
    clip = 10.0
    if vec_path and os.path.exists(vec_path):
        from stable_baselines3.common.vec_env import VecNormalize
        with open(vec_path, "rb") as f:
            import pickle
            vn = pickle.load(f)
        if isinstance(vn, VecNormalize) and vn.norm_obs:
            mean = vn.obs_rms.mean
            var = vn.obs_rms.var
            clip = vn.clip_obs
    return model, mean, var, clip


def policy_controller(model, mean=None, var=None, clip=10.0,
                      deterministic=True):
    def controller(env):
        obs = env.current_obs()
        if mean is not None:
            obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = model.predict(obs.astype(np.float32), deterministic=deterministic)
        return np.asarray(a, dtype=float).ravel()
    controller.label = "PPO"
    return controller


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_controller(controller, n_eval, plant="full", families=None,
                        keep_logs=0, **env_kwargs):
    """Run one controller over n_eval held-out fields.

    keep_logs > 0 retains that many full per-step episode logs, which is what
    the mechanism and tradeoff figures are built from.
    """
    env = QuadSaddleEnv(families=families, ideal_plant=(plant == "ideal"),
                        **env_kwargs)
    rows, logs = [], []
    for i in range(n_eval):
        r = run_episode(env, controller, EVAL_SEED0 + i)
        r["seed"] = EVAL_SEED0 + i
        rows.append(r)
        if i < keep_logs:
            lg = {k: np.asarray(v).tolist() for k, v in env.episode_log.items()}
            lg["family"] = env.fld.family
            lg["saddle"] = env.fld.saddle.tolist()
            lg["hess"] = env.fld.hess.tolist()
            lg["params"] = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                            for k, v in env.fld.params.items()}
            lg["seed"] = EVAL_SEED0 + i
            # Enough to reconstruct the exact field in visualize.py:
            # default_rng(seed) then sample_field(rng, families) reproduces the
            # same draw the environment made, because reset() seeds its own
            # generator and takes the field from it first.
            lg["families"] = families
            logs.append(lg)
    env.close()
    return rows, logs


def per_family(rows):
    out = {}
    for fam in sf.FAMILY_NAMES:
        sub = [r for r in rows if r["family"] == fam]
        if sub:
            out[fam] = summarize(sub)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.path.join(OUT_DIR, "ppo_final.zip"))
    p.add_argument("--vecnorm", default=os.path.join(OUT_DIR, "ppo_vecnormalize.pkl"))
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--keep-logs", type=int, default=8)
    p.add_argument("--baselines-json", default=os.path.join(OUT_DIR, "baselines.json"))
    p.add_argument("--out", default=os.path.join(OUT_DIR, "evaluation.json"))
    p.add_argument("--obs-mode", choices=["raw", "raw+est"], default="raw")
    p.add_argument("--action-mode",
                   choices=["full", "no_rot", "fixed_size"], default="full")
    p.add_argument("--tag", default="PPO")
    p.add_argument("--reward-mode", choices=["shaped", "pure", "metric"],
                   default="shaped",
                   help="must match how the checkpoint was trained; the env's "
                        "own reward is unused at eval time (info['e'] scores "
                        "it) but kept in sync for episode_log completeness")
    p.add_argument("--skip-policy", action="store_true",
                   help="baselines only, for when no checkpoint exists yet")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {"n_eval": args.n_eval, "eval_seed0": EVAL_SEED0, "runs": {}}
    logs_out = {}

    print("=" * 112)
    print(f"EVALUATION on {args.n_eval} held-out fields "
          f"(seeds {EVAL_SEED0}..{EVAL_SEED0 + args.n_eval - 1})")
    print("=" * 112)

    # -- baselines --------------------------------------------------------
    best_gains = None
    if os.path.exists(args.baselines_json):
        with open(args.baselines_json) as f:
            bj = json.load(f)
        best_gains = bj.get("sweep_best")

    runs = []
    runs.append(("do-nothing (floor)", zero_controller(), "full", {}))
    runs.append(("rot-hessian single, default gains",
                 rotating_hessian(mode="single"), "full", {}))
    runs.append(("rot-hessian none, default gains",
                 rotating_hessian(mode="none"), "full", {}))
    if best_gains:
        runs.append((
            f"rot-hessian single, BEST gains "
            f"(k_rot={best_gains['k_rot']:g}, k_trans={best_gains['k_trans']:g}, "
            f"r0={best_gains['r0']:g})",
            rotating_hessian(mode="single", k_rot=best_gains["k_rot"],
                             k_trans=best_gains["k_trans"], r0=best_gains["r0"],
                             max_omega=min(2.0, 0.3 / best_gains["r0"])),
            "full", {}))
    runs.append(("rot-hessian single, 0-dynamics ceiling",
                 rotating_hessian(mode="single"), "ideal", {}))

    for label, ctl, plant, kw in runs:
        rows, logs = evaluate_controller(ctl, args.n_eval, plant=plant,
                                         keep_logs=args.keep_logs, **kw)
        s = summarize(rows)
        payload["runs"][label] = dict(summary=s, per_family=per_family(rows),
                                      plant=plant)
        logs_out[label] = logs
        _print_summary(label, s)

    # -- policy -----------------------------------------------------------
    if not args.skip_policy:
        if not os.path.exists(args.checkpoint):
            print(f"\n  no checkpoint at {args.checkpoint}, skipping the policy")
        else:
            model, mean, var, clip = load_policy(args.checkpoint, args.vecnorm)
            ctl = policy_controller(model, mean, var, clip)
            rows, logs = evaluate_controller(
                ctl, args.n_eval, plant="full", keep_logs=args.keep_logs,
                obs_mode=args.obs_mode, action_mode=args.action_mode,
                reward_mode=args.reward_mode)
            s = summarize(rows)
            payload["runs"][args.tag] = dict(summary=s,
                                             per_family=per_family(rows),
                                             plant="full")
            logs_out[args.tag] = logs
            print()
            _print_summary(args.tag, s)

    # -- per-family breakdown ---------------------------------------------
    print()
    print("=" * 112)
    print("MEDIAN FINAL DISTANCE BY FIELD FAMILY")
    print("=" * 112)
    fams = sf.FAMILY_NAMES
    print(f"  {'controller':46s}" + "".join(f"{f[:11]:>12s}" for f in fams))
    print("  " + "-" * 108)
    for label, d in payload["runs"].items():
        cells = ""
        for f in fams:
            v = d["per_family"].get(f)
            cells += f"{v['e_final_median']:12.3f}" if v else f"{'-':>12s}"
        print(f"  {label[:46]:46s}{cells}")

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    logs_path = args.out.replace(".json", "_logs.json")
    with open(logs_path, "w") as f:
        json.dump(logs_out, f)
    print(f"\nwrote {args.out}")
    print(f"wrote {logs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
