"""
Hand-derived baselines: the rotating-Hessian law from saddle_point_4_robot.ipynb.

The law, unchanged from the notebook:

    Estimate (H, g) from the four readings by the C(4,3) plane-fit method.
    Take the saddle-seeking Newton step  d = -H^-1 g.
    Translate along d.
    Servo the formation angle onto the heading of d.

The notebook's own summary table, for its three rotation modes on the
log-sum-exp field with no actuator dynamics:

    none          final distance 9.81   diverges
    single (pi)   final distance 0.20   converges
    quad (pi/2)   final distance 0.81   limit cycle

All three are reproduced here, and then run through the real plant.

Both baselines drive the SAME QuadSaddleEnv that PPO trains against, through
the same action interface, reward, and termination logic.  The only difference
between a baseline run and a policy run is which function produces the action,
which is the only way the comparison means anything.

The controller is given the robot positions and the four readings.  It is not
given the true saddle location; that is used solely for scoring.
"""
import numpy as np

from estimator import plane_fit_estimate, newton_step, formation_angle
from quad_saddle_env import (QuadSaddleEnv, V_MAX_CMD, OMEGA_MAX,
                             R_MIN, R_MAX)


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------

def _radius_action(r0):
    """Inverse of the env's action-to-radius map, so size is held at r0."""
    return 2.0 * (np.clip(r0, R_MIN, R_MAX) - R_MIN) / (R_MAX - R_MIN) - 1.0


def rotating_hessian(mode="single", k_trans=0.3626, k_rot=1.0, r0=0.15,
                     max_speed=0.3513, max_omega=1.0):
    """The notebook law as an env controller.

    Args:
        mode:      'single' wraps the angle error to [-pi, pi] (the mode the
                   notebook found converges), 'quad' wraps to [-pi/4, pi/4]
                   exploiting the square's 4-fold symmetry (limit-cycles),
                   'none' disables rotation entirely (diverges).
        k_trans:   translation gain, m/s per unit Newton step.  Default is the
                   value baked into src/control/quad_primitives.py.
        k_rot:     rotation gain, rad/s per radian of angle error.
        r0:        fixed ring radius; the notebook law has no size control.
        max_speed: translation clamp, m/s.
        max_omega: rotation clamp, rad/s.
    """
    a3 = _radius_action(r0)

    def controller(env):
        xy = env._robot_xy()
        z = env._readings(xy)
        H, g = plane_fit_estimate(xy, z)
        d = newton_step(H, g)

        n = np.linalg.norm(d)
        if n > 1e-10:
            speed = np.clip(k_trans * n, 0.0, max_speed)
            vx, vy = speed * d[0] / n, speed * d[1] / n
        else:
            vx = vy = 0.0

        if mode == "none":
            om = 0.0
        else:
            target = np.arctan2(d[1], d[0])
            raw = target - formation_angle(xy)
            if mode == "single":
                err = np.arctan2(np.sin(raw), np.cos(raw))
            elif mode == "quad":
                err = (raw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
            else:
                raise ValueError(f"unknown rotation mode {mode!r}")
            om = float(np.clip(k_rot * err, -max_omega, max_omega))

        return np.array([vx / V_MAX_CMD, vy / V_MAX_CMD,
                         om / OMEGA_MAX, a3], dtype=float)

    controller.label = (f"rot-hessian[{mode}] k_rot={k_rot:g} "
                        f"k_trans={k_trans:g} r0={r0:g}")
    return controller


def handover(primary, fallback, patience=120, tol=0.30):
    """Run `primary`; hand over to `fallback` once it has clearly stalled.

    Motivation, measured on 1000 paired fields: the tuned analytic law and a
    from-scratch PPO policy score 66.5% and 64.7%, a statistical tie, but they
    are COMPLEMENTARY rather than redundant.  The analytic law alone succeeds
    on 11.8% of fields, the policy alone on 10.0%, and either-succeeds is
    76.5%.  So ~10 points sit on the table for any rule that can tell, online,
    which one is failing.

    The stall test uses only quantities a controller legitimately has: the
    spread of its own four readings as a proxy for local gradient, and whether
    the formation has stopped making progress.  It never looks at the true
    saddle.  Concretely: track the best (smallest) reading-spread seen so far;
    if `patience` steps pass with no improvement and the cluster is not already
    close to converged, switch permanently to `fallback`.

    Args:
        primary:  controller to start with
        fallback: controller to hand over to
        patience: steps without improvement before switching
        tol:      spread below which the primary is considered to be working,
                  which suppresses handover
    """
    state = {}

    def controller(env):
        if state.get("switched"):
            return fallback(env)
        if not state:
            state.update(best=np.inf, since=0, switched=False)

        xy = env._robot_xy()
        z = env._readings(xy)
        spread = float(np.linalg.norm(z - z.mean()))
        # Scale-free progress signal: reading spread shrinks as the cluster
        # approaches any critical point, so a flat spread means "stuck".
        if spread < state["best"] * 0.98:
            state["best"] = spread
            state["since"] = 0
        else:
            state["since"] += 1

        if state["since"] > patience and spread > tol * state["best"]:
            state["switched"] = True
            return fallback(env)
        return primary(env)

    controller.label = f"handover({getattr(primary,'label','?')} -> " \
                       f"{getattr(fallback,'label','?')})"
    return controller


def zero_controller():
    """Do nothing. Sanity floor for the metrics."""
    def controller(env):
        return np.array([0.0, 0.0, 0.0, _radius_action(0.15)])
    controller.label = "do-nothing"
    return controller


# --------------------------------------------------------------------------
# Rollout harness
# --------------------------------------------------------------------------

def run_episode(env, controller, seed):
    """One episode. Returns a metrics dict plus the env's per-step log."""
    env.reset(seed=seed)
    e_hist = []
    last_info = {}
    while True:
        _, _, term, trunc, info = env.step(controller(env))
        if "e" in info:
            e_hist.append(info["e"])
        if term or trunc:
            last_info = info
            break
    e_hist = np.asarray(e_hist) if e_hist else np.array([np.nan])
    log = env.episode_log
    om_ach = np.asarray(log["omega_ach"]) if log["omega_ach"] else np.array([0.0])
    om_cmd = np.asarray(log["omega_cmd"]) if log["omega_cmd"] else np.array([0.0])
    return dict(
        family=env.fld.family,
        outcome=last_info.get("outcome", "timeout"),
        success=bool(last_info.get("is_success", False)),
        time_in_tol=float(last_info.get("time_in_tol", 0.0)),
        e_final=float(e_hist[-1]),
        e_min=float(np.nanmin(e_hist)),
        steps=len(e_hist),
        r_final=float(log["radius"][-1]) if log["radius"] else np.nan,
        omega_cmd_abs=float(np.mean(np.abs(om_cmd))),
        omega_ach_abs=float(np.mean(np.abs(om_ach))),
        omega_sat_frac=float(np.mean(np.abs(om_cmd) > 0.98 * OMEGA_MAX)),
        frozen_frac=float(np.mean(np.asarray(log["robot_speed"]) < 1e-9))
        if log["robot_speed"] else 0.0,
    )


def evaluate(controller, n_episodes=30, plant="full", families=None,
             seed0=10_000, **env_kwargs):
    """Run a controller over n_episodes fields. Same seeds give same fields."""
    env = QuadSaddleEnv(families=families, ideal_plant=(plant == "ideal"),
                        **env_kwargs)
    rows = [run_episode(env, controller, seed0 + i) for i in range(n_episodes)]
    env.close()
    return rows


def summarize(rows):
    ef = np.array([r["e_final"] for r in rows])
    em = np.array([r["e_min"] for r in rows])
    return dict(
        n=len(rows),
        success_rate=float(np.mean([r["success"] for r in rows])),
        time_in_tol=float(np.mean([r.get("time_in_tol", 0.0) for r in rows])),
        e_final_median=float(np.median(ef)),
        e_final_p25=float(np.percentile(ef, 25)),
        e_final_p75=float(np.percentile(ef, 75)),
        e_min_median=float(np.median(em)),
        omega_cmd_abs=float(np.mean([r["omega_cmd_abs"] for r in rows])),
        omega_sat_frac=float(np.mean([r["omega_sat_frac"] for r in rows])),
        frozen_frac=float(np.mean([r["frozen_frac"] for r in rows])),
        steps_mean=float(np.mean([r["steps"] for r in rows])),
    )


def _print_summary(label, s):
    print(f"  {label:44s} succ={s['success_rate']:5.1%}  "
          f"in_tol={s.get('time_in_tol', 0.0):5.1%}  "
          f"e_final={s['e_final_median']:6.3f} "
          f"[{s['e_final_p25']:.3f},{s['e_final_p75']:.3f}]  "
          f"e_min={s['e_min_median']:6.3f}  "
          f"|om|={s['omega_cmd_abs']:.2f} sat={s['omega_sat_frac']:4.0%}  "
          f"frozen={s['frozen_frac']:4.0%}")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def experiment_modes(n_episodes, families):
    """Reproduce the notebook's three rotation modes, both plants."""
    print("=" * 110)
    print("ROTATION MODES, notebook gains (k_trans=0.3626, k_rot=1.0, r0=0.15)")
    print("=" * 110)
    out = {}
    for plant in ("ideal", "full"):
        print(f"\n  plant = {plant}"
              f"{'   (alpha=0, no stiction, no cap)' if plant == 'ideal' else '   (alpha=0.7, stiction=0.025, cap=0.3)'}")
        for mode in ("none", "single", "quad"):
            ctl = rotating_hessian(mode=mode)
            rows = evaluate(ctl, n_episodes, plant, families)
            s = summarize(rows)
            out[(plant, mode)] = s
            _print_summary(f"mode={mode}", s)
    print()
    print("  Read: 'single' is the mode the notebook found converges without")
    print("  dynamics.  Compare its two plant rows to size the dynamics gap.")
    return out


def experiment_sweep(n_episodes, families, plant="full"):
    """Small gain sweep, so the baseline is reported at its best cell.

    The default gains in src/control/quad_primitives.py were never retuned for
    the velocity form of this law.  Without a sweep, any PPO win confounds
    'dynamics broke the law' with 'the gains were wrong', so the baseline is
    reported at its best cell rather than its default one.
    """
    k_rots = [0.5, 1.0, 2.0, 4.0]
    k_transs = [0.2, 0.3626, 0.8, 1.5]
    r0s = [0.10, 0.15, 0.25]

    print("=" * 110)
    print(f"GAIN SWEEP, mode=single, plant={plant}, "
          f"{len(k_rots)}x{len(k_transs)}x{len(r0s)} cells "
          f"x {n_episodes} fields")
    print("=" * 110)

    results = []
    for r0 in r0s:
        omega_ceiling = 0.3 / r0
        print(f"\n  r0={r0:.2f}  (actuator ceiling omega_max = 0.3/r0 = "
              f"{omega_ceiling:.2f} rad/s)")
        print(f"    {'k_rot':>7s} {'k_trans':>8s} {'succ':>6s} "
              f"{'e_final med':>12s} {'e_min med':>10s} {'sat':>5s}")
        for k_rot in k_rots:
            for k_trans in k_transs:
                ctl = rotating_hessian(mode="single", k_rot=k_rot,
                                       k_trans=k_trans, r0=r0,
                                       max_omega=min(OMEGA_MAX, omega_ceiling))
                s = summarize(evaluate(ctl, n_episodes, plant, families))
                s.update(k_rot=k_rot, k_trans=k_trans, r0=r0)
                results.append(s)
                print(f"    {k_rot:7.2f} {k_trans:8.4f} "
                      f"{s['success_rate']:6.1%} {s['e_final_median']:12.3f} "
                      f"{s['e_min_median']:10.3f} {s['omega_sat_frac']:5.0%}")

    best = min(results, key=lambda s: s["e_final_median"])
    print()
    print("-" * 110)
    print(f"  BEST CELL: k_rot={best['k_rot']}, k_trans={best['k_trans']}, "
          f"r0={best['r0']}")
    _print_summary("best", best)
    return results, best


def main():
    import argparse
    import json
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plant", choices=["full", "ideal", "both"], default="both")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--sweep", action="store_true", help="run the gain sweep")
    p.add_argument("--sweep-episodes", type=int, default=12)
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--out", default="outputs/baselines.json")
    args = p.parse_args()

    payload = {}

    if args.plant == "both":
        modes = experiment_modes(args.episodes, args.families)
        payload["modes"] = {f"{k[0]}/{k[1]}": v for k, v in modes.items()}
    else:
        print("=" * 110)
        print(f"ROTATION MODES, plant={args.plant}")
        print("=" * 110)
        d = {}
        for mode in ("none", "single", "quad"):
            s = summarize(evaluate(rotating_hessian(mode=mode),
                                   args.episodes, args.plant, args.families))
            d[f"{args.plant}/{mode}"] = s
            _print_summary(f"mode={mode}", s)
        payload["modes"] = d

    if args.sweep:
        print()
        results, best = experiment_sweep(args.sweep_episodes, args.families,
                                         plant="full")
        payload["sweep"] = results
        payload["sweep_best"] = best

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
