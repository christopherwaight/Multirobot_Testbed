"""
saddle_tune.py
==============
Tune the four gains of `scalar_newton_with_rotation` for saddle-point finding
on the hyperbolic saddle field (phi = x^2 - y^2, saddle at origin).

Method
------
* N independent simulated-annealing chains, run in parallel via multiprocessing.
* Within each chain: random initial gains -> propose perturbed candidate ->
  evaluate on a small batch of random starts -> accept if mean batch cost is
  better. Perturbation scale decays exponentially over the chain.
* After all chains finish, every chain's best gains are re-evaluated on a
  larger FIXED validation set (same random starts for every chain), and the
  global winner is selected by validation cost. This guards against a chain
  "getting lucky" on easy starts during search.

Outputs
-------
* Console: top-5 gain sets + suggested values at the end.
* tuning_results/tuning_log.json   -- full record of every candidate
* tuning_results/best_gains.json   -- just the winning gains

Usage
-----
Place alongside `main_omni.py` (i.e. inside the scripts/ directory of the
project, so that `os.path.dirname(os.path.dirname(__file__))` points to the
project root). Then:

    python saddle_tune.py

All knobs to edit are in the CONFIGURATION block below.
"""

import sys
import os
import json
import time
import contextlib
from multiprocessing import Pool, cpu_count

# Project root setup -- mirrors main_omni.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalScalarField
from src.fields.environments.Scalar_Fields import hyperbolic_saddle
import src.control.quad_primitives as qcp


# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Search ranges (low, high) for each gain ---
GAIN_RANGES = {
    "translation_gain": (0.1, 1.0),
    "rotation_gain":    (0.1, 1.0),
    "max_speed":        (0.1, 1.0),
    "max_omega":        (0.1, 1.0),
}

# --- Simulated annealing schedule ---
N_CHAINS                  = 8     # Independent SA chains (also parallelism)
N_CANDIDATES_PER_CHAIN    = 100    # Proposals per chain
N_SAMPLES_PER_CANDIDATE   = 4      # Random starts averaged per candidate

# Perturbation scale as fraction of each gain's range; anneals from init -> final
SIGMA_FRAC_INIT  = 0.30
SIGMA_FRAC_FINAL = 0.02

# --- Final validation pass (run on every chain's winner) ---
N_VALIDATION_SAMPLES = 30          # Fixed seeds, same for every chain
VALIDATION_SEED_BASE = 999_000     # Just needs to not collide with chain seeds

# --- Cost function ---
SUCCESS_DIST_THRESHOLD = 0.25      # meters: within this = "converged"
FAILURE_PENALTY        = 1.0       # added to non-converged runs
OSCILLATION_WEIGHT     = 1.0       # cost += W * std(distance over last quarter)

# --- Simulation ---
SIM_STEPS        = 2500            # 200 s simulated at dt = 0.1 s
FORMATION_CONFIG = "config/formations/quad_square.yaml"
TARGET           = np.array([0.0, 0.0])  # Hyperbolic saddle is at origin

# --- Parallelism / output ---
N_WORKERS  = min(N_CHAINS, max(1, cpu_count() - 1))
OUTPUT_DIR = "tuning_results"
MASTER_SEED = None                   # Set to None for nondeterministic runs


# ============================================================================
# Low-level helpers
# ============================================================================

@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Mute the noisy prints inside QuadCluster.__init__ and the primitive."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _make_primitive(gains):
    """Wrap scalar_newton_with_rotation with a fixed set of gains."""
    tg, rg = gains["translation_gain"], gains["rotation_gain"]
    ms, mo = gains["max_speed"],        gains["max_omega"]

    def primitive(cluster):
        return qcp.scalar_newton_with_rotation(
            cluster,
            translation_gain=tg,
            rotation_gain=rg,
            max_speed=ms,
            max_omega=mo,
        )
    return primitive


def run_one_simulation(gains, run_seed):
    """
    Run one saddle-finding simulation with the given gains. The start position
    and orientation are randomized inside QuadCluster.__init__ using numpy.random,
    so we seed numpy here for reproducibility.

    Returns a dict with the centroid trajectory and start/end points.
    """
    #np.random.seed(run_seed)
    np.random.seed(run_seed % (2**32))

    try:
        with _suppress_stdout_stderr():
            field = AnalyticalScalarField(hyperbolic_saddle)
            cluster = QuadCluster(FORMATION_CONFIG, field)
            primitive = _make_primitive(gains)
            for _ in range(SIM_STEPS):
                cluster.move(primitive)
        traj = cluster.get_center_history()  # (T, 2) numpy array
        return {
            "ok": True,
            "initial_pos": traj[0].tolist(),
            "final_pos":   traj[-1].tolist(),
            "trajectory":  traj,
        }
    except Exception as exc:
        # Treat any sim crash as a worst-case failure -- don't kill the tuner.
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "initial_pos": [None, None],
            "final_pos":   [10.0, 10.0],
            "trajectory":  np.array([[10.0, 10.0]]),
        }


def compute_cost(traj_data):
    """
    Convert one trajectory into a scalar cost.

    Converged (final dist <= threshold):
        cost = final_dist + OSCILLATION_WEIGHT * std_of_distance_in_last_quarter
    Failed:
        cost = FAILURE_PENALTY + final_dist
    """
    final_pos  = np.array(traj_data["final_pos"])
    trajectory = traj_data["trajectory"]

    final_dist = float(np.linalg.norm(final_pos - TARGET))

    last_q = trajectory[3 * len(trajectory) // 4:]
    dists  = np.linalg.norm(last_q - TARGET, axis=1)
    oscillation = float(np.std(dists))

    converged = final_dist <= SUCCESS_DIST_THRESHOLD
    if converged:
        cost = final_dist + OSCILLATION_WEIGHT * oscillation
    else:
        cost = FAILURE_PENALTY + final_dist

    return {
        "cost":        float(cost),
        "final_dist":  final_dist,
        "oscillation": oscillation,
        "converged":   bool(converged),
    }


def evaluate_candidate(gains, n_samples, seed_base):
    """
    Evaluate a candidate over n_samples random starts seeded deterministically
    from seed_base. Returns mean cost + per-run records (no full trajectories).
    """
    runs = []
    for k in range(n_samples):
        traj = run_one_simulation(gains, run_seed=seed_base + k)
        cost_info = compute_cost(traj)
        runs.append({
            "initial_pos": traj["initial_pos"],
            "final_pos":   traj["final_pos"],
            **cost_info,
        })
    return {
        "mean_cost":   float(np.mean([r["cost"] for r in runs])),
        "n_converged": int(sum(r["converged"] for r in runs)),
        "runs":        runs,
    }


# ============================================================================
# Simulated annealing chain
# ============================================================================

def _sample_random_gains(rng):
    return {n: float(rng.uniform(lo, hi)) for n, (lo, hi) in GAIN_RANGES.items()}


def _perturb_gains(gains, sigma_frac, rng):
    new = {}
    for name, (lo, hi) in GAIN_RANGES.items():
        sigma = sigma_frac * (hi - lo)
        new[name] = float(np.clip(gains[name] + rng.normal(0, sigma), lo, hi))
    return new


def run_chain(chain_id, n_candidates, seed):
    """A single annealed random-search chain."""
    rng = np.random.default_rng(seed)

    # Initial point
    best_gains = _sample_random_gains(rng)
    best_eval  = evaluate_candidate(best_gains,
                                    N_SAMPLES_PER_CANDIDATE,
                                    seed_base=seed * 10_000)
    log = [{"iter": 0, "gains": best_gains, "accepted": True, **best_eval}]

    # Progress markers every ~10%
    progress_every = max(1, n_candidates // 10)

    for t in range(1, n_candidates + 1):
        frac  = t / n_candidates
        sigma = SIGMA_FRAC_INIT * (SIGMA_FRAC_FINAL / SIGMA_FRAC_INIT) ** frac

        candidate = _perturb_gains(best_gains, sigma, rng)
        cand_eval = evaluate_candidate(
            candidate,
            N_SAMPLES_PER_CANDIDATE,
            seed_base=seed * 10_000 + t * 100,
        )

        accepted = cand_eval["mean_cost"] < best_eval["mean_cost"]
        if accepted:
            best_gains, best_eval = candidate, cand_eval

        log.append({
            "iter": t,
            "gains": candidate,
            "sigma_frac": float(sigma),
            "accepted": accepted,
            **cand_eval,
        })

        if t % progress_every == 0:
            print(f"[chain {chain_id:2d}] iter {t:4d}/{n_candidates}  "
                  f"best={best_eval['mean_cost']:.4f}  "
                  f"conv={best_eval['n_converged']}/{N_SAMPLES_PER_CANDIDATE}  "
                  f"sigma={sigma:.3f}",
                  flush=True)

    return {
        "chain_id":         chain_id,
        "seed":             int(seed),
        "best_gains":       best_gains,
        "best_mean_cost":   best_eval["mean_cost"],
        "best_n_converged": best_eval["n_converged"],
        "log":              log,
    }


def _chain_worker(args):
    chain_id, n_candidates, seed = args
    t0 = time.time()
    result = run_chain(chain_id, n_candidates, seed)
    result["elapsed_s"] = time.time() - t0
    print(f"[chain {chain_id:2d}] DONE in {result['elapsed_s']/60:5.2f} min  "
          f"best_cost={result['best_mean_cost']:.4f}",
          flush=True)
    return result


def _validation_worker(args):
    chain_id, gains = args
    val = evaluate_candidate(gains, N_VALIDATION_SAMPLES, seed_base=VALIDATION_SEED_BASE)
    print(f"[val   {chain_id:2d}] cost={val['mean_cost']:.4f}  "
          f"conv={val['n_converged']}/{N_VALIDATION_SAMPLES}",
          flush=True)
    return chain_id, val


# ============================================================================
# Main
# ============================================================================

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    total_sims = N_CHAINS * (N_CANDIDATES_PER_CHAIN + 1) * N_SAMPLES_PER_CANDIDATE \
                 + N_CHAINS * N_VALIDATION_SAMPLES

    print("=" * 72)
    print(" SADDLE-POINT GAIN TUNING")
    print("=" * 72)
    print(f" Field:               hyperbolic_saddle (target = origin)")
    print(f" Gains tuned:         {list(GAIN_RANGES)}")
    print(f" Ranges:")
    for n, (lo, hi) in GAIN_RANGES.items():
        print(f"     {n:<20} [{lo}, {hi}]")
    print(f" Chains:              {N_CHAINS}")
    print(f" Candidates / chain:  {N_CANDIDATES_PER_CHAIN}")
    print(f" Samples / candidate: {N_SAMPLES_PER_CANDIDATE}")
    print(f" Validation samples:  {N_VALIDATION_SAMPLES} (fixed seeds, same for all chains)")
    print(f" Total simulations:   {total_sims:,}")
    print(f" Parallel workers:    {N_WORKERS}")
    print(f" Sim steps per run:   {SIM_STEPS}")
    print("=" * 72)
    print()

    # Independent seeds for each chain (reproducible if MASTER_SEED is set)
    master_rng  = np.random.default_rng(MASTER_SEED)
    chain_seeds = master_rng.integers(0, 2**31 - 1, size=N_CHAINS)
    tasks = [(i, N_CANDIDATES_PER_CHAIN, int(s)) for i, s in enumerate(chain_seeds)]

    # --- Phase 1: run SA chains in parallel ---
    print(">>> Phase 1: simulated annealing chains")
    chain_results = []
    with Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(_chain_worker, tasks):
            chain_results.append(result)

    chain_results.sort(key=lambda r: r["chain_id"])
    phase1_elapsed = time.time() - t_start
    print(f"\n>>> Phase 1 complete in {phase1_elapsed/60:.2f} min\n")

    # --- Phase 2: re-evaluate every chain's winner on a fixed validation set ---
    print(">>> Phase 2: final validation on fixed-seed test set")
    val_tasks = [(r["chain_id"], r["best_gains"]) for r in chain_results]
    val_lookup = {}
    with Pool(N_WORKERS) as pool:
        for chain_id, val in pool.imap_unordered(_validation_worker, val_tasks):
            val_lookup[chain_id] = val

    for r in chain_results:
        v = val_lookup[r["chain_id"]]
        r["val_mean_cost"]   = v["mean_cost"]
        r["val_n_converged"] = v["n_converged"]
        r["val_runs"]        = v["runs"]

    # Re-rank by validation cost
    chain_results.sort(key=lambda r: r["val_mean_cost"])

    elapsed = time.time() - t_start

    # =====================================================================
    # Final report
    # =====================================================================
    print("\n" + "=" * 72)
    print(" RESULTS")
    print("=" * 72)
    print(f" Total time: {elapsed/60:.2f} min")
    print()
    top_k = min(5, N_CHAINS)
    print(f" Top-{top_k} gain sets (ranked by validation cost):")
    print("-" * 72)
    header_names = "".join(f"{n[:13]:<15}" for n in GAIN_RANGES)
    print(f" {'rank':<5}{'val_cost':<11}{'val_conv':<11}{'search_cost':<13}{header_names}")
    print("-" * 72)
    for rank, r in enumerate(chain_results[:top_k], 1):
        gains_str = "".join(f"{r['best_gains'][n]:<15.4f}" for n in GAIN_RANGES)
        print(f" {rank:<5}{r['val_mean_cost']:<11.4f}"
              f"{r['val_n_converged']}/{N_VALIDATION_SAMPLES:<8}"
              f"{r['best_mean_cost']:<13.4f}{gains_str}")

    print()
    print(f" Consensus across top {top_k} chains:")
    print("-" * 72)
    for name in GAIN_RANGES:
        vals = [r["best_gains"][name] for r in chain_results[:top_k]]
        print(f"  {name:<22} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"range=[{min(vals):.4f}, {max(vals):.4f}]")

    print()
    print("=" * 72)
    print(" SUGGESTED GAIN VALUES (best chain by validation cost):")
    print("=" * 72)
    best = chain_results[0]
    for name in GAIN_RANGES:
        print(f"   {name:<22} = {best['best_gains'][name]:.4f}")
    print(f"   val_mean_cost          = {best['val_mean_cost']:.4f}")
    print(f"   val_convergence_rate   = {best['val_n_converged']}/{N_VALIDATION_SAMPLES} "
          f"({100.0 * best['val_n_converged'] / N_VALIDATION_SAMPLES:.0f}%)")
    print("=" * 72)

    # ---- Save full log ----
    log_path = os.path.join(OUTPUT_DIR, "tuning_log.json")
    payload = {
        "config": {
            "gain_ranges":             GAIN_RANGES,
            "n_chains":                N_CHAINS,
            "n_candidates_per_chain":  N_CANDIDATES_PER_CHAIN,
            "n_samples_per_candidate": N_SAMPLES_PER_CANDIDATE,
            "sigma_frac_init":         SIGMA_FRAC_INIT,
            "sigma_frac_final":        SIGMA_FRAC_FINAL,
            "n_validation_samples":    N_VALIDATION_SAMPLES,
            "validation_seed_base":    VALIDATION_SEED_BASE,
            "success_dist_threshold":  SUCCESS_DIST_THRESHOLD,
            "failure_penalty":         FAILURE_PENALTY,
            "oscillation_weight":      OSCILLATION_WEIGHT,
            "sim_steps":               SIM_STEPS,
            "formation_config":        FORMATION_CONFIG,
            "field":                   "hyperbolic_saddle",
            "master_seed":             MASTER_SEED,
        },
        "chains":    chain_results,
        "elapsed_s": elapsed,
    }
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2, default=_to_jsonable)
    print(f"\n Full log saved to:   {log_path}")

    # ---- Save just the winner ----
    best_path = os.path.join(OUTPUT_DIR, "best_gains.json")
    summary = {
        "best_gains":     best["best_gains"],
        "val_mean_cost":  best["val_mean_cost"],
        "val_n_converged": best["val_n_converged"],
        "val_total":      N_VALIDATION_SAMPLES,
        "top_5_chains": [
            {
                "gains":         r["best_gains"],
                "val_mean_cost": r["val_mean_cost"],
                "val_converged": r["val_n_converged"],
            }
            for r in chain_results[:5]
        ],
    }
    with open(best_path, "w") as f:
        json.dump(summary, f, indent=2, default=_to_jsonable)
    print(f" Best gains saved to: {best_path}")


if __name__ == "__main__":
    main()