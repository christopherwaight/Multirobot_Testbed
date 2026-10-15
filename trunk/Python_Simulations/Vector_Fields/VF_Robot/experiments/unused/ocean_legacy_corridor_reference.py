"""
ocean_legacy_corridor_reference.py

Freezes the D and s1 paths of the CURRENTLY PUBLISHED ocean figure, the ones
produced under the pre-2026-08-03 anisotropic map from start
(34.411906N, -120.392016W), and writes them to
experiments/outputs/oecs/legacy_corridor_reference.json.

Why this exists: the map fix changes every ocean trajectory, so the new
start-point search needs a fixed definition of "the existing corridor" to score
against. That definition is the pair of paths in the paper today. Forcing
isotropic_map: false here means the reference is reproducible from the current
code at any time rather than being a snapshot that rots.

Also records the same paths under the new square map from the same start, which
is the honest before/after at a fixed start point and the baseline the search
has to beat.

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_legacy_corridor_reference.py
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C


def summarize(tag, d_path, s1_path):
    dD, fD = C.closest_approach(d_path, C.E0)
    dS, fS = C.closest_approach(s1_path, C.E0)
    print(f"  {tag}:")
    print(f"    D  end=({d_path[-1][0]:.4f},{d_path[-1][1]:.4f})  "
          f"closest to E0 {dD:.2f} km @ {fD:.0%}  len={C.path_length_km(d_path):.1f} km  "
          f"branch={C.branch_of(d_path[-1])}")
    print(f"    s1 end=({s1_path[-1][0]:.4f},{s1_path[-1][1]:.4f})  "
          f"closest to E0 {dS:.2f} km @ {fS:.0%}  len={C.path_length_km(s1_path):.1f} km  "
          f"branch={C.branch_of(s1_path[-1])}")
    print(f"    J = max(dD, dS1) = {max(dD, dS):.2f} km")
    return {
        "dD_closest_km": dD, "dD_closest_frac": fD,
        "dS1_closest_km": dS, "dS1_closest_frac": fS,
        "J_km": max(dD, dS),
        "D_end": [float(d_path[-1][0]), float(d_path[-1][1])],
        "s1_end": [float(s1_path[-1][0]), float(s1_path[-1][1])],
        "D_branch": C.branch_of(d_path[-1]), "s1_branch": C.branch_of(s1_path[-1]),
        "D_length_km": C.path_length_km(d_path), "s1_length_km": C.path_length_km(s1_path),
        "D_smoothness_rad": C.mean_turning_angle(d_path),
        "s1_smoothness_rad": C.mean_turning_angle(s1_path),
    }


def run_pair(isotropic, start):
    field, cluster, prim_d, prim_s1 = C.build_trial(isotropic_map=isotropic)
    d_path = C.run_traj(field, cluster, prim_d, *start)
    s1_path = C.run_traj(field, cluster, prim_s1, *start)
    # Reproducibility guard: a second D run from the same start must land
    # identically, or the cluster is carrying state between runs.
    d_again = C.run_traj(field, cluster, prim_d, *start)
    assert np.allclose(d_path[-1], d_again[-1]), \
        "D endpoint did not reproduce; cluster state is leaking between runs"
    return d_path, s1_path


def main():
    start = C.LEGACY_START
    print(f"Start: ({start[0]:.6f}N, {-start[1]:.6f}W)\n")

    print("Legacy map (isotropic_map=false, the published figure)...")
    legacy_d, legacy_s1 = run_pair(False, start)
    legacy_stats = summarize("legacy", legacy_d, legacy_s1)

    print("\nSquare map (isotropic_map=true, same start)...")
    square_d, square_s1 = run_pair(True, start)
    square_stats = summarize("square", square_d, square_s1)

    print(f"\n  Corridor deviation, square vs legacy, from the same start:")
    print(f"    D  mean {C.path_dev_km(legacy_d, square_d):.2f} km, "
          f"max {C.path_dev_max_km(legacy_d, square_d):.2f} km")
    print(f"    s1 mean {C.path_dev_km(legacy_s1, square_s1):.2f} km, "
          f"max {C.path_dev_max_km(legacy_s1, square_s1):.2f} km")

    C.atomic_write_json(C.LEGACY_REFERENCE_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/ocean_legacy_corridor_reference.py",
        "note": ("D and s1 centroid paths of the published ocean figure, run under "
                 "the pre-2026-08-03 anisotropic map (isotropic_map: false) from the "
                 "start published in Draft_6a. This is the definition of 'the existing "
                 "corridor' used by ocean_square_map_start_search.py. square_map_* "
                 "holds the same start's paths under the new isotropic map, which is "
                 "the do-nothing baseline the search has to beat."),
        "start": [start[0], start[1]],
        "operating_point": {
            "v_max": C.V_MAX, "sim_steps": C.SIM_STEPS, "control_gain": C.CONTROL_GAIN,
            "momentum_alpha": C.MOMENTUM_ALPHA, "stiction": C.STICTION_THRESHOLD,
            "time_warp": C.TIME_WARP, "formation": C.FORMATION_CONFIG,
        },
        "E0_target": [C.E0[0], C.E0[1]],
        "legacy_stats": legacy_stats,
        "square_map_stats": square_stats,
        "square_vs_legacy_dev_km": {
            "D_mean": C.path_dev_km(legacy_d, square_d),
            "D_max": C.path_dev_max_km(legacy_d, square_d),
            "s1_mean": C.path_dev_km(legacy_s1, square_s1),
            "s1_max": C.path_dev_max_km(legacy_s1, square_s1),
        },
        "d_path": legacy_d.tolist(),
        "s1_path": legacy_s1.tolist(),
        "square_map_d_path": square_d.tolist(),
        "square_map_s1_path": square_s1.tolist(),
    })
    print(f"\nSaved: {C.LEGACY_REFERENCE_JSON}")


if __name__ == "__main__":
    main()
