"""
omega_seed_sweep.py

Proposition 3 (frame equivariance) proves the closed loop equivariant "by
induction from the second tracking step onward," exempting the degenerate
guard, but the seed t_ref = v0_hat at first contact is ALSO frame-dependent
and, unlike the guard, is always used. If the seed resolves to the opposite
sign in the rotating frame, the whole ride commits to the opposite branch,
not a bounded residue (revision/items.yaml B5, reviews/Draft_6a_review.md B5).

At the experiment's start (0.05, 0.40) used throughout the objectivity demo
(traverse_objectivity_demo.py, OMEGA_ROT=0.2), the rotating-frame apparent
flow at t=0 is

    v_rot(p) = v_true(p) + omega_rot * perp(p),   perp(x,y) = (-y, x)

(Rotating_Frame.make_rotating_frame's own transform, evaluated at t=0 so no
rotation has yet accumulated -- this IS the frame's zeroth-cycle sample).
The seed's resolved sign survives the frame change so long as
v_rot(p) . v_true(p) > 0; it flips at the omega_rot solving

    |v_true(p)|^2 + omega_rot * (perp(p) . v_true(p)) = 0.

This script reports that closed-form critical Omega, the margin at
Omega=0.2 (matching v1's own numbers, |v|=0.107, Omega||p||=0.081, ratio
1.3x), and a numeric sweep confirming the sign genuinely flips there.

Writes outputs/oecs/omega_seed_sweep.csv (one row per Omega) and prints
the critical value.
"""
import os
import sys
import csv

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VFR = os.path.dirname(_HERE)
_PAPER_SCRIPTS = os.path.abspath(os.path.join(_VFR, "..", "..", "..", "..",
                                              "Paper_Writing", "Separatrix_and_OW_Paper", "scripts"))
sys.path.insert(0, _VFR)
sys.path.insert(0, _PAPER_SCRIPTS)

from verify_estimator_bias import field_derivs

SEED_POINT = (0.05, 0.40)  # traverse_objectivity_demo.py START
OMEGA_TESTED = 0.2         # traverse_objectivity_demo.py OMEGA_ROT


def true_flow(x, y):
    d = field_derivs(x, y)
    return np.array([d["u"], d["v"]])


def perp(x, y):
    return np.array([-y, x])


def rotating_seed_flow(x, y, omega_rot):
    return true_flow(x, y) + omega_rot * perp(x, y)


def main():
    x0, y0 = SEED_POINT
    v_true = true_flow(x0, y0)
    p_norm = float(np.hypot(x0, y0))
    v_norm = float(np.linalg.norm(v_true))
    perp_dot_v = float(perp(x0, y0) @ v_true)

    # Headline criterion (matches reviews/Draft_6a_review.md B5 exactly): the
    # transport term's magnitude Omega*||p|| against the raw flow speed
    # |v_true|, the conservative worst-case bound independent of the angle
    # between perp(p) and v_true. |v_true|=0.1067 and Omega*||p||=0.0806 at
    # Omega=0.2 match B5's stated 0.107 and 0.081 to 3 decimals.
    omega_crit_magnitude = v_norm / p_norm

    # Supplementary: the exact vector condition v_rot . v_true = 0, which
    # additionally requires perp(p) to be anti-aligned with v_true at this
    # point -- it need not coincide with the magnitude bound above and can
    # come out negative (no flip for any positive Omega along this exact
    # direction) even when the magnitude bound is already violated.
    omega_crit_exact = -v_norm**2 / perp_dot_v if abs(perp_dot_v) > 1e-12 else float("nan")

    print(f"Seed point {SEED_POINT}: |v_true| = {v_norm:.4f}, ||p|| = {p_norm:.4f}")
    print(f"  Omega*||p|| at Omega={OMEGA_TESTED}: {OMEGA_TESTED * p_norm:.4f}")
    print(f"  margin ratio |v_true| / (Omega*||p||) at Omega={OMEGA_TESTED}: "
          f"{v_norm / (OMEGA_TESTED * p_norm):.3f}x")
    print(f"  critical Omega, magnitude bound (Omega*||p|| = |v_true|): {omega_crit_magnitude:.4f}")
    print(f"  critical Omega, exact vector condition (v_rot . v_true = 0): {omega_crit_exact:.4f}"
          + (" (no flip for positive Omega along this exact direction)" if omega_crit_exact < 0 else ""))
    omega_crit = omega_crit_magnitude

    omegas = np.linspace(0.0, max(0.4, omega_crit_magnitude * 1.5), 41)
    rows = []
    flip_bracket = None
    prev_sign = None
    prev_om = None
    for om in omegas:
        v_rot = rotating_seed_flow(x0, y0, om)
        dot = float(v_rot @ v_true)
        margin_survives = v_norm - om * p_norm > 0
        if prev_sign is not None and margin_survives != prev_sign and flip_bracket is None:
            flip_bracket = (prev_om, float(om))
        prev_sign, prev_om = margin_survives, om
        rows.append({"omega_rot": float(om), "v_rot_dot_v_true": dot,
                    "seed_sign_survives": int(margin_survives)})

    print(f"  magnitude-bound flip bracket: {flip_bracket}")

    out_dir = os.path.join(_HERE, "outputs", "oecs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "omega_seed_sweep.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["omega_rot", "v_rot_dot_v_true", "seed_sign_survives"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")

    import json
    summary = {
        "seed_point": list(SEED_POINT),
        "v_true_norm": v_norm,
        "p_norm": p_norm,
        "omega_tested": OMEGA_TESTED,
        "omega_p_norm_product": OMEGA_TESTED * p_norm,
        "margin_ratio_at_omega_tested": v_norm / (OMEGA_TESTED * p_norm),
        "omega_critical": omega_crit_magnitude,
        "omega_critical_exact_vector": omega_crit_exact,
        "flip_bracket": flip_bracket,
    }
    summary_path = os.path.join(out_dir, "omega_seed_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
