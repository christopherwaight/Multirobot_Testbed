"""
ocean_rank_params.py

Ranks the operating-point search (ocean_param_search.py) and prints the
tradeoff it exposes.

Hugging the ridge and completing the transit pull against each other. The
tightest, smoothest tracking comes from a low speed cap, and a low speed cap
covers less ground inside the fixed 168-step record, so those runs do not reach
the middle-island landfall. Ranking on the ridge score alone therefore returns
runs that trace the structure beautifully and stop halfway down the channel.

So three tables are printed:

  ridge         best ridge score outright, whatever it costs elsewhere
  ridge+reach   best ridge score among runs whose s1 also closes on the
                landfall (J <= REACH_J_KM), which is the paper's existing claim
  frontier      the Pareto set of (ridge score, J), so the shape of the
                tradeoff is visible rather than asserted

Running:
    cd trunk/Python_Simulations/Vector_Fields/VF_Robot
    venv/bin/python3 experiments/ocean_rank_params.py
    venv/bin/python3 experiments/ocean_rank_params.py --reach-j-km 12
"""
import os
import sys
import csv
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _ocean_run_common as C

CSV_PATH = os.path.join(C.OUT_DIR, "ocean_param_search.csv")
OUT_JSON = os.path.join(C.OUT_DIR, "ocean_param_candidates.json")
TOP_N = 6
DEDUP_KM = 1.0

HDR = (f"{'score':>6} {'ridge':>6} {'turn':>6} {'J':>6} {'D_cls':>6} {'s1_cls':>7} "
       f"{'start':<22} {'rho_km':>6} {'hdg':>5} {'k':>5} {'c_max':>6}")


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            for k in r:
                if k not in ("D_branch", "s1_branch", "gated"):
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def line(r):
    return (f"{r['score']:>6.3f} {r['ridge_mean_km']:>6.2f} {r['turn_mean_rad']:>6.3f} "
            f"{r['J_km']:>6.1f} {r['dD_closest_km']:>6.2f} {r['dS1_closest_km']:>7.2f} "
            f"{r['start_lat']:.4f},{r['start_lon']:.4f}  "
            f"{r['rho_km']:>6.2f} {r['heading_deg']:>5.0f} {r['gain']:>5.1f} "
            f"{r['v_max']:>6.3f}")


def dedup(rows, n=TOP_N):
    out = []
    for r in rows:
        p = (r["start_lat"], r["start_lon"])
        if all(C.km(p, (o["start_lat"], o["start_lon"])) >= DEDUP_KM for o in out):
            out.append(r)
        if len(out) >= n:
            break
    return out


def pareto(rows):
    """Non-dominated set on (score, J), both minimised."""
    front = []
    for r in sorted(rows, key=lambda x: x["score"]):
        if all(r["J_km"] < o["J_km"] for o in front):
            front.append(r)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach-j-km", type=float, default=16.0,
                    help="J bar for counting as reaching the landfall; the old "
                         "published run was 14.05 km")
    args = ap.parse_args()

    rows = load()
    ok = [r for r in rows if not r["gated"]]
    print(f"{len(rows)} evaluations, {len(ok)} ungated\n")
    if not ok:
        print("Nothing ungated yet.")
        return

    by_score = sorted(ok, key=lambda r: r["score"])
    reach = [r for r in ok if r["J_km"] <= args.reach_j_km]

    print("=== best ridge score, unconstrained ===")
    print(HDR)
    best_ridge = dedup(by_score)
    for r in best_ridge:
        print(line(r))

    print(f"\n=== best ridge score among runs with J <= {args.reach_j_km} km "
          f"({len(reach)} of {len(ok)} qualify) ===")
    print(HDR)
    best_reach = dedup(sorted(reach, key=lambda r: r["score"])) if reach else []
    for r in best_reach:
        print(line(r))
    if not reach:
        print("  none. Hugging the ridge and reaching the landfall are not both "
              "available in what has been sampled.")

    print("\n=== Pareto frontier, ridge score against J ===")
    print(HDR)
    for r in pareto(ok)[:12]:
        print(line(r))

    C.atomic_write_json(OUT_JSON, {
        "generated": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/ocean_rank_params.py",
        "source_csv": CSV_PATH,
        "n_evaluations": len(rows), "n_ungated": len(ok),
        "reach_j_km": args.reach_j_km,
        "best_ridge": best_ridge,
        "best_ridge_with_reach": best_reach,
        "pareto": pareto(ok)[:12],
    })
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
