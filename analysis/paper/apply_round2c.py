"""Round 2c (2026-11-17): second length pass toward 13 pages. Drops Table III(d) (its numbers stay in
Sec. V-D), shortens three sentences, and re-copies the shortened Figs. 7 and 8.
Exact-match edits; aborts before writing if one fails. Run after apply_round2b.py.
Usage: python3 analysis/paper/apply_round2c.py [--dry-run]
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apply_round2 as r2  # noqa: E402

EDITS = [
    (r" (d) Share of orbit cycles on which $\hat{\mathbf{J}}$ had the type of the printed field, and in the vortex the share labeled rotational.}", "}"),
    (r"on every cycle of the orbit runs (Table~\ref{tab:estimation_results}(d)).", "on every cycle of the orbit runs."),
    (r"that demonstrate repeatable critical point estimation, attraction, and orbit control, and it compares each run with a simulation from the same start.",
     r"that demonstrate repeatable critical point estimation, attraction, and orbit control."),
    (r", and a fit to either field or either formation predicts the other within \rev{0.0004}~m/s.", r"."),
    (r"The model also treats every robot alike, yet their fitted gains range from \rev{0.80} to \rev{0.89}, and a robot held out of the field reconstruction read the map with a bias of up to \rev{5}$^\circ$.",
     r"The model also treats every robot alike, though their fitted gains range from \rev{0.80} to \rev{0.89} and an unseen robot read the map with a bias of up to \rev{5}$^\circ$."),
]


def main(dry):
    s = r2.apply(open(r2.TEX).read(), EDITS, "paper")
    s = r2.replace_table(s, "tab:estimation_results", open(os.path.join(HERE, "table_hardware.tex")).read())
    print(f"paper: {len(EDITS)} edits and Table III without (d)")
    if dry:
        return
    open(r2.TEX, "w").write(s)
    for src, dst in list(r2.FIGS.items())[:2]:
        shutil.copy(src, os.path.join(r2.PAPER, dst))
    print("written; Figs. 7 and 8 re-copied")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
