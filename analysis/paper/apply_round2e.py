"""Round 2e (2026-11-17): fourth length pass toward 13 pages. Fig. 6 at 0.85 column width, shorter VI-A
robot-differences sentence and conclusion clause; re-copies the shortened Figs. 7 and 9.
Exact-match edits; aborts before writing if one fails. Run after apply_round2d.py.
Usage: python3 analysis/paper/apply_round2e.py [--dry-run]
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apply_round2 as r2  # noqa: E402

EDITS = [
    (r"\includegraphics[width=0.90\columnwidth]{fig_reconstruction_3x2_2026.png}",
     r"\includegraphics[width=0.85\columnwidth]{fig_reconstruction_3x2_2026.png}"),
    (r"The model also treats every robot alike, though their fitted gains range from \rev{0.80} to \rev{0.89} and an unseen robot read the map with a bias of up to \rev{5}$^\circ$.",
     r"The model also treats every robot alike, though their fitted gains range from \rev{0.80} to \rev{0.89}."),
    (r"; at a commanded 0.10~m the vortex orbit settled onto a limit cycle sized by the delay and lag of the robots.",
     r"; at 0.10~m the vortex orbit settled onto a limit cycle sized by the robots' delay and lag."),
]


def main(dry):
    s = r2.apply(open(r2.TEX).read(), EDITS, "paper")
    print(f"paper: {len(EDITS)} edits")
    if dry:
        return
    open(r2.TEX, "w").write(s)
    shutil.copy(os.path.join(r2.ROOT, "analysis", "hardware", "figures", "fig_hw_attraction.pdf"),
                os.path.join(r2.PAPER, "fig_hw_attraction_2026.pdf"))
    shutil.copy(os.path.join(r2.ROOT, "analysis", "limit_cycle", "figures", "fig_limit_cycle_column.pdf"),
                os.path.join(r2.PAPER, "fig_limit_cycle_2026.pdf"))
    print("written; Figs. 7 and 9 re-copied")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
