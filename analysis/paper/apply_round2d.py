"""Round 2d (2026-11-17): third length pass toward 13 pages. Shorter III-B square sentence, V-D four-robot
spread sentence (the numbers are in Table III(c)), VI-A lead sentence and Fig. 8 caption; re-copies Figs. 8 and 9.
Exact-match edits; aborts before writing if one fails. Run after apply_round2c.py.
Usage: python3 analysis/paper/apply_round2d.py [--dry-run]
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apply_round2 as r2  # noqa: E402

EDITS = [
    (r" The four-robot square is parameterized by its diagonals $d_1$ and $d_2$, the fractions $r_1$ and $r_2$ of each diagonal at which they cross, and the angle $\phi$ between them, with $r_1 = r_2 = 0.5$ and $\phi = 90^\circ$ \cite{50}.",
     r" The four-robot square is parameterized by its two diagonals, the fraction of each at which they cross, and the angle between them \cite{50}."),
    (r"in both fields, and its within-run standard deviations ranged from \rev{0.017} to \rev{0.057}~m in the vortex and \rev{0.022} to \rev{0.076}~m in the saddle.",
     r"in both fields and a similar within-run spread (Table~\ref{tab:estimation_results}(c))."),
    (r"a lead of unidentified cause that the model omits; adding it to the model brings the vortex orbits at $r_d = 0.01$ and 0.10~m within \rev{7}~mm of the hardware.",
     r"a lead the model omits; added to the model, it brings the small-radius vortex orbits within \rev{7}~mm of the hardware."),
    (r"\caption{Hardware orbits against simulations matched to each run (Section~V-B), three robots (blue) and four (orange). (a) Radial bias, hardware mean and $\pm 1$ between-trial spread (markers), with the matched simulations on the reconstructed (solid) and analytical (dashed) field. (b) Within-run spread.}",
     r"\caption{Hardware orbits (markers, with $\pm 1$ between-trial spread) against simulations matched to each run (Section~V-B), three robots (blue) and four (orange). (a) Radial bias. (b) Within-run spread.}"),
]


def main(dry):
    s = r2.apply(open(r2.TEX).read(), EDITS, "paper")
    print(f"paper: {len(EDITS)} edits")
    if dry:
        return
    open(r2.TEX, "w").write(s)
    shutil.copy(os.path.join(r2.ROOT, "analysis", "simulation", "figures", "fig_sim_vs_hw.pdf"),
                os.path.join(r2.PAPER, "fig_sim_vs_hw_2026.pdf"))
    shutil.copy(os.path.join(r2.ROOT, "analysis", "limit_cycle", "figures", "fig_limit_cycle_column.pdf"),
                os.path.join(r2.PAPER, "fig_limit_cycle_2026.pdf"))
    print("written; Figs. 8 and 9 re-copied")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
