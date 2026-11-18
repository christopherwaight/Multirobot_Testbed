"""Round 3 (2026-11-17): number-accuracy fixes. Exact-text edits; stops on the first mismatch.

Sources: nonlinear_summary.json (irrotational-saddle orbits kept 100% at 0.30 m, 3% at 0.40 m);
nonlinear.py (orbits 60 s); extra_numbers.json and Eq. (16) (simulated floor 0.047, closed form 0.049);
orbit_per_run.csv (revolutions at r_d 0.01: 5.6 to 13.1); limit_cycle.py (heading 0); estimate lead along
the motion at r_d 0.01, all four groups: 17 to 28 mm (computed from runs2026.pkl, 3 s trim); the lead-in-model
check (7 mm) used the superseded 75 ms first fit, so it is cut. Sensors: palette calibration not redone.
"""
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import paths  # noqa: E402

TEX = Path(paths.PAPER) / "cwaight_systems_paper_with_revisions2.tex"

EDITS = [
    ("and about half the irrotational-saddle orbits at 0.3~m were lost past the points $(0, \\pm 0.58)$~m.",
     "and nearly all irrotational-saddle orbits from 0.4~m were lost past the points $(0, \\pm 0.58)$~m."),
    ("then the orbital controller (\\ref{eq:11}) was used for \\rev{300} time steps.",
     "then the orbital controller (\\ref{eq:11}) was used for \\rev{300} time steps (\\rev{600} on the nonlinear fields)."),
    ("There $r_{\\min} = \\rev{0.047}$~m, so the error stays below 0.05~m at every commanded radius. "
     "Lowering the orbital speed $k_t$ to 0.1~m/s cuts $r_{\\min}$ to \\rev{0.031}~m,",
     "In simulation the floor is \\rev{0.047}~m, against \\rev{0.049}~m from (\\ref{eq:orbit_cf}), so the error "
     "stays below 0.05~m at every commanded radius. Lowering the orbital speed $k_t$ to 0.1~m/s cuts the simulated "
     "floor to \\rev{0.031}~m,"),
    ("long enough for at least one full orbit at the largest radius and up to eight at the smallest.",
     "long enough for at least one full orbit at the largest radius and five to thirteen at the smallest."),
    ("and in simulation on the reconstructed vortex the curve has mean radius \\rev{0.111}~m.",
     "and in simulation on the reconstructed vortex, with the formation at one heading, the curve has mean radius "
     "\\rev{0.111}~m."),
    ("In these runs the live estimate ran \\rev{1.5} to \\rev{2}~cm ahead of the cluster along its motion, a lead "
     "the model omits; added to the model, it brings the small-radius vortex orbits within \\rev{7}~mm of the hardware.",
     "In these runs the live estimate ran \\rev{1.7} to \\rev{2.8}~cm ahead of the cluster along its motion, a lead "
     "the model omits."),
    ("All robots were calibrated after receiving preventive maintenance and servicing for this study.",
     "All robots received preventive maintenance and servicing for this study, and the color sensors kept their "
     "calibration of \\cite{22}."),
]


def main():
    text = TEX.read_text()
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"abort, {text.count(old)} matches for: {old[:70]}")
        text = text.replace(old, new)
    backup = Path(__file__).with_name("backup_cwaight_systems_paper_with_revisions2_before_round3.tex")
    shutil.copy(TEX, backup)
    TEX.write_text(text)
    print(f"applied {len(EDITS)} edits; backup {backup.name}")


if __name__ == "__main__":
    main()
