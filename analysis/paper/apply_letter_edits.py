"""Numeric edits to Response_to_Reviewers.tex, matched on exact text (abort on any mismatch).

Changed or verified values are marked \\rev{}. Claims that changed are queued in queued_prose_batch.md (section R).
Run from the repo root:  python3 analysis/paper/apply_letter_edits.py [--dry-run]
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402

LETTER = os.path.join(os.path.dirname(paths.PAPER), "Response_to_Reviewers.tex")

EDITS = [   # (exact old, new, count)
    ("orbit per field in simulation, 10 per start and \\open{three} per radius on hardware), with a stated first-10~s exclusion.",
     "orbit per field in simulation, 10 per start and \\rev{ten} per radius on hardware), with a stated first-\\rev{3}~s exclusion.", 1),
    ("in all \\newnum{2000} [\\ldots] trials''", "in all \\rev{2000} [\\ldots] trials''", 1),
    ("$r_d \\ge \\syn{0.40}$~m (vortex) and $\\syn{0.50}$~m (saddle) on hardware",
     "$r_d \\ge \\rev{0.20}$~m (vortex) and $\\rev{0.20}$~m (saddle) on hardware", 1),
    ("repeat each radius (\\rev{100} and \\open{three} times) with the same first-10~s exclusion;",
     "repeat each radius (\\rev{100} and \\rev{ten} times) with the same first-\\rev{3}~s exclusion;", 1),
    ("over \\newnum{500} attraction and \\newnum{50} orbit trials per setting.",
     "over \\rev{500} attraction and \\rev{50} orbit trials per setting.", 1),
    ("over \\newnum{500} attraction and \\newnum{50} orbit trials per setting,",
     "over \\rev{500} attraction and \\rev{50} orbit trials per setting,", 1),
    ("``reached $\\mathbf{p}^*$ in \\newnum{7}\\% of trials''", "``reached $\\mathbf{p}^*$ in \\rev{6}\\% of trials''", 2),
    ("with the time history in Fig.~3 and \\syn{Fig.~7}", "with the time history in Fig.~3 and \\rev{Fig.~7(c)}", 2),
    ("at most \\rev{1.7}~mm of temporal effect) and on hardware in Section~VI-A \\open{campaign block E}.",
     "at most \\rev{3.0}~mm of temporal effect) and on hardware in Section~VI-A (search ``between cycles'').", 1),
    ("``the center is labeled correctly on \\rev{90}\\% of cycles''", "``the center is labeled correctly on \\rev{86}\\% of cycles''", 1),
    ("from \\newnum{0.16}~mm [\\ldots] to \\newnum{25}~mm''", "from \\rev{0.15}~mm [\\ldots] to \\rev{25}~mm''", 1),
    ("Each condition is repeated (\\rev{100} and \\open{three} times);", "Each condition is repeated (\\rev{100} and \\rev{ten} times);", 1),
]


def main(dry):
    s = open(LETTER).read()
    for old, new, n in EDITS:
        c = s.count(old)
        if c != n:
            sys.exit(f"ABORT: {old[:70]!r} found {c} times, expected {n}")
        s = s.replace(old, new)
    if dry:
        print("dry run: all", len(EDITS), "edits matched")
        return
    shutil.copy2(LETTER, os.path.join(HERE, "backup_Response_to_Reviewers_before_2026-11-16.tex"))
    open(LETTER, "w").write(s)
    print("applied", len(EDITS), "edits; backup in analysis/paper/")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
