"""Measure how much of the submitted paper survives verbatim in the revision.

Survival is the share of the submitted body's words that appear in the revised
body inside identical runs of four or more words, matched in order (a moved
paragraph counts as deleted and re-added). The body runs from \\begin{document}
to the bibliography, with comments and tabular contents removed, so swapping a
table's numbers does not count against the prose.

Usage (from anywhere):
    python3 survival.py                      # working revision vs submitted paper
    python3 survival.py path/to/other.tex    # any revised file vs submitted paper

Targets set by Chris for the 2026-11 revision: 80% overall, 70% floor.
"""
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMITTED = HERE.parent / "Systems Paper" / "cwaight_systems_paper" / "cwaight_systems_paper.tex"
REVISION = HERE / "cwaight_systems_paper" / "cwaight_systems_paper_with_revisions.tex"
MIN_RUN = 4
TARGET, FLOOR = 80.0, 70.0


def body(path):
    text = Path(path).read_text()
    text = text[text.index("\\begin{document}"):text.index("\\begin{thebibliography}")]
    text = re.sub(r"(?<!\\)%.*", "", text)
    return re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.S)


def tokens(text):
    """Words tagged with the submitted-style section and subsection they sit in."""
    out, sec, sub = [], "Front matter", ""
    for line in text.split("\n"):
        s = line.strip()
        m = re.match(r"\\(sub)?section\*?\{([^}]*)\}", s)
        if m:
            if m.group(1):
                sub = m.group(2)
            else:
                sec, sub = m.group(2), ""
        elif s.startswith("\\appendices"):
            sec, sub = "Appendices", ""
        out.extend((w, sec, sub) for w in line.split())
    return out


def survival(old_toks, new_toks):
    sm = difflib.SequenceMatcher(None, [t[0] for t in old_toks], [t[0] for t in new_toks], autojunk=False)
    kept_old, kept_new = [False] * len(old_toks), [False] * len(new_toks)
    for a, b, n in sm.get_matching_blocks():
        if n >= MIN_RUN:
            kept_old[a:a + n] = [True] * n
            kept_new[b:b + n] = [True] * n
    return kept_old, kept_new


def table(toks, kept):
    rows = {}
    for (_, sec, sub), k in zip(toks, kept):
        for key in ((sec, ""), (sec, sub)) if sub else ((sec, ""),):
            n, s = rows.get(key, (0, 0))
            rows[key] = (n + 1, s + k)
    return rows


def main():
    revised = Path(sys.argv[1]) if len(sys.argv) > 1 else REVISION
    old, new = tokens(body(SUBMITTED)), tokens(body(revised))
    kept_old, kept_new = survival(old, new)

    print(f"Submitted: {SUBMITTED.relative_to(HERE.parent)}")
    print(f"Revised:   {revised}\n")
    print(f"{'Submitted section / subsection':52s} {'words':>6s} {'kept':>7s}")
    for (sec, sub), (n, s) in table(old, kept_old).items():
        label = f"  {sub}" if sub else sec
        flag = "" if sub or 100 * s / n >= FLOOR else "   < floor"
        print(f"{label[:52]:52s} {n:6d} {100 * s / n:6.1f}%{flag}")

    total = 100 * sum(kept_old) / len(old)
    carried = 100 * sum(kept_new) / len(new)
    status = "on target" if total >= TARGET else "acceptable" if total >= FLOOR else "BELOW FLOOR, stop and ask Chris"
    print(f"\nSubmitted words surviving: {total:.1f}%  ({status}; target {TARGET:.0f}%, floor {FLOOR:.0f}%)")
    print(f"Revised body: {len(new)} words vs {len(old)} submitted; {carried:.1f}% of it is carried-over text")


if __name__ == "__main__":
    main()
