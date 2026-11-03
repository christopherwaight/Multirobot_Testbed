"""Measure how much of the submitted paper survives verbatim in the revision.

Survival is the share of the submitted body's words that appear in the revised
body inside identical runs of four or more words. Runs are first matched in
order. Runs that were moved (a swapped paragraph, a sentence relocated to another
section) are then matched wherever they landed, so a move does not count as
deleted text. Both figures are printed. The body runs from \\begin{document}
to the bibliography, with comments and tabular contents removed, so swapping a
table's numbers does not count against the prose.

Usage (from anywhere):
    python3 survival.py                      # working revision vs submitted paper
    python3 survival.py path/to/other.tex    # any revised file vs submitted paper

Targets set by Chris for the 2026-11 revision: 80% overall, 55% hard floor (lowered from 70% on 2026-11-03).
"""
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMITTED = HERE.parent / "Systems Paper" / "cwaight_systems_paper" / "cwaight_systems_paper.tex"
REVISION = HERE / "cwaight_systems_paper" / "cwaight_systems_paper_with_revisions2.tex"
MIN_RUN = 4
TARGET, FLOOR = 80.0, 55.0


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


def moved(old_w, new_w, kept_old, kept_new, min_run=MIN_RUN):
    """Match runs of min_run or more identical words that sit in a different order in the
    two texts. Only words still unmatched in both are used, longest runs first."""
    while True:
        index = {}
        for j in range(len(new_w) - min_run + 1):
            if not any(kept_new[j:j + min_run]):
                index.setdefault(tuple(new_w[j:j + min_run]), []).append(j)
        cands = []
        for i in range(len(old_w) - min_run + 1):
            if any(kept_old[i:i + min_run]):
                continue
            for j in index.get(tuple(old_w[i:i + min_run]), ()):
                if i and j and not kept_old[i - 1] and not kept_new[j - 1] and old_w[i - 1] == new_w[j - 1]:
                    continue  # not the start of a maximal run
                n = min_run
                while (i + n < len(old_w) and j + n < len(new_w) and not kept_old[i + n]
                       and not kept_new[j + n] and old_w[i + n] == new_w[j + n]):
                    n += 1
                cands.append((n, i, j))
        if not cands:
            return
        cands.sort(reverse=True)
        took = False
        for n, i, j in cands:
            if any(kept_old[i:i + n]) or any(kept_new[j:j + n]):
                continue  # overlaps a longer run taken this round
            kept_old[i:i + n] = [True] * n
            kept_new[j:j + n] = [True] * n
            took = True
        if not took:
            return


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
    in_order = 100 * sum(kept_old) / len(old)
    moved([t[0] for t in old], [t[0] for t in new], kept_old, kept_new)

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
    print(f"  in order only: {in_order:.1f}%; moved runs add {total - in_order:.1f} points")
    print(f"Revised body: {len(new)} words vs {len(old)} submitted; {carried:.1f}% of it is carried-over text")


if __name__ == "__main__":
    main()
