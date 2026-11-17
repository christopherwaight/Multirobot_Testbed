"""List every \\syn{}, \\rev{}, \\newnum{} and \\open{} marker in the revision tex, with its section and
context, to analysis/paper/markers.csv (one row per occurrence, in document order)."""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import paths  # noqa: E402

TEX = os.path.join(paths.PAPER, "cwaight_systems_paper_with_revisions2.tex")


def balanced(s, i):
    """Index just past the brace group starting at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError("unbalanced")


def main():
    s = open(TEX).read()
    sec = [(m.start(), m.group(2)) for m in re.finditer(r"\\(sub)*section\*?\{([^}]*)\}", s)]
    rows = []
    for m in re.finditer(r"\\(syn|rev|newnum|open)\{", s):
        end = balanced(s, m.end() - 1)
        body = s[m.end():end - 1]
        line = s.count("\n", 0, m.start()) + 1
        section = [name for pos, name in sec if pos < m.start()]
        ctx = s[max(0, m.start() - 90):min(len(s), end + 60)].replace("\n", " ")
        rows.append(dict(n=len(rows) + 1, marker=m.group(1), line=line, section=section[-1] if section else "",
                         value=body if len(body) < 60 else body[:57] + "...", is_table=body.lstrip().startswith("\\begin{tabular}"),
                         context=ctx))
    with open(os.path.join(HERE, "markers.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(len(rows), "markers;", Counter(r["marker"] for r in rows))
    print(Counter(r["section"] for r in rows).most_common())


if __name__ == "__main__":
    main()
