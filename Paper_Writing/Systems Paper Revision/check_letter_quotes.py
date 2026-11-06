"""Check that every passage Response_to_Reviewers.tex quotes from the revision still matches it.

Checks each newtext block and each ``inline quote`` of 25 characters or more. References in the
paper are replaced by their compiled numbers from the .aux, so compile the paper first.
Usage: python3 check_letter_quotes.py
"""
import re
from pathlib import Path

here = Path(__file__).resolve().parent
paper_tex = here / "cwaight_systems_paper" / "cwaight_systems_paper_with_revisions2.tex"
paper_aux = paper_tex.with_suffix(".aux")
letter_tex = here / "Response_to_Reviewers.tex"

labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{(.*?)\}\{", paper_aux.read_text()))
labels = {k: v.replace("\\mbox  {", "").replace("}", "") for k, v in labels.items()}


def norm(s):
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace("\\allowbreak", "").replace("~", " ").replace("\\,", "")
    s = re.sub(r"\\cite\{[^}]*\}", lambda m: "[" + "], [".join(m.group(0)[6:-1].split(",")) + "]", s)
    return re.sub(r"\s+", " ", s).strip()


def pieces(q):
    # Split an elided quote into its verbatim fragments. "[\ldots]" (bracketed
    # or bare) marks an elision; each surrounding fragment must appear verbatim.
    q = q.replace("[\\ldots]", "\\ldots")
    for part in q.split("\\ldots"):
        part = norm(part).strip(" .,[]")
        if len(part) >= 8:   # skip trivial fragments like a lone bracket
            yield part


paper = norm(re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "??"), paper_tex.read_text()))
letter = letter_tex.read_text()

quotes = [q for q in re.findall(r"\\begin\{newtext\}(.*?)\\end\{newtext\}", letter, re.S)]
quotes += [q for q in re.findall(r"``(.*?)''", letter, re.S) if len(q) >= 25]

bad = 0
for q in quotes:
    for piece in pieces(q):
        if piece not in paper:
            bad += 1
            print("NO MATCH:", piece[:160])
print(f"{len(quotes)} quotes checked, {bad} mismatches")
