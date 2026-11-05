"""Batch c (2026-11-05): slop-removal pass from an outside review, with pushback as noted.

Every old string must occur exactly once in the source, or the batch stops.
Usage: python3 batch_2026-11-05c.py SRC OUT_TEX OUT_MD
"""
import sys

E = []
def sub(i, why, old, new): E.append((i, why, old, new))

sub("A1", "VI-C Second. Repeated push/pushes/pulls removed; \"advect\" is R1.7's word. (Suggested fix rejected: banned \"moreover\", and it compared a velocity to the gain k.)",
r"The printed fields also do not push the robots, and a physical flow that pushes the cluster away from the critical point faster than the gain $k$ pulls it back would destabilize the attraction law.",
r"The printed fields also do not advect the robots, and a physical flow that carries the cluster away from the critical point at a rate above the gain $k$ would destabilize the attraction law.")

sub("A2", "VI-C Third. No personified footprint; \"measurement errors\" kept as the fixed term",
r"With $n > 3$ robots the least-squares fit averages measurement errors and survives the loss of one, but a wider footprint asks the affine model to hold over more of the field's curvature.",
r"With $n > 3$ robots the least-squares fit averages measurement errors and tolerates the loss of one robot, but the wider footprint requires the affine model to hold over a larger region of the curved field.")

sub("B1", "II-B. Comma appositive removed",
r"The $\mathcal{O}(\ell^2)$ term carries the field's curvature, the second-order Taylor terms that a plane cannot represent.",
r"The $\mathcal{O}(\ell^2)$ term captures the field's second-order curvature, which a plane cannot represent.")

sub("B2", "II-B. Comma appositive replaced by a relative clause",
r"$c_w \to 1/3$, the variance factor of the mean of three independent readings.",
r"$c_w \to 1/3$, which matches the variance of the mean of three independent readings.")

sub("B3", "III-A. Comma appositive removed",
r"and the last term is Coulomb friction, an extra command of magnitude $v_c$ opposing the motion.",
r"and the last term models Coulomb friction as an opposing command of magnitude $v_c$.")

sub("B4", "III-D. Stacked appositives removed",
r"so the cluster comes to rest within the friction ring, a disk of radius $v_c/k$ about the estimate, 0.025~m at $k = 1$~s$^{-1}$.",
r"so the cluster comes to rest inside a friction ring of radius $v_c/k$ (0.025~m at $k = 1$~s$^{-1}$) around the estimate.")

sub("B5a", "VI-B. Appositive removed while keeping the submitted phrase \"reduces the flow to its magnitude\"",
r"reduces the flow to its magnitude, the speed $S_i = \|\mathbf{v}(\mathbf{p}_i)\|$ at each robot, and a plane fit to the three speeds gives their gradient $\mathbf{g}$.",
r"reduces the flow to its magnitude, and a plane fit to the speeds $S_i = \|\mathbf{v}(\mathbf{p}_i)\|$ at the three robots gives their gradient $\mathbf{g}$.")

sub("B5b", "VI-B. Appositive removed",
r"The second, the oracle, takes the true critical point as the target of (\ref{eq:10}) and the center of the radial term of (\ref{eq:11}).",
r"The second uses an oracle (the true critical point) as the target of (\ref{eq:10}) and the center of the radial term of (\ref{eq:11}).")

sub("C1", "III-D Orbits. \"simply\" dropped; \"could\" kept at Chris's request (hardware rule undecided; simulation steps randomly)",
r"the controller could simply take a step in a random direction",
r"the controller could take a step in a random direction")

sub("C2", "IV-A. Conversational \"showed in\" replaced",
r"In the nonlinear fields, curvature showed in the localization error and in which critical point the cluster reached.",
r"In the nonlinear fields, curvature affected both the localization error and which critical point the cluster reached.")

sub("C3", "VI-B. Present-to-past tense shift fixed",
r"That estimate never lets the cluster rest during attraction, however, and the cluster still moved at \newnum{0.12}~m/s after 10~s.",
r"During attraction, however, that estimate never let the cluster settle, and the cluster still moved at \newnum{0.12}~m/s after 10~s.")


def apply(text):
    log = []
    for i, why, old, new in E:
        n = text.count(old)
        assert n == 1, (i, n, old[:80])
        text = text.replace(old, new)
        log.append((i, why, old, new))
    return text, log


if __name__ == "__main__":
    src, out_tex, out_md = sys.argv[1:4]
    new_text, log = apply(open(src).read())
    open(out_tex, "w").write(new_text)
    with open(out_md, "w") as f:
        f.write("# Batch c, 2026-11-05 (slop-removal pass)\n\nTarget: `cwaight_systems_paper/cwaight_systems_paper_with_revisions2.tex`. Exact-text match; stops on any miss.\n"
                "Not applied by design: C4 (\"The architecture itself is not a contribution of this work\") kept because it answers the AE and R2.1 in their own terms.\n\n")
        for i, why, old, new in log:
            f.write(f"## {i}\n{why}\n\nOLD:\n```latex\n{old}\n```\nNEW:\n```latex\n{new}\n```\n\n")
    print(f"applied {len(log)} edits")
