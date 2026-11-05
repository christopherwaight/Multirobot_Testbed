"""Batch b (2026-11-05): roadmap restored, friction paragraph simplified, limitations shortened.

Every old string must occur exactly once in the source, or the batch stops.
Usage: python3 batch_2026-11-05b.py SRC OUT_TEX OUT_MD
"""
import sys

E = []
def sub(i, why, old, new): E.append((i, why, old, new))

sub("B1", "Roadmap restored (Chris). Submitted text with two accuracy edits: \"distributed\" dropped (R1.7 scope) and the nonlinear fields added",
"% [OPEN] campaign trial count; the submitted 169 is superseded",
"% [OPEN] campaign trial count; the submitted 169 is superseded\n\n"
"The remainder of this paper is organized as follows. Section II develops the mathematical framework for Jacobian estimation and critical point prediction, and presents the control strategies for approach and orbit. Section III describes the control architecture used to evaluate these strategies in simulation and hardware. Section IV presents simulation results across six vector field types and three nonlinear fields. Section V describes the hardware testbed and presents validation results. Section VI compares performance against alternative navigation strategies and discusses limitations and future work. Section VII concludes.")

sub("B2", "Friction and Estimate Error simplified (Chris): the Lyapunov inequality and the derivative condition stated in words; the final-error bound kept",
r"The attraction law commands $k\|\mathbf{e} - \boldsymbol{\delta}\| = k\|\mathbf{p}_c - \hat{\mathbf{p}}^*\|$, so the cluster comes to rest inside the friction ring, a disk of radius $v_c/k$ about the estimate, 0.025~m at $k = 1$~s$^{-1}$. The estimate error shrinks as the cluster nears the critical point (\ref{eq:bound}) and settles at the value it has there. Under ideal continuous kinematics, $\dot{\mathbf{e}} = -k(\mathbf{e} - \boldsymbol{\delta})$, and $V = \frac{1}{2}\|\mathbf{e}\|^2$ satisfies $\dot{V} \le -k\|\mathbf{e}\|(\|\mathbf{e}\| - \|\boldsymbol{\delta}\|)$, so the error decreases while the cluster is farther from the critical point than the estimate is. Provided that moving the cluster a given distance changes the estimate error by less than that distance, $\|\partial\boldsymbol{\delta}/\partial\mathbf{p}_c\| < 1$, the changing estimate causes neither instability nor continued hunting, and the cluster comes to rest. It rests within $v_c/k$ of the estimate, which lies $\|\boldsymbol{\delta}_{\text{final}}\|$ from the critical point, so $\|\mathbf{e}_{\text{final}}\| \le v_c/k + \|\boldsymbol{\delta}_{\text{final}}\|$.",
r"The attraction law commands a speed of $k$ times the distance to the estimate, so the cluster comes to rest within the friction ring, a disk of radius $v_c/k$ about the estimate, 0.025~m at $k = 1$~s$^{-1}$. The estimate moves as the cluster moves, but its error shrinks as the cluster nears the critical point (\ref{eq:bound}). Under ideal kinematics, as long as the estimate moves less than the cluster does, the cluster closes on it and comes to rest without hunting. Its final error is then at most the ring radius plus the estimate error there, $\|\mathbf{e}_{\text{final}}\| \le v_c/k + \|\boldsymbol{\delta}_{\text{final}}\|$.")

sub("B3", "Limitations, Second: advection math folded into one plain sentence (Chris)",
r"but does not capture dynamic field behavior, and the printed fields do not push the robots. A physical flow advects the cluster and adds its own velocity $\mathbf{J}\mathbf{e}$ to the centroid near $\mathbf{p}^*$. For an ideal centroid with an exact estimate, the error then obeys $\dot{\mathbf{e}} = (\mathbf{J} - k\mathbf{I})\mathbf{e}$, which is unstable when an eigenvalue of $\mathbf{J}$ has real part above $k$.",
r"but does not capture dynamic field behavior. The printed fields also do not push the robots, and a physical flow that pushes the cluster away from the critical point faster than the gain $k$ pulls it back would destabilize the attraction law.")

sub("B4", "Limitations, Third: two sentences merged",
"With $n > 3$ robots the least-squares fit averages measurement errors across robots and survives the loss of one. However, a wider footprint asks the affine model to hold over more of the field's curvature.",
"With $n > 3$ robots the least-squares fit averages measurement errors and survives the loss of one, but a wider footprint asks the affine model to hold over more of the field's curvature.")

sub("B5", "Limitations, Finally: central-computer description points to V-A, which already gives it; dropout and latency sentence shortened",
"Finally, the computation is centralized. A central computer receives OptiTrack positions and field readings from every robot, forms the estimate, and returns velocity commands. The estimate needs only the positions and readings of one cycle in a common frame, so a robot receiving them from its neighbors could compute it. Onboard relative localization and inter-robot communication were not tested. A dropout can leave fewer than three readings, and the fit then has no unique solution, while latency mixes readings from different times into one fit.",
"Finally, the computation is centralized (Section~V-A). The estimate needs only the positions and readings of one cycle in a common frame, so a robot receiving them from its neighbors could compute it, but onboard relative localization and inter-robot communication were not tested. A dropout that leaves fewer than three readings leaves the fit without a unique solution, and latency mixes readings from different times into one fit.")

sub("B6", "Format. Extra blank lines before Finally removed",
"one formation side, were not tested.\n\n\n\nFinally,", "one formation side, were not tested.\n\nFinally,")


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
        f.write("# Batch b, 2026-11-05\n\nTarget: `cwaight_systems_paper/cwaight_systems_paper_with_revisions2.tex`. Exact-text match; stops on any miss.\n\n")
        for i, why, old, new in log:
            f.write(f"## {i}\n{why}\n\nOLD:\n```latex\n{old}\n```\nNEW:\n```latex\n{new}\n```\n\n")
    print(f"applied {len(log)} edits")
