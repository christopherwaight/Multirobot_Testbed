# Reviewer feedback: status against Draft 10

Both review files (`reviewer1_feedback.md`, `reviewer2_feedback.md`) were written
against **Draft 8a**. The chain since is 8a, 9, 10. Many items are already closed in
Draft 10 and several others were declined deliberately. Read this file before acting on
anything in either review, and do not re-raise a closed or declined item.

`Reviewer1_Response.md` is **superseded**. It rebuts items the paper no longer contests
(see E9 below). Do not send it as written.

Last checked against Draft_10.tex on 2026-09-23.

---

## Closed in Draft 10

Verified present in the current file. No action needed.

| Item | Where it is handled now |
|---|---|
| **E1 / C7** condition-number normalization | II-C reports the radius-normalized 8.26; IV-G (line ~1155) now reports the ocean formation at its nominal 8.26 as well. The raw 688 is gone, so the paper no longer reads as if the ocean formation were more degenerate than the pathological example. |
| **E3** band `B` mislabeled | II-C now states outright that the band "surrounds $\{D = 0\}$, not the separatrix," and names both thresholds (raw depth, curvature-normalized). |
| **E5** unbiasedness scoped | Estimator Sensitivity now separates the two channels: measurement noise leaves $\hat{D}$ unbiased, position noise correlates the components and injects a bias scaling with the local Jacobian. |
| **E6** traversal bound near stagnation | III-C states the bound on a segment "excluding an $\varepsilon$ neighborhood of its endpoints," and ties the excluded neighborhood to the terminal test. |
| **E7** `D` tracker sign reference | The sign test is specified, and (\ref{eq:d_capture}) $\lambda_1\lambda_2 \geq 0$ is given explicitly as the primitive's terminal condition. |
| **E9** $\lambda_1 < 0 < \lambda_2$ premise | The caveat is now at the premise (III-C, ~line 629), not buried in Limitations: it "fails over the outer half of each separatrix segment of the benchmark, where the true $\mathbf{H}_D$ is positive definite." See the correction note below. |
| **E12** `e_H` undefined | The symbol no longer appears anywhere in Draft 10. |
| **Continuity** `D` tracker terminal test missing from Section III | Now specified in III-C at (\ref{eq:d_capture}). |
| **"straddle retention" undefined** | Defined in IV-E (~line 1043). |
| **"trench-network distance" undefined** | The term no longer appears in Draft 10. |
| **R2-2 / "four experiment families"** | Corrected to five (line ~864). |
| **R2-2 `k = 1.8` branch** | IV-G now states the branch is the one `k = 1.8` selects on the evolving ridge. |

### Correction to the E9 rebuttal

`Reviewer1_Response.md` Part 1 item 1 claims the reviewer's sign evaluation was
"backwards." It was not. On the traversed segment $x_f = 1$, so
$\cos 2\pi x_f = +1$ and the first entry of
$\mathbf{H}_D = 2\pi^6 A^2 \operatorname{diag}(\cos 2\pi x_f, \cos 2\pi y_f)$ is
positive everywhere on that line. An indefinite Hessian therefore requires
$\cos 2\pi y_f < 0$, i.e. $0.25 < y_f < 0.75$, the **inner** half. Outside it both
entries are positive and the Hessian is positive definite, which is what the reviewer
said and what Draft 10's own III-C and Limitations both state. The rebuttal contradicted
the paper it was defending. Draft 10 needs no change; only the response document is wrong.

---

## Declined on scope

Legitimate asks for a different, longer paper. Draft 10 was deliberately narrowed to the
$D$ / $s_1$ contrast and the controllers that ride it, and these do not serve that claim.
Declining them is a standing decision, not an oversight.

- **A1, the gain ladder table.** Reviewer 1's top-priority addition. Cut on purpose:
  nothing downstream consumes $\gamma = 8/\sqrt{10}$, and the noise result is carried by
  the Monte Carlo sweep. The intro sentence that advertised it ("noise gains are exact and
  isotropic in formation heading, one scaling rule per derivative order") was removed on
  2026-09-23 so the paper no longer promises a table it does not print.
- **E4**, one gain per derivative order. Moot once the ladder and its intro sentence were
  cut. The underlying distinction (the $O(\rho^{-q})$ scaling rule versus the constant)
  was defensible but is no longer claimed.
- **A3** experiment matrix, **A6** algorithm boxes, **A8** state-machine figure,
  **A2** parameter table.
- **A5** symmetric $s_1$ runs for the three `D`-only experiments.
- **A9** feasibility sentence, **A10** compute cost, **A11** data availability.
- **Promoting the reachable-set estimate** and **restructuring the abstract around IV-E**.
  Both would swap the paper's claim for the reviewer's favorite passage.
- **Reviewer 2 Section 8** (the writing grades) in full. Its rewrites insert em-dashes,
  "Crucially," "Ultimately," "Conversely," and intensifiers, all against the house style in
  CLAUDE.md, and its proposed abstract is longer than what it replaces.

---

## Live

Short list. Nothing here is an addition request.

- **C6**, "closed-loop frame equivariance" in the contribution list versus what Appendix A
  proves (equivariant commands, bounded path divergence, one non-objective seeded step).
  The rebuttal's distinction between the control mapping and the trajectory integral is
  correct; confirm Draft 10's wording matches Appendix A rather than overstating it.
- **C2**, selection rule "by field and mission" versus the conclusion's "observing
  platform, not the flow." Verify Draft 10's intro and conclusion now agree.
- **Figure hygiene.** `rho_sweep_findings.md` notes
  `figures/estimator_accuracy_vs_noise.png` and its `.meta.json` were modified but
  uncommitted, so the embedded Fig. 3 may not match any committed state. Independent of
  the reviews, worth resolving before resubmission.

---

## The rho sweep

`rho_sweep_findings.md` answers Reviewer 1's A4, which it called the addition that would
"move this from a good paper to a hard-to-reject one." It was run (25,600 trials) and
came back with a **better** result than predicted: no interior optimum, success rises
monotonically with $\rho$, and the trade lives on a different axis (robustness against
parking accuracy). Both exponents confirm the gain ladder in closed loop, 0.98 and 0.94
against a predicted 1, 2.02 and 1.95 against a predicted 2. New claim available: both
primitives are second-order-limited even though the $s_1$ tracker rides a first-order
tangent.

**Status: not in the .tex, and not slated for Draft 10.** It is a gain-ladder result, and
the ladder is out of scope here. Most likely the seed of the next paper. Numbers still
need author sign-off before entering any .tex (CLAUDE.md).
