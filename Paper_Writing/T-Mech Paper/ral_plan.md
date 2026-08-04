# RA-L Retargeting Plan for the Vector Field Paper

Execution plan for converting `Paper_Draft_5A.tex` (desk-rejected at T-Mech on scope,
editor suggested T-RO or RA-L) into an RA-L submission. The strategy decided with the
author is RA-L for this paper, T-RO for the separatrix/OW paper (Draft 6a). 

**Scope Pivot**: To align with RA-L's focus on robotics applications and autonomy, the central narrative will pivot from a static estimation problem to autonomous environmental monitoring in **dynamic, time-varying environments**. Experiments involving time-varying flows and field morphing will be the primary motivation. 
Due to the strict 8-page limit, detailed mathematical derivations and extended plots for the new time-varying experiments will be placed in a new supplemental document (`ral_supplemental.tex`). This document serves as the working spec for the agent doing that conversion.

## Venue facts

- RA-L page limit is 6 pages plus at most 2 extra pages at a charge, so 8 pages hard max,
  references included.
- RA-L is fully open access (APC roughly $2800 for 2026 submissions). Decision within
  6 months of submission.
- RA-L uses its own template (conference-style, ieeeconf-based), not the lettersize
  journal IEEEtran that Draft 5A uses. Page counts do not transfer between the two
  formats. Download the current RA-L template, reflow the paper into it, and measure the
  real page count before doing any trim arithmetic. Verify current page and template
  policy at https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
  at submission time.

## Ground rules (read before doing anything)

1. The author does not write code. Agents write everything. The author reviews designs
   and numbers.
2. **Gate A, design review.** Before writing any experiment code, present each
   experiment design to the author in Sutton and Barto pseudocode style (initialization
   block, named quantities, update rules, loop in plain form, parameters at top) with a
   plain-English explanation. Wait for sign-off.
3. **Gate B, numbers review.** Never insert experiment numbers into any .tex file
   without the author's explicit approval. Report results as prose and tables (in
   conversation or a results .md in this folder). The author reviews, then approves
   insertion. This step is not automated.
4. Report results faithfully. The expectations noted per experiment below are
   hypotheses. If a baseline performs better than expected or a claim does not hold,
   report that plainly. Do not tune until the expected story appears.
5. Other agents may be working in this repo concurrently and the author edits files in
   the IDE. Re-grep before editing, do not trust line numbers from earlier in a session,
   and do not revert changes you did not make.
6. Do not modify `Paper_Draft_5A.tex`. It is the record of the T-Mech submission. All
   RA-L work happens in a new file (see Trim section).
7. Prose style in the paper follows repo CLAUDE.md. IEEE register, no em-dashes, no
   pseudocode in the paper, hand-formatted integer-key `\bibitem` entries with
   `\cite{N}`, no forward-pointing scaffolding sentences in related work.

## Boundary with Draft 6a (hard constraints)

The separatrix/OW paper (`Paper_Writing/Separatrix_and_OW_Paper/Draft_6a.tex`) is under
active development for T-RO. The RA-L paper must not overlap it.

- No double gyre, in any form. No ocean HFR data. All new experiments use the six
  canonical fields only.
- No separatrix, manifold, trench, Lagrangian coherent structure, or Okubo-Weiss
  content or vocabulary anywhere in the RA-L paper.
- The saddle field appears only as a static or time-varying attraction/orbit target.
  No manifold tracking of any kind.
- Do not build a standalone estimator-error-versus-noise headline figure. Draft 6a
  owns `estimator_accuracy_vs_noise`. Noise appears here only as an axis inside the
  classification experiment.
- Do not touch anything in `Paper_Writing/Separatrix_and_OW_Paper/`.
- Do not write into `experiments/outputs/unsteady_gyre/` or other output folders the
  6a experiments use. New experiment outputs go in new `experiments/outputs/ral_*`
  subfolders.
- `control_architecture` figures are shared between papers. Do not edit the figure
  images. If prose and figure disagree, fix the prose.

## Part 1. Experiment package

Four items, in priority order. Each follows the two gates above. New scripts go in
`trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/` with an `ral_` prefix.
Use the shared venv (`trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/`).
Existing infrastructure pointers are in repo CLAUDE.md (entry point `main_omni.py`,
fields in `src/fields/environments/`, runner in `src/simulation/runner.py`).

### Experiment A. Quantitative primitive comparison (the novelty anchor)

Claim it supports. The paper currently shows one qualitative figure (outward spiral
drift of alternatives in the vortex field). Replace assertion with numbers showing
direction-only primitives structurally cannot do the task.

Design sketch, to be detailed for Gate A:

- Controllers compared, all three sharing the same cluster and robot layers so
  differences are algorithmic: (1) the proposed critical point controller, (2) the
  vector-sum primitive (move along the average perceived flow), (3) the
  vector-to-scalar primitive (cited as [48] in Draft 5A; magnitude descent, with the
  tangential variant for orbit, matching how Section VI-B of Draft 5A describes both).
- Tasks: attraction and orbit, across all six canonical fields.
- Trials: batch of random initial positions per field per controller (propose a count
  at Gate A; enough for stable success percentages). Noise-free primary; optionally one
  hardware-calibrated noise level (sigma_v = 0.02 m/s per Draft 5A Section V) as a
  secondary condition.
- Metrics: success rate (converged within a threshold distance of the true critical
  point inside a time budget), time to converge, final error. For orbit: mean radial
  error and radial drift rate, or DNF.
- Expected story (hypothesis, see ground rule 4): alternatives fail on saddle and
  source for attraction and drift without bound in orbit; the proposed controller
  succeeds across all six.

Output. One table. This is the most important single addition to the paper.

### Experiment B. Time-varying canonical fields

Claim it supports. Draft 5A's Discussion distinguishes two time-varying cases (field
structure changes versus critical point translates) and defers both to future work.
This experiment converts that paragraph into results. Key theoretical point to state
and verify: for fields affine at each instant, the estimator is exact at every instant
regardless of variation rate, because all robots sample simultaneously and the
estimator carries no state. All degradation is control-loop lag.

Design sketch, three representative cases rather than the full mode-by-field grid:

1. Translating vortex. Critical point drifts at constant speed U, Jacobian fixed.
   Sweep U as a fraction of robot speed capability. Metrics: steady-state tracking
   error of the attraction controller, and orbit radius error while orbiting the
   moving center. Derive the predicted lag law from the attraction law as written in
   Draft 5A Section II-D (present the derivation at Gate A) and overlay it on the
   sweep. Model the translation as pattern translation (the field zero moves with the
   pattern). No background advection term; if the author wants it later, note that a
   vortex embedded in ambient flow U has its instantaneous zero offset by J^{-1}U,
   worth one acknowledging sentence in the paper at most.
2. Pulsing sink. J(t) = (1 + a sin(omega t)) J0 with the critical point fixed. Sweep
   a and/or omega. Metrics: position error at the critical point, and a numerical
   check that the noise-free estimate stays exact through the pulsing.
3. Sanity check first, before either sweep: verify zero noise-free estimation error at
   every instant on a fast-varying affine field. If this fails, something is wrong in
   the implementation, not the theory. Report it and stop.

Implementation. Follow the pattern in `src/fields/environments/Double_Gyre.py` (time
parameter threaded through evaluation, config-driven parameters). The runner already
passes simulation time. Add new time-varying field variants or named instances. Do not
mutate the existing static field modules; other work depends on them.

Output. One combined figure (trajectory snapshot panel plus error-versus-rate panel
with the lag-law line), and a small table if needed.

### Experiment C. Classification accuracy, static and morphing

Claim it supports. The abstract and intro claim the method recovers location and
identity. Identity is never exercised in Draft 5A (its own Future Work says so).

Design sketch:

- Static half. Across the six canonical fields, log estimated Jacobian eigenvalues
  during runs at several noise levels (multiples of the hardware-calibrated
  sigma_v = 0.02 m/s), classify per the eigenvalue rules of Draft 5A Table I, report a
  confusion matrix or accuracy versus noise. Classifying center versus spiral under
  noise requires a dead-band threshold on the eigenvalue real part; propose the
  threshold criterion at Gate A. Frame everything as classification accuracy, not
  estimator error versus noise (see Draft 6a boundary).
- Morphing half. One field whose type changes during the run, vortex to sinking vortex,
  with the eigenvalue real part alpha(t) ramping through zero. Produce a timeline of
  estimated eigenvalues with the true crossing marked, and report detection latency
  (time from the true crossing of the classifier dead-band to the classifier
  switching). Reuses Experiment B's time-varying machinery.

Output. Confusion matrix or accuracy table, plus the transition timeline panel.

### Experiment D. Four-robot overdetermined estimation (space permitting)

Claim it supports. Draft 5A claims arbitrary team sizes through the overdetermined
formulation and lists team size as a limitation. `src/robot/quad_cluster.py` and
`src/control/quad_primitives.py` already exist.

Design sketch. Repeat the estimation accuracy measurement with N = 4 versus N = 3
under matched conditions and noise levels. Fold results into existing tables as added
rows rather than a new section. Include only if the page budget survives Experiments
A through C.

## Part 2. Trim plan

Draft 5A is 11 pages in journal IEEEtran with 45 references and two appendices. Target
is 8 pages in the RA-L template with roughly one page of new material from Part 1, so
net cuts must exceed three journal pages. Reflow into the RA-L template first, then
re-measure, then cut in this order.

Working file. Copy `Paper_Draft_5A.tex` to a new file in this folder (suggested name
`RAL_Draft_1A.tex`), reflowed into the RA-L template. Keep a running changelog in this
folder (`RAL_CHANGELOG.md`), matching the existing `CONDENSE_5A_CHANGELOG.md`
convention.

Cuts, largest first:

1. Both appendices (Vector Field Environments; Three-Robot Cluster Controller
   Equations with forward kinematics, inverse kinematics, inverse Jacobian). Cut
   entirely from the RA-L file. Roughly 1.5 to 2 journal pages. This material moves to
   the arXiv extended version (below), which the RA-L paper cites for derivations.
2. Section III, Multilayer Control Architecture. The robot controller and cluster
   space controller layers are previously published Kitts-lab work. Compress both to a
   short paragraph each with citations. Keep the adaptive navigation layer, which is
   this paper's contribution. Keep the architecture figure unmodified. Roughly half a
   page.
3. Fold Discussion into Results, the move already made in Draft 6a. Error Analysis
   compresses to its two strongest findings (dynamics-limited convergence confirmed by
   the idealized rerun; the sigma_p noise scaling matched against hardware scatter).
   The Comparison with Other Primitives subsection is superseded by Experiment A's
   table and shrinks to framing around it. Roughly half a page.
4. Simulation section (Section IV). Keep summary tables, cut per-field narrative
   walkthroughs. Roughly half a page.
5. Limitations and Future Work. Rewrite rather than trim blindly. Three current
   limitation/future-work items become results (time variation, classification,
   team size N = 4) and must move out of these lists. What remains: planar fields
   only, no flow advection on the platforms, update rate and dropout robustness
   untested, affine approximation at cluster scale.
6. References only if still over budget. Consolidate multi-citation clusters; do not
   cut the comparators or the works the novelty claim is positioned against.

Do not cut:

- The hardware section. The 157 convergence trials at 100 percent success and the 12
  orbital trials are the paper's strongest asset at RA-L. Tighten prose, keep all
  numbers and both result tables.
- The minimality argument (three robots). It can compress but must survive, since it
  is a listed contribution.

arXiv extended version. After the RA-L draft stabilizes, assemble an extended version
containing everything cut in items 1 through 4 plus the full experiment details, and
prepare it for arXiv posting. The author decides when to post. The RA-L paper cites it
for the cut derivations.

## Part 3. Integration and framing

1. Narrative placement. Experiment A lands where Section VI-B (primitive comparison)
   currently sits, promoted from qualitative to quantitative. Experiments B and C form
   a short new results subsection after the static simulation results, framed as
   simulation-only extensions, with one sentence noting the hardware validation is
   static printed fields.
2. Contribution list. Rewrite the four contributions to include time-varying
   validation and type identification. Candidate novelty sentence: "To our knowledge,
   this is the first method to recover both the location and topological type of
   vector field critical points from instantaneous distributed measurements." Before
   adding it, check the claim against the reference corpus using the paper-search MCP
   server (search for critical point localization, stagnation point estimation,
   distributed flow feature estimation, and adjacent phrasings) and report findings to
   the author. The author decides whether the sentence goes in. Do not claim "first
   time-varying" anything; manifold-tracking work also operates in time-varying flows.
3. Abstract. Rewrite after results are in and approved, adding classification and
   time-varying results in one sentence each.
4. In the paper, all time-varying claims are about instantaneous critical points. No
   transport or material-structure language.

## Part 4. Sequencing and checkpoints

| Phase | Work | Gate |
|---|---|---|
| 0 | Get RA-L template, reflow copy, measure real page count | Report count to author |
| 1 | Write all experiment designs (A, B, C, D) | Gate A sign-off per experiment |
| 2 | Implement and run approved experiments, write results memo | Gate B numbers review |
| 3 | Trim pass per Part 2 (can run parallel to Phase 2) | Author reviews compiled PDF and changelog |
| 4 | Integrate approved numbers, new figures and tables | Gate B applies to every number |
| 5 | Contribution list, novelty claim corpus check, abstract | Author decides on claim |
| 6 | Final compile (pdflatex twice), verify all figure paths resolve, page budget check | Author review |
| 7 | Assemble arXiv extended version | Author decides posting |

Build commands and venv paths are in repo CLAUDE.md. Note that
`docs/verify-figures.sh` takes the .tex path as an argument and can be pointed at the
new RA-L file.
