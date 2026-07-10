# Thesis Execution Plan

## SESSION COMPLETE

`thesis.tex` (in this same folder) is fully assembled: all 6 chapters
(Introduction, Decabot Testbed, Cluster Builder, Vector Field, Separatrix/
Okubo-Weiss, Conclusions), 4 appendices (Vector Field Environments,
Double-Gyre Analytic Forms, Cluster Kinematics for the Formations Used in
This Dissertation, Four-Robot Generalization), and a consolidated 66-entry
bibliography. Compiles clean at 150 pages (two `pdflatex` passes, zero `! `
errors, zero undefined or multiply-defined references; the only warning
present is the pre-existing, harmless `\IfDocumentMetadataT` hyperref/kernel
notice already confirmed present in the untouched `phd_thesis_template.tex`
before this session began, i.e. not a regression).

All 8 tasks from the original Step-by-Step Opus Prompts (Section 8 below)
are done: scaffold + notation (Prompt 1), Introduction (Prompt 2), Decabot
Testbed (Prompt 3), Cluster Builder including the added "Prior Work: The
Cluster Tree Inverse Jacobian" section crediting the author's Master's
thesis (Prompt 4, see addendum above), Vector Field paper (Prompt 5),
Separatrix/OW paper (Prompt 6), the 4-robot supplementary appendix
(Prompt 7), and Conclusions plus full bibliography consolidation (Prompt 8).

### What remains: 25 in-text `%% TODO: SYNC WITH LATEST DRAFT` markers

Grep the file for the exact string `TODO: SYNC WITH LATEST DRAFT` to find
all 25. They fall into four categories, none of which block a first full
read-through of the thesis:

1. **Real figures not yet generated for the thesis** (the majority, ~18
   markers): every figure in Chapters 3-5 is currently a `\framebox`
   placeholder with a comment pointing at the exact source script that
   generates it (e.g. `experiments/separatrix_clean_runs.py`,
   `experiments/ocean_hfr_2km_panel_progression.py`). Running those scripts
   and dropping the output PNGs into a `figures/` directory, then swapping
   `\framebox{...}` for `\includegraphics{...}`, closes these out
   mechanically.
2. **User decisions explicitly deferred, not resolved silently** (per this
   plan's original Section 6 and Section 10): the stiction value discrepancy
   (0.05 m/s text vs. 0.025 m/s code, Chapter 4), the Fig. 6 testbed photo
   possibly showing four robots while captioned three, and whether to
   finally derive the two "omitted for brevity" 6x6/12x12 kinematic
   Jacobians now that Chapter 3's general assembly algorithm could produce
   them mechanically.
3. **Chapter 3's four incomplete `[FILL]` bibliography entries** inherited
   from the source IDETC-II draft (a Mas and Kitts follow-on paper, the
   "2016 SCU toolbox thesis" likely but not confirmed to be entry 66 the
   Master's thesis, a virtual-structure citation, and a testbed
   self-citation already covered by entry 33). These need their actual
   titles/venues/years before they can be assigned permanent numbers.
4. **Cosmetic/front-matter placeholders**: the Acknowledgements and Abstract
   were drafted fresh for this session (not copied from a reviewed source)
   and are flagged for the user's own pass; the Glossary of Terms table
   lacks page numbers pending final chapter pagination.

None of these 25 items required a judgment call beyond "user must decide"
or "mechanical follow-up"; nothing was resolved by guessing.

### Notation and structural decisions made this session (for continuity)

- Cluster Builder (Chapter 3) kept as the general theory chapter, with a
  new "Prior Work" section crediting the Master's thesis, rather than being
  split into a thin narrative chapter plus a comprehensive appendix (user
  chose the first option when asked).
- SAS triangle notation `p, q, beta` preserved unchanged (Kitts-lab
  lineage); only the genuine three-way collisions (`r`, `alpha`, `J`) were
  renamed thesis-wide, exactly as Section 2 of this plan specifies.
- The 4-robot appendix is appendix-only with no forward-reference from
  Chapter 4's main body, and its discussion paragraph deliberately widens
  the "why hardware underperformed" explanation beyond calibration bias
  alone to include network/real-time factors, per the user's explicit
  instruction.
- Bibliography consolidation used Chapter 4's existing 51-entry numbering
  as the base (unchanged), appended Chapter 5's 14 unique entries as
  52-65, added the Master's thesis as 66, and dropped Chapter 5's
  incomplete self-citation of Chapter 4 (former entry 92) entirely, per
  the plan's own Section 7 recommendation. All of Chapter 5's in-text
  citation numbers were mechanically renumbered to match (verified by
  script, not by eye).


**Addendum (post-execution correction):** Chapter 3 is not new work invented for
this dissertation; it directly evolves the author's Master's thesis chapter,
"An Algorithm for Calculating the Inverse Jacobian of Multirobot Systems in a
Cluster Space Formulation" (source at
`trunk/Python_Simulations/Vector_Fields/VF_Robot/cluster_builder/Original Master's
Thesis Work/jacobian_propagation_paper_thesis version.tex`). That thesis's
"Cluster Tree Inverse Jacobian" algorithm is almost certainly the "2016 toolbox"
referenced anonymously in `cluster_kinematics3.html` and the IDETC-II draft's
`[FILL]` bibliography item 3. Chapter 3 as executed now includes a new Section
"Prior Work: The Cluster Tree Inverse Jacobian" immediately after the
Introduction, which presents that HTM/frame-propagation algorithm, states its
$O(\text{depth}^2)$ limitation in the thesis's own terms, and frames the three
PhD-level contributions (compositional grammar, single-pass assembly, consensus
gauge) as direct, explicit advances over it rather than free-standing new
theory. If bibliography item 3 (the `[FILL]` "2016 SCU toolbox thesis" entry,
see Section 6/7 below) is ever filled in, it should very likely cite this same
Master's thesis rather than a separate, unidentified 2016 document — confirm
with the user before assuming, since the HTML's "2016" date does not match this
Master's thesis's own likely date and may refer to an even earlier, separate
antecedent. This structural decision was made mid-execution (Chapter 3 chosen
to remain the general chapter with a new lineage section, rather than moving
the full chart library to an appendix) after the user confirmed the Master's
thesis connection; the general chart library, compositional grammar, and
validation remain in the Chapter 3 body as originally planned, unchanged in
their content, only with the new prior-work framing added ahead of them.


Blueprint for assembling `thesis.tex` from four existing papers plus supplementary
material. This document is written for an execution AI (Opus) to read and act on.
It is not the thesis itself; it contains no thesis-body LaTeX. Read this entire
file before touching any source paper.

Author: Christopher Waight. Advisor: Christopher A. Kitts. Santa Clara University,
Mechanical Engineering PhD. Template precedent: Shae Taylor Hart's 2023 SCU ME
PhD thesis (same advisor, same lab, same swarm-robotics lineage), found at
`Paper_Writing/PhD Thesis/ShaeThesisForexample of template.pdf`. The blank
class file at `Paper_Writing/PhD Thesis/phd_thesis_template.tex` is the actual
LaTeX skeleton to build inside (SCU `report` class, front matter macros, chapter
structure already stubbed) — use its packages, title-page macros, and TOC/LOF/LOT
setup as-is. Do not restart from Shae's PDF structurally; use it only to confirm
chapter-level hierarchy and front-matter ordering, since Opus cannot see its
LaTeX source, only the compiled PDF.

---

## 1. The Thesis Narrative (for the Introduction chapter)

**One-paragraph story.** This dissertation develops a progression of tools for
multirobot estimation and navigation in physical fields, moving from the hardware
substrate upward to increasingly higher-order field information. Chapter 2
establishes the physical testbed (Decabots, HSV-encoded printed vector fields,
neural-network sensor calibration) that every later experimental result depends
on. Chapter 3 generalizes the geometric machinery underneath every multirobot
formation used in the rest of the dissertation: a compositional grammar for
cluster-space kinematics that replaces one-off, per-formation Jacobian derivations
with a small library of composable charts, assembled in a single pass regardless
of team size. Chapter 4 shows what three robots in the minimal SAS triangle chart
from Chapter 3 can do with only first-order (Jacobian) field information: locate
and classify a vector field's critical points (sinks, sources, saddles, vortices)
using nothing but local measurements, then navigate to or orbit them. Chapter 5
extends this to second order: six robots in a pentagon-plus-center chart recover
the field's local Hessian structure, letting the team track dynamic, moving
boundaries, Okubo-Weiss zero-contours and separatrices, that no single critical
point can describe. The arc is deliberately monotonic in both team size and
estimation order (3 robots/first order to 6 robots/second order) and in
generality of the geometric substrate (one hand-derived triangle to an arbitrary
composition tree), so each chapter's control law is a strictly higher-order
sibling of the one before it, standing on the same cluster-space formalism.

**Contributions list (for Ch.1 Sec. "Contributions").**
1. A functional, validated hardware testbed for multirobot vector-field
   navigation using HSV-encoded printed fields and neural-network sensor
   calibration (R²=0.96 direction, 0.91 magnitude), removing the RGB
   channel-interference failure of the prior testbed generation.
2. A compositional grammar for cluster-space kinematics: any multirobot
   formation is a tree of three base charts (pair, SAS triangle, k-body star);
   the resulting inverse Jacobian is complete and square for any tree (Theorem 1)
   and assembles in a single depth-first pass, O(N) at bounded depth / O(N log N)
   balanced, replacing the prior method's O(depth^2) rotation-product
   recomputation. A consensus orientation gauge additionally removes the
   single-pointer-robot fragility near field critical points.
3. A distributed, model-free method for three robots to estimate a 2D vector
   field's local Jacobian from simultaneous local measurements alone (no prior
   field knowledge), proven minimal at exactly three robots, enabling closed-form
   critical point localization, eigenvalue-based type classification, and
   attraction/orbital control laws; validated in simulation (8 fields, 1000 Monte
   Carlo trials each, 100% convergence) and on hardware (157 convergence + 12
   orbital trials, 100% success, sub-centimeter bias).
4. A distributed, model-free method for six robots (pentagon-plus-center) to
   estimate a 2D flow's local Jacobian, Okubo-Weiss parameter, and Hessian from
   simultaneous local measurements, proven minimal at exactly six robots
   (Proposition 1) and almost-always sufficient (Proposition 2, ruling out the
   degenerate hexagon/conic trap), enabling two modified-Newton controllers that
   track the Okubo-Weiss boundary and the separatrix trench respectively, with a
   proven stability/traversal guarantee and validation on both a canonical
   double-gyre model and real ocean HFR current data.

**Publication map (for frontmatter "Acknowledgements" / list of publications).**

| # | Working title | Venue (target/actual) | Thesis chapter | Status |
|---|---|---|---|---|
| P1 | "A Functional Indoor Testbed for Multirobot Adaptive Navigation in Vector Field Environments" | IDETC/CIE 2025 (DETC2025-167604) | Ch. 2 | Submitted |
| P2 | "A Compositional Grammar for Cluster-Space Kinematics with Single-Pass Inverse Jacobian Assembly" | IDETC/CIE (Idetc-Paper-II), target venue TBD | Ch. 3 | Draft, mostly unwritten (see Sec. 3 below) |
| P3 | "Adaptive Navigation of Multirobot Systems to Critical Points in 2D Vector Fields" | IEEE/ASME Transactions on Mechatronics | Ch. 4 | Near-submission (Paper_Draft_4A.tex, audited) |
| P4 | "[Separatrix/Okubo-Weiss tracking title, see Paper_Draft_2A.tex]" | IEEE Transactions on Robotics (target) | Ch. 5 | Active draft, body complete |

Note for the Acknowledgements chapter (mirroring Shae Hart's template, which
states "A portion of this work has been published in [1] and [2], and portions
will be used for future publications"): P1 and P3 are submitted/near-submission
and can be cited as such; P2 and P4 should be described as "in preparation."

---

## 2. Global Notation Glossary

The four source papers were written independently and never cross-checked
against each other. Three real collisions exist; the rest of each paper's
notation is either already consistent or scoped narrowly enough (only used in
one chapter) that no change is needed. Per your decision, **do not rename `p`,
`q`, `beta` for the SAS triangle** — these are inherited directly from the
Kitts-lab cluster-space lineage (Kitts & Mas 2009, bibitem in every one of the
four papers) and should read identically across all four chapters. Fix only the
following three collisions, thesis-wide:

### Collision 1: the symbol `r`

Three unrelated meanings across the corpus:
- Orbital radial vector / magnitude, Ch. 4 (`r = p* - p_c`, Eq. 11 context).
- Third side length of the SAS triangle (the 1-3 diagonal), Ch. 4 Appendix B,
  Eq. 23 context (audit item E10 already flags this and recommends the fix
  below).
- Polar radius in the six canonical field definitions, Ch. 4 Appendix A.
- (Also used generically for "robot stacked-position vector" in the Ch. 3
  HTML/tex source — `r = [x_1,y_1,...]^T` — a fourth, looser usage.)

**Thesis-wide fix:**
- Keep `r` for the Ch. 4 orbital radial vector/magnitude (`\mathbf{r}`, `r`) —
  this is the most load-bearing use (appears in the named control law, Eq. 11)
  and is already how docs/notation.md defines it.
- Rename the SAS triangle's third side (1-3 diagonal) to `s` in Ch. 4 Appendix
  B, exactly as the paper's own audit (item E10) already recommends. This
  also resolves the reused polar-radius question below.
- Rename the polar radius in Ch. 4 Appendix A's field definitions to `\rho`
  (matches the pentagon-formation ring radius symbol already used in Ch. 5,
  keeping "radius of a circular/polar construction" visually distinct from
  "radius of an orbit" thesis-wide).
- The Ch. 3 stacked robot-position vector: rename to `\mathbf{q}_r` (robot-space
  vector) if it must appear at all in the thesis prose; prefer using `\mathbf{p}_i`
  stacked notation (`[\mathbf{p}_1;\mathbf{p}_2;\ldots]`) instead, consistent
  with Ch. 4/5's per-robot position notation, since the HTML's `r` was never
  disambiguated from cluster-state `q` internally (see Collision 3 below) and a
  thesis reader should not have to hold that ambiguity.

### Collision 2: the symbol `alpha` (three-way, not two-way)

CLAUDE.md already documents a two-way collision (`alpha_eig` vs `alpha_mom`).
Extracting Ch. 4's Appendix B surfaces a third, silent use: the SAS triangle's
other two vertex angles (audit item E10 calls these out and recommends
renaming). Ch. 3's HTML material has a fourth, structurally unrelated use
(`alpha_i = theta_ci - theta_v`, a spine-relative shape-angle offset in the
optional composition-tree extension) that never made it into the tex draft.
**All four are in scope for the thesis and must be kept visually and verbally
distinct:**

| Use | Where | Thesis symbol |
|---|---|---|
| Eigenvalue real part of field Jacobian J (spiral classification, Table I) | Ch. 4 Sec. II-A/Table I | `\alpha_{\text{eig}}` (as CLAUDE.md already mandates) |
| Momentum/discretization coefficient, `exp(-dt/tau)` | Ch. 4 Sec. III-A (robot dynamics), Ch. 5 Sec. IV-A (identical role) | `\alpha_{\text{mom}}` (as CLAUDE.md already mandates; Ch. 5's own paper already disambiguates this correctly in-text, per the extraction report, so no change needed there) |
| SAS triangle's other two vertex angles (unused elsewhere in the paper) | Ch. 4 Appendix B, Eqs. 25-26 | `\phi_1, \phi_3` (audit's own suggestion; these angles are not referenced anywhere else in the paper, so renaming is a pure textual substitution with no equation-logic changes) |
| Spine-relative shape-angle offset (composition-tree extension) | Ch. 3, if the optional extension is written into thesis prose at all | `\alpha_{\text{spine}}`, and flag explicitly in a footnote that it is unrelated to `\alpha_{\text{eig}}` and `\alpha_{\text{mom}}` |

Add a single consolidated "alpha collision" callout box at the start of the
Notation chapter (Ch. 1 Sec. 2, see Section 5 of this plan) listing all four,
since a reader hitting Ch. 3 after Ch. 4 will have `\alpha_{\text{eig}}` and
`\alpha_{\text{mom}}` already primed and needs the warning before hitting
`\phi_1/\phi_3` or `\alpha_{\text{spine}}`.

### Collision 3: the symbol `J` (and the cluster-space `q` / SAS `q` collision)

- **`J` collision:** field Jacobian (Ch. 4, Ch. 5: `J = [[u_x,u_y],[v_x,v_y]]`)
  vs. the 6x6 (Ch. 4 Appendix B) / 12x12 (Ch. 5 Appendix B) kinematic
  (inverse) Jacobian mapping cluster-shape velocities to robot velocities.
  Audit item E10 already recommends renaming the kinematic one. **Thesis-wide
  fix:** field Jacobian stays `\mathbf{J}` everywhere (it is the dissertation's
  central mathematical object — first-order in Ch. 4, embedded in the
  second-order estimator in Ch. 5). Rename every kinematic/cluster-space
  Jacobian, in Ch. 3, Ch. 4 Appendix B, and Ch. 5 Appendix B, to `\mathbf{J}_c`.
  This is one consistent renaming across three chapters, not three separate
  ones, since Ch. 3's compositional grammar produces exactly this kinematic
  Jacobian as its main deliverable.
- **`q` collision (found during Ch. 3 extraction, not previously flagged
  anywhere):** bold `\mathbf{q}` is the general cluster-space state vector in
  Ch. 3's HTML/tex source (`\mathbf{q} = J_c \mathbf{r}`, pose+shape stacked);
  scalar `q` is the SAS triangle's second side length (2-3 distance), which
  per your instruction is being preserved unrenamed. These do not actually
  collide on the page (bold vs. scalar, same rule structure as the bold-p /
  scalar-p hard rule already in CLAUDE.md) but the Ch. 3 source materials
  never state this disambiguation explicitly the way CLAUDE.md's p-rule does.
  **Fix:** add a new hard rule to the thesis Notation chapter, modeled exactly
  on the existing bold-p/scalar-p rule: "bold **q** is always the full
  cluster-space state vector (pose + shape, dimension 2N); scalar *q* is
  always the SAS triangle's second side length. There is no unbolded q for
  the state vector and no bold q for the side length." This is a new rule,
  not previously written anywhere in the repo, needed only because Ch. 3
  introduces the general cluster-state vector for the first time in the
  thesis's chapter order.

### Everything else: no change needed

- SAS triangle `p, q, beta` (per your instruction): unchanged across Ch. 4 and
  Ch. 5 (both use the same three symbols for the same triangle role; Ch. 5's
  12-state formation is three SAS pairs, each internally using this same
  `p/beta/q`-style parameterization per pair, so consistency is automatic, not
  imposed).
- Ch. 4's Table I (critical point classification: Center/Stable Spiral/Stable
  Node/Unstable Node/Unstable Spiral/Saddle by eigenvalue signature): carries
  into Ch. 5 unchanged as background, since Ch. 5's own classification is
  structural (D>0/D<0/D=0 regions) rather than a second eigenvalue table, and
  the two schemes do not overlap in claims (Ch. 5 never re-classifies isolated
  critical points, only the boundary/trench between regions).
- `D = \det(\mathbf{J})`, the Okubo-Weiss parameter (Ch. 5 only): no collision
  with anything in Ch. 2-4.
- Robot dynamics parameters `\tau` (actuator time constant), `\Delta t`
  (control period), `v_{max}`, stiction floor: consistent across Ch. 4 and
  Ch. 5 already (both papers use the identical momentum model). Note for the
  thesis text only (not a renaming, a factual reconciliation): Ch. 4's audit
  item E2 flags that its own text says stiction=0.05 m/s while its code
  default is 0.025 m/s; Ch. 5 separately uses stiction=0.002 for its ocean-HFR
  simulation (a different, larger-timescale operating point, not an error).
  State this explicitly in the thesis robot-dynamics section so a reader
  does not think Ch. 4 and Ch. 5 disagree about hardware.

### Table: full symbol reference for the thesis Notation chapter

Build this table directly from `docs/notation.md` (already the authoritative
reference for Ch. 4's own notation) plus the additions below it. Do not
re-derive docs/notation.md's content by hand; copy it, then append:

- Ch. 3 additions: `\mathbf{J}_c` (kinematic Jacobian, replaces bare `J` in
  cluster-kinematics contexts), `\mathbf{q}` (cluster-space state vector, bold,
  new hard rule above), tree/grammar symbols (`n`, `N`, `m`, `k`, `S_v`, `n_v`
  per the Ch. 3 extraction report Section 2), consensus-gauge symbols (`C, S,
  \rho_{\text{gauge}}`, — note this `\rho` is the *gauge alignment magnitude*,
  unrelated to the polar-radius `\rho` introduced above or the pentagon
  ring-radius `\rho` in Ch. 5; flag this as a fourth, minor rho-overload if all
  three appear in the same thesis — recommend subscripting all non-pentagon
  uses, e.g. `\rho_{\text{gauge}}`, `\rho_{\text{field}}`, reserving bare `\rho`
  for the Ch. 5 pentagon ring radius since that is the most-cited use across
  the largest number of equations).
- Ch. 5 additions: `D, Q` (Okubo-Weiss parameter and its incompressible
  identity), `\mathbf{H}_D` (Hessian of D), `\boldsymbol{\phi}(\tilde{\mathbf{p}})`
  (quadratic basis vector), `\boldsymbol{\Phi}` (6x6 formation matrix, capital
  Phi — distinct from lowercase `\phi_1,\phi_3` vertex angles introduced in the
  Collision 2 fix; flag this near-collision explicitly in the glossary since
  capital/lowercase phi look similar in some fonts), `D_{\text{capture}}`,
  `\alpha_{\text{mom}}` (already covered above), pentagon ring radius `\rho`.

Equate `\mathbf{p}^*` (bold-p-star, used in Ch. 4 for the vector field critical
point) explicitly with the Ch. 5 saddle-point notation the first time Ch. 5
uses it: Ch. 5's separatrix work refers to "the saddle points" of the double
gyre without consistently bold-starring them the way Ch. 4 does. Add one
sentence early in Ch. 5 (see Sec. 3 chapter mapping below): "As in Chapter 4,
we write a critical point of the flow, here a saddle, as \mathbf{p}^*."

---

## 3. Chapter-by-Chapter Mapping

Chapter numbers follow the template's existing numbering
(`phd_thesis_template.tex`) with the middle chapters expanded from its generic
3/4/5 into four content chapters. Front matter (title, copyright, signature,
Abstract, Acknowledgements) and the Glossary-of-Terms front-matter chapter
(present in Shae Hart's template, not yet in `phd_thesis_template.tex` — add it,
see Section 5) stay in the template's existing order.

### Chapter 1: Introduction
- **New material, not copy-paste.** Use Section 1 of this plan (Narrative and
  Contributions) as the drafting brief.
- Subsections: Motivation, Problem Statement, Contributions (four items above),
  Publication List (table above), Organization (one paragraph per chapter,
  restating the narrative arc in miniature).
- Do NOT copy any paper's own Introduction section wholesale; each paper's
  Introduction argues for that paper's specific contribution against that
  paper's specific related work, which is Ch. 1's Section "Problem Statement"
  and Ch. 2-5's own "Related Work" subsections, not Ch. 1's job. Ch. 1 argues
  for the dissertation as a whole.

### Chapter 2: The Decabot Hardware Testbed
- **Primary source:** `Paper_Writing/IDETC-Paper1/Multirobot_vector_test_bed_paper_Submitted.pdf`
  (DETC2025-167604 — confirm this is the intended file; the other
  similarly-named PDF in that folder, "IDETCE...submitted march12.pdf", is a
  DIFFERENT, simulation-only 2024 paper and must not be used here, see the flag
  in Section 6 below).
- Since the source is a PDF (no LaTeX available), Opus must retype the content
  as thesis LaTeX from the extracted report, not copy-paste raw source. Preserve
  the paper's own section order (Introduction -> Cluster Control Architecture
  -> Test Bed Design -> Results and Discussion -> System Limitations -> Future
  Work -> Conclusion) but fold "In this paper" framing into "In this chapter."
- Map: paper Sec. 2 (Cluster Control Architecture) -> Ch. 2 Sec. "Control
  Architecture Overview" (this becomes the FIRST place the reader sees the
  3-layer robot/cluster/navigation stack — Ch. 4 and Ch. 5 both reuse this
  exact architecture, so Ch. 2 should present it in enough generality that
  Ch. 4/5 can say "as described in Chapter 2" rather than re-deriving it).
- Map: paper Sec. 3 (Test Bed Design, incl. HSV encoding and NN calibration) ->
  Ch. 2 Sec. "Field Representation and Sensor Calibration."
- Map: paper Sec. 4 (Results: Tables 1-3, hue/saturation R², formation-holding
  accuracy) -> Ch. 2 Sec. "Validation."
- Map: paper Sec. 5-7 (Limitations, Future Work, Conclusion) -> compress into
  a short Ch. 2 closing section; do not give System Limitations its own
  full subsection since Ch. 4/5's own Limitations sections supersede it for
  everything except the physical/optical constraints (2D-only HSV, workspace
  bounds, battery life), which ARE still Ch. 2's unique content and should stay.
- **Strip:** the paper's own Abstract (redundant with thesis Abstract).
- Bibliography: merge this paper's 24 references into the thesis-wide
  bibliography (see Section 4 numbering directive below).

### Chapter 3: A Compositional Grammar for Cluster-Space Kinematics
- **Primary sources, in this priority order:**
  1. `trunk/Python_Simulations/Vector_Fields/VF_Robot/cluster_builder/cluster_kinematics3.html`
     — the actual math (per your decision: draft new LaTeX from this directly,
     since the IDETC-II tex is mostly unwritten).
  2. `Paper_Writing/Idetc-Paper-II/cluster_kinematics_idetc.tex` — use its
     Abstract, Sec. VI (Validation, fully written), Sec. VII (Conclusion, fully
     written), and its 9-entry bibliography verbatim; use its Sec. I-V as a
     SECTION-HEADING SKELETON ONLY (the prose/equations under those headings
     are `\fillnote` placeholders and must be replaced, not copied).
- This is the one chapter requiring net-new LaTeX authorship rather than
  primarily copy-paste-and-bridge. Budget accordingly — this chapter is
  larger-effort than its source-material page count suggests.
- Section mapping:
  - Ch. 3 Sec. "Introduction" <- combine IDETC-II Sec. I's stated motivation
    (chart concept, cluster-of-clusters, the O(depth^2) gap) with the HTML's
    framing prose.
  - Ch. 3 Sec. "Cluster-Space Charts and the Atom Library" <- HTML Sec. 1-4
    (SAS-3, Cross-Diagonal-4, H-Frame-4, Pentagon-6 full forward/inverse
    equations) + the signed-angle footnote (HTML-only) + all four numeric
    round-trip worked examples (HTML-only, port at least the SAS one already
    partially in the tex, ideally all four).
  - Ch. 3 Sec. "A Compositional Grammar for Formations" <- IDETC-II's Theorem 1
    statement/proof-sketch (already written) + HTML's two worked composition
    trees (9-robot, 8-robot) including the dimension-count annotations; note
    the HTML's inline SVG figures need to be redrawn as thesis figures (TikZ or
    similar), not screenshotted.
  - Ch. 3 Sec. "Single-Pass Inverse Jacobian Assembly" <- IDETC-II's Algorithm 1
    box (already written) + HTML's explicit leaf-rule matrices (size-1/2/3) and
    internal-node chain-rule formula (HTML-only) + the reusable `J_pair`/
    `J_pair^{-1}` 4x4 closed forms (HTML-only).
  - Ch. 3 Sec. "Orientation Gauge: Pointer versus Consensus" <- HTML Sec. 5 in
    full (Procrustes-optimum formula, circular-mean special case, companion
    stationarity identity, explicit partial-derivative formulas including the
    collapsed 1/(5r_i) form) — this entire section is HTML-only content, the
    tex has nothing here but placeholders.
  - Ch. 3 Sec. "Validation" <- IDETC-II Sec. VI verbatim (already fully
    written): round-trip exactness, Jacobian consistency, assembly scaling,
    block sparsity, gauge conditioning near a critical point. Three figures
    here are still framebox placeholders in the tex (`fig:scaling`,
    `fig:sparsity`, `fig:gauge`) — generate real plots from the
    `cluster_builder/` test suite before finalizing this chapter (see the TODO
    directive in Section 6).
  - Ch. 3 Sec. "Discussion / Limitations" <- HTML Sec. 8 (Edge cases: non-convex
    formations, collinear degeneracy, handedness/mirror ambiguity, coincident-
    robot singularity) + HTML's "cluster of one" / identity-node discussion +
    HTML's "choosing a structure" prose, none of which are in the tex at all.
  - Ch. 3 Sec. "Conclusion" <- IDETC-II Sec. VII verbatim (already written),
    update its Future Work list against what Ch. 4/5 actually deliver (e.g. the
    "dynamics layer" extension it lists as future work is still future work
    after this thesis; say so plainly).
- **Formation-type framing:** explicitly connect this chapter's N-robot
  generalization to the specific formations used later: Ch. 4 uses the SAS-3
  chart (3 robots), Ch. 5 uses the Pentagon-6 (well, pentagon-plus-center)
  chart's hub-and-spoke atom (6 robots). State this connection as its own short
  subsection ("Formations Used in This Dissertation") near the chapter's end,
  since it is the mechanical link between Ch. 3's generality and Ch. 4/5's
  specific instances.

### Chapter 4: First-Order Estimation: Critical Point Navigation
- **Primary source:** `Paper_Writing/Vector Field Paper/Paper_Draft_4A.tex`,
  read and copy-pasted section by section, in its own section order.
- Map: paper's own Introduction -> DELETE (superseded by Ch. 1 and by a new,
  short Ch. 4-specific "Related Work" paragraph that keeps only the
  vector-field-specific prior-art discussion, i.e., the parts of the paper's
  Introduction that argue specifically against scalar-field-only gradient
  methods and heading-only guidance, not the general multirobot survey
  material already covered in Ch. 1/Ch. 2).
- Map: Sec. II (Critical Point Detection and Control, all four subsections
  including Table I) -> copy near-verbatim, applying the Collision 1/2/3
  notation fixes above (the `r`, `alpha`, `J` renamings do NOT touch this
  section's content, only its symbols in Appendix B, so Sec. II itself is a
  clean copy).
- Map: Sec. III (MultiLayer Control Architecture) -> shorten substantially;
  the three-layer architecture and robot dynamics model are now Ch. 2's
  content (per the Ch. 2 mapping above) and Ch. 3's cluster-space-controller
  layer is now Ch. 3's content. Ch. 4 Sec. III should become a short "as
  established in Chapters 2-3, applied here with formation SAS-3" bridge,
  not a re-derivation. This is the paper's largest compression in the thesis.
- Map: Sec. IV (Simulation Results, Tables II-III) -> copy verbatim, applying
  the Sec. E2/E3/E4/E5 audit corrections (see the resolution list below).
- Map: Sec. V (Hardware Validation, Tables IV-V) -> copy verbatim, applying
  audit corrections E5, E9, E12, E13.
- Map: Sec. VI (Discussion: Error Analysis, Comparison, Limitations, Future
  Work) -> copy verbatim, applying audit correction E14; note this chapter's
  own "Limitations" (2D-only, static fields, 3-robot only) sets up Ch. 5's
  "second order / dynamic boundary" framing directly, so keep the
  three-robot-only limitation prominent here as the explicit hook Ch. 5 answers.
- Map: Appendix A (Vector Field Environments) -> becomes a thesis-level
  appendix (shared reference for Ch. 4's six analytical fields; Ch. 5's
  double-gyre field is DIFFERENT and gets its own appendix entry, do not merge
  them since they're mathematically unrelated field families).
- Map: Appendix B (Three-Robot Cluster Controller Equations) -> MOVE into
  Chapter 3 rather than keeping as a Ch. 4 appendix, since it is properly an
  instance of Ch. 3's SAS-3 atom (Ch. 3 Sec. "Cluster-Space Charts") and Ch. 3
  already derives the general triangle chart from which this is a special
  case. Cross-reference from Ch. 4 instead of duplicating. Apply the `s`,
  `\phi_1/\phi_3`, `\mathbf{J}_c` renamings here specifically, since this is
  where Collision 1 and Collision 2's third row actually live.
- **4-robot appendix:** see the dedicated instruction in Section 4 below (this
  is lost data being ADDED, not part of the paper's own content, so it's a
  new appendix, not a mapped section).
- **Audit corrections to apply while copying** (from `AUDIT_REPORT_4A.md`,
  confirmed still open per the extraction report): B1 (theta_c pi-disagreement,
  one-line atan2 fix), E1 (soften "distinct eigenvalues" claim, sink/source are
  repeated-eigenvalue star nodes), E2 (reconcile stiction 0.05 vs. 0.025 — ask
  the user which is correct before the thesis goes final, do not silently pick
  one), E3 (alpha=0.7 vs 0.717 sim assertion), E4 (disclose the 100-step
  transient discard in simulation orbital stats; note hardware Table V does NOT
  discard a transient, and say so), E5 (fix "within 1 m" to the correct 1x1 m
  box / 0.71 m max-distance description), E6 (state the actual convergence
  success criterion), E7 (explain noise-free orbital trajectory identity
  mechanistically), E8 (fix the k_t/r_d mislabel; add the actuator-lag/
  discretization breakdown), E9 (rounding reconciliation across Table IV/
  Conclusion), E12 (fix the backwards alpha=1 claim to alpha=0, and the 1e-15
  to the correct 1e-6 deadband), E13 (add the translation-invariance caveat to
  the kappa(A) optimality claim), E14 (replace the "suspected" large-radius
  bias explanation with the audit's proposed mechanism). E10's renamings are
  handled by the global notation fix in Section 2 above, not repeated here.
- **Do NOT apply:** the Phase 3 literature additions and the Fig. 6
  photo/caption mismatch fix are lower priority; flag both with `%% TODO: SYNC
  WITH LATEST DRAFT` rather than resolving them silently, since they involve
  editorial judgment calls (which citations to add, whether to reshoot a
  testbed photo) that are the user's call, not a mechanical copy-paste
  decision.

### Chapter 5: Second-Order Estimation: Separatrix and Okubo-Weiss Tracking
- **Primary source:** `Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex`,
  copy-pasted section by section, in its own section order (this paper's body
  is complete, per the extraction report, so this chapter is the most
  straightforward copy-paste of the four).
- Map: paper's own Introduction Sec. I-A through I-E (LCS/objectivity, OW/
  Q-criterion lineage, prior robotic manifold-tracking critique, distributed
  gradient/Hessian estimation lineage, cluster-space control lineage) -> keep
  MOST of this (unlike Ch. 4, this Introduction's related-work content is
  genuinely chapter-specific: LCS theory and Okubo-Weiss lineage appear nowhere
  else in the dissertation) but delete the redundant cluster-space-control
  paragraph (I-E), since Ch. 3 now owns that lineage discussion in full.
- Map: Sec. II (Problem Formulation, incl. double-gyre running example and
  Problem Statement) -> copy verbatim.
- Map: Sec. III (Distributed Second-Order Estimation: minimality Prop 1/2,
  Corollary 1, noise gains Lemma 1, closed-form recovery, sensitivity analysis)
  -> copy verbatim; this is the chapter's mathematical core and pairs directly
  against Ch. 4's Sec. II (three-robot first-order estimation) as the
  dissertation's central "here is the second-order sibling of Chapter 4's
  method" moment — consider adding one explicit cross-reference sentence here
  ("Where Chapter 4 recovered J from three robots and a linear model, we now
  recover J, D, and the Hessian H_D from six robots and a quadratic model").
- Map: Sec. IV (Adaptive Navigation Controllers: architecture, modified Newton
  step, both controllers, gains table) -> copy verbatim, but shorten Sec. IV-A
  (Control Architecture) since it repeats the three-layer stack already
  established in Ch. 2 and reused in Ch. 4 — same compression directive as
  Ch. 4 Sec. III above.
- Map: Sec. V (Stability Analysis, all theorems/lemmas) -> copy verbatim.
  Flag internally (do not silently rename) that this section's "Lemma 1"
  (common transverse contraction) is a DIFFERENT Lemma 1 than Sec. III's Lemma 1
  (noise gains) — same-numbered lemma in two different sections of the SAME
  paper. Recommend renumbering thesis-wide as Theorem/Lemma numbers unify
  across chapters (i.e., the thesis should have ONE running theorem/lemma
  counter, not per-chapter resets, if the SCU template's `report` class
  supports it — check `phd_thesis_template.tex`'s theorem-environment setup,
  currently absent, and add one, e.g. via `amsthm`, before this chapter is
  typeset, so this internal collision resolves automatically).
- Map: Sec. VI-VII (Simulation/Methods and Results) -> copy verbatim.
- Map: Sec. VIII-X (Discussion, Limitations, Future Work) -> copy verbatim;
  Limitations' basin-of-attraction paragraph (47.9%, n=1000) is a fixed,
  already-approved fact per plan.md, do not alter its wording.
- Map: Sec. XI (Conclusion) -> copy verbatim.
- Map: Appendix A (Double-Gyre Analytic Forms) -> becomes its OWN thesis
  appendix (do not merge with Ch. 4's six-field appendix, per the Ch. 4 mapping
  note above).
- Map: Appendix B (Pentagon Cluster Kinematics) -> MOVE into Chapter 3
  alongside Ch. 4's triangle appendix, for the same reason (it is an instance
  of Ch. 3's Pentagon-6 / hub-and-spoke atom). Apply the `\mathbf{J}_c`
  renaming here too (the paper's own text already says the 12x12 Jacobian's
  "explicit entries are omitted here for brevity" — this is a genuinely
  unfinished derivation, not just unported prose; either derive it in Ch. 3
  using the general assembly algorithm, which should make it a natural
  byproduct rather than new work, or keep the omission and say so honestly).
- **Front-matter note:** two `IEEEbiography` blocks in the source are `% TODO:
  insert bio` placeholders and the self-citation bibliography entry (key 92,
  the Ch. 4 paper citing itself) is incomplete (no volume/issue/page/year).
  Neither belongs in the thesis chapter itself (author bios and forward
  self-citations are conference/journal-paper furniture, not thesis content) —
  drop both when porting, do not carry the TODOs forward.
- **Known internal doc-drift to flag, not resolve:** the extraction report
  found that `plan.md`'s own last status update (2026-07-03) still lists Phase
  5/6 items as in-progress, but the actual `Paper_Draft_2A.tex` content shows
  those items (ocean HFR methods rewrite, Results B/C/D) already complete.
  Before finalizing this chapter, ask the user to confirm plan.md is simply
  stale (most likely) rather than the .tex being ahead of what was actually
  reviewed/approved. This is exactly the kind of judgment call that should NOT
  be resolved silently by an execution AI.

---

## 4. Integrating the Lost 4-Robot Data

Per your decision: **appendix only, no forward-reference from the main Ch. 4
body.** The appendix should present the material honestly as a real, substantial,
but not peer-reviewed hardware effort, and should explicitly foreground your
stated framing: simulation showed the 4-robot configuration matching or
exceeding 3-robot theoretical accuracy (noise-averaging benefit of an
overdetermined system), but hardware underperformed both the 3-robot hardware
baseline and the 4-robot simulation prediction, and the gap is most plausibly a
calibration, bandwidth, or other mechanical/implementation issue rather than a
flaw in the estimation theory itself.

**Appendix title:** "Four-Robot Generalization: Simulation and Hardware
Trials (Unpublished)."

**Contents, drawn from the extraction report:**
1. **Motivation paragraph.** State plainly that this generalizes Chapter 4's
   three-robot minimality result (which proves 3 is necessary, not that more
   robots can't help) to an overdetermined 4-robot case, and that this material
   predates the final, 3-robot-only submitted paper and was not peer-reviewed.
2. **Three estimator variants**, each described with its governing equations
   (available in `src/control/quad_primitives.py`, functions
   `dual_jacobian_center_finder`/`dual_jacobian_center_finder_advanced`,
   `estimate_center_planar`/`four_planar_center_finder`, and the cut Four-Robot
   Cluster Space Controller subsection from Older Paper Drafts 6-9's Appendix
   B):
   - **Planar Center** (single overdetermined least-squares plane fit through
     all 4 robots) — this is the one carried through to the final 258-trial
     hardware campaign in Drafts 7-9.
   - **Dual Jacobian** (splits the 4-robot square into two overlapping
     triangles, robots {1,2,3} and {2,3,4}, and averages two independent
     3-robot Jacobian estimates) — present in Drafts 4-6 only, claimed superior
     at tight orbital radii (0.111 m error, "58% improvement") but dropped by
     Draft 7; note this drop explicitly and, if the reason is known from your
     own recollection, ask you before asserting a reason in the text.
   - **Four-Robot Cluster Space Controller** (diagonal-based quadrilateral
     kinematic parameterization, 8 shape parameters: d1, d2, r1, r2, phi, x_c,
     y_c, theta_c) — the kinematic layer these estimators sit inside; present
     its forward equations from Drafts 6-9's Appendix B. Cross-reference this
     as a natural instance of Chapter 3's Cross-Diagonal-4 atom (the diagonal-
     based parameterization is exactly Ch. 3's Cross-Diagonal-4 chart), which
     lets this appendix additionally serve as a second worked example of
     Chapter 3's grammar.
3. **Hardware campaign numbers**, quoted exactly from Drafts 7-9 and confirmed
   against the surviving data in `trunk/robots_4/`:
   - 258 total hardware trials (236 convergence + 22 orbital), planned as 80
     3-robot-triangle trials/field plus 80 additional 4-robot-square vortex
     trials (3 saddle + 1 vortex lost to file-management errors, matching the
     same kind of loss pattern already disclosed for the 3-robot campaign in
     the main Ch. 4 text).
   - 4-robot vortex RMSE: 34.67 mm (vs. approximately 25 mm for the 3-robot
     hardware baseline in comparable fields).
   - Confirmed on-disk: `trunk/robots_4/4robot_vortex_stats.csv`
     (NumRuns=80, Bias_mm=16.14, Precision_mm=30.69) as the numerical backing
     for the RMSE claim; cite this file directly as the appendix's data source
     rather than re-deriving from raw run logs.
   - Simulation comparison: 4-robot simulated RMSE (13.44-15.48 mm) at or below
     3-robot simulated RMSE (12.65-18.45 mm) — this is the "theory says 4
     should help" half of the story.
4. **The honest discussion paragraph** (this is the one place in this appendix
   where you should write actual prose, not just port drafts): state that the
   drafts attributed the hardware gap to "motion capture calibration bias" and
   a "systematic y-direction offset reaching -0.086 m at 0.400 m commanded
   radius" (Draft 6's language), but per your own framing for this thesis,
   broaden this beyond a single blamed cause: the gap is consistent with
   calibration error, but could equally reflect WiFi/TCP bandwidth contention
   with a fourth robot on the same control loop, additional latency in the
   10 Hz cycle from a larger Simulink model
   (`trunk/robots_4/chris_4R_all_primitives.slx`), or some other unquantified
   mechanical/implementation factor. Do not assert a single root cause more
   strongly than the surviving data supports. This paragraph should read as
   a research-integrity statement, not a rationalization.
5. **Figures/tables**, sourced from `trunk/robots_4/Results/` (per-group
   trajectory plots, `average_trajectories.fig/png`, boxplot/histogram
   outputs) — regenerate as thesis-quality figures rather than reusing the
   `.fig`/`.png` files directly, since MATLAB `.fig` figure styling will not
   match the thesis's LaTeX figure conventions.
6. Do not include the scalar-field Hessian/Newton-saddle 4-robot material from
   `Paper_draft_0.tex` in this appendix — that thread belongs to the dead
   scalar-field paper (per existing project memory) and mixing it in here
   would misrepresent this appendix's scope as broader than "4-robot
   generalization of the vector-field estimator."

---

## 5. The "Copy-Paste and Bridge" Directive

Applies to Chapters 2, 4, and 5 (Chapter 3 is majority new-authorship per
Section 3 above, and per your decision on the Cluster Builder gap).

1. **"In this paper" -> "In this chapter."** Apply globally, including
   variants: "this work presents" -> "this chapter presents," "our paper" ->
   "our chapter" / "this dissertation," "the authors" -> stays as-is only if
   referring to a still-distinct external work being cited, otherwise -> "we."
2. **Strip every copied paper's own Abstract.** The thesis has one Abstract
   (front matter); no chapter needs its own. If a paper's Abstract contains a
   headline number not stated anywhere else in that paper's body (check before
   deleting), move that number into the chapter's own Introduction or
   Conclusion rather than losing it.
3. **Strip every copied paper's own Acknowledgements section** (Ch. 2's source
   paper has one, naming Scot Tomer, Jiayi Wang, Michael Waight, Sara Alvarez)
   — fold these names into the thesis's single front-matter Acknowledgements
   chapter instead, do not create per-chapter acknowledgement blocks.
4. **Write transitional paragraphs at every chapter boundary**, roughly
   150-250 words each, that do the following specific work (do not write generic
   "in the next chapter we will..." filler):
   - End of Ch. 2 -> start of Ch. 3: the testbed is fixed at 3 robots in this
     chapter's validation; the next chapter asks what changes if the team size
     or formation shape is different, motivating the general kinematic grammar.
   - End of Ch. 3 -> start of Ch. 4: having established that any formation
     tree assembles a valid inverse Jacobian, the simplest non-trivial instance
     (a single SAS-3 triangle, three robots) is exactly the formation the next
     two chapters use; Ch. 4 asks what a three-robot team can infer about the
     FIELD it's sitting in, not just about its own shape.
   - End of Ch. 4 -> start of Ch. 5: Ch. 4's explicit Limitations section
     already states "three-robot only" and "static fields, isolated critical
     points only" as open limitations; Ch. 5's transition should explicitly
     quote or closely paraphrase that limitation and state that six robots and
     a quadratic (not linear) field model is exactly the generalization that
     answers it, then immediately state what a single critical point cannot
     describe that a boundary/trench can (i.e., motivate WHY a separatrix
     needs second-order information, not just why more robots exist).
   - End of Ch. 5 -> start of Ch. 6 (Conclusion): synthesize, do not repeat;
     look back across all four chapters' "Future Work" sections and identify
     any that now point at each other (e.g., Ch. 4's Future Work already
     mentions "separatrices/heteroclinic orbits via eigenvector alignment" as
     future work — flag explicitly in the Conclusion that Ch. 5 IS that future
     work, delivered).
5. **Do not silently merge sections that only superficially match.** Ch. 2's
   "Cluster Space Controller Layer" and Ch. 4's "Cluster Space Controller
   Layer" describe the SAME architecture but were written for different
   papers; when compressing Ch. 4's copy per Section 3's instruction, verify
   the SAS-3-specific numbers (p=q=0.35 m, beta=1.05 rad from Ch. 2's testbed
   paper) match Ch. 4's own numbers before treating them as redundant. If they
   differ, do not silently pick one; flag with `%% TODO: SYNC WITH LATEST
   DRAFT` and ask.

---

## 6. Draft Syncing and Placeholders

Insert `%% TODO: SYNC WITH LATEST DRAFT` (exact string, so it is
`grep`-able across the whole thesis file) at every one of the following
locations, identified during source extraction:

1. **Ch. 3, three Validation-section figures** (`fig:scaling`, `fig:sparsity`,
   `fig:gauge` in the IDETC-II source): currently unfilled `\framebox`
   placeholders even in the "fully written" Validation section. Generate real
   plots from the `cluster_builder/tests/` suite before removing this TODO.
2. **Ch. 3, HTML-sourced SVG figures** (the two worked composition trees,
   9-robot and 8-robot; the hub-and-spoke star diagram): need redrawing as
   proper thesis figures (TikZ recommended, to stay text-searchable and
   stylistically consistent with the rest of the document), not screenshots
   of the HTML's inline SVGs.
3. **Ch. 3, IDETC-II bibliography entries 2, 3, 8, 9**: marked `[FILL]` in the
   source (missing title/venue/year for a Mas & Kitts follow-on paper, the
   2016 SCU thesis that originated the O(depth^2) toolbox, a virtual-structure/
   formation-control citation, and the self-citation of the testbed paper).
   Resolve these against the actual bibliography before the thesis bibliography
   is assembled (see Section 7 numbering note) — they cannot be merged into a
   single numbered thesis bibliography while incomplete.
4. **Ch. 4, Appendix B's 6x6 kinematic Jacobian**: paper states its "explicit
   entries are omitted... for brevity." Decide whether the thesis appendix
   should finally write these out (natural, since Ch. 3's general assembly
   algorithm should produce them as a specific instance) or keep the omission
   with a forward-reference to Ch. 3's general method as the reason it's safe
   to omit.
5. **Ch. 4, audit item E2** (stiction 0.05 vs. 0.025 m/s): explicitly listed
   above as a user decision, not a mechanical fix. Do not resolve silently.
6. **Ch. 4, Fig. 6 photo/caption mismatch** (`testbed_with_four.png` appears to
   show four robots, captioned "three Decabot rovers"): flag, do not silently
   swap the photo or reword the caption, since the correct fix depends on which
   photo actually exists and matches the described setup.
7. **Ch. 5, self-citation bibliography entry** (key 92, the Ch. 4 paper citing
   itself with no volume/issue/page/year): resolve once Ch. 4's own T-Mech
   submission has those details, or drop the forward self-citation from the
   thesis version entirely, since within a single thesis Ch. 5 can just say
   "as shown in Chapter 4" instead of citing it as an external paper.
8. **Ch. 5, plan.md vs. Paper_Draft_2A.tex sync status**: explicitly flagged
   above in Section 3; the execution AI should raise this to the user rather
   than assume either document is authoritative.
9. **Ch. 5, Appendix B's 12x12 inverse kinematic Jacobian**: same "omitted for
   brevity" situation as item 4 above, same resolution options.
10. **Any thesis-wide theorem/lemma renumbering** needed to resolve Ch. 5's
    internal same-numbered "Lemma 1" collision (estimation-section noise gains
    vs. stability-section transverse contraction) — flagged in Section 3 above,
    needs an `amsthm`-style running counter added to `phd_thesis_template.tex`
    before Ch. 5 is typeset.

---

## 7. Bibliography Consolidation

The four source papers share a citation pool (many entries appear in two or
three of them verbatim, since they come from the same lab lineage), but each
paper currently numbers its bibliography independently (`\bibitem{1}`...
`\bibitem{51}` in Ch. 4; a non-consecutive key set `1,2,3,4,5,7,8,9,10,12,...`
up to `111` in Ch. 5, apparently drawn from a shared numbering pool with Ch. 4
already). Per the existing CLAUDE.md citation rules (hand-formatted
`\bibitem{N}` blocks, no BibTeX, plain `[N]` inline citations, no `\cite{}`):

1. Build ONE thesis-wide `thebibliography` block at the end of the document
   (after Ch. 5 / before Appendices, matching the template's existing
   structure).
2. De-duplicate: several entries are IDENTICAL across papers (e.g., Kitts &
   Mas 2009 "Cluster space specification and control" appears in Ch. 2's
   24-entry list, Ch. 4's 51-entry list, AND Ch. 5's list). Assign each unique
   reference exactly one thesis-wide number; every chapter's inline `[N]`
   citations must be renumbered to match the unified list.
3. Recommended ordering: preserve rough first-appearance order by chapter
   (Ch. 2's unique references first, then Ch. 3's, then Ch. 4's remaining
   unique ones, then Ch. 5's remaining unique ones), rather than alphabetizing,
   to match the existing hand-numbered convention already used in each source
   paper.
4. The four `[FILL]` entries from Ch. 3's bibliography (item 3 in Section 6
   above) must be resolved before this consolidation can complete; do not
   assign them a permanent thesis number while incomplete.
5. Ch. 5's incomplete self-citation (key 92) should likely be DROPPED rather
   than merged, per Section 6 item 7 above (within a thesis, cite the chapter,
   not an external paper for a work that is itself a chapter of the same
   document).

---

## 8. Step-by-Step Opus Prompts

Use these in order. Each assumes the prior prompt's output is already written
to `thesis.tex` (or a chapter-specific file, if you choose to draft chapters
separately before merging — recommended, given the size of this project).
Each prompt should be pasted verbatim as a new instruction; do not summarize
them into shorter task requests, since the specificity is what keeps the
execution AI from improvising decisions that belong to the user.

**Prompt 1 (scaffold + notation chapter):**
"Using `Paper_Writing/PhD Thesis/phd_thesis_template.tex` as the base LaTeX
skeleton and `Paper_Writing/PhD Thesis/ShaeThesisForexample of template.pdf`
as the structural precedent for front-matter ordering, set up `thesis.tex`
with: title page, copyright, signature page, Abstract (placeholder), Acknowledgements
(placeholder), a new Glossary-of-Terms chapter modeled on Shae Hart's template
(a single alphabetized symbol table, distinct from the inline notation
discussion inside Chapter 1), Table of Contents, List of Figures, List of
Tables. Then write Chapter 1's Section 'Notation and Definitions' using the
full Global Notation Glossary from Section 2 of Thesis_Execution_Plan.md,
including the four-way alpha collision callout, the r/J renamings, and the new
bold-q/scalar-q hard rule. Add an `amsthm`-based running theorem/lemma counter
to the preamble (needed later for Chapter 5). Do not write any chapter body
content yet."

**Prompt 2 (Chapter 1, Introduction):**
"Write Chapter 1 (Introduction) of thesis.tex using Section 1 of
Thesis_Execution_Plan.md verbatim as your drafting brief: the one-paragraph
narrative, the four numbered contributions, and the publication map table.
Do not copy any sentence directly from any of the four source papers' own
Introductions; this chapter argues for the dissertation as a whole, and each
source paper's Introduction argues for that paper's narrower contribution
against that paper's specific prior art, which belongs in that chapter's own
Related Work subsection instead. Include Motivation, Problem Statement,
Contributions, and Organization sections."

**Prompt 3 (Chapter 2, Decabot Testbed):**
"Write Chapter 2 (The Decabot Hardware Testbed) of thesis.tex by retyping the
content of `Paper_Writing/IDETC-Paper1/Multirobot_vector_test_bed_paper_Submitted.pdf`
(DETC2025-167604) as thesis LaTeX, following the Chapter 2 mapping in Section 3
of Thesis_Execution_Plan.md exactly: fold 'In this paper' into 'In this
chapter,' strip the paper's own Abstract, fold its Acknowledgements names into
the thesis's front-matter Acknowledgements instead of a chapter-level one,
compress its Limitations/Future Work/Conclusion into one short closing section,
and keep only the physical/optical constraints from its Limitations (the rest
is superseded by Chapters 4 and 5's own Limitations sections). Do NOT use
`Paper_Writing/IDETC-Paper1/IDETCE - Adaptive Navigation of Multirobot Systems
in 2D Vector Fields - submitted march12.pdf` as a source; that is a different,
simulation-only 2024 paper. Merge its 24-entry bibliography into a running
thesis bibliography list (do not number permanently yet, that happens after
all four chapters are drafted, per Section 7)."

**Prompt 4 (Chapter 3, Cluster Builder — the effortful one):**
"Write Chapter 3 (A Compositional Grammar for Cluster-Space Kinematics) of
thesis.tex. This is the one chapter requiring substantial new authorship, not
copy-paste: per Section 3 of Thesis_Execution_Plan.md, the actual math lives
almost entirely in
`trunk/Python_Simulations/Vector_Fields/VF_Robot/cluster_builder/cluster_kinematics3.html`
(an HTML export with the full forward/inverse kinematics for four base charts,
worked numeric examples, the consensus-gauge derivation, and edge-case
discussion), while `Paper_Writing/Idetc-Paper-II/cluster_kinematics_idetc.tex`
has only its Abstract, Validation section (VI), Conclusion (VII), and
bibliography actually written; its Sections I-V are `\fillnote` placeholders.
Use the tex file's Abstract/Validation/Conclusion/bibliography verbatim (with
notation fixes applied), and draft fresh thesis prose and equations for
everything else directly from the HTML, following the detailed section-by-
section content map in Section 3 of Thesis_Execution_Plan.md (charts and atom
library, compositional grammar and Theorem 1, single-pass Jacobian assembly,
orientation gauge). Redraw the HTML's inline SVG figures as TikZ figures rather
than screenshotting them. Insert the `%% TODO: SYNC WITH LATEST DRAFT` markers
listed in Section 6, items 1-4, at the three unfilled validation figures and
the incomplete bibliography entries. Apply the r/alpha/J notation fixes from
Section 2 throughout, and add the new bold-q/scalar-q hard rule the first time
the cluster-space state vector appears."

**Prompt 5 (Chapter 4, Vector Field paper):**
"Write Chapter 4 (First-Order Estimation: Critical Point Navigation) of
thesis.tex by copy-pasting `Paper_Writing/Vector Field Paper/Paper_Draft_4A.tex`
section by section, following the Chapter 4 mapping in Section 3 of
Thesis_Execution_Plan.md: delete the paper's own Introduction in favor of a
short chapter-specific Related Work paragraph, substantially compress Section
III (Control Architecture) since it duplicates Chapter 2's content, keep
Sections II/IV/V/VI close to verbatim, and move Appendix B (Three-Robot
Cluster Controller Equations) into Chapter 3 instead of keeping it here (cross-
reference it from this chapter). Apply every audit correction listed in
Section 3's 'Audit corrections to apply while copying' bullet, EXCEPT
E2 (stiction value) and the Fig. 6 photo/caption mismatch, both of which must
be flagged with `%% TODO: SYNC WITH LATEST DRAFT` and left for the user to
decide rather than resolved. Apply the r-to-s, alpha-to-phi_1/phi_3, and
J-to-J_c renamings specifically in the (now-relocated) Appendix B content.
Write a transitional paragraph at the end of this chapter per Section 5, item
4, of Thesis_Execution_Plan.md (bullet 'End of Ch. 4 -> start of Ch. 5'),
quoting or closely paraphrasing this chapter's own stated three-robot/static-
field limitation as the explicit hook for Chapter 5."

**Prompt 6 (Chapter 5, Separatrix/Okubo-Weiss paper):**
"Write Chapter 5 (Second-Order Estimation: Separatrix and Okubo-Weiss
Tracking) of thesis.tex by copy-pasting
`Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex` section by section,
following the Chapter 5 mapping in Section 3 of Thesis_Execution_Plan.md: keep
most of the Introduction's related-work content (it is chapter-specific, not
duplicative, except for its cluster-space-control paragraph I-E, which
duplicates Chapter 3 and should be cut), compress Section IV-A (Control
Architecture) since it duplicates Chapter 2, keep everything else close to
verbatim, and move Appendix B (Pentagon Cluster Kinematics) into Chapter 3
instead of keeping it here. Apply the J-to-J_c renaming in the relocated
appendix. Flag, do not resolve, the same-numbered-but-different 'Lemma 1' in
the Estimation section versus the Stability section (Section III's noise-gains
lemma vs. Section V's transverse-contraction lemma) using the thesis-wide
theorem counter set up in Prompt 1. Drop both `IEEEbiography` placeholder
blocks and the incomplete self-citation of the Chapter 4 paper (bibliography
key 92); within a thesis, refer to Chapter 4 directly instead of citing it as
an external work. Insert the plan.md-vs-tex sync flag from Section 6, item 8,
of Thesis_Execution_Plan.md as a `%% TODO: SYNC WITH LATEST DRAFT` comment
near this chapter's Results section, and ask the user to confirm plan.md is
simply stale before removing it."

**Prompt 7 (4-robot appendix):**
"Write the appendix titled 'Four-Robot Generalization: Simulation and Hardware
Trials (Unpublished)' following Section 4 of Thesis_Execution_Plan.md exactly:
present the Planar Center, Dual Jacobian, and Four-Robot Cluster Space
Controller variants with their governing equations (sourced from
`src/control/quad_primitives.py` and the cut Appendix B in
`Paper_Writing/Vector Field Paper/Older Paper Drafts/Paper_Draft_7.tex` through
`Paper_Draft_9.tex`), the 258-trial hardware campaign numbers exactly as
quoted in this plan, and the honest discussion paragraph about the sim/hardware
gap: note simulation showed the 4-robot configuration matching or exceeding
3-robot accuracy, but hardware underperformed both the 3-robot baseline and the
4-robot simulation prediction, and state that the gap is consistent with
motion-capture calibration bias but could equally reflect network bandwidth
contention or other unquantified mechanical/implementation factors — do not
assert a single cause more strongly than the data supports. Cross-reference the
Four-Robot Cluster Space Controller's diagonal parameterization as an instance
of Chapter 3's Cross-Diagonal-4 atom. Do NOT forward-reference this appendix
from Chapter 4's main body. Do not include the scalar-field Hessian/Newton
material from `Paper_draft_0.tex`; that belongs to the separate, dead
scalar-field paper thread, not this appendix."

**Prompt 8 (Conclusion + bibliography consolidation + final pass):**
"Write Chapter 6 (Conclusions) of thesis.tex: Summary (synthesize, do not
repeat, the four chapters' individual conclusions; explicitly note that
Chapter 4's own Future Work item 'separatrices/heteroclinic orbits via
eigenvector alignment' is answered by Chapter 5), Future Work (aggregate
across all four chapters' Future Work sections, removing duplicates), and
Final Thoughts. Then consolidate the bibliography per Section 7 of
Thesis_Execution_Plan.md: merge all four chapters' `thebibliography` entries
into one thesis-wide numbered list, de-duplicating identical references (e.g.
Kitts & Mas 2009 appears in three of the four source papers), renumbering every
inline `[N]` citation across all chapters to match, dropping Chapter 5's
incomplete self-citation of Chapter 4, and leaving Chapter 3's four `[FILL]`
entries unresolved with a `%% TODO: SYNC WITH LATEST DRAFT` marker pending user
input. Finally, grep the whole document for `%% TODO: SYNC WITH LATEST DRAFT`
and produce a numbered list of every remaining open item as your final output,
so the user can resolve them in one pass."

---

## 9. Files Referenced in This Plan (for quick lookup)

- `Paper_Writing/PhD Thesis/phd_thesis_template.tex` — LaTeX skeleton to build in.
- `Paper_Writing/PhD Thesis/ShaeThesisForexample of template.pdf` — structural precedent.
- `Paper_Writing/IDETC-Paper1/Multirobot_vector_test_bed_paper_Submitted.pdf` — Ch. 2 source (DETC2025-167604).
- `trunk/Python_Simulations/Vector_Fields/VF_Robot/cluster_builder/cluster_kinematics3.html` — Ch. 3 primary math source.
- `Paper_Writing/Idetc-Paper-II/cluster_kinematics_idetc.tex` — Ch. 3 skeleton + Validation/Conclusion/bibliography.
- `Paper_Writing/Vector Field Paper/Paper_Draft_4A.tex` — Ch. 4 source.
- `Paper_Writing/Vector Field Paper/AUDIT_REPORT_4A.md` — Ch. 4 open audit items.
- `docs/notation.md` — base notation table, extend per Section 2.
- `Paper_Writing/Separatrix_and_OW_Paper/Paper_Draft_2A.tex` — Ch. 5 source.
- `Paper_Writing/Separatrix_and_OW_Paper/teaching_notes.tex` — optional pedagogical reference for Ch. 5 prose.
- `plan.md` (repo root) — Ch. 5 working tracker; check for staleness per Section 6, item 8.
- `Paper_Writing/Vector Field Paper/Older Paper Drafts/Paper_Draft_7.tex` through `Paper_Draft_9.tex` — 4-robot lost-data source.
- `src/control/quad_primitives.py`, `src/robot/quad_cluster.py`, `src/control/quad_kinematics.py` (under `trunk/Python_Simulations/Vector_Fields/VF_Robot/`) — 4-robot code, appendix source.
- `trunk/robots_4/4robot_vortex_stats.csv`, `trunk/robots_4/Results/` — 4-robot hardware data.

---

## 10. Open Questions for the User (not resolved by this plan)

1. Ch. 4 audit item E2: is the correct stiction value 0.05 m/s (paper text) or
   0.025 m/s (code default)? Needed before Tables II/III can be called final.
2. Ch. 4 Fig. 6: does a photo of exactly three Decabots on the printed map
   exist, or does the testbed_with_four.png image need to be reshot/relabeled?
3. Ch. 5: confirm plan.md's "Phase 5/6 still in progress" status is stale
   relative to the actual (apparently complete) state of Paper_Draft_2A.tex.
4. Ch. 3 bibliography: the four `[FILL]` entries (Mas & Kitts follow-on paper,
   the 2016 SCU thesis, a virtual-structure citation, the testbed self-
   citation) need their actual titles/venues/years before the thesis
   bibliography can be consolidated.
5. Ch. 3 Appendix B (Ch. 4's relocated triangle kinematics) and Ch. 5's
   relocated pentagon kinematics both have "omitted for brevity" Jacobian
   entries. Should the thesis finally write these out in full (recommended,
   since Chapter 3's general assembly algorithm should produce them
   mechanically), or preserve the omission with a forward-reference?
