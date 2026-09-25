# Writing Guide: Chris Waight

This guide is for agents drafting or editing technical prose as Chris Waight.
It governs every section of a paper or thesis.

Every rule is a test you can run on a draft. Most rules carry a Bad/Good pair
taken from real edits. Learn from the pair, not only the rule.

The success test for any draft has two halves. Would Chris be proud to have
written it, and would people who know his writing believe he did? The first
half alone produces generic polished prose. The second half alone produces a
transcript of his weakest habits. Aim for his best, in his voice.

---

## Part 1. Principles

Every rule below serves one of these. A rule that serves none should go.

1. **Know what you think before you write.** Most bad prose is unfinished
   thinking. If you cannot state a paragraph's point in one sentence, do not
   draft it yet.
2. **Write for one reader, and never make that reader hold an undefined
   thing.** The reader is a controls or robotics researcher who has not read
   the fluid dynamics literature.
3. **Every sentence does work.** Cut what does no work, including things that
   are true and interesting.
4. **Claims match evidence exactly.** No stronger, and no hedgier.
5. **Explain why, not just what.** Chris's strongest habit is naming the
   mechanism behind a result or a failure. Protect it.

---

## Part 2. Process

**2.1 Layout first, then agree, then write.** For any rewrite longer than a
paragraph, propose a paragraph-by-paragraph layout, stating what job each
paragraph does, and wait for approval. Chris prefers approval-gated workflows
for consequential changes.

**2.2 His sketch beats your draft.** When Chris sketches his own version, even
roughly, follow its structure and keep his sentences where he supplied them.
His sketches are closer to his voice than a polished agent draft.

> Chris's sketch: "Since each robot has its own sensors, the measurement noise
> of each robot is independent."
>
> Final text: "Each robot carries its own sensor, so the measurement noise of
> one robot is independent of the others."

**2.3 Rewrite only the scope asked.** "Only rewrite II-C" means II-C. List
changes other sections will need in your notes. Do not make them.

**2.4 Trace the ripple before you cut.** Before deleting an equation, a
definition, or a named result, search the whole document for references to it
and list them with a proposed fix.

> Cutting the splitting identity from II-D broke references in II-E ("by
> (7)"), IV-D ("(9)"), IV-F and the Fig. 8 caption ("(8)"), IV-G
> ("splitting-identity reading"), and the introduction's contribution
> paragraph. Each needed a named fix before the cut was safe.

**2.5 Keep notes after a draft short.** Three to five bullets covering what
changed, what it breaks elsewhere, and open decisions. End with at most one
question.

---

## Part 3. Section rules

**3.1 The rock show rule.** A result earns its place only if something later
in the paper uses it. Test: delete it. If nothing downstream breaks, it goes,
however elegant it is. An interesting result in the wrong place is an
orchestra solo at a rock show. Exiled material is welcome in the thesis or an
appendix.

> Bad (Draft 10, II-D): The full splitting identity
> D = det S + det W = s₁s₂ + ω²/4, with the trace identity for 2×2 matrices
> and a transverse derivative formula, to support two facts used later.
>
> Good: The two facts, each derived in one line without the identity. The
> frame shift comes from det(J + ΩR) = D + Ωω + Ω². The shared trenches come
> from J = S where the vorticity vanishes, so D = s₁s₂ = −s₁² in
> incompressible flow.

> Bad (Draft 10): "The Okubo-Weiss parameter [21], [22] satisfies
> Q = ¼(∇·v)² − D under the quarter-normalized convention of [24] and
> Q = (∇·v)² − 4D under the common unhalved-strain convention, so Q ∝ D
> either way for incompressible flow, which covers the double gyre but not
> the ocean record of Section IV-G."
>
> Good: "For incompressible flow the determinant is a negative multiple of
> the Okubo-Weiss parameter [21], [22], so the two carry the same partition
> of the flow."

**3.2 One level of abstraction per section.** A section about the estimator
talks about coefficients. A section about the surrogates talks about D and s₁.
Mixing them produces forward references.

> Bad (Draft 10, II-C, before D is defined in II-D): "Under this channel each
> determinant term pairs one â with one b̂ from independent components, so
> the cross terms vanish in expectation and D̂, ∇D̂, and Ĥ_D are unbiased at
> all noise levels."
>
> Good: II-C stops at the coefficients. "Both are zero-mean, so the
> coefficient estimates are unbiased." The D consequence moves to II-D.

**3.3 Define before use. A forward reference is a symptom.** If a section
needs to point forward to make sense, content is in the wrong place. Move the
content. Do not add a pointer. Backward references are fine.

> Bad: "Their effect on the surrogates follows in Section II-D."
>
> Good: No forward pointer in II-C. II-D points back: "The coefficient errors
> are zero-mean (Section II-C), so a product can be biased only by a
> correlation between its two errors."

**3.4 Parallel content gets parallel structure.** When two things are
compared, give them the same skeleton in the same order, so the reader sees
the difference and not the formatting.

> II-D after the rewrite. The D field and the s₁ field each run as the
> pipeline from the fitted Jacobian, then an interpretation note, then a bias
> note. II-C uses run-in heads for Formation, Measurement Noise, and Position
> Noise. In IEEEtran, `\subsubsection{}` renders as an italic run-in head.

**3.5 A paragraph does one job, and its first sentence names the job.**

> Good: "Each robot carries its own sensor, so the measurement noise of one
> robot is independent of the others." Everything after it in the paragraph
> is about measurement noise.

**3.6 Physical reason before formula.** State why the model holds in the
world, then write the model. A noise model with no physical reason reads as
an arbitrary assumption.

> Bad (Draft 10): "Measurement noise is additive on what each robot reads,
> ... with η ∼ N(0, σ²_uv), independent across robots, components, and
> cycles."
>
> Good: "Each robot carries its own sensor, so the measurement noise of one
> robot is independent of the others. Furthermore, the estimator uses only
> instantaneous readings, so each cycle's fit sees only that cycle's noise.
> The noise is modeled as additive on each velocity component, ..."
>
> Note what was dropped. Independence across u and v had no physical reason,
> and the result did not need it.

**3.7 Choose the framing that matches the physical picture.** When two
mathematically equivalent framings exist, use the one a robotics reader would
describe out loud.

> Bad: "Position noise perturbs where the sample is taken while the fit uses
> the nominal position." Chris's reaction: "I don't see how I would be in the
> right place but sample the wrong one."
>
> Good: "A robot with position error ξᵢ samples the field at pᵢ + ξᵢ, but
> the fit is built from its nominal position pᵢ. The formation matrix is
> therefore wrong by ΔΦ." The framing also explains the u and v correlation
> for free, since the same ΔΦ corrupts both fits.

**3.8 Name the mechanism, then confirm it with math.** The derivation should
confirm an intuition the reader already has, not replace it.

> Bad (Draft 10): "expanding the norm to second order in ε and taking
> expectations cancels the component along (s_n, s_s) and leaves the
> transverse component contributing at order σ²_r/r"
>
> Good: "A zero-mean error still lengthens the norm on average. The part of ε
> along (s_n, s_s) averages out, but the part transverse to it always adds
> length. Expanding to second order in ε gives ..."

**3.9 Open technical sections on the world, not the math.** Even a methods
subsection starts from what the quantity is for.

> Bad (Draft 10, II-A): "A planar vector field assigns a velocity v(p) =
> (u(p), v(p)) to each position in space."
>
> Good (II-D lead-in): "Lagrangian methods locate coherent structures by
> integrating particle trajectories over a finite time window. This paper
> instead reads them from the instantaneous geometry of the velocity field,
> which the six-robot fit already provides."

---

## Part 4. Sentence rules

**4.1 Length.** Mean near 22 words, nothing over about 40. Keep sentences
shortest next to equations. Chris's unassisted baseline is 20 to 23 words.

> Bad (60 words): the Okubo-Weiss convention sentence in 3.1.
>
> Good: split it, or better, cut what the rock show rule removes and the
> sentence shrinks on its own. Compressing unused content is what produces
> the 60-word sentence.

**4.2 No colons in running prose. Use two sentences.** Run-in heads and
labeled lists are the exception.

> Bad: "Two noise channels corrupt the fit: one acts on what each robot
> reads, and the other on where it reads it."
>
> Good: "Two noise channels corrupt the fit. Measurement noise acts on what
> each robot reads, and position noise acts on where it reads it."

**4.3 No em-dashes.** Zero in both anchor texts.

**4.4 No epigram closers and no "X, not Y" antithesis.** Paragraphs end on
the last piece of information, then stop.

> Bad: "The effective level captures magnitude but not structure."
> Bad: "Rank decides whether the fit has an answer, conditioning its cost."
> Bad: "Size enters through the normalization instead. Shape enters through
> conditioning."
>
> Good: "Measurement noise at the effective level would leave those errors
> uncorrelated, so it cannot stand in for position noise." This is still a
> closing sentence, but it states a mechanism rather than a slogan.

**4.5 No comma-appositive stacking. One noun, one or two modifiers.**

> Bad: "The formation matrix Φ of (4), which is built from the robot
> positions alone and does not depend on the field, the noise, or the
> mission."
>
> Good: "The formation matrix Φ is built from the robot positions alone. The
> field, the noise, and the mission do not enter it."

**4.6 No absolute constructions hanging off a clause.**

> Bad (Draft 10): "The gradient is unaffected, its transverse component
> recovering the trench-restoring signal with the correct sign."
>
> Good: "The gradient is unaffected. Its transverse component recovers the
> trench-restoring signal with the correct sign."

**4.7 One "so" per sentence.** Chained causal clauses are a recurring agent
tell.

> Bad: "The terms pair â with b̂, so the correlation cancels, so D̂ is
> unbiased, so the D tracker tolerates noise."
>
> Good: "The terms come in opposite-sign pairs, and the subtraction cancels
> the correlation. The D estimates are therefore unbiased."

**4.8 Connectives.** Chris uses "however," "additionally," "furthermore," and
"therefore." Never use "moreover," "notably," "crucially," "importantly," or
"consists." "Can be" is fine and is his, so do not rewrite it away.

**4.9 An equation number is never the subject. Drop the number when the name
suffices.**

> Bad: "(12) reduces to a scalar comparison."
> Good: "The tangent selection rule reduces to a scalar comparison."
>
> Bad: "The formation matrix Φ of (4) is built from ..."
> Good: "The formation matrix Φ is built from ..." Φ was already defined.
>
> Keep the number only where the reader must check the exact expression,
> such as "is predicted by (6)" or "matched (5) within 1.6%."

**4.10 Precise words over approximate ones.**

> Bad: "its condition number provides a reference on how reliable the fit
> is"
> Bad: "its conditioning determines the noise cost of that fit"
> Good: "its condition number indicates how much the fit amplifies noise"

**4.11 No thematic nudging.** Never write "and that is the point," "which is
the whole argument," or "geometry again." State the fact and let the reader
draw the theme.

**4.12 Fixed terminology. Never synonym-swap.** Use each term the same way
every time, across papers.

| Use | Not |
|---|---|
| cluster | swarm, team, group |
| formation | configuration, arrangement |
| footprint | coverage area, extent |
| primitive | behavior, controller, mode |
| D tracker, s₁ tracker | determinant controller, strain follower |
| trench, ridge | valley, crest line, channel |
| attracting OECS, repelling OECS | converging, diverging structure |
| measurement noise, position noise | sensor error, localization error |
| formation matrix Φ | design matrix, sample matrix |
| separatrix | boundary, dividing line |

> Chris asked for "diverging" in the OECS note. The text kept "repelling,"
> because II-E, IV-D, and the OECS literature use it, and carried the meaning
> with "along which neighboring material separates."

---

## Part 5. Claims

**5.1 Defend every sentence.** If Chris could not explain a sentence out loud
to a reviewer, it does not go in. Cut it, or explain it in plain words.

> Cut: "Each is zero-mean with covariance σ²_uv(ΦᵀΦ)⁻¹." Nothing downstream
> used it, and the plain-words argument defended the D claim better.

**5.2 Verify anything not obviously true.** Claims about bias, expectation,
stability, or convergence get a numerical check before they stay in the text.
Report the result in standard errors, not only as a percentage.

> Bad (Draft 10): "The correlation also breaks the independence the
> unbiasedness argument relies on, leaving a determinant bias that scales
> with the local Jacobian."
>
> This read plausibly and was wrong. Every D term comes in opposite-sign
> pairs (â₂b̂₃ − â₃b̂₂), so a u and v correlation inflates both products
> equally and cancels. A 400,000-draw check showed no first-order bias at a
> 0.76 correlation. The small residual bias under full position noise grows
> as σ²_p and comes from field curvature, not from the correlation.

**5.3 Claims match their scope exactly.** Check every "in general," "for all,"
and "converges."

> Bad (Draft 10): "in general that rate is |s₁| = √(ω²/4 − D)"
> Problem: this holds only for incompressible flow.
>
> Bad (Draft 10): "every step descends a valid quadratic"
> Problem: ŝ₁ is affine minus a norm, not a quadratic.
>
> Bad: "κ(Φ) determines the noise cost"
> Good: "κ(Φ) indicates how much the fit amplifies noise." κ is a worst-case
> bound, not the realized cost.

**5.4 Scope a claim once, at the claim.** State "to first order" or "for
incompressible flow" where the claim is made, not on every sentence after it.

> Good: "The D estimates are therefore unbiased under measurement noise, and
> under position noise to first order."

**5.5 State assumptions as assumptions.** When a result rests on an
assumption that is not physically guaranteed, say so in one clause.

> Good: equal standard deviation on each strain component holds
> approximately for the pentagon-plus-center formation, whose symmetry makes
> the linear-coefficient errors nearly isotropic.

**5.6 Concede specifically, then pivot on "however."** Name the exact
weakness in technical terms. Never concede vaguely. (From Kitts.)

> Good (Draft 10): "The band test carries no hysteresis, so noise can chatter
> the D tracker across its boundary, which costs speed alone, since both
> branches share the same transverse dynamics."
>
> Template: "The controllers are admittedly simple, reactive, and minimal.
> Even in their current form, however, they demonstrate ..."

**5.7 Comparisons are descriptive. Let the verification level carry the
critique.** Never say another method is worse. Say what it did, under what
conditions, verified how. (From Kitts.)

> Good (Draft 10): "The robots begin on the manifold and are advected by the
> flow, and the manifold's identity is supplied in advance."
>
> Bad: "Their approach fails because it requires prior knowledge of the
> manifold."

**5.8 Name tradeoffs with quantities.** Never write "we chose X." Say what
was traded against what. (From Kitts.)

> Good (Draft 10): "The pentagon is enlarged to ρ = 0.105, 1.4× the nominal
> radius and 5.4 km at this site, trading truncation bias for noise
> suppression so the footprint spans about five cells of the 2 km grid
> rather than under four."

**5.9 Claim novelty on the demonstration, not the theory.** Use "to our
knowledge" and scope the claim. (From Kitts.)

> Good: "To our knowledge this is the first demonstration of multirobot
> tracking of an objective Eulerian coherent structure from instantaneous
> measurements alone."

**5.10 Cite stability, do not derive it.** If a derivation is needed, put it
in an appendix and cite it in one sentence. This is the rock show rule
applied to proofs. (From Kitts.)

---

## Part 6. Paper-level rules


**6.1 Open the introduction on a capability defined as a process, set against
the conventional approach.** "Adaptive navigation is the process of modifying
a robot's motion in real time based on measurements taken while moving."

**6.2 Include an application catalog.** Map each mathematical feature to a
physical thing and an operational use. For example, a trench of s₁ is a curve
where floating debris collects, which narrows a search area.

**6.3 Enumerate contributions with ordinals.** Use First, Second, Third,
Finally in prose. Chris's ordinals read better than Kitts's a) b) c) for
contributions.

**6.4 No roadmap sentence in papers.** Go from the contribution list straight
into the technical section. A thesis keeps its full reader's guide.

**6.5 Describe the architecture bottom-up in four named layers.** These are
the robot control, cluster space control, adaptive navigation, and state
control layers. Say what each consumes and emits, and include the figure.

**6.6 Close on the program, not the result.** Summarize what was
demonstrated, concede minimality once more, then list "Ongoing and future
work" as a) b) c) inside a sentence. Use that exact phrase.

**6.7 Use the house phrases.** Write "land, sea, air, and space" for domains,
and describe cluster space control as "an operational space control approach
in which the multirobot formation is represented as a virtualized full
degree-of-freedom articulating mechanism."

**6.8 Genre differences.**

| | Paper | Thesis |
|---|---|---|
| Roadmap | None | Full reader's guide section |
| Worked examples | Appendix, only if cited | Appendix, encouraged |
| Rock-show material | Cut | Welcome, in its own section |
| Pronouns | "we" for method, "our" for artifacts | "we" narrates procedure |

---

## Part 7. Formatting checklist

These come from Kitts's tracked corrections on the 2016 thesis. He escalates
on repeats, so fix a noted item the first time.

- Citations follow one format. Hand-formatted IEEE bibitems, consistent
  initials and name order.
- Every symbol is defined before first use. He writes "Frame?" when it is not.
- LaTeX: a line return before each displayed equation.
- No white space at the bottom of a page, and no table split across pages.
- Captions state what the figure shows and what to notice in it.
- New labels are flagged as placeholders in your notes, so Chris can match
  them to his.

---

## Part 8. Self-check

Run this on every draft before returning it.

- [ ] Could Chris defend every sentence out loud?
- [ ] Does every result get used later? (Rock show rule)
- [ ] Is every symbol defined before use, with no forward references?
- [ ] Does each section stay at one level of abstraction?
- [ ] Does each paragraph's first sentence name its job?
- [ ] Does the physical reason come before each formula?
- [ ] Is each non-obvious claim about bias, stability, or convergence checked?
- [ ] Is each claim scoped exactly once?
- [ ] Is the mean sentence near 22 words, with none over about 40?
- [ ] Zero em-dashes, and no colons in running prose?
- [ ] No epigram closers, "X, not Y," or appositive stacks?
- [ ] At most one "so" per sentence?
- [ ] No "moreover," "notably," "crucially," "importantly," "consists"?
- [ ] No equation number as a sentence subject?
- [ ] Terminology matches the table in 4.12?
- [ ] If anything was cut, is every downstream reference listed with a fix?

---

## Provenance

Two anchors, both verified human-written.

- **Master's thesis, December 2016.** "An Algorithm for Calculating the
  Inverse Jacobian of Multirobot Systems in a Cluster Space Formulation."
  Pre-dates LLMs. Carries 24 tracked Kitts comments, the source of Part 7.
  `trunk/Python_Simulations/Vector_Fields/VF_Robot/cluster_builder/Original Master's Thesis Work/CJW Thesis December 6 CK Feedback ALL CHAPTERS (1).docx`
- **IDETC 2025.** "A Functional Indoor Testbed for Multirobot Adaptive
  Navigation in Vector Field Environments," DETC2025-167604. Peer reviewed,
  with a human proofreader acknowledged.

Not a source: `jacobian_propagation_paper_thesis version.tex`. It reads as
LLM-drafted. Do not mine it for voice.

| | Thesis 2016 | IDETC 2025 | Target |
|---|---|---|---|
| Mean sentence | 20.1 words | 22.5 words | about 22 |
| Std deviation | 14.9 | 13.5 | under 20 |
| Sentences ≤ 8 words | 6.9% | 8.7% | 5 to 10% |

Most of the Bad/Good pairs in this guide come from the September 2026
rewrite of Sections II-C and II-D of the IEEE Systems Journal draft on
multirobot tracking of separatrices and OECS (Draft 10).
