# Writing in the Waight voice

Companion to `Kitts_style.md`. That file describes the voice you write *toward*
when the audience is Kitts or the lab's publication lineage. This file describes
the voice you actually have, extracted from work written before heavy LLM use.

## Source texts

Two anchors, both verified human-written.

* **Master's thesis, December 2016.** "An Algorithm for Calculating the Inverse
  Jacobian of Multirobot Systems in a Cluster Space Formulation."
  `trunk/Python_Simulations/Vector_Fields/VF_Robot/cluster_builder/Original Master's Thesis Work/CJW Thesis December 6 CK Feedback ALL CHAPTERS (1).docx`
  Carries 24 tracked Kitts comments, which double as a record of what he corrects
  in your writing. Pre-dates LLMs entirely.
* **IDETC 2025.** "A Functional Indoor Testbed for Multirobot Adaptive Navigation
  in Vector Field Environments," DETC2025-167604. Peer reviewed, published,
  acknowledgements thank a human proofreader.

**Not a source:** `jacobian_propagation_paper_thesis version.tex` in the same
folder. That is a later paper-ized rewrite of the thesis and reads as LLM-drafted
(em-dash in the abstract, 200-word unbroken abstract paragraph, mean sentence 35.5
words with zero sentences under 8 words, "the key insight is," "leveraging the
hierarchical structure"). Do not mine it for voice.

## Baseline numbers

| | Thesis 2016 | IDETC 2025 | Draft 10 (current) |
|---|---|---|---|
| mean sentence | 20.1 words | 22.5 words | 33.7 words |
| std deviation | 14.9 | 13.5 | 90.9 |
| sentences <= 8 words | 6.9% | 8.7% | 5.6% |

**Your natural sentence is 20 to 23 words.** Draft 10 runs 50% longer than
anything you have written unassisted. The standard deviation of 90.9 is the
clearest single tell in the current draft: it means a few enormous sentences are
sitting next to short ones, which is what happens when generated prose gets
patched into hand-written prose. Target mean 22, sd under 20.

---

## A. How you open

**Rule 1. You zoom in from the world, not out from the math.** The thesis opens
"Robots have been instrumental in developing many industries in modern culture."
IDETC opens "Multirobot systems offer unique advantages for autonomous navigation
of vector fields." Both start wide and narrow over several paragraphs. Neither
opens on a definition, an assumption block, or a variable.

This matches Kitts Rule 3 and is worth protecting. Draft 10's Section II opens on
"A planar vector field assigns a velocity..." which is the estimator-first
instinct the Kitts guide warns you about at the very end.

**Rule 2. You state the gap as a deficiency in what exists, plainly.** From
IDETC: "Adaptive Navigation through vector fields remains largely theoretical,
compared to extensive experimental confirmation in scalar fields." From the
thesis: "this process is not only tedious but also error-prone, with mistakes
difficult to detect." No hedging, no throat-clearing. Name what does not work.

**Rule 3. You tell an origin story when there is one.** IDETC explains that the
previous testbed used red and green for orthogonal vectors and suffered sensor
channel interference, worst at small magnitudes where minor value changes cause
large direction shifts. This is the most human paragraph in your corpus. Specific
prior failure, specific mechanism, specific consequence. Keep doing this.

## B. Structural habits

**Rule 4. Numbered contributions in prose, using ordinals.** IDETC: "First, the
development of an HSV color space representation... Second, the implementation of
a supervised learning neural network calibration technique... Third, a method
to... Finally, experimental validation of..."

You use First/Second/Third/Finally where Kitts uses a) b) c). Both are fine.
Yours is the more readable of the two. Draft 10 currently has one compressed
contribution paragraph with no enumeration at all, which is a departure from both.

**Rule 5. You write a reader's guide, and you should keep it in the thesis and
drop it in papers.** The thesis has a full section 1.5 walking through every
chapter. IDETC has the compressed journal version: "The paper is organized as
follows: Section 2 covers... Section 3 describes..."

Note the Kitts guide observes that neither [4] nor [25] carries a roadmap
sentence. Draft 10 correctly has none. Keep it that way for the Systems Journal
papers, keep the full version for the PhD thesis.

**Rule 6. Worked examples in an appendix.** The thesis carries a complete
two-robot cluster example showing every step. This is a genuine strength and it
is how you think. When a derivation is too long for the body, an appendix worked
example is your move, not a compression.

## C. Sentence construction

**Rule 7. Your connectives are "however," "additionally," "furthermore."** The
thesis uses however 10 times, furthermore 6, in order to 4. IDETC uses
additionally 4. These are plain and slightly formal, and they are yours. You do
not naturally write "moreover," "notably," "crucially," or "importantly." If one
of those appears in a draft, it was not you.

**Rule 8. You use "can be" heavily and it is fine.** 49 occurrences in the thesis
body. "A multirobot system can be defined as..." "The inverse Jacobian can be
found by..." This is unpretentious technical English. Do not let anyone talk you
into replacing every instance with something more active.

**Rule 9. Pronouns: "our" for the artifact, "we" for the method.** IDETC uses
"our" 21 times against "we" 5, almost always "our testbed," "our research," "our
contributions." The thesis inverts it, "we" 15 times, because a thesis narrates
procedure. Both are correct for their genre. Draft 10 uses "our" 6 and "we" 3,
which is thin for a paper claiming a first demonstration.

**Rule 10. You explain the mechanism, not just the result.** Thesis: "two robots
with similar 'follow' instructions can compete to occupy the same space and
result in collision." IDETC: minor value changes causing large direction shifts
at small magnitudes. You consistently say *why* a thing fails. This is your
strongest single habit. Protect it.

## D. What you are not

These are the AI-isms to hunt, defined as constructions absent from both anchors.

**Anti-rule 1. No aphoristic paragraph closers.** Neither anchor ends a paragraph
on a short balanced epigram. Draft 10 Section II has roughly six: "Rank decides
whether the fit has an answer, conditioning its cost." "Unbiasedness belongs to
the measurement channel alone." "One scaling rule holds per derivative order, not
one gain per order." "The second is not a relabeling of the first."

The tell is the **X, not Y** construction and antithesis across a semicolon. Your
real paragraphs end on the last piece of information, then stop.

**Anti-rule 2. No em-dashes.** Zero in the thesis body, zero in IDETC. Already in
CLAUDE.md; the anchors confirm it is a real preference and not an affectation.

**Anti-rule 3. No comma-appositive stacking.** You do not hang three modifiers off
one noun. "Everything the estimator does is fixed by the formation matrix Phi of
(6), built from the robot positions alone, not the field, the noise, or the
mission" is not a sentence you would write unassisted. You would write two
sentences.

**Anti-rule 4. No thematic nudging.** "geometry again," "and that is the point,"
"which is the whole argument." You state facts and let the reader draw the theme.

**Anti-rule 5. No forward-pointing scaffolding.** Already in CLAUDE.md. The
anchors confirm it: the thesis reader's guide points forward *once*, in its own
labeled section, and then never again.

## E. Kitts's standing corrections to you

From the 24 tracked comments on the 2016 thesis. These are what he actually
catches, so pre-empt them.

* **Citation style consistency.** "Your citations need to follow a consistent
  style, they currently don't. For ex, [1] has first initial then last name, [7]
  has last name then first initial." This is why `CLAUDE.md` is strict about
  hand-formatted IEEE bibitems. It is a scar.
* **Whitespace and pagination.** Five separate comments: "No white space at
  bottom of page!!!!", "Table goes across 2 pages", "Yeah... this should be a new
  page." He reads layout as carefully as prose.
* **Define your symbols before use.** "Have V and w been defined? Are they the
  robot space velocity vectors?" and two bare "Frame?" comments.
* **Line return before equations.** "Add a line return before equations, here and
  throughout."
* **He escalates on repeats.** "Still bad wording, you didn't make a change based
  on my last comment.... Please do so!" Fix a noted item the first time.
* **He does praise.** "Captions are MUCH BETTER with this new formatting,"
  "Excellent!!!" Layout fixes and clear captions land well with him.

## F. Where your voice and the Kitts voice conflict

The Kitts guide closes by naming this: you lead with the estimator's mathematical
properties, he leads with what the system does in the world.

Both anchors show that **your unassisted instinct is actually closer to his than
your recent drafts are.** The 2016 thesis opens on robots in industry. IDETC
opens on multirobot sensing advantages. Neither opens on a Jacobian. The
estimator-first opening in Draft 10 Section II is not your natural voice; it is
what the writing drifted into.

So the correction is not "write less like yourself to please Kitts." It is
"write more like your 2016 and 2025 self, and the Kitts alignment follows."

## Quick self-check

* Is the mean sentence near 22 words, with no sentence over about 45?
* Is the standard deviation under 20, meaning no giant sentence next to a short one?
* Does every paragraph end on information rather than an epigram?
* Are contributions enumerated First/Second/Third/Finally?
* Did you explain *why* the prior approach fails, with a mechanism?
* Zero em-dashes, zero "moreover/notably/crucially"?
* Are all symbols defined before first use, and is there a line return before each equation?
* Are the citations in one consistent format?
