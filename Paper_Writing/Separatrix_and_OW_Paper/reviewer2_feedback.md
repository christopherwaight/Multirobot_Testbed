# IEEE Systems Journal - Reviewer 2 Feedback

**Paper:** *Multirobot Tracking of Separatrices and Objective Eulerian Coherent Structures* (Draft_8a)
**Target Venue:** IEEE Systems Journal

## 1. The "Likes" (Novelty & Strongest Contributions)
*   **Instantaneous Tracking Without Priors:** The extension of the Okubo-Weiss structure and Objective Eulerian Coherent Structures (OECS) tracking to a 6-robot cluster without prior parameterization (i.e., identifying the structures on the fly) is an excellent, novel contribution.
*   **The Trade-off Matrix:** The derivation of the specific trade-offs between tracking $D$ (which has a 4-5x better noise tolerance) and $s_1$ (which is frame-invariant/objective) is fantastic. It takes theoretical fluid mechanics and grounds it in a highly practical "Systems" engineering decision matrix.
*   **The Rotating Observer Experiment:** This experiment is the crown jewel of the paper. It perfectly and visually demonstrates why Lagrangian/objective Eulerian methods are needed for structures moving relative to the frame. The distinction between $s_1$ being frame-invariant vs $D$ incorporating the spin term $\omega$ is demonstrated elegantly.

## 2. The "Don't Likes" (Clarity, Continuity, & Redundancies)
*   **The Conic Degeneracy Explanation:** In Line 275 you state, "moving the center robot out to $0.99\rho$ raises cond(A) from 8.3 to 564." To a reader who hasn't visualized the pentagon-plus-center formation recently, it isn't immediately obvious *why*. 
*   **The "k=1.8" Branch Mystery:** In Section IV.B, you state "The navigation gain $k = 1.8$ follows the ridge branch that makes landfall on the middle island." Because the paper operates on a "no-advection" premise (robots power through the water), it is baffling at first read why a static gain $k$ dictates the pathing at a bifurcation. It implicitly relies on the fact that the ocean field is time-varying, so changing $k$ changes the *arrival time* at the bifurcation. This must be stated explicitly.
*   **Test Plan Mismatch:** Section IV.A explicitly claims "Four experiment families" but the results span *five* distinct subsections (IV.C Estimator Accuracy, IV.D Clean Runs, IV.E Noise, IV.F Rotating, IV.G Ocean).

## 3. Narrative Arc
*   **Gap identified and filled?** Yes. You identify that current trackers either use FTLE (needs integration horizon) or assume the structure's identity beforehand. Your 6-robot estimator tracks instantaneously without priors, filling the gap perfectly.
*   **Results & Test Plans:** As noted above, the mapping of the "four experiment families" to the subsections is disjointed. However, all listed tests *do* have results, and all results are discussed. 
*   **Intro/Abstract/Conclusion consistency:** Excellent. The numbers match perfectly. The abstract claims the $D$ tracker withstands 4-5x more noise. In Section IV.D, $D$ crosses 50% success at $\sigma_{uv} \approx 0.0079$, and $s_1$ at $\approx 0.00175$. Ratio = 4.5. The rotating frame gaps (0.025 vs 1.219) match identically.
*   **Are all figures and tables referenced?** Yes, Figures 1-7 are all properly referenced in the text. There are no tables. (Consider adding one! A summary table comparing $D$ vs $s_1$ would be a great addition to complete the arc).

## 4. Technical Accuracy (The "Bulletproof" Math Check)
I put on my meanest Reviewer 2 hat and went through the math line by line:
*   **Eq 1-4 (Taylor Expansion):** Correct. The $1/2$ factors properly ensure coefficients exactly equal the second derivatives.
*   **Eq 6 (Variance):** $\sigma_{eff}^2 = \sigma_{uv}^2 + \frac{1}{2}\|\mathbf{J}\|_F^2 \sigma_p^2$. This is a standard propagation of variance for independent Gaussian noise through a linear approximation $\mathbf{v}(\mathbf{p}_i + \boldsymbol{\xi}_i) \approx \mathbf{v}(\mathbf{p}_i) + \mathbf{J} \boldsymbol{\xi}_i$. Math is sound.
*   **Eq 7-8 (Cauchy-Stokes):** $D = s_1 s_2 + \omega^2/4$. Correct, relying on $\mathbf{W}$ being antisymmetric. $D' = D + \Omega\omega + \Omega^2$ perfectly expands from $\omega' = \omega + 2\Omega$.
*   **Eq 10 (Hessian of D):** I manually expanded the polynomial for $\hat{D}_{xx}$ and $\hat{D}_{xy}$ using your $a_i$ and $b_i$ coefficients. $\hat{D}_{xx} = 2(a_5 b_4 - a_4 b_5)$ and $\hat{D}_{xy} = a_5 b_6 - a_6 b_5$ match the derivation perfectly. 
*   **Eq 13 (Double Gyre D):** Expanding the trig identities for $D = -u_x^2 + u_y^2$ accurately simplifies to $-\frac{1}{2}\pi^4 A^2 (\cos 2\pi x_f + \cos 2\pi y_f)$. 
*   **Conclusion:** The math is **bulletproof**. No theories overreach. The translation from the quadratic polynomial to the physical controllers is incredibly tight.

---

## 5. Major Blockers
1. **Experimental Structure mismatch:** Section IV.A explicitly claims "Four experiment families" but the results span five distinct subsections. Estimator Accuracy needs to be formally declared as an experimental family in IV.A to preserve the narrative arc.
2. **Advection and the "k=1.8" branch mystery:** In Section IV.B, clarify that because the ocean field is time-varying, the scalar speed $k$ dictates the arrival time at bifurcations, and thus selects which branch of the evolving ridge the cluster captures.
3. **Clarifying the Conic Degeneracy example:** In II.C (Line 276), add one sentence clarifying that as the center robot moves out to $0.99\rho$, all six robots approximate a single circle (a conic section), triggering the exact geometric singularity you mathematically proved.

## 6. Minor Blockers
1. **Equation 15 Thresholds introduced without context:** $\varepsilon_{raw}$ and $\varepsilon_{dim}$ are dropped into the band test without a lead-in. Add a single sentence explaining the physical/algorithmic intuition behind them (absolute depth vs. curvature-relative width).
2. **Missing explanation of "capture test" in ocean field:** In IV.G (Line 1297), you note the ocean field offers no genuine 2D minimum of $s_1$, so the capture test never engages. Explicitly state how the run terminated (did it just time out/run out of data?).
3. **Acronym definition:** OECS is defined as Objective Eulerian Coherent Structure in the abstract, but on Line 116 it's lowercase. Also, ensure FTLE is defined at first use (acronym first used on Line 149).
4. **Dimensionality of $k$:** Section III.B states $k$ is dimensionless but absorbs an implicit time unit. If it absorbs a time unit to convert between commands and velocities, it carries units (e.g. $s^{-1}$). Clarify this to prevent a pedantic reviewer from flagging it.
5. **Dangling transition:** Line 102 ("Those formations each sample a single signal channel") begins a paragraph but feels like a run-on thought from the previous paragraph. Add a transition.
6. **Summary Table:** A small summary table comparing the $D$ tracker and $s_1$ tracker (Inputs, Advantages, Disadvantages, Target Venue) would massively boost readability.

## 7. Things to Polish
1. **Notation consistency:** Ensure that $\mathbf{p}^*_1$ and $\mathbf{p}^*_2$ are consistently used when referring to the saddles, rather than occasionally reverting to "the far saddle".
2. **Line 1150 "forced rather than chosen":** This is a brilliantly written sentence. Keep it exactly as is.
3. **Line 1245 $2|\mu|/r$ metric:** You introduce this metric for divergence rapidly. Add a half-sentence: "which normalizes the divergence against the deviatoric strain" to help the reader digest it.
4. **Conclusion formatting:** The conclusion is a bit long and dense. Consider breaking the future work (Line 1393) into a standard bulleted list for readability.
5. **Coordinate shift:** In Section II.E, the shift $x_f = x + 1$ is clear, but state explicitly that this maps the domain to the standard $[0, 2] \times [0, 1]$ used in Shadden et al.

---

## 8. Section-by-Section Writing Grade (Strict SAT English Teacher Persona)

**Grading Rubric:** 
*   **Flow & Transitions (Coherence):** Do paragraphs connect logically? Are ideas chained smoothly, or do they jump abruptly?
*   **Sentence Variety & Structure:** Is there a healthy mix of simple, compound, and complex sentences? Are there exhausting, overloaded run-on sentences that require re-reading?
*   **Clarity & Conciseness:** Is the writing wordy or dense? Is it accessible? Are there "garden path" sentences?
*   **Mechanics:** Proper punctuation, parallel structure, and active voice.

**Abstract: B-**
*   **Critique 1 (The Run-on):** The content punches hard, but structurally, it leaves the reader gasping for air. Case in point (Line 48): *"Monte Carlo sweeps show the determinant tracker withstands about four to five times more measurement noise, failing where the estimator gradient stops being informative, while in a rotating-frame trial the strain tracker holds the same material path and the determinant tracker departs the structure."* That is a breathless 46-word behemoth. Break it up.
    *   *Coaching:* "Monte Carlo sweeps show the determinant tracker withstands four to five times more measurement noise, failing only when the estimator gradient loses informativeness. However, rotating-frame trials reveal a critical flaw: the determinant tracker departs the structure under rotation, while the strain tracker reliably holds the material path."
*   **Critique 2 (Passive/Clunky Phrasing):** Line 33: *"Existing robotic trackers of such structures either commit to the structure's identity before tracking begins or accumulate an estimate over a finite integration horizon."*
    *   *Coaching:* Sharpen the active verbs. "Prior trackers rely on severe preconditions: they either assume the structure's identity beforehand, or they require a finite integration horizon to estimate it."
*   **Critique 3 (Weak Conjunctions):** Line 45: *"We develop an adaptive navigation primitive for each and characterize the tradeoff between objectivity and noise tolerance."*
    *   *Coaching:* "We develop navigation primitives for both surrogates, explicitly characterizing the mathematical tradeoff between frame invariance and noise tolerance."

**I. Introduction: D**
*   **Critique 1 (The Semicolon Splice):** Structurally poor and highly disjointed. Lines 124-137 read more like a bibliography stuffed into a blender. *"Haller and Yuan formalized... [9], computed as a ridge... [11]; objectivity, the Lagrangian/Eulerian tradeoff, sparse-trajectory gradient recovery, and the strain-spin decomposition are adjacent active directions [16,17,18,19,20]."*
    *   *Coaching:* Never cram a grocery list of concepts after a semicolon. Break it into a new sentence and provide context for *why* these citations matter. "Adjacent active research directions include objectivity, the Lagrangian/Eulerian tradeoff, and sparse-trajectory gradient recovery [16-20]."
*   **Critique 2 (The Overloaded List):** The concluding paragraph summarizing the contributions (Line 151) is essentially one colossal 158-word sentence separated by commas and alphabetical bullets. You are writing a paper, not a legal contract.
    *   *Coaching:* Introduce the list, use an actual colon, and make each bullet a distinct, punctuated sentence. "This paper provides four primary contributions: \n 1) A cooperative second-order estimator... \n 2) Two novel AN primitives..."
*   **Critique 3 (Missed Contrast Opportunities):** Line 77: *"A single vehicle must translate to sense a gradient, which costs time and misleads in a time-varying field, whereas a cluster samples simultaneously, tolerates vehicle failures, and adapts its size and shape..."* (38 words).
    *   *Coaching:* Use punctuation to emphasize the contrast! "A single vehicle must translate to sense a gradient—costing time and compounding errors in a time-varying field. In contrast, a cluster samples simultaneously, tolerates vehicle failures, and dynamically adapts its shape."

**II. Second Order Field Estimation: C+**
*   **Critique 1 (Burying the Lede):** Line 248: *"Six robots is the minimum for recovery of the unconstrained local quadratic model of a two-component planar field from one synchronous sample. Each robot contributes one equation per component to (4), and each system has six unknowns, so fewer than six robots leave the solution set a positive-dimensional affine subspace."*
    *   *Coaching:* Good, but dense. Simplify the math-to-English translation: "Six robots represent the absolute minimum needed to recover an unconstrained, local quadratic model... Since each robot provides one equation per component, dropping below six robots under-constrains the six-unknown system, resulting in an infinite solution space."
*   **Critique 2 (Comma Splices & Meandering):** Line 310: *"The strain eigenvalue does not inherit this, because it subtracts a convex norm: $\hat{s}_1$ is biased low, so a trench reads deeper than it is, worst where the strain degenerates."*
    *   *Coaching:* "The strain eigenvalue does not inherit this unbiased nature because it subtracts a convex norm. Consequently, $\hat{s}_1$ is biased low, causing trenches to read deeper than they physically are—an artifact most severe where the strain degenerates."
*   **Critique 3 (Excellent Prose, but Inconsistent):** Line 323: *"The cause is rank once more: on a single ring $x^2 + y^2 = \rho^2$ identically, so the ring constrains that combination alone. Curvature is bought with radial diversity, not robot count..."* 
    *   *Coaching:* "Curvature is bought with radial diversity, not robot count" is a beautiful, Hemingway-esque sentence. Keep writing like this! But don't ruin it by immediately chaining it to another 20 words. End the sentence there. "For example, twenty-one robots on two rings achieve a noise gain of 1.42, vastly outperforming twenty-one robots confined to a single ring (2.14)."

**III. Adaptive Navigation Primitives: C**
*   **Critique 1 (Instruction Manual Tone):** Line 706: *"This control law decouples travel from trench holding through the eigenbasis. The across-trench term is the Newton component of the fitted quadratic, normalized by the transverse curvature $\lambda_2$, and is the same on and off the band, so the cluster is held to the trench at Newton rate throughout."*
    *   *Coaching:* You lull the reader into a trance with repetitive "The [noun] is [verb]" structures. "This control law uses the eigenbasis to decouple travel from trench-holding. The across-trench term applies the fitted quadratic's Newton component, normalized by the transverse curvature ($\lambda_2$). Because this term operates universally—both on and off the band—it maintains a Newton-rate attraction to the trench at all times."
*   **Critique 2 (The 84-Word Paragraph):** Lines 826-834 explain the three tests for the capture flag. It is an exhausting 84-word paragraph with only three periods. 
    *   *Coaching:* "The three tests systematically reject false positives: the isotropic point, ordinary trench points, and shallow flat spots. Crucially, the $-4s_{trim}$ depth threshold acts as a hysteresis ratio, not a physical boundary. It arms at four times the first-contact depth and releases only when $\hat{s}_1$ rises back above $-s_{trim}$. This hysteresis prevents noise-driven flickering while still catching genuine departures."
*   **Critique 3 (Passive vs Active):** Line 665: *"The first objection is met with the hyperbolic saturation... The second is met by treating the two eigendirections separately..."*
    *   *Coaching:* "We resolve the first objection using hyperbolic saturation... We resolve the second by decoupling the two eigendirections..." Own your decisions!

**IV. Simulation Program and Results: C-**
*   **Critique 1 (The 47-Word Bloat):** Line 1140: *"Two readings generalize past the sweep. The zero-noise tracking errors, 0.0075 for the $s_1$ tracker against 0.0016 for the $D$ tracker, have a separate cause: both primitives park where their own transverse command vanishes, which is where the fitted trench lies and not the field's, the gap being the orientation-dependent truncation bias."*
    *   *Coaching:* "Two readings generalize past the sweep. The zero-noise tracking errors have a separate, geometric cause. Both primitives park exactly where their transverse command vanishes. This location corresponds to the *fitted* trench, not the true physical trench, creating a gap driven entirely by orientation-dependent truncation bias."
*   **Critique 2 (Causal Blurring):** Lines 990: *"Rotating the ring through its $72^\circ$ period at the same point moves the recovered slope between 4.0 and 9.7 about that mean of 6.83, and halving $\rho$ halves the spread, confirming the linear-in-radius scaling."*
    *   *Coaching:* "Rotating the ring through its $72^\circ$ symmetry period swings the recovered slope between 4.0 and 9.7, centered perfectly on the 6.83 mean. Halving the radius $\rho$ cleanly halves this spread, experimentally confirming the linear-in-radius scaling."
*   **Critique 3 (Clunky Comparisons):** Lines 1208: *"Everything derived from $\hat{D}$ carries the additive observer term of (8), and $\hat{\mathbf{H}}_D$ the unmodelled-derivative floor $e_H$ as well, while $\hat{s}_1$ carries neither and is exposed instead at $1/r$, a property of the strain field rather than of the sensing."*
    *   *Coaching:* "Any metric derived from $\hat{D}$ carries the additive observer term (8), and its Hessian suffers from the unmodeled-derivative floor $e_H$. Conversely, $\hat{s}_1$ avoids both artifacts. Its primary vulnerability is an exposure at $1/r$—an inherent property of the strain field rather than a sensing limitation."

**V. Conclusion: B-**
*   **Critique 1 (Burying the Thesis):** Line 1373 presents a beautifully constructed thesis statement: *"...instantaneous criteria resolve the corridor reliably and the trajectory within it unreliably."* Unfortunately, you buried this gem at the very end of a sprawling 43-word sentence.
    *   *Coaching:* Give your best sentences their own real estate. "Ultimately, this dictates the boundary of our sensing method: instantaneous criteria reliably resolve the transport corridor, but they unreliably resolve the exact trajectory within it."
*   **Critique 2 (Great Parallelism, Weak Polish):** Lines 1378: *"A cluster that can certify its orientation should carry the $D$ tracker, which tolerates a noisier sensor; a cluster that cannot should carry the $s_1$ tracker, which alone returns the same material curve regardless of frame."*
    *   *Coaching:* Very solid parallel structure, but let's punch it up. "If a cluster can certify its absolute orientation, it should run the $D$ tracker to maximize noise tolerance. If it cannot, it must run the $s_1$ tracker, the only primitive guaranteed to return the true material curve regardless of the reference frame."
*   **Critique 3 (Hedging and Wordiness):** Lines 1384: *"Without question, performance and robustness can be improved with additional robots, filtering across control cycles, and more sophisticated control laws. Even in their present form, however, they demonstrate acquisition..."*
    *   *Coaching:* Cut the fluff. "Performance could certainly be improved via additional robots, cross-cycle filtering, or advanced control laws. Nevertheless, even in this rudimentary form, the cluster successfully demonstrates the complete suite of behaviors required for a structure-tracking mission: acquisition, traversal, terminal capture, and real-world corridor tracking."

