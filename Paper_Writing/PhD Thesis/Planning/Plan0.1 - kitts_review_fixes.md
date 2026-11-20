# Plan 0.1: Fixes from the mock Kitts review

Date: 2026-11-19. Source: the mock advisor review of `thesis_main.pdf` (148 pp, 2026-11-18 build), four points. This plan sits beside `Plan0 - thesis_review.pdf` and covers only what Plan0 does not, except where both touch the same sentence (table below).

## How edits land

- `chapters/*.tex` are generated. Edit `sources/assemble/chNN.py` instead: a ported passage `P(...)` gets a `subs=[(old, new)]` entry; new text `T(...)` is edited in place. Do not touch the source papers (the revision paper is frozen; Draft_13 wording stays Draft_13's).
- The abstract and glossary are hand-written in `thesis_main.tex`.
- Rebuild: `python3 sources/assemble/all.py`, then `build_bib.py`, then three `pdflatex` passes with `TEXMFHOME=/nonexistent`.
- Queue every prose change as one batch until Chris says apply. No new number enters the .tex until Chris approves it.

## Overlap with Plan0

| Item here | Plan0 entry | Handle together |
|---|---|---|
| 1a, 1b, 1d "this paper", "Section II-B / IV" | Sec. 2 rows "Sec. 9.1.1" and "Fig. 4.1 caption" (Required) | Same edits |
| 2a 169 to 459 | Sec. 2 row 1; also "distributed" to "cooperative" in the same sentence | One rewrite of the contributions paragraph |
| 1c, 1e duplicate testbed and calibration text | Ch. 2 "consolidate repeated testbed descriptions" | Same edit |
| 4 scope Chapter 7 as simulation | 3E "Ensure the abstract and conclusion match" | Same edit |
| 4 hardware campaign | Plan0: "No additional hardware campaign is recommended" | Agrees with recommendation 4-B |

The contributions paragraph (`ch01.py`, Contributions `T` block) gets edited by Plan0 (wording), 2a (count) and 4b (scope). Write it once with all three.

## Decisions Chris needs to make first

| # | Question | Recommendation |
|---|---|---|
| D1 | Chapter 7: run a six-robot hardware campaign, or scope the claims as simulation? | Scope now (4-B). The campaign needs a six-robot Simulink model, two more calibrated robots and a new map; keep it as future work (a) in Ch. 10. |
| D2 | Dead zone: is 0.05 m/s (IDETC era) a real difference from 0.025 m/s (2026 refit), e.g. before the maintenance in Sec. 2.4.4? | If yes, say so once in Sec. 2.6 with the date. If no, replace 0.05 with 0.025. |
| D3 | Re-run the Sec. 8.1 closed-loop test with the 2026-11-16 refit model? (CLAUDE.md: ask before any 3-robot sim) | Yes. The current numbers predate the 2026-10-28 formation sign fix and use the old model. |
| D4 | Which version survives for each duplicated passage (1c, 1e, 1f, 1h)? | As listed under each item. |
| D5 | Wording of the proposed replacement sentences below. | Proposals only; Chris rewrites. |

## 1. Seams between the three papers

- [ ] **1a. "this paper" in the novelty claim.** Ch. 1, Sec. 1.2.3: "To our knowledge, this paper gives the first hardware demonstration...". Fix: in `ch01.py`, add a sub on `P('sys', 'Work that senses the physical field falls into two camps.')`: `this paper gives` to `this dissertation gives`. Mechanical.
- [ ] **1b. "this paper" twice in Ch. 4.**
  - Sec. 4.1, from `P('sep', 'The control architecture is the established multilayer approach of prior', ...)`: "this paper provides the quadratic fit that fills the feature estimation block and the D and s1 trackers...". This is Draft_13's scope inside a chapter that serves all three rungs. Proposed sub: "this dissertation fills the feature estimation block with the fits of Chapters 5 to 7 and the navigation controller block with the primitives built on them."
  - Fig. 4.1 caption, from `P('sys', '\\caption{Hierarchical control architecture...')`: "new in this paper". Proposed: "are the contributions of this dissertation."
- [ ] **1c. Testbed described twice (Sec. 2.2).** The IDETC paragraph (`ch02.py`, `T(IDC.format('Sec. 3.1') ...)`) says three rovers and 2.3 m² maps. The revision paragraph (`P('sys', "The testbed used for validation is SCU's...")`) and Fig. 2.4 say three or four robots on 1.6 m × 1.6 m maps. Fix: one description of the testbed, then one sentence giving each campaign's robot count and map size. Chris confirms whether the IDETC maps (2.3 m²) and the Chapter 6 maps (2.56 m²) really differ.
- [ ] **1d. IEEE section numbers in Ch. 9.** Sec. 9.1.1, from `P('sys', 'Several limitations constrain...')` in `ch09.py`. Add subs: `(Section~II-B)` to `(Section~\ref{sec:ch6:estimation})`, and `Section~IV probes` to `Section~\ref{sec:ch6:sim_attraction} probes` (both spellings checked against the revision source, line 579). Mechanical.
- [ ] **1e. Calibration described twice (Sec. 2.4).** Secs. 2.4.1 to 2.4.3 (IDETC, seven inputs) and Sec. 2.4.4 (revision passage, four inputs) each describe the palette, the 24 intervals and the two networks. Fix: in `ch02.py`, trim the Sec. 2.4.4 passage with subs so it keeps only what changed: four raw channels instead of seven, pooled training, and the 6.1° / 3.8% / 12 to 14% results.
- [ ] **1f. Cluster space control defined twice (Sec. 4.1.2).** "an operational space control approach in which the multirobot formation is represented as a virtualized full degree-of-freedom articulating mechanism" appears in the IDETC block and again in the six-robot paragraph. Fix: a sub on the six-robot passage drops the second definition. In the same IDETC block, "developed by Dr. Chris Kitts of Santa Clara University" could become "developed by Kitts and colleagues at Santa Clara University" (Chris's choice).
- [ ] **1g. Ch. 1, Sec. 1.2.2 states the same gap twice.** The IDETC paragraph ("remains largely theoretical... artificial vector fields... only validated in simulation") and `P('sys', 'By contrast, only a few works address...')` say the same thing with different citations. Recommendation: keep the IDETC paragraph, keep the revision's sentence on field campaigns [33], and move the revision's citations onto the matching IDETC sentences after checking what each cited work did.
- [ ] **1h. Ch. 5 opens twice.** The new Sec. 5.1 text and the IDETC "To evaluate our platform, we implemented two adaptive navigation primitives..." overlap. Proposed sub: drop "To evaluate our platform," so the paragraph reads as the baseline the chapter's new opening describes.
- [ ] **1i. Ch. 6 summary ends after its closing.** In `ch06.py`, the new sentence "Offline, the hardware attraction estimates also named the type..." sits after "...the natural next step toward field deployment." Move it before "Together these results establish...".
- [ ] **1j. Ch. 7, Sec. 7.2: "In this work, we extend".** Sub to "This chapter extends". Mechanical.
- [ ] **1k. Read it once, straight through, as a book.** Last step, after everything else. Log any further seams at the bottom of this file.

## 2. Numbers that disagree

- [ ] **2a. 169 vs 459 hardware runs.** `ch01.py`, Contributions: "in 169 hardware experiments". Proposed: "in 459 hardware runs with three and four robots" (239 attraction trials and 220 orbit runs, as in Table 10.1). Not a new number: it is already in the abstract and in Chs. 6 and 10.
- [ ] **2b. τ in the glossary, and two names for one coefficient.** The thesis uses two robot models. The refit (Sec. 2.6, used in Ch. 6) has τ = 0.02 s and L = 0.055 s. The earlier momentum model (Sec. 7.4, used in Ch. 7) has τ = 0.28 s. The glossary gives only 0.28 s. Separately, the same quantity e^(-Δt/τ) is α in eq. (2.2) and Sec. 6.4.1 but α_mom in Ch. 7 and the glossary, and Sec. 6.6.6 uses α for the real part of an eigenvalue. Fix:
  1. Glossary (`thesis_main.tex`): τ is "actuator time constant: 0.02 s in the refit model (Sec. 2.6), 0.28 s in the earlier momentum model (Ch. 7)". Add L, g and v_c, which Sec. 2.6 introduces and the glossary omits.
  2. Use α_mom everywhere for e^(-Δt/τ): subs on the `ch02.py` passage `P('sys', 'In simulation, each robot is modeled...')` and on the Sec. 6.4.1 passage in `ch06.py`. That leaves α for the eigenvalue only.
  3. Structural option (recommended): move the earlier momentum model's description from Sec. 7.4 into Sec. 2.6, next to the refit, so Ch. 2 defines both models and states which chapter uses which. Ch. 7 then refers back. This also defines the model before Ch. 5 needs it (item 3).
- [ ] **2c. Dead zone 0.025 vs 0.05 m/s.** 0.025 m/s in Sec. 2.6 (refit). 0.05 m/s in Sec. 2.9 (`ch02.py`, "a limited velocity range of 0.05 to 0.3 m/s") and Sec. 5.4.3 (`ch05.py`, "velocity commands less than 0.05 m/s that are not sufficient..."), both IDETC text. Related: Secs. 2.3.3 and 2.5 say saturation below 0.3 (about 0.09 m/s) did not overcome static friction. Fix depends on D2. If the IDETC-era robots really needed more, say so once in Sec. 2.6 with the date and the maintenance, and leave the IDETC sentences as they are. Otherwise correct them to 0.025 m/s.
- [ ] **2d. Ch. 8 closed-loop numbers vs Table 6.1.** Sec. 8.1 says the attraction law finished within 0.015 m in 95% of trials. Table 6.1 puts the analytical-field mean final error at 22.6 mm, on the 0.025 m friction ring. Source: `VF_Robot/experiments/interesting/thesis_ch8_magnitude_saddle.py`, run 2026-09-30, before the 2026-10-28 formation sign fix, with the old model (0.33 m triangle, momentum α 0.7). Steps, after D3:
  1. Point the script at the refit defaults: 0.30 m side, k = 1, the `OmniCluster` defaults of 2026-11-16. The script moved to `interesting/`, so its documented run path is stale; fix the path or the import before running.
  2. Re-run 1000 trials × 3 fields × 2 primitives.
  3. Report old vs new as prose and a table: median and 95th-percentile final error per field and primitive.
  4. After approval, update the Sec. 8.1 `T` text in `ch08.py` (the "0.006 m", "0.048 m", "91%" and "0.015 m" sentences) and regenerate Fig. 8.1 into `figures/`. Its caption already claims "the robot model of Chapter 6", which becomes true only after the re-run.
  5. Expect the attraction law to settle near the friction ring (about 0.023 m), so the comparison may need restating as bias (magnitude descent off the saddle) against the ring, not "within 0.015 m". Secs. 8.3 and 8.4 are fits without robot dynamics (`thesis_ch8_radius_sweep.py`, `thesis_ch8_geometry.py`) and do not need a re-run.
- [ ] **2e. Ch. 5 formation precision.** `ch05.py`: "The values stayed within 1 cm of desired parameters... within the resolution limits of the OptiTrack motion capture system". Table 5.1 shows standard deviations of 0.9 to 3.2 cm in p and q and 0.045 to 0.12 rad in β; OptiTrack resolves well below that. Proposed: "The mean side lengths stayed within 1 cm of the commanded 0.35 m and the mean angle within 0.03 rad of 1.05 rad, with standard deviations of 0.9 to 3.2 cm and 0.045 to 0.12 rad." Drop the OptiTrack clause. These numbers come from Table 5.1, but they still go in the batch for approval.

## 3. Orbit drift: momentum (Ch. 5) vs delay and lag (Ch. 6)

Ch. 5 says the vector-sum orbit drifts outward because of "momentum effects not captured in previous simulation models". Sec. 6.4.4 derives the drift from the total response delay T ≈ L + τ + Δt/2: a cluster that turns late drifts outward at T V²/r. Under the refit, the lag τ is the smallest part of T. The two chapters describe one mechanism in two ways.

- [ ] **3a.** `ch05.py`, IDETC block in Sec. 5.4.3: replace "while experiencing momentum effects not captured in previous simulation models" with the mechanism in plain words. Proposal: "...always moving tangent to the circle, but each robot answers a new heading late, after a delay and a lag that the simulations of [26] did not model, so every step leaves the cluster slightly outside the circle it was on." Scope "a behavior not predicted in simulation studies" to the simulations of [26] in the same pass.
- [ ] **3b.** `ch05.py`, Sec. 5.6 (new text): "so momentum carried the cluster steadily outward" becomes "so the late turns carried the cluster steadily outward".
- [ ] **3c.** Sec. 6.4.4: add one backward reference after the drift sentence (new `T` in `ch06.py`): "This is the outward drift of the vector-sum primitive in Chapter 5; the radial term of (6.11) balances it." Backward references are allowed. Do not add a forward pointer in Ch. 5.
- [ ] **3d. Optional check, if the IDETC vector-sum logs can be found.** The closed form predicts d(r²)/dt = 2 T V², so r² should grow linearly. Fit the slope of r² against time from the logs and compare it with T from the refit (0.125 s) and from the momentum model. Report only. This also informs D2: if the IDETC robots match the slower model, the 0.05 m/s floor was probably real for that era.

## 4. Chapter 7 has no hardware

**4-B, scope as simulation (recommended):**

- [ ] **4a. Abstract** (`thesis_main.tex`, around "A second-order fit from six robots..."): say "in simulation" once, at the first second-order result. The Monte Carlo and Santa Barbara sentences then inherit the scope.
- [ ] **4b. Contributions** (`ch01.py`): the Fifth and Finally items name simulation as the evidence, as the Fourth names hardware. Write this together with 2a and Plan0's wording change.
- [ ] **4c. Conclusion** (`ch10.py`): the Ch. 7 paragraph ported from Draft_13 does not say simulation; Table 10.1 does. Add the scope to the paragraph's first sentence.
- [ ] **4d. Check, no change expected:** Sec. 1.7 ("separatrix tracking in simulation"), the Sec. 7.1 bullets, Sec. 9.1.2 ("limited to kinematic simulation"), Table 10.1 and future work (a) already scope it.

**4-A, run the campaign (not recommended before the defense).** Needs:

- six calibrated Decabots (only R1 to R4 have data; the pooled model reads R4's magnitude 12 to 14% low)
- a printed double gyre or saddle-pair map sized for the pentagon on the 1.6 m map or the 3 m × 6 m floor
- a six-robot Simulink model with the pentagon-plus-center kinematics of App. C (only 3- and 4-robot `.slx` models exist; MATLAB stays preserved, not modernized)
- the D and s1 trackers in MATLAB
- matched six-robot simulations on the refit dynamics (the `Omnibot` defaults are still legacy)

This is weeks of lab work, and Plan0 recommends against it.

## Also found while skimming (not in the review)

- [ ] Sec. 2.1: "Each is powered by an Arduino". Sec. 2.2 names the Intel Edison as the onboard computer. Shae's Sec. 2.1 describes an Intel Edison on an Arduino breakout board with a Teensy 3.2 for the encoders. Make Sec. 2.1 match (`ch02.py`, IDETC block).
- [ ] Earlier memory flags 1 to 4 (offline classification provenance, the Ch. 7 legacy-model sentence, the Ch. 8 numbers (now 2d), and the stale `T()` text in Apps. C, D and F) stay open.

## Order of work

1. **Mechanical, no decision needed:** 1a, 1d, 1i, 1j, plus the subs part of 2b.2. Rebuild and compile.
2. **Decisions:** D1 to D5 from Chris.
3. **Prose batch:** 1b, 1c, 1e to 1h, 2a to 2c, 2e, 3a to 3c, 4a to 4c, and the Sec. 2.1 fix. Present it as one batch with old and new text side by side; apply on Chris's word.
4. **Re-run:** 2d after D3, reported before any .tex change. 3d if the logs turn up.
5. **Finish:** rebuild, three passes, then 1k.

## Done when

- Three passes compile with no undefined references and no new warnings.
- This search returns nothing outside intended uses (the Ch. 7 momentum model, and "momentum" in equation labels):
  `grep -n -E 'this paper|Section~?(II|III|IV|V)[-~ ,)]|169 hardware|momentum effects' chapters/*.tex thesis_main.tex`
- The glossary, Sec. 2.6, Sec. 2.9, Sec. 5.4.3 and Table 6.1 agree on τ, L and the dead zone.
- The Sec. 8.1 numbers and Fig. 8.1 come from the refit model.
- The abstract, the contributions, Ch. 10 and Table 10.1 all scope Chapter 7 the same way.
- The straight-through read (1k) is done and its notes are logged below.

## Read-through notes

(empty)
