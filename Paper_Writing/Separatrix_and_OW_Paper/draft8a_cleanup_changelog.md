# Draft_8a cleanup: change log

Edited `Draft_8a.tex` in place (GATE 0). Baseline and final both compile at
**15 pages** with **zero undefined references** and **zero `??`** in the PDF.
Three `pdflatex -interaction=nonstopmode` passes, TeX Live 2025 BasicTeX.

## Baseline (before any edit)

- 15 pages. Zero undefined refs, zero `??`, zero overfull boxes >15pt.
- Layout: II p2, III p5, IV p7, Conclusion p13, Appendix A p14, references to p15.
- Style clean: 0 em-dashes, 0 "consists".
- **Brief section 2.2 was obsolete.** `rem:conic`, `prop:unbiased`, `eq:s1_bias`
  appear nowhere in the file. The lines the brief attributed to them cite
  `sec:sensitivity` and resolve. No blocking equation was needed (GATE 2 moot).

## Changes made

| # | Anchor | Change | Words | Page delta |
|---|--------|--------|-------|-----------|
| A1 | `\markboth{IEEE Transactions on Robotics}` (line 28) | to `\markboth{IEEE Systems Journal}` | 0 | 0 |
| A2 | preamble `\newtheorem` block | added `\usepackage{amsthm}`, `\theoremstyle{plain}` / `\theoremstyle{remark}`. No `\proof` collision, no `\let\proof\relax` needed. | 0 | 0 |
| A2 | `\subsection{Minimality}` (line 281) | added `\label{sec:minimality}`; wrapped the statement as `\begin{proposition}\label{prop:minimality}...`; kept the incompressibility discussion as the following paragraph unchanged. Did **not** touch the conic paragraph (cites `\cite{2}`/`\cite{4}`, not the `\cite{4}`/`\cite{6}` in the brief snippet). | +30 | 0 |
| A2 | "No usable Hessian of $s_1$..." (line 507) | wrapped the two-sentence statement as `\begin{remark}\label{rem:s1hess}...`; kept the third-derivative / per-cycle-refit discussion as the following paragraph. | +5 | 0 |
| A2 | 3 sites (688, 1518, 1596) | `Section~\ref{prop:minimality}` to `Section~\ref{sec:minimality}` so they render "Section II-B" not "Section Proposition 1". The `Proposition~\ref{prop:minimality}` at the Conclusion and all 5 `Remark~\ref{rem:s1hess}` sites now render "Proposition 1" / "Remark 1" correctly. | 0 | 0 |
| A3 | "reachability condition of Problem 2" (line 1549) | reworded to "the terminal condition (\ref{eq:capture_test}) never engages"; dropped the Problem framing (GATE 3, per Chris: not framed as Problems). No "Problem" string remains anywhere. | -8 | 0 |
| A4 | "By the fourth consequence of the splitting identity" (line 1146) | to "By (\ref{eq:trench_coincide}, Section~\ref{sec:detj_field})". Kept the existing "coincide" geometry (the brief's "tangent" wording would contradict the next sentence). | -6 | 0 |
| B1 | end of V-C, after "...says nothing about the noisy case." (line 1370) | inserted the margin-hold paragraph (brief's wording verbatim, "same acquisition trial" instead of "same single-far-saddle trial" to match V-C's own terms). | +100 | see below |
| B1 | Limitations 4th paragraph (line ~1598) | replaced the implemented-rule description + stats with the shortened framing pointing to `sec:disc_noise` and keeping the two unimplemented changes. "four limitations" still correct. | -95 | see below |
| B2 | Conclusion, "...an FTLE ridge neither could see." (line 1649) | extended with the corridor-vs-trajectory scope statement and the "bound this sensing carries" sentence. | +33 | see below |
| B3 | V-C, after "...without any single comparison being reversed." (line 1402) | new paragraph "That reliance is forced rather than chosen..." assembling the three-references argument. Cites `rem:s1hess` and `app:equivariance` (Appendix A does state the measured flow is not objective). Skipped the brief's extra V-D cross-reference sentence: V-D already closes pointing at `sec:disc_noise`. | +92 | see below |
| C-5.2 | V-C, "with $\gamma_0 = 1$ for a" (line ~1335) | restored "since the fit interpolates at the center robot". | +7 | 0 |
| C-5.2 | V-C, "signal-to-noise ratio of $1.4$" (line ~1347) | restored "at $\sigma_{uv} = 0.002$, against the combined noise of the two compared coefficients". | +13 | 0 |
| D-C1 | double-gyre "Three facts follow directly" (line 587) | compressed to a one-clause Okubo-Weiss statement + "Two further facts carry into Sections...". Dropped "First/Second/Third" enumeration labels. **Kept** the separatrix-trench-of-$D$ fact and the $s_1$-transverse-trench fact in full. Did not move `eq:det_analytic_main` / `eq:gradhess_analytic_main` inline (each is referenced once elsewhere; inlining a 2x2 Hessian would hurt readability for little gain). | -20 | included below |
| D-C2 | Ocean HFR setup, derived-units footnote (line 1036) | footnote folded to two inline sentences; dropped the discrete-stability-bound clause. | -25 | included below |
| D-C2 | Ocean HFR setup, tangent-plane similarity argument (line 1049) | cut from ~9 lines to 3 sentences. | -35 | included below |
| D-C3 | V-C "Two readings sit alongside it" (line 1293) | tightened; numbers retained. | -15 | included below |
| D-C3 | V-C zero-noise tracking-error paragraph, discrete-bound tail (line 1392) | compressed the "signature of the discrete bound" digression from 6 lines to 3. | -30 | included below |

## Page-count trace

| Checkpoint | Pages | Note |
|---|---|---|
| Baseline | 15 | body ends ~40% down p14; refs [13]-[40] on p15 |
| After A + B + C-5.2 | 15 | amsthm + two theorem environments add ~`\topsep` padding; content adds ~140 words |
| After D-C1 | 15 | |
| After D-C2 + D-C3 | 15 | refs [15]-[40] now on p15 (was [13]-[40]); ~2 references pulled onto p14 |
| Final | 15 | body + Appendix A occupy p1-p13; p14 = Conclusion tail + refs [1]-[14]; **p15 = refs [15]-[40] only** |

**Net page count unchanged at 15.** The D-group trims (~125 words) recovered
what the amsthm theorem-environment spacing and the B-group additions (~225
words) cost, and pushed roughly two references from p15 onto p14. The paper is
effectively 13.3 pages of body + appendix with a reference list spanning the
bottom of p14 through p15.

Reaching a true 14 pages would need about one more freed column on p14, i.e.
~0.5 page of body reduction beyond C1-C3. Every remaining lever (C4 param
table, cutting mechanism paragraphs, trimming the appendix or the protected
experiments) is either established not to help the reference-overflow
constraint or is protected content. Per GATE 7, stopping here.

## Consistency sweep

- 5.1 / GATE 8: **no edit.** Verified against Draft_7b (which carries the
  identical wording, so it does not adjudicate) and the draft's own data:
  `s_1` 50% crossing 0.0015-0.002 (V-C), `D` crossing 0.0079 (V-C), ratio
  4.0x-5.3x, and the noise grid step spans the same gap. Abstract "four to
  five times" and body "a full grid step" are both correct descriptions of
  the same measurement. Chris confirmed: keep 4-5x, abstract untouched.
- 5.3: cross-checked and consistent. 0.025 / 1.219 (V-D fig, V-D text,
  Conclusion); 44.6 / 84.1 (V-C moved paragraph, no longer in Limitations);
  2.1% (est-accuracy, Conclusion); 1.8 / 2.4 km (V-E, Conclusion); 4.5deg vs
  41deg (est-accuracy fig text and V-D); six-robot minimum (abstract,
  contribution (a), II-B, Conclusion). The 44.6 -> 45.8% at line 1348 is the
  separate seed-only ablation, not the margin-hold result; same 44.6%
  baseline, consistent.
- 5.4: 0 em-dashes, 0 "consists" in the final file.
- 5.5: all 40 `\bibitem` keys cited, all 40 cited keys resolve.

## Theorem numbering (verified in compiled PDF)

Only one Proposition and one Remark exist (the brief's `prop:unbiased` and
`rem:conic` were moot). Renders as **Proposition 1** (minimality, II-B) and
**Remark 1** ($s_1$ Hessian, II-D). All 7 call sites resolve; no "Section
Proposition" or "II-D" leakage.
