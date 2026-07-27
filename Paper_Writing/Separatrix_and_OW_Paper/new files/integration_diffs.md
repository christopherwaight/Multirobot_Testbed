# Integration diffs for Draft_5c.tex

Each entry gives the search string and its replacement. Line numbers are
from the uploaded `Draft_5c.tex`. Apply after dropping in the three
replacement sections.

**New labels introduced:** `eq:sigma_eff`, `eq:gain_ladder`, `eq:groups`,
`eq:error_budget`, `tab:budget` (§III-D); `eq:transverse_rates`,
`tab:bandwidth` (§IV-D).

**Labels retired:** `lem:gains`, `eq:gain_grad`, `eq:gain_hess`,
`rem:eigvec`, `eq:oecs_n_dynamics`, and the symbol `M_3`.

**New symbols:** `\nu` (relative noise), `\tilde\rho` (relative radius),
`\gamma_q` (geometry gains), `a_\perp` (transverse rate), `q` (derivative
order). All were checked against the current draft and none collide.
`\varepsilon` was deliberately avoided, since the draft already uses it for
double-gyre unsteadiness and for the two band thresholds.

**Table renumbering:** `tab:budget` and `tab:bandwidth` land ahead of
`tab:params`, so table numbers shift. Nothing in the draft hard-codes a
table number, so this is safe.

---

## D1 — §III-C, retire the `lem:gains` reference (line ~750)

```
-(Lemma~\ref{lem:gains}), one full derivative order better in noise gain
-than the $\hat{\mathbf{H}}_D$ eigenvectors the $D$ tracker uses for the
-same purpose.
+one derivative order better in noise gain than the
+$\hat{\mathbf{H}}_D$ eigenvectors the $D$ tracker uses for the same
+purpose, and free of the floor those eigenvectors carry
+(Table~\ref{tab:budget}).
```

## D2 — §V-A, hook the exponent reading to the sweep (line ~1371)

```
-The estimator-accuracy sweep holds
-the formation static and draws independent noise per trial.
+The estimator-accuracy sweep holds
+the formation static and draws independent noise per trial; it is read
+on log--log axes, where the exponents of (\ref{eq:error_budget}) appear
+directly as slopes.
```

## D3 — §V-D-2, isotropy prediction (line 1516)

```
-heading, testing the isotropy prediction of Lemma~\ref{lem:gains}.
+heading, testing the isotropy prediction of (\ref{eq:gain_ladder}).
```

## D4 — §VI-A, name the floor (line 1611)

```
-This traces to the structural bias of $\hat{\mathbf{H}}_D$ (Section~\mbox{III-D}):
+This traces to the floor $e$ on $\hat{\mathbf{H}}_D$ (Section~\mbox{III-D}):
```

## D5 — §VI-A, point at the effective-noise equation (line 1693)

```
-scales the same curve by $\|\mathbf{J}\|$, per the effective-noise
-argument of Section~\mbox{III-D} ($\sigma_p = 0.01$ alone and
+scales the same curve by $\|\mathbf{J}\|$, per (\ref{eq:sigma_eff})
+($\sigma_p = 0.01$ alone and
```

## D6 — §VI-A, isotropy (lines 1705 and 1721)

```
-75.9--77.1\% success per quartile, confirming the isotropy of
-Lemma~\ref{lem:gains}.
+75.9--77.1\% success per quartile, confirming the isotropy of
+(\ref{eq:gain_ladder}).
```

```
-produces here, while Lemma~\ref{lem:gains}'s estimator-error isotropy
+produces here, while the estimator-error isotropy of (\ref{eq:gain_ladder})
```

## D7 — §VI-B-1, retire `rem:eigvec` (line 1748)

```
-onto the crossing wall trench, the divergence Remark~\ref{rem:eigvec}
-already predicts for the $D$ tracker's own command.
+onto the crossing wall trench, the divergence the floor $e$ of
+Section~\mbox{III-D} already predicts for the $D$ tracker's own command.
```

## D8 — §VI-B-3, reattribute the zero-noise tracking error (line 1801)

**This is the diff that carries the new stability finding into Results.**

```
-zero-noise tracking error is 0.0075, about five times the $D$ tracker's
-0.0016, the gradient-rate cost predicted by Remark~\ref{rem:s1hess}.
+zero-noise tracking error is 0.0075, about five times the $D$ tracker's
+0.0016. The ultimate bound (\ref{eq:iss_bound}) predicts the opposite
+ranking, since $a_{\perp,s_1} > a_{\perp,D}$ at this operating point;
+the ordering instead matches the discrete stability condition of
+Table~\ref{tab:bandwidth}, which the $s_1$ tracker crosses over the
+outer half of the structure.
```

## D9 — §VI-B-3, feed the open-loop numbers into Table `tab:bandwidth` (line 1905)

```
-transverse curvature $6.9$ at that point, the right sign and the same
-order though Newton normalization is unavailable; and at
+transverse curvature $6.9$ at that point, the two values entering
+Table~\ref{tab:bandwidth}; and at
```

## D10 — §VIII Conclusion, retire `rem:eigvec` (line 2228)

```
-in magnitude but not reliably in sign (Remarks~\ref{rem:s1hess} and
-\ref{rem:eigvec}); its frame
+in magnitude but not reliably in sign (Remark~\ref{rem:s1hess},
+Table~\ref{tab:budget}); its frame
```

## D11 — §VIII Future Work, add the discriminating run (line 2250)

```
-Online formation reshaping would protect the conditioning of
-$\boldsymbol{\Phi}$.
+Online formation reshaping would protect the conditioning of
+$\boldsymbol{\Phi}$, and a step-halving sweep would separate the $s_1$
+tracker's discretization limit cycle from its estimation error
+(Table~\ref{tab:bandwidth}).
```

## D12 — Appendix A, rehost the ladder derivation (lines 2264, 2284)

```
-\begin{IEEEproof}[Proof of Lemma~\ref{lem:gains}]
+\begin{IEEEproof}[Derivation of the noise ladder (\ref{eq:gain_ladder})]
```

```
-(\ref{eq:gain_grad}) and (\ref{eq:gain_hess}). No power sum depends on
+(\ref{eq:gain_ladder}). No power sum depends on
```

## D13 — Appendix A, Theorem 1 proof Step 1 (lines 2300-2310)

`eq:normal_form` now carries $\kappa_\perp$ for a generic surrogate, so the
proof follows suit and names $a_\perp$.

```
-(\ref{eq:normal_form}) the Hessian is diagonal in
-$(\hat{\mathbf{e}}_\parallel, \hat{\mathbf{e}}_\perp)$ with transverse
-eigenvalue $\lambda_\perp > 0$ and transverse gradient component
-$\lambda_\perp n$, so the Newton argument in that direction is
-$-\lambda_\perp n/\lambda_\perp = -n$ and the saturated command is
-(\ref{eq:n_dynamics}).
+(\ref{eq:normal_form}) the Hessian is diagonal in
+$(\hat{\mathbf{e}}_\parallel, \hat{\mathbf{e}}_\perp)$ with transverse
+eigenvalue $\kappa_\perp > 0$ and transverse gradient component
+$\kappa_\perp n$, so the Newton argument in that direction is
+$-\kappa_\perp n/\kappa_\perp = -n$, giving $a_{\perp,D} = 1$ in
+(\ref{eq:n_dynamics}).
```

```
-n\tanh(1)/c_{\max}$, hence $\dot{V}_\perp \leq -2\tanh(1)V_\perp$ and
+n\tanh(1)/c_{\max}$, hence $\dot{V}_\perp \leq -2a_\perp\tanh(1)V_\perp$ and
```

---

# Optional diffs

## O1 — §I, contribution 1 (line ~155)

```
-quadratic model and with exact, heading-isotropic noise gains.
+quadratic model, exact heading-isotropic noise gains, and an error
+budget that separates noise, truncation, and two constants no formation
+reduces.
```

## O2 — §IV-C, point the gradient-rate remark forward (line 1116)

```
-transverse channel contracts at gradient rate rather than at Newton
-rate.
+transverse channel contracts at gradient rate rather than at Newton
+rate, $a_\perp = g_\perp\kappa_\perp$ in Section~\mbox{IV-D}.
```

## O3 — §VII-D Limitations (line ~2166)

No edit required: `thm:separatrix` and `ass:trench` both survive. If you
want the discretization caveat here rather than in §IV-D, add one
sentence after "the noise sweeps of Table~II start from a formation
straddling the separatrix":

```
+Both are continuous-time results, and Table~\ref{tab:bandwidth} shows
+the $s_1$ tracker's implemented loop does not realize them over the
+outer half of the structure.
```

## O4 — Abstract

No change required. The abstract reports 0.0016 and 0.0075 as
measurements without attributing a cause, so D8 does not contradict it.

---

# Two things to check before this goes out

1. **D8 makes a claim you have not yet measured.** Saying the $s_1$
   tracker's larger tracking error is "more likely a discretization
   effect" invites the obvious question, and the answer is one config
   line: rerun the zero-noise clean runs at $\Delta t = 0.05$. If the
   error roughly halves it is the limit cycle and D8 stands as written;
   if it does not move, soften both D8 and the "two consequences"
   paragraph in §IV-D to a bandwidth caveat rather than an explanation.
   The $D$ tracker at $\Delta t\,k\,a_\perp = 0.3$ is the built-in
   control.

2. **§III-D asserts the exponents are visible as slopes in Fig. 2.**
   They should be, since both panels are already log--log, but read
   them off before the sentence in D2 ships: $+1$ against $\sigma_{uv}$,
   and $-q$ / $+(p{+}1{-}q)$ either side of the minimum against $\rho$.

---

# Length

Prose words, excluding floats and display math:

| Section | Draft 5c | Replacement | Change |
|---|---|---|---|
| §III-D estimator sensitivity | 694 | 543 | −22% |
| §IV-D stability properties | 930 | 818 | −12% |
| §VII-B + §VII-C discussion | 391 | 334 | −15% |

§IV-D misses the 25% target because it absorbs net-new material: the
bandwidth conditions, `tab:bandwidth`, and the two consequences run to
about 165 words plus a table. Cut that block and §IV-D lands at roughly
650 words, a 30% reduction, but the reframe goes with it.

Three further levers if you need the pages:

- **Drop `tab:budget`** and carry the six rows as one sentence each in
  the two paragraphs that precede it. Saves a float; costs the
  skimmable object the discussion refers back to twice.
- **Demote Theorem 1** to a stated property with no environment and no
  appendix proof. Saves about 50 words in §IV-D and 230 in Appendix A.
  It is currently claimed as a contribution in the abstract and §I, so
  this one needs those edited too.
- **Fold (\ref{eq:groups}) inline.** Saves a display, costs nothing else.
