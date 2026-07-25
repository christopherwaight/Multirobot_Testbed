# Archive: 500 trials/cell, pre-objectivity-fix

Frozen 2026-07-24. Read only. Kept for comparison against the 10,000-trial
runs that replaced them.

## What this is

The last Monte Carlo sweep of the OECS traverser (Primitive 11) run before
the descent channel was made rotation equivariant. Generated 2026-07-24
~15:07 PDT, 500 trials per cell.

## Why it was superseded

`_descend_s1` in `src/control/pentagon_primitives.py` saturated the descent
command per Cartesian component:

    vx = -v_max * tanh(g_perp * grad_s1[0] / v_max)
    vy = -v_max * tanh(g_perp * grad_s1[1] / v_max)

That is frame dependent. Rotating the coordinate frame changes both the
descent direction (up to 39 degrees of skew) and the commanded speed (a
factor of sqrt(2) between an axis-aligned and a diagonal gradient), so the
closed-loop trajectory depended on how the x-axis happened to be drawn.
The paper's objectivity argument does not survive that: s1 is objective,
but a frame-dependent controller acting on it discards the property at the
last step.

Replaced by vector saturation, which preserves direction and saturates
magnitude:

    a = -g_perp * grad_s1
    scale = v_max * tanh(|a| / v_max) / |a|

This is exactly rotation equivariant (residual ~1e-17, verified in
`tests/test_oecs_objectivity.py`) and matches the PARK Lyapunov result
already stated in the paper.

## Caveats when comparing

- The CSV headers here stamp `git_commit: 0e2e370`, but that commit does
  NOT contain the fix (the fix landed in e6cbaa6). Whether the fix was in
  the working tree at run time cannot be determined from these files. Treat
  these numbers as pre-fix, and do not cite them.
- Both laws are valid Lyapunov descents (s1_dot <= 0 either way), so the
  difference is objectivity and speed, not stability.
- The old law commanded up to 41 percent more speed on diagonal gradients.
  Expect the new runs to be somewhat SLOWER, and possibly slightly worse
  near the sigma_uv cliff. That is the cost of frame independence, not a
  regression.

## Files

| File | Contents |
|---|---|
| `trials_fixed.csv` | one row per trial, two-target traverse |
| `summary_fixed.csv` | per-cell rates, sigma_uv x sigma_p grid |
| `summary_single_target.csv` | single far-saddle variant |
| `flip_resolution.csv` | flip resolution vs sigma_uv, sigma_p = 0 |
| `flip_resolution_sigma_p.csv` | flip resolution vs sigma_p |
