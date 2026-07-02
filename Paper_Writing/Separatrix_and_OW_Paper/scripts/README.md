# scripts/

Figure generation pipeline for `Paper_Draft_1A.tex`.

Each script in this directory produces one figure, writes a JSON sidecar, and
(by default) recompiles the paper. All scripts are run from the paper root:

```
cd "Paper_Writing/Separatrix and OW Paper"
```

---

## Running a single figure

```bash
python3 scripts/fig_double_gyre_streamlines.py
```

Outputs:
- `figures/double_gyre_streamlines.png` -- overwrites the stub
- `figures/double_gyre_streamlines.meta.json` -- provenance record
- `Paper_Draft_1A.pdf` -- recompiled (two-pass pdflatex)

To regenerate without recompiling the paper:

```bash
python3 scripts/fig_double_gyre_streamlines.py --no-compile
```

To print the canonical parameters without running anything:

```bash
python3 scripts/fig_double_gyre_streamlines.py --show-params
```

---

## Regenerating all figures

```bash
python3 scripts/regen_all.py
```

Flags:

| Flag | Effect |
|------|--------|
| `--skip-slow` | Skip figures marked `slow: true` in `figures.yaml` (the Monte Carlo sweeps) |
| `--only NAME...` | Only regenerate the named figures (space-separated) |
| `--no-compile` | Do not recompile the paper at the end |
| `--dry-run` | Print what would run without running it |

Examples:

```bash
# All fast figures, then compile once:
python3 scripts/regen_all.py --skip-slow

# Only the double-gyre and separatrix trajectory figures:
python3 scripts/regen_all.py --only double_gyre_streamlines separatrix_trajectories

# Quick preview of what would run:
python3 scripts/regen_all.py --dry-run
```

---

## Figure scripts

| Script | Output | Slow |
|--------|--------|------|
| `fig_double_gyre_streamlines.py` | Streamlines + det(J)=0 contour | no |
| `fig_detJ_contour.py` | Filled contour of det(J) over the domain | no |
| `fig_detJ_trench_cross_section.py` | |det(J)| vs x at fixed y values | no |
| `fig_pentagon_formation.py` | Pentagon-of-pairs cluster diagram | no |
| `fig_estimator_accuracy_vs_noise.py` | Box plots of det(J) and grad det(J) estimation error | no |
| `fig_separatrix_trajectories.py` | Centroid trajectories under Logic C controller | no |
| `fig_separatrix_success_rate.py` | Success-rate heatmap over (sigma_uv, sigma_p) grid | yes |
| `fig_ow_trajectories.py` | Centroid trajectories under Newton-step OW controller | no |
| `fig_ow_success_rate.py` | Success-rate heatmap over (sigma_uv, sigma_p) grid | yes |
| `fig_controller_comparison.py` | Basin-of-attraction side-by-side | no |

---

## Slow figures and caching

`fig_separatrix_success_rate.py` and `fig_ow_success_rate.py` run Monte Carlo sweeps
(200 trials per cell by default) and take several minutes each. Results are cached:

```
figures/separatrix_success_rate.cache.npz
figures/ow_success_rate.cache.npz
```

To redraw from the cache without re-running the simulation:

```bash
python3 scripts/fig_separatrix_success_rate.py --use-cache
python3 scripts/fig_ow_success_rate.py --use-cache
```

To run a quick smoke-test version (5 trials per cell):

```bash
python3 scripts/fig_separatrix_success_rate.py --n-trials-per-cell 5
```

---

## The .meta.json sidecar

Every figure script writes a JSON sidecar alongside the PNG. Example
(`figures/separatrix_trajectories.meta.json`):

```json
{
  "figure_name": "separatrix_trajectories",
  "source_script": "scripts/fig_separatrix_trajectories.py",
  "generated_utc": "2026-06-05T18:42:10Z",
  "git_commit": "ac24cf6",
  "git_dirty": false,
  "python": "3.11.7",
  "numpy": "1.26.4",
  "params": {
    "seed": 0,
    "sim_steps": 200,
    "sigma_uv": 0.0,
    "sigma_p": 0.0,
    "control_gain": 3.0
  },
  "controllers": {
    "primitive": "separatrix_logic_c_step",
    "primitive_file": "trunk/Python_Simulations/Vector_Fields/VF_Robot/src/control/pentagon_primitives.py",
    "primitive_file_sha1": "..."
  },
  "extra": {
    "final_distances_to_saddle": [0.012, 0.018, 0.009, 0.014, 0.021],
    "n_trials": 5
  }
}
```

Fields:
- `git_commit` / `git_dirty` -- exact repo state when the figure was generated.
  If `git_dirty: true`, the working tree had uncommitted changes; the figure may not
  be reproducible from the commit alone.
- `primitive_file_sha1` -- SHA1 of `pentagon_primitives.py` at run time. If this
  changes after a figure was generated, the figure may be stale.
- `extra` -- numerical summaries available for the paper. These are not injected
  automatically; copy values into the .tex manually after reviewing them.

---

## figures.yaml manifest

`figures.yaml` at the paper root lists every figure with its canonical parameters.
`regen_all.py` reads this file to discover scripts; the scripts themselves own their
`PARAMS` dicts and only use `figures.yaml` as a reference. To add a new figure, add
an entry here and create the corresponding script.

---

## Upstream code changes

Two small additions were made to the shared simulation library to support noise sweeps.
Both are backward-compatible (default values reproduce original behavior).

**`trunk/Python_Simulations/Vector_Fields/VF_Robot/src/fields/field_types.py`**

`AnalyticalField` now accepts `noise_std=0.0`. When nonzero, independent Gaussian noise
is added to each field component at every `get_value` call:

```python
field = AnalyticalField(double_gyre_static, noise_std=0.005)
```

**`trunk/Python_Simulations/Vector_Fields/VF_Robot/src/control/pentagon_primitives.py`**

`_sample_vector_at_robots` now reads `cluster.position_noise_std` (defaults to 0.0 if
not set). When nonzero, each robot's position is perturbed by independent Gaussian noise
before the field is sampled, modeling GPS/motion-capture uncertainty:

```python
cluster.position_noise_std = 0.01   # set before running
```

The true robot state is not modified; noise is applied only at measurement time.

---

## Python environment

Scripts use the shared venv at
`trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/`. If that venv exists,
`regen_all.py` invokes it automatically. To run scripts directly, activate first:

```bash
source "trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/activate"
python3 scripts/fig_double_gyre_streamlines.py
```
