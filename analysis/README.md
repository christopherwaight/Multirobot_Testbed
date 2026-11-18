# analysis/

Analysis scripts and outputs for the vector field revision (2026-11). Raw data stay where they are:
`New_Hardware_Data/`, `Extras/`, `data_simulation_campaign/`. Results are summarized in `REPORT.md`.

Python: `trunk/Python_Simulations/Vector_Fields/VF_Robot/venv/bin/python3` (written `$VP` below). Run everything from
the repo root. Scripts build their paths from `common/paths.py`.

## Layout

| Folder | Contents |
|---|---|
| `common/` | `paths.py` (all folders); `centres.py` (hue decode, field-zero fit); `hw2025.py`, `hw2026.py` (run loaders; 2026 reads only `log_legend` variables); `frames.py` (session to net frame registration); `map_readings.py` (map readings and 1 cm cells in the 2026 net frame); `figstyle.py` (paper figure style) |
| `hardware/` | `settling_2026.py` (transient trim), `score_hardware.py` (all hardware metrics), `figures_hardware.py`; `outputs/` CSVs and `hardware_summary.json`, `figures/` |
| `simulation/` | `rigidsim.py` (vectorized 3R and 4R sim, validated by `validate_rigidsim.py`), `paired_runs.py` (sim from each hardware start, held formation), `figure_paired.py` (paper Fig. 8), `theory_check.py` (stability limit, orbit floor); `revision_metrics/` (results, tables, figures written by the campaign through `REVISION_OUT`), `paired/` |
| `limit_cycle/` | `limit_cycle.py` (scan), `figure_limit_cycle.py`; `outputs/`, `figures/` |
| `rbf/` | Thin-plate-spline alternative to the 2026 nets (2026-11-17, not used by the paper): `rbf_fields.py` (same cells and held-out runs as the nets), `rbf_smoothing_scan.py`, `rbf_paired.py` (hardware-matched runs on the RBF maps); `outputs/`, `figures/` |
| `paper/` | `extract_markers.py` (marker list), `extra_numbers.py` (numbers not in the summaries), `table_hardware.py`, `fig_reconstruction.py`, `apply_paper_edits.py`, `apply_letter_edits.py`, `apply_prose_batch.py`, `apply_round2.py` to `apply_round2e.py` (2026-11-17), `numbers_map.csv` (every marker: old, new, source), `queued_prose_batch.md` (waiting for approval), backups of the tex and letter before the 2026-11-16 edits |

## Order to rerun

```
$VP analysis/hardware/settling_2026.py
$VP analysis/hardware/score_hardware.py
$VP analysis/hardware/figures_hardware.py
$VP analysis/simulation/validate_rigidsim.py
$VP analysis/simulation/paired_runs.py
$VP analysis/simulation/theory_check.py
$VP analysis/limit_cycle/limit_cycle.py
$VP analysis/limit_cycle/figure_limit_cycle.py
$VP analysis/simulation/figure_paired.py
$VP analysis/paper/fig_reconstruction.py
$VP analysis/paper/table_hardware.py
$VP analysis/paper/extra_numbers.py
python3 analysis/paper/extract_markers.py
python3 analysis/paper/apply_paper_edits.py --dry-run     # then without --dry-run
python3 analysis/paper/apply_letter_edits.py --dry-run
```

The edit scripts match exact text and stop on the first mismatch. After the 2026-11-16 edits have been applied, they will
stop at once, because the old values are gone. Use them as a record, or to replay the edits on the backups.

The simulation campaign (revision metrics and v4 sweep) runs from `data_simulation_campaign/run_campaign.sh`. Its
README has the layout and the resume command. The revision lane writes into `analysis/simulation/revision_metrics/`.
