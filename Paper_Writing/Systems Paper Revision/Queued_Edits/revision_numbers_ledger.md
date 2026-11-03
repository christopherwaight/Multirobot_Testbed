# Revision numbers ledger (Session 1, surgical revision, 2026-11-02)

Generated from the markers in `cwaight_systems_paper/cwaight_systems_paper_with_revisions.tex`, one row per marker occurrence in document order. Section names follow the compiled numbering. Table-level markers (`\rev{\begin{tabular}...}`) cover every cell of that table and are listed once.

## Source files behind the tags

| Tag in the .tex comment | File | Authorized in REVISION_PROMPT_1? |
|---|---|---|
| revision_metrics/tables/table_II_revised.tex, table_III_revised.tex, table_classification_revised.tex, table_baselines_revised.tex | `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/revision_metrics/tables/` (Coulomb model, after the 2026-10-28 sign fix) | yes, `\rev` |
| table_IV_revised.tex (hardware attraction) | same folder; old three-robot logs scored by revision_metrics | yes, `\syn` (campaign replaces) |
| V_VIA numbers.tex, tab_orbit_body.tex | `Queued_Edits/V_VIA_synthetic_draft/results/` (three-robot columns only; orbit rows fabricated) | yes, `\syn` |
| hyperparameter study v3 (E3k, E4, E5, E6, closed form, table 3), start_distance_bins.txt | `.../experiments/hyperparameter_study/results_v3/report_numbers.txt`, `start_distance_bins.txt`, `hyperparameter_study.md` | yes, `\rev` |
| vib_baselines_note | `vib_baselines_note/vib_baselines.tex` (PENDING sentence left out) | yes, `\rev` |
| error note | `error_note/error_analysis.tex` classification percentages | yes, `\rev` |
| stability_note | `stability_note/stability.tex` | **no**, drafted note, `\rev` |
| prose batch N1 | `Queued_Edits/prose_framing_batch.tex` note N1 (parked-robot reading change) | **no**, drafted note, `\rev` |
| [OPEN] | `trunk/hardware_campaign_2026-11/CAMPAIGN_PROMPT.md` sections 4.2, 4.3, 6.2 | undecided settings |

## Markers

| # | Marker | Section | Value | Source (from the .tex comment) |
|---|---|---|---|---|
| 1 | `\syn` | Abstract | `100` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 2 | `\syn` | Abstract | `86` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 3 | `\syn` | Abstract | `158` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 4 | `\syn` | Abstract | `0.037` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 5 | `\open` | Abstract | `36` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 6 | `\open` | Abstract | `six` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 7 | `\syn` | Abstract | `0.40` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 8 | `\syn` | Abstract | `0.50` | [SYNTHETIC] hardware numbers; [OPEN] orbit trial count and radii |
| 9 | `\open` | I Introduction | `campaign` | [OPEN] campaign trial count; the submitted 169 is superseded |
| 10 | `\rev` | III-D Closed-Loop Stability and Robustness | `1.0003` | [REVIEW] stability_note (outside the authorized source list) |
| 11 | `\open` | III-D Closed-Loop Stability and Robustness | `holds the last valid estimate` | [OPEN] invalid-estimate behavior, CAMPAIGN_PROMPT 4.3 |
| 12 | `\rev` | IV-A Convergence to Critical Points | `100` | [REVIEW] revision_metrics/tables/table_II_revised.tex |
| 13 | `\rev` | IV-A Convergence to Critical Points | `95` | [REVIEW] revision_metrics/tables/table_II_revised.tex |
| 14 | `\rev` | IV-A Convergence to Critical Points (fig:sim_convergence caption) | `six` | [REVIEW] hyperparameter study v3 E1hw saved trajectories |
| 15 | `\rev` | IV-A Convergence to Critical Points | `all cells of tab:simulation_estimation_results` | [REVIEW] revision_metrics/tables/table_II_revised.tex (six identical analytical rows merged, transposed); classification rows: table_classification_revised.tex, attraction columns |
| 16 | `\rev` | IV-B Orbital Control | `100` | [REVIEW] revision_metrics/tables/table_III_revised.tex caption (100 trials per field and radius) |
| 17 | `\rev` | IV-B Orbital Control | `all cells of tab:simulation_orbital_results` | [REVIEW] revision_metrics/tables/table_III_revised.tex (RMSE columns omitted; linear std < 0.001 and Ctrl. = Bias) |
| 18 | `\rev` | IV-B Orbital Control | `0.001` | [REVIEW] revision_metrics/tables/table_III_revised.tex |
| 19 | `\rev` | IV-B Orbital Control | `0.3` | [REVIEW] revision_metrics/tables/table_III_revised.tex |
| 20 | `\rev` | IV-B Orbital Control | `0.4` | [REVIEW] revision_metrics/tables/table_III_revised.tex |
| 21 | `\rev` | IV-C Nonlinear Fields | `6.6` | [REVIEW] hyperparameter study v3 E4, E5 (report_numbers.txt) |
| 22 | `\rev` | IV-C Nonlinear Fields | `0.18` | [REVIEW] hyperparameter study v3 E4, E5 (report_numbers.txt) |
| 23 | `\rev` | IV-C Nonlinear Fields | `99.89` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 24 | `\rev` | IV-C Nonlinear Fields | `451,508` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 25 | `\rev` | IV-C Nonlinear Fields | `74` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 26 | `\rev` | IV-C Nonlinear Fields | `17` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 27 | `\rev` | IV-C Nonlinear Fields | `99.6` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 28 | `\rev` | IV-C Nonlinear Fields | `44` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 29 | `\rev` | IV-C Nonlinear Fields | `4` | [REVIEW] hyperparameter study v3 E5 basins, start_distance_bins.txt, E1hw gyre orbits |
| 30 | `\rev` | IV-C Nonlinear Fields | `1.7` | [REVIEW] hyperparameter study v3 E6 |
| 31 | `\rev` | IV-C Nonlinear Fields | `1.004` | [REVIEW] hyperparameter study v3 E6 |
| 32 | `\rev` | IV-C Nonlinear Fields | `0.98` | [REVIEW] hyperparameter study v3 E6 |
| 33 | `\rev` | IV-C Nonlinear Fields | `1.04` | [REVIEW] hyperparameter study v3 E6 |
| 34 | `\rev` | IV-C Nonlinear Fields | `66` | [REVIEW] hyperparameter study v3 E6 |
| 35 | `\rev` | IV-C Nonlinear Fields (fig:nonlinear caption) | `six` | [REVIEW] hyperparameter study v3: E1hw saved trajectories |
| 36 | `\rev` | IV-D Classification | `90` | [REVIEW] revision_metrics/tables/table_classification_revised.tex, error note |
| 37 | `\rev` | IV-D Classification | `95` | [REVIEW] revision_metrics/tables/table_classification_revised.tex, error note |
| 38 | `\rev` | IV-D Classification | `100` | [REVIEW] revision_metrics/tables/table_classification_revised.tex, error note |
| 39 | `\rev` | IV-D Classification | `100` | [REVIEW] revision_metrics/tables/table_classification_revised.tex, error note |
| 40 | `\open` | V-B Validation Test Plan | `20` | [OPEN] run length, formation side (CAMPAIGN_PROMPT 4.2, 6.2) |
| 41 | `\open` | V-B Validation Test Plan | `0.33 or 0.30` | [OPEN] run length, formation side (CAMPAIGN_PROMPT 4.2, 6.2) |
| 42 | `\open` | V-B Validation Test Plan | `six` | [OPEN] number of radii, repeats, orbit run length (CAMPAIGN_PROMPT 6.2) |
| 43 | `\open` | V-B Validation Test Plan | `three` | [OPEN] number of radii, repeats, orbit run length (CAMPAIGN_PROMPT 6.2) |
| 44 | `\open` | V-B Validation Test Plan | `1800` | [OPEN] number of radii, repeats, orbit run length (CAMPAIGN_PROMPT 6.2) |
| 45 | `\syn` | V-C Attraction Controller Results | `80` | [SYNTHETIC] run counts of the old 3R logs, table_IV_revised.tex; replaced by the campaign |
| 46 | `\syn` | V-C Attraction Controller Results | `78` | [SYNTHETIC] run counts of the old 3R logs, table_IV_revised.tex; replaced by the campaign |
| 47 | `\syn` | V-C Attraction Controller Results | `100` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 48 | `\syn` | V-C Attraction Controller Results | `86` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 49 | `\syn` | V-C Attraction Controller Results | `77` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 50 | `\syn` | V-C Attraction Controller Results | `34` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 51 | `\syn` | V-C Attraction Controller Results | `0.018` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 52 | `\syn` | V-C Attraction Controller Results | `0.036` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 53 | `\syn` | V-C Attraction Controller Results | `0.007` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 54 | `\syn` | V-C Attraction Controller Results | `0.024` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 55 | `\open` | V-C Attraction Controller Results | `campaign` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 56 | `\open` | V-C Attraction Controller Results | `campaign` | [SYNTHETIC] old 3R logs scored by revision_metrics (table_IV_revised.tex, V_VIA numbers.tex), replaced by the campaign; [OPEN] hardware classification |
| 57 | `\syn` | V-C Attraction Controller Results (fig:vortextomiddle caption) | `[Placeholder]` | (see paragraph comment) |
| 58 | `\syn` | V-C Attraction Controller Results (tab:estimation_results caption) | `(placeholder data)` | [SYNTHETIC] table_IV_revised.tex run counts (old 3R logs) |
| 59 | `\syn` | V-C Attraction Controller Results (tab:estimation_results caption) | `78` | [SYNTHETIC] table_IV_revised.tex run counts (old 3R logs) |
| 60 | `\syn` | V-C Attraction Controller Results (tab:estimation_results caption) | `80` | [SYNTHETIC] table_IV_revised.tex run counts (old 3R logs) |
| 61 | `\syn` | V-C Attraction Controller Results | `all cells of tab:estimation_results` | [SYNTHETIC] revision_metrics/tables/table_IV_revised.tex (old 3R logs), replaced by the campaign |
| 62 | `\open` | V-D Orbital Control Results | `1800` | [OPEN] orbit run length (CAMPAIGN_PROMPT 6.2) |
| 63 | `\syn` | V-D Orbital Control Results | `0.166` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 64 | `\syn` | V-D Orbital Control Results | `0.018` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 65 | `\syn` | V-D Orbital Control Results | `0.027` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 66 | `\syn` | V-D Orbital Control Results | `0.119` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 67 | `\syn` | V-D Orbital Control Results | `0.111` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 68 | `\syn` | V-D Orbital Control Results | `0.088` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 69 | `\syn` | V-D Orbital Control Results | `0.018` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 70 | `\syn` | V-D Orbital Control Results | `0.095` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 71 | `\syn` | V-D Orbital Control Results | `0.10` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 72 | `\syn` | V-D Orbital Control Results | `0.40` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 73 | `\syn` | V-D Orbital Control Results | `0.50` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 74 | `\syn` | V-D Orbital Control Results | `One` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 75 | `\syn` | V-D Orbital Control Results | `three` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 76 | `\syn` | V-D Orbital Control Results | `0.40` | [SYNTHETIC] V_VIA numbers.tex and tab_orbit_body.tex (three-robot columns, six radii) |
| 77 | `\syn` | V-D Orbital Control Results (tab:orbital_radius_comparison caption) | `(placeholder data)` | [SYNTHETIC] V_VIA results/tab_orbit_body.tex |
| 78 | `\syn` | V-D Orbital Control Results | `all cells of tab:orbital_radius_comparison` | [SYNTHETIC] V_VIA results/tab_orbit_body.tex, three-robot columns, rows 0.05 and 0.15 m omitted |
| 79 | `\rev` | VI-A Error Analysis | `0.03` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 80 | `\rev` | VI-A Error Analysis | `1971` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 81 | `\rev` | VI-A Error Analysis | `0.133` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 82 | `\rev` | VI-A Error Analysis | `0.31` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 83 | `\rev` | VI-A Error Analysis | `0.058` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 84 | `\rev` | VI-A Error Analysis | `0.067` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 85 | `\rev` | VI-A Error Analysis | `0.059` | [REVIEW] hyperparameter study v3 report_numbers.txt (closed form, table 3) |
| 86 | `\rev` | VI-A Error Analysis | `$2\times10^{-5}$` | [REVIEW] prose batch N1 (outside the authorized source list) |
| 87 | `\syn` | VI-A Error Analysis | `0.177` | [SYNTHETIC] V_VIA numbers.tex; [OPEN] static noise runs |
| 88 | `\syn` | VI-A Error Analysis | `0.166` | [SYNTHETIC] V_VIA numbers.tex; [OPEN] static noise runs |
| 89 | `\open` | VI-A Error Analysis | `campaign block E` | [SYNTHETIC] V_VIA numbers.tex; [OPEN] static noise runs |
| 90 | `\rev` | VI-B Comparison with Matched Baselines | `0.12` | [REVIEW] vib_baselines_note |
| 91 | `\rev` | VI-B Comparison with Matched Baselines | `four` | [REVIEW] vib_baselines_note |
| 92 | `\rev` | VI-B Comparison with Matched Baselines | `0.07` | [REVIEW] vib_baselines_note |
| 93 | `\rev` | VI-B Comparison with Matched Baselines (fig:effort caption) | `three` | [REVIEW] run_speed_traces.py: Table VI attraction batches rerun with recording (seed 0, Coulomb model); effort and final error match baselines_attraction.csv; traces in revision_metrics/results/baselines_speed_traces.npz |
| 94 | `\rev` | VI-B Comparison with Matched Baselines (fig:effort caption) | `0.49 to 0.56` | [REVIEW] run_speed_traces.py: Table VI attraction batches rerun with recording (seed 0, Coulomb model); effort and final error match baselines_attraction.csv; traces in revision_metrics/results/baselines_speed_traces.npz |
| 95 | `\rev` | VI-B Comparison with Matched Baselines (tab:baselines caption) | `500` | [REVIEW] revision_metrics/tables/table_baselines_revised.tex caption (trial counts) |
| 96 | `\rev` | VI-B Comparison with Matched Baselines (tab:baselines caption) | `50` | [REVIEW] revision_metrics/tables/table_baselines_revised.tex caption (trial counts) |
| 97 | `\rev` | VI-B Comparison with Matched Baselines | `all cells of tab:baselines` | [REVIEW] revision_metrics/tables/table_baselines_revised.tex; identical oracle rows and the analytical plane-fit row merged |
| 98 | `\syn` | VII Conclusion | `158` | [SYNTHETIC] hardware numbers |
| 99 | `\syn` | VII Conclusion | `100` | [SYNTHETIC] hardware numbers |
| 100 | `\syn` | VII Conclusion | `86` | [SYNTHETIC] hardware numbers |
| 101 | `\syn` | VII Conclusion | `0.037` | [SYNTHETIC] hardware numbers |
| 102 | `\syn` | VII Conclusion | `0.40` | [SYNTHETIC] hardware numbers |
| 103 | `\syn` | VII Conclusion | `0.50` | [SYNTHETIC] hardware numbers |

Totals: 45 `\syn`, 45 `\rev`, 13 `\open`.

