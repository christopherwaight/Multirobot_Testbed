# Revision numbers ledger

Written 2026-11-02 by the Session 1 paper agent from the markers in `cwaight_systems_paper_with_revisions.tex` (one row per marker occurrence, in document order). Regenerate by re-running the extraction if the .tex changes. Section names are the paper's headings.

Sources, by tag in the .tex comments:
- `revision_metrics Table II/III/classification/baselines`: `trunk/Python_Simulations/Vector_Fields/VF_Robot/experiments/revision_metrics/tables/*_revised.tex` (Coulomb model, after the 2026-10-28 sign fix).
- `study ...`, `hyperparameter study v3 report_numbers.txt`: `.../experiments/hyperparameter_study/results_v3/report_numbers.txt`, `hyperparameter_study.md` (Tables 3, A, B; Sections 4, 6, 7, 8), and `results_v3/start_distance_bins.txt`.
- `V_VIA numbers.tex`: `Queued_Edits/V_VIA_synthetic_draft/results/numbers.tex` and `tab_orbit_body.tex` (three-robot columns only). Orbit values are synthetic; three-robot attraction values are the old 3R logs.
- `table_IV_revised.tex`: the old three-robot attraction logs scored by revision_metrics; marked \syn because the campaign replaces them.
- `stability_note`, `prose batch N1`, `error note`: drafted notes. **These are outside the source list authorized in REVISION_PROMPT_1_paper.md** (only the error note's classification percentages were listed), so they are marked \rev and need your explicit approval.
- `[OPEN]`: undecided campaign settings from `trunk/hardware_campaign_2026-11/CAMPAIGN_PROMPT.md`.

| # | Marker | Section | Float | Value | Source (from the .tex comment) |
|---|---|---|---|---|---|
| 1 | \syn red | Abstract |  | `100` | [SYNTHETIC] hardware numbers |
| 2 | \syn red | Abstract |  | `86` | [SYNTHETIC] hardware numbers |
| 3 | \syn red | Abstract |  | `0.023` | [SYNTHETIC] hardware numbers |
| 4 | \syn red | Abstract |  | `0.037` | [SYNTHETIC] hardware numbers |
| 5 | \syn red | Abstract |  | `0.40` | [SYNTHETIC] hardware numbers |
| 6 | \syn red | Abstract |  | `0.50` | [SYNTHETIC] hardware numbers |
| 7 | \syn red | Introduction |  | `a hardware campaign on two printed fields` | [SYNTHETIC] campaign not yet run |
| 8 | \rev blue | Multilayer Control Architecture / Closed-Loop Stability and Robustness |  | `1.0003` | [REVIEW] stability_note |
| 9 | \open magenta | Multilayer Control Architecture / Closed-Loop Stability and Robustness |  | `holds the last valid estimate` | [OPEN] invalid-estimate behavior, CAMPAIGN_PROMPT 4.3 |
| 10 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.0249` | [REVIEW] revision_metrics Table II |
| 11 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.0008` | [REVIEW] revision_metrics Table II |
| 12 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.0222` | [REVIEW] revision_metrics Table II |
| 13 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.0137` | [REVIEW] revision_metrics Table II |
| 14 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `45` | [REVIEW] revision_metrics Table II |
| 15 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `62` | [REVIEW] revision_metrics Table II |
| 16 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `95` | [REVIEW] revision_metrics Table II |
| 17 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `100` | [REVIEW] revision_metrics Table II |
| 18 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `(placeholder data, Section~V-C)` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 19 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `[whole table body]` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 20 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `Saddle (hardware, 78 runs)` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 21 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.007` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 22 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.024` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 23 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.023 $\pm$ 0.009` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 24 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.024 [0.037]` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 25 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.018` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 26 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `3.3` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 27 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.205` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 28 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `77/99/100` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 29 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `Vortex (hardware, 80 runs)` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 30 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.018` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 31 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.036` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 32 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.037 $\pm$ 0.017` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 33 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.036 [0.070]` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 34 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.028` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 35 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `3.6` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 36 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `0.187` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 37 | \syn red | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_estimation_results | `34/60/86` | [REVIEW] sim rows: revision_metrics/tables/table_II_revised.tex (six identical linear rows merged); [SYNTHETIC] hardware rows: table_IV_revised.tex (old 3R logs, replaced by the campaign) |
| 38 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields | tab:simulation_orbital_results | `[whole table body]` | [REVIEW] revision_metrics/tables/table_III_revised.tex |
| 39 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.129` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 40 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.033` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 41 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.03` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 42 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `1,971` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 43 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.133` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 44 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.31` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 45 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.058` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 46 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.067` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 47 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.059` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 48 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.3` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 49 | \rev blue | Control Performance in Simulation / Convergence and Orbits on the Canonical Fields |  | `0.4` | [REVIEW] 0.129, 0.033 and the practical range 0.3/0.4 m: Table III (revision_metrics); 0.03 mm, 1,971, r_min values: hyperparameter study v3 report_numbers.txt and Table 3 |
| 50 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `0.16` | [REVIEW] study E4, E5 |
| 51 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `6.6` | [REVIEW] study E4, E5 |
| 52 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `17` | [REVIEW] study E4, E5 |
| 53 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `21` | [REVIEW] study E4, E5 |
| 54 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `2` | [REVIEW] study E4, E5 |
| 55 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `+0.013` | [REVIEW] study E4, E5 |
| 56 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `$-0.047$` | [REVIEW] study E4, E5 |
| 57 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `+0.18` | [REVIEW] study E4, E5 |
| 58 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `99.89` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 59 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `451,508` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 60 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `180,771` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 61 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `74` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 62 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `17` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 63 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `99.6` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 64 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `$-0.044$` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 65 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `+0.051` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 66 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `44` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 67 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `4` | [REVIEW] study E5 basins, start_distance_bins.txt, E1hw |
| 68 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `21.3` | [REVIEW] study E6 |
| 69 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `22.5` | [REVIEW] study E6 |
| 70 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `14.1` | [REVIEW] study E6 |
| 71 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `15.8` | [REVIEW] study E6 |
| 72 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `1.004` | [REVIEW] study E6 |
| 73 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `0.98` | [REVIEW] study E6 |
| 74 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `1.04` | [REVIEW] study E6 |
| 75 | \rev blue | Control Performance in Simulation / Nonlinear Fields |  | `66` | [REVIEW] study E6 |
| 76 | \rev blue | Control Performance in Simulation / Classification |  | `100` | [REVIEW] error note, revision_metrics classification table |
| 77 | \rev blue | Control Performance in Simulation / Classification |  | `96` | [REVIEW] error note, revision_metrics classification table |
| 78 | \rev blue | Control Performance in Simulation / Classification |  | `97` | [REVIEW] error note, revision_metrics classification table |
| 79 | \rev blue | Control Performance in Simulation / Classification |  | `95` | [REVIEW] error note, revision_metrics classification table |
| 80 | \rev blue | Control Performance in Simulation / Classification |  | `96` | [REVIEW] error note, revision_metrics classification table |
| 81 | \rev blue | Control Performance in Simulation / Classification |  | `90` | [REVIEW] error note, revision_metrics classification table |
| 82 | \rev blue | Control Performance in Simulation / Classification |  | `64` | [REVIEW] error note, revision_metrics classification table |
| 83 | \rev blue | Control Performance in Simulation / Classification |  | `65` | [REVIEW] error note, revision_metrics classification table |
| 84 | \rev blue | Control Performance in Simulation / Classification | tab:classification | `[whole table body]` | [REVIEW] revision_metrics/tables/table_classification_revised.tex |
| 85 | \open magenta | Hardware Validation / Validation Test Plan |  | `20` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 86 | \open magenta | Hardware Validation / Validation Test Plan |  | `0.33 or 0.30~m` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 87 | \open magenta | Hardware Validation / Validation Test Plan |  | `\syn{8}` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 88 | \syn red | Hardware Validation / Validation Test Plan |  | `8` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 89 | \open magenta | Hardware Validation / Validation Test Plan |  | `\syn{10}` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 90 | \syn red | Hardware Validation / Validation Test Plan |  | `10` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 91 | \open magenta | Hardware Validation / Validation Test Plan |  | `60` | [OPEN] run length, formation side, radii, repeats, orbit length (CAMPAIGN_PROMPT 4.2, 6.2); inner \syn = the synthetic data's 8 radii x 10 repeats |
| 92 | \syn red | Hardware Validation / Attraction Controller Results |  | `80` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 93 | \syn red | Hardware Validation / Attraction Controller Results |  | `78` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 94 | \syn red | Hardware Validation / Attraction Controller Results |  | `100` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 95 | \syn red | Hardware Validation / Attraction Controller Results |  | `77` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 96 | \syn red | Hardware Validation / Attraction Controller Results |  | `86` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 97 | \syn red | Hardware Validation / Attraction Controller Results |  | `34` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 98 | \syn red | Hardware Validation / Attraction Controller Results |  | `23` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 99 | \syn red | Hardware Validation / Attraction Controller Results |  | `37` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 100 | \syn red | Hardware Validation / Attraction Controller Results |  | `7` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 101 | \syn red | Hardware Validation / Attraction Controller Results |  | `18` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 102 | \syn red | Hardware Validation / Attraction Controller Results |  | `3.3` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 103 | \syn red | Hardware Validation / Attraction Controller Results |  | `3.6` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 104 | \syn red | Hardware Validation / Attraction Controller Results |  | `18` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 105 | \syn red | Hardware Validation / Attraction Controller Results |  | `28` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 106 | \open magenta | Hardware Validation / Attraction Controller Results |  | `campaign` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 107 | \open magenta | Hardware Validation / Attraction Controller Results |  | `campaign` | [SYNTHETIC] old 3R logs via revision_metrics Table IV / V_VIA numbers.tex; [OPEN] hardware classification, no logs yet |
| 108 | \syn red | Hardware Validation / Attraction Controller Results | fig:attr_ecdf | `(placeholder data)` |  |
| 109 | \syn red | Hardware Validation / Orbital Control Results |  | `0.166` | [SYNTHETIC] V_VIA numbers.tex |
| 110 | \syn red | Hardware Validation / Orbital Control Results |  | `0.018` | [SYNTHETIC] V_VIA numbers.tex |
| 111 | \syn red | Hardware Validation / Orbital Control Results |  | `0.111` | [SYNTHETIC] V_VIA numbers.tex |
| 112 | \syn red | Hardware Validation / Orbital Control Results |  | `0.088` | [SYNTHETIC] V_VIA numbers.tex |
| 113 | \syn red | Hardware Validation / Orbital Control Results |  | `0.017` | [SYNTHETIC] V_VIA numbers.tex |
| 114 | \syn red | Hardware Validation / Orbital Control Results |  | `0.036` | [SYNTHETIC] V_VIA numbers.tex |
| 115 | \syn red | Hardware Validation / Orbital Control Results |  | `0.014` | [SYNTHETIC] V_VIA numbers.tex |
| 116 | \syn red | Hardware Validation / Orbital Control Results |  | `0.041` | [SYNTHETIC] V_VIA numbers.tex |
| 117 | \syn red | Hardware Validation / Orbital Control Results |  | `0.027` | [SYNTHETIC] V_VIA numbers.tex |
| 118 | \syn red | Hardware Validation / Orbital Control Results |  | `0.119` | [SYNTHETIC] V_VIA numbers.tex |
| 119 | \syn red | Hardware Validation / Orbital Control Results |  | `0.018` | [SYNTHETIC] V_VIA numbers.tex |
| 120 | \syn red | Hardware Validation / Orbital Control Results |  | `0.095` | [SYNTHETIC] V_VIA numbers.tex |
| 121 | \syn red | Hardware Validation / Orbital Control Results |  | `0.05` | [SYNTHETIC] V_VIA numbers.tex |
| 122 | \syn red | Hardware Validation / Orbital Control Results |  | `0.40` | [SYNTHETIC] V_VIA numbers.tex |
| 123 | \syn red | Hardware Validation / Orbital Control Results |  | `0.50` | [SYNTHETIC] V_VIA numbers.tex |
| 124 | \syn red | Hardware Validation / Orbital Control Results |  | `1` | [SYNTHETIC] V_VIA numbers.tex |
| 125 | \syn red | Hardware Validation / Orbital Control Results |  | `80` | [SYNTHETIC] V_VIA numbers.tex |
| 126 | \syn red | Hardware Validation / Orbital Control Results |  | `3` | [SYNTHETIC] V_VIA numbers.tex |
| 127 | \syn red | Hardware Validation / Orbital Control Results |  | `80` | [SYNTHETIC] V_VIA numbers.tex |
| 128 | \syn red | Hardware Validation / Orbital Control Results |  | `0.40` | [SYNTHETIC] V_VIA numbers.tex |
| 129 | \syn red | Hardware Validation / Orbital Control Results | tab:orbital_radius_comparison | `(placeholder data)` | [SYNTHETIC] V_VIA results/tab_orbit_body.tex, three-robot columns only |
| 130 | \syn red | Hardware Validation / Orbital Control Results | tab:orbital_radius_comparison | `[whole table body]` | [SYNTHETIC] V_VIA results/tab_orbit_body.tex, three-robot columns only |
| 131 | \rev blue | Discussion / Error Analysis |  | `6.5` | [REVIEW] study E3 |
| 132 | \rev blue | Discussion / Error Analysis |  | `2.3` | [REVIEW] study E3 |
| 133 | \rev blue | Discussion / Error Analysis |  | `$2\times10^{-5}$` | [REVIEW] prose batch N1 (3R parked intervals) |
| 134 | \syn red | Discussion / Error Analysis |  | `0.177` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 135 | \syn red | Discussion / Error Analysis |  | `0.166` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 136 | \syn red | Discussion / Error Analysis |  | `0.075` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 137 | \syn red | Discussion / Error Analysis |  | `0.197` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 138 | \syn red | Discussion / Error Analysis |  | `0.037` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 139 | \syn red | Discussion / Error Analysis |  | `0.157` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 140 | \syn red | Discussion / Error Analysis |  | `The between-trial spread stays small while the within-run spread grows with $r_d$.` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 141 | \open magenta | Discussion / Error Analysis |  | `campaign block E` | [SYNTHETIC] V_VIA numbers.tex; the within/between split is an input of the synthetic generator |
| 142 | \rev blue | Discussion / Comparison with Matched Baselines |  | `0.05--0.06` | [REVIEW] |
| 143 | \rev blue | Discussion / Comparison with Matched Baselines |  | `0.06` | [REVIEW] |
| 144 | \rev blue | Discussion / Comparison with Matched Baselines |  | `0.21` | [REVIEW] |
| 145 | \rev blue | Discussion / Comparison with Matched Baselines |  | `0.12` | [REVIEW] |
| 146 | \rev blue | Discussion / Comparison with Matched Baselines |  | `four` | [REVIEW] |
| 147 | \rev blue | Discussion / Comparison with Matched Baselines |  | `0.07` | [REVIEW] |
| 148 | \rev blue | Discussion / Comparison with Matched Baselines | tab:baselines | `[whole table body]` | [REVIEW] revision_metrics/tables/table_baselines_revised.tex |
| 149 | \syn red | Conclusion |  | `100` | [SYNTHETIC] hardware numbers are placeholders until the campaign |
| 150 | \syn red | Conclusion |  | `86` | [SYNTHETIC] hardware numbers are placeholders until the campaign |
| 151 | \syn red | Conclusion |  | `0.023` | [SYNTHETIC] hardware numbers are placeholders until the campaign |
| 152 | \syn red | Conclusion |  | `0.037` | [SYNTHETIC] hardware numbers are placeholders until the campaign |
| 153 | \syn red | Conclusion |  | `0.40` | [SYNTHETIC] hardware numbers are placeholders until the campaign |
| 154 | \syn red | Conclusion |  | `0.40` | [SYNTHETIC] hardware numbers are placeholders until the campaign |
| 155 | \syn red | Conclusion |  | `0.50` | [SYNTHETIC] hardware numbers are placeholders until the campaign |

Totals: 79 \syn, 67 \rev, 9 \open occurrences.

Notes
- The bodies of Table II (simulation rows), Table III, Table IV (classification) and Table VI (baselines) are wrapped whole in \rev, so every cell in them is unapproved simulation output. The two hardware rows of Table II are \syn cell by cell (old three-robot logs, replaced by the campaign).
- Practical-range radii 0.3 and 0.4 m in IV-B are read from Table III by the 0.25 r_d criterion, not taken from a file.
- Model parameters stated without a marker (tau 0.28 s, v_c 0.025 m/s, v_max 0.4 m/s, alpha 0.70, the 0.3 m/s per-axis hardware limit, the 0.5 m/s simulation command limit from revision_metrics/sim.py, epsilon_c = 0.2) come from the robot model note, the one-pager and the simulation code. Confirm you consider them approved.
- Numbers kept from the submitted paper (calibration RMSE, map size, 10 Hz, gains, the 10^-6 m deadband, the 0.02 m/s calibration error) are unmarked.
