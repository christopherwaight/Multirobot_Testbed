# 🏆 Robot Report Card: which Decabot is best, and why

**From the 2026-11 hardware campaign** (vortex 3R/4R on 2026-11-13, saddle 3R on 2026-11-15). Compiled by Claude (AI assistant) on 2026-11-15. The user asked for this to be a prominent note.

## The verdict

> **🥇 pacific-blue is the best robot overall: the best sensor and the strongest, most consistent motion.**
> **🥈 celeste is a close second, and the *most predictable* robot.**
> 🥉 tidal comes third: solid, but its battery died during the saddle runs.
> 4th: **redwood**, for an expected reason. It is the robot the sensor net **never saw** (the IDETC unseen-robot test).

## The evidence

| | Sensor direction error (11/13) | Speed reached, healthy runs (actual/commanded) | Battery this campaign | Motion fit: gain / lag α (with the 0.1 s delay) | Motion predictability (free-run error, m/s) | Stalls |
|---|---|---|---|---|---|---|
| **🥇 pacific-blue** | **5.1°** (best) | **0.87–0.96** (highest) | never dropped | **0.89** / 0.09 | 0.035 (least predictable, mostly the stalls) | **every** brief stall: 0.2–0.3 s, 6 on the saddle and 2 on the vortex, always 7–14 cm from the map centre |
| **🥈 celeste** | **5.2°** | 0.78–0.88 | slow fade on 11/15 (0.85 → 0.77) | 0.85 / 0.17 (laggiest, slight friction 0.009 m/s) | **0.027** (most predictable) | one dead stop in the failed 4R r_d 0.5 run, caused by the mat edge, not the robot |
| **🥉 tidal** | 5.7° | 0.83–0.88 | **died on 11/15**: 0.72 → 0.55 over 9 min, then 1.01 on new batteries | 0.88 / 0.06 | 0.032 | none notable |
| **redwood** | 6.8–7.8° (worst), magnitude ~20% low | 0.81–0.92 | **died on 11/13**: 0.81 → 0.34 over ~10 min, then 0.92 on new batteries | 0.88 / 0.09 | 0.028 | none notable; about −7° heading offset in Motive |

## Why pacific-blue wins
- **Best sensor (tied):** it has the lowest direction error. Sensor accuracy is what drives the critical-point estimate, and so the orbit.
- **Strongest and most consistent motion:** it reaches the highest share of its commanded speed, and it never had a battery drop in the campaign.
- **Its one flaw: short stalls.** All the brief 0.2–0.3 s stalls in the campaign are pacific-blue's, on **both** maps, and always near the centre. That points to its **wheels**, so check them for hair or debris, as well as the old mat. The stalls are also most of why its motion is the hardest to predict.

## Why celeste is a close second
- Its sensor is just as good (5.2°), and it is the **most predictable** robot. That makes it the best one to model.
- **But:** it reaches the lowest share of its commanded speed (about 0.85), has the most lag, and its battery faded slowly through the saddle session.

## Why redwood is last, and why that's interesting
- **The net was never trained on redwood, deliberately.** It is the IDETC unseen-robot test: can a sensor net be deployed on a robot it never saw? Its 7–8° error against about 5° for the trained robots **is the generalisation gap.** That's a result, not a fault.
- It also has a Motive heading offset of about −7° and had the 11/13 battery failure.

## For the paper / thesis
- **Trained robots:** about 5–5.7° median sensor direction error. **Untrained robot:** about 7–8°, with magnitude about 20% low. The net generalises, at the cost of about 2–3°.
- **All four robots move alike once batteries are healthy:** gain 0.85–0.89, almost no lag after a 0.1 s delay, and no kinetic friction. See `2026-11-15_robot_response_model.md`.
- **Battery state matters more than robot identity** for motion: two of the four robots had a battery die mid-session. Screen each run with the speed ratio (healthy 0.8–0.93, swap below about 0.7).

## Caveats
- The sensor ranking comes from the 11/13 **vortex** data; the saddle hasn't been ranked yet.
- The battery-dependent columns reflect battery state as much as the robot.
- The motion columns use healthy-battery runs only (speed ratio ≥ 0.75).

## Sources
- **Sensor ranking:** `2026-11-13_vortex_orbit_campaign.md` §8; script `scripts/robot_accuracy.m`.
- **Motion fits:** `2026-11-15_robot_response_model.md`; script `scripts/fit_coulomb_delay.m`.
- **Batteries:** `../robots_3/README_hardware.md` §5.7 and §5.9, and `../robots_4/README_four_robot.md` (redwood).
- **Stalls:** session scan on 2026-11-15 (scratchpad `snag_scan.m`). Snags are accepted as real-world variability, not excluded.
