"""Closed-loop check of candidate robot models against hardware.

Attraction: the 157 hardware attraction starts (2025, 10 s, k = 1), mean final distance (hardware 23.4 mm).
Orbit: 2026-11 hardware Orbiter gains (k_t 0.15, k_r 1, radial clip 0.3), side 0.30, linear vortex,
radius about the true centre averaged over t > 10 s (hardware 3R: 0.071 m at r_d 0.01, 0.130 m at 0.1,
speed 0.13 to 0.15 m/s).
"""
import os
import sys

import numpy as np

VF = sys.argv[1]
sys.path.insert(0, os.path.join(VF, "experiments", "hyperparameter_study"))
import vecsim as vs  # noqa: E402
from stiction_hw_check import hardware, stats  # noqa: E402

MODELS = {
    "2026-10-29 (coulomb, tau .28, cap .4)":
        dict(model="coulomb", vk=None, P=dict(alpha=np.exp(-1 / 2.8), gain=1.0, delay=0.0, vmax=0.4)),
    "2026-11-15 note (a .11, g .85, d 0.1 s, vk 0)":
        dict(model="coulomb", vk=0.0, P=dict(alpha=0.11, gain=0.85, delay=0.1, vmax=0.3)),
    "new, coulomb vk 0 (a 0, g .86, d .075 s)":
        dict(model="coulomb", vk=0.0, P=dict(alpha=0.0, gain=0.86, delay=0.075, vmax=0.3)),
    "NEW DEFAULT, dead zone (a 0, g .86, d .075 s)":
        dict(model="deadzone", vk=None, P=dict(alpha=0.0, gain=0.86, delay=0.075, vmax=0.3)),
}
REPS = 20

print("Attraction, hardware starts x 20 headings, 10 s. Hardware: mean 23.4 mm (vortex 23.5, saddle 23.3)")
for name, m in MODELS.items():
    row = []
    for field in ("vortex", "saddle"):
        start, _, _ = hardware(field)
        starts = np.repeat(start, REPS, axis=0)
        n = len(starts)
        th = np.random.default_rng(0).uniform(0, 2 * np.pi, n)
        P = vs.params(n, **m["P"])
        M, _ = vs.simulate(vs.make_field(field), "attract", "fit", P, 10.0, seed=0, starts=starts,
                           thetas=th, stiction_model=m["model"], vk=m["vk"])
        fin = np.stack([M["final_x"], M["final_y"]], 1) - vs.make_field(field).target
        mean, bias, prec = stats(fin)
        row.append(f"{field} {mean*1000:5.1f} mm")
    print(f"  {name:<46} " + "  ".join(row))

print("\nOrbit, linear vortex, hardware Orbiter gains, 30 s, radius over t > 10 s (hardware 3R: 0.071, 0.130)")
for name, m in MODELS.items():
    row = []
    for rd in (0.01, 0.1, 0.2):
        n = 20
        P = vs.params(n, side=0.30, kt=0.15, kr=1.0, rclamp=0.3, rd=rd, **m["P"])
        _, _ = None, None
        out, _ = vs.simulate(vs.make_field("vortex"), "orbit", "fit", P, 30.0, seed=1,
                             stiction_model=m["model"], vk=m["vk"], record=True)
        c = out["rec"]["c"][:, 101:]
        r = np.hypot(c[..., 0], c[..., 1]).mean()
        sp = (np.hypot(*np.diff(c, axis=1).transpose(2, 0, 1)) / 0.1).mean()
        row.append(f"r_d {rd}: {r:.3f} m ({sp:.3f} m/s)")
    print(f"  {name:<46} " + "  ".join(row))
