import os, sys
import numpy as np
sys.path.insert(0, os.path.join(sys.argv[1], "experiments", "hyperparameter_study"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vecsim_lead as vs
for lead in (0.0, 0.015, 0.0175, 0.02):
    vs.LEAD = lead
    row = []
    for rd in (0.01, 0.1):
        P = vs.params(20, side=0.30, kt=0.15, kr=1.0, rclamp=0.3, rd=rd)
        out, _ = vs.simulate(vs.make_field("vortex"), "orbit", "fit", P, 30.0, seed=1, record=True)
        c = out["rec"]["c"][:, 101:]
        row.append(f"r_d {rd}: {np.hypot(c[..., 0], c[..., 1]).mean():.3f} m")
    print(f"  new model + estimate lead {lead*100:.2f} cm: " + "  ".join(row))
