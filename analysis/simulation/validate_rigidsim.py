"""Check rigidsim against the reference simulators.

3R: rigidsim with vecsim's triangle against vecsim.simulate (formation control included) on linear,
nonlinear and reconstructed fields, attraction and orbit, from identical starts and headings.
4R: rigidsim's square against QuadCluster (src/robot/quad_cluster.py, cluster-space kinematics and
shape control, Omnibot robots with the 2026-11-16 model) driven by the same 4-point least-squares
attraction law, from the same start pose.
Run from the repo root with the VF_Robot venv.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rigidsim as rs  # noqa: E402
import vecsim as vs  # noqa: E402
import paths  # noqa: E402

STEPS = 300


def check_3r():
    rng = np.random.default_rng(3)
    for name, task in (("vortex", "attract"), ("saddle", "orbit"), ("gyre", "attract"), ("vortex_nn_2026", "orbit"),
                       ("saddle_nn_2026", "attract"), ("sinking_vortex", "orbit")):
        F = vs.make_field(name)
        n = 5
        th = rng.uniform(0, 2 * np.pi, n)
        st = rng.uniform(-0.4, 0.4, (n, 2)) if task == "attract" else 0.2 * np.stack([np.cos(th), np.sin(th)], 1)
        P = vs.params(n, rd=0.2)
        out, _ = vs.simulate(F, task, "fit", P, STEPS * 0.1, starts=st, thetas=th, record=True)
        C, _ = rs.simulate(F, task, "fit", P, STEPS * 0.1, rs.triangle(th, P["side"]), st)
        d = np.nanmax(np.abs(out["rec"]["c"] - C))
        print(f"3R {name:16s} {task:7s}: max centroid difference rigidsim vs vecsim over {STEPS} steps x {n} = {d:.1e} m"
              f"  (vecsim max side error {out['max_side_err'].max():.1e})")


def check_4r():
    from src.robot.quad_cluster import QuadCluster
    from src.fields.field_types import AnalyticalField
    cfg = os.path.join(paths.VF, "config", "formations", "quad_square_default.yaml")
    for name in ("vortex", "saddle", "vortex_nn_2026"):
        F = vs.make_field(name)
        worst = 0.0
        for start, th in (((0.35, -0.25), 0.4), ((-0.3, 0.3), 2.0), ((0.1, 0.4), -1.1)):
            with contextlib.redirect_stdout(io.StringIO()):
                cl = QuadCluster(cfg, AnalyticalField(lambda x, y: tuple(float(a) for a in F.fn(x, y))))
            off = rs.square(np.array([th]))[0]
            for r, o in zip(cl.robots, off):
                r.set_position(*(np.array(start) + o))
                r.reset_dynamics()
            cl.center_history = []
            P1 = vs.params(1)

            def prim(cluster):
                pos = np.array([r.get_position() for r in cluster.robots])[None]
                c = pos.mean(1)
                U, V = F.fn(pos[..., 0], pos[..., 1])
                gux, guy, ub = vs._fit(pos, c, np.asarray(U, float))
                gvx, gvy, vb = vs._fit(pos, c, np.asarray(V, float))
                det = gux * gvy - guy * gvx
                ph = c - np.stack([(gvy * ub - guy * vb) / det, (-gvx * ub + gux * vb) / det], 1)
                w = ph - c
                dist = np.hypot(*w.T)
                spd = np.minimum(P1["k"] * dist, P1["vcmd_max"])
                v = np.clip(w / dist[:, None] * spd[:, None], -P1["vaxis"][0], P1["vaxis"][0])[0]
                return float(v[0]), float(v[1])
            for _ in range(STEPS):
                cl.move(prim)
            ref = np.vstack([np.array(start)[None], cl.get_center_history()])
            C, _ = rs.simulate(F, "attract", "fit", P1, STEPS * 0.1, rs.square(np.array([th])), np.array([start]))
            worst = max(worst, float(np.nanmax(np.abs(ref - C[0]))))
        print(f"4R {name:16s} attract: max centroid difference rigidsim vs QuadCluster over {STEPS} steps x 3 = {worst:.1e} m")


if __name__ == "__main__":
    check_3r()
    check_4r()
