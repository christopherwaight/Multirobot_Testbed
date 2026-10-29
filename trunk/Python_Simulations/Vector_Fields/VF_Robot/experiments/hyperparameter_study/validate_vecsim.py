"""
Check vecsim against the reference classes (OmniCluster + Omnibot +
critical_point_plane_fitting / critical_point_orbiter_plane_fitting) on
identical starts and headings, and the batched MLP fields against NNField.
Run from VF_Robot:  venv/bin/python3 experiments/hyperparameter_study/validate_vecsim.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import vecsim as vs
import _field_tests_common as ft
from src.fields.field_types import NNField
from src.fields.environments.Sink import sink1
from src.fields.environments.Source import source1
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Saddle import saddle1
from src.fields.environments.Sinking_Vortex import sinking_vortex1
from src.fields.environments.Spewing_Vortex import spewing_vortex1

STEPS = 400


def ref_run(fieldfn, task, start, theta, side, alpha, vsn, rd, model="legacy", vk=None):
    spec = ft.FieldSpec("ref", 0.0, lambda x, y: tuple(float(a) for a in fieldfn(x, y)),
                        [("t", 0.0, 0.0)], lambda x, y: True, 0.5)
    cl = ft.make_cluster(spec, start, side, alpha, vsn, theta)
    for robot in cl.robots:
        robot.stiction_model = model
        robot.kinetic_threshold = vsn if vk is None else vk
    prim = ft.attract_primitive("vector") if task == "attract" else ft.orbit_primitive("vector", rd)
    for _ in range(STEPS):
        cl.move(prim)
    return cl.get_center_history()


def vec_run(field, task, start, theta, side, alpha, vsn, rd, model="legacy", vk=None):
    P = vs.params(1, side=side, alpha=alpha, vs=vsn, rd=rd)
    _, tr = vs.simulate(field, task, "fit", P, STEPS * 0.1, starts=np.array([start]),
                        thetas=np.array([theta]), keep=1, keep_stride=1,
                        stiction_model=model, vk=vk)
    out = tr[0, :STEPS]          # traj[k] is the centroid after step k, as center_history[k]
    assert np.isfinite(out).all()
    return out


if __name__ == "__main__":
    # linear fields equal the src definitions
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (50, 2))
    for name, f in [("sink", sink1), ("source", source1), ("vortex", vortex1), ("saddle", saddle1),
                    ("sinking_vortex", sinking_vortex1), ("spewing_vortex", spewing_vortex1)]:
        F = vs.make_field(name)
        err = max(np.hypot(*(np.array(f(x, y)) - np.array(F.fn(x, y)))) for x, y in pts)
        print(f"field {name:15s} max |v_src - v_vec| = {err:.1e}")
    for name, d in [("vortex_nn", "vortex_predictors"), ("saddle_nn", "saddle_predictors")]:
        F = vs.make_field(name)
        ref = NNField(d)
        u, v = F.fn(pts[:, 0] * 0.6, pts[:, 1] * 0.6)
        err = max(np.hypot(*(np.array(ref.get_value(x * 0.6, y * 0.6)) - np.array([u[i], v[i]])))
                  for i, (x, y) in enumerate(pts))
        print(f"field {name:15s} max |NNField - batched| = {err:.1e}")

    cases = [("vortex", "attract", 0.33, 0.7, 0.025, 0.3), ("saddle", "attract", 0.10, 0.0, 0.0, 0.3),
             ("gyre", "attract", 0.33, 0.7, 0.025, 0.3), ("sinking_vortex", "orbit", 0.33, 0.7, 0.025, 0.2),
             ("gyre", "orbit", 0.33, 0.5, 0.0, 0.3), ("saddle", "orbit", 0.2, 0.9, 0.0, 0.1),
             ("curved_saddle", "orbit", 0.1, 0.01, 0.0, 0.3), ("vortex_nn", "attract", 0.33, 0.7, 0.025, 0.3),
             ("saddle_nn", "orbit", 0.33, 0.7, 0.025, 0.3)]
    for name, task, side, alpha, vsn, rd in cases:
        F = vs.make_field(name)
        worst = 0.0
        for trial in range(3):
            if task == "attract":
                start = rng.uniform(-0.4, 0.4, 2)
            else:
                ph = rng.uniform(0, 2 * np.pi)
                start = rd * np.array([np.cos(ph), np.sin(ph)])
            theta = rng.uniform(0, 2 * np.pi)
            if name.endswith("_nn"):
                ref = NNField("vortex_predictors" if name == "vortex_nn" else "saddle_predictors")
                fn = ref.get_value
            else:
                fn = F.fn
            a = ref_run(fn, task, start, theta, side, alpha, vsn, rd)
            b = vec_run(F, task, start, theta, side, alpha, vsn, rd)
            assert a.shape == b.shape, (a.shape, b.shape)
            worst = max(worst, float(np.abs(a - b).max()))
        print(f"{name:15s} {task:7s} side {side:4.2f} alpha {alpha:4.2f} vs {vsn:5.3f}: "
              f"max centroid difference over {STEPS} steps x 3 trials = {worst:.1e} m")

    # each stiction rule; coulomb at the hardware fit (v_k 0.09) and with a static excess
    rules = [("legacy", 0.025, None), ("deadzone", 0.025, None), ("scaled", 0.025, None),
             ("coulomb", 0.09, None), ("coulomb", 0.12, 0.09), ("coulomb", 0.05, 0.0)]
    for model, vsn, vk in rules:
        worst = 0.0
        for name, task, rd in [("vortex", "attract", 0.3), ("saddle", "attract", 0.3),
                               ("vortex", "orbit", 0.3)]:
            F = vs.make_field(name)
            for trial in range(3):
                if task == "attract":
                    start = rng.uniform(-0.4, 0.4, 2)
                else:
                    ph = rng.uniform(0, 2 * np.pi)
                    start = rd * np.array([np.cos(ph), np.sin(ph)])
                theta = rng.uniform(0, 2 * np.pi)
                a = ref_run(F.fn, task, start, theta, 0.33, 0.7, vsn, rd, model, vk)
                b = vec_run(F, task, start, theta, 0.33, 0.7, vsn, rd, model, vk)
                worst = max(worst, float(np.abs(a - b).max()))
        print(f"stiction {model:8s} v_s {vsn:5.3f} v_k {str(vk):5s}: max centroid difference, "
              f"vortex/saddle attract and vortex orbit x 3 trials = {worst:.1e} m")
