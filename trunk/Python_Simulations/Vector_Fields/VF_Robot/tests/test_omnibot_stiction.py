"""
Omnibot stiction models, mainly "coulomb" (static plus kinetic friction).
Run from VF_Robot:  venv/bin/python3 -m pytest tests/test_omnibot_stiction.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.robot.omnibot import Omnibot

TAU = 0.28
VS, VK = 0.12, 0.09


def bot(dt=0.1, vs=VS, vk=VK, alpha=None, vmax=10.0):
    a = np.exp(-dt / TAU) if alpha is None else alpha
    return Omnibot(0.0, 0.0, timestep=dt, momentum_alpha=a, max_velocity=vmax,
                   stiction_threshold=vs, stiction_model="coulomb", kinetic_threshold=vk)


def run(b, u, seconds):
    for _ in range(int(round(seconds / b.timestep))):
        b.command_velocity(*u)
    return b


def test_parked_robot_holds_up_to_static_threshold():
    b = run(bot(), (VS, 0.0), 5.0)
    assert b.stuck and not b.velocity.any() and not b.position.any()


def test_breakaway_just_above_static_threshold():
    b = run(bot(), (VS + 1e-3, 0.0), 5.0)
    assert not b.stuck
    assert b.velocity[0] == pytest.approx(VS + 1e-3 - VK, rel=1e-6)


def test_moving_robot_settles_at_command_minus_kinetic():
    b = run(bot(), (0.3, 0.0), 5.0)
    assert np.linalg.norm(b.velocity) == pytest.approx(0.3 - VK, rel=1e-6)


def test_hysteresis_band_keeps_a_moving_robot_moving():
    # v_k < |u| <= v_s: moving robot keeps going, parked robot stays put
    u = (0.5 * (VK + VS), 0.0)
    moving = run(run(bot(), (0.3, 0.0), 3.0), u, 5.0)
    assert not moving.stuck
    assert moving.velocity[0] == pytest.approx(u[0] - VK, rel=1e-6)
    parked = run(bot(), u, 5.0)
    assert parked.stuck and not parked.position.any()


def test_stops_below_kinetic_and_does_not_restart_inside_band():
    b = run(run(bot(), (0.3, 0.0), 3.0), (0.5 * VK, 0.0), 3.0)
    assert b.stuck and not b.velocity.any()
    x = b.position.copy()
    run(b, (0.5 * (VK + VS), 0.0), 3.0)
    assert b.stuck and np.array_equal(b.position, x)


@pytest.mark.parametrize("dt", [0.2, 0.1, 0.05, 0.01, 0.001])
def test_coast_velocity_is_exact_at_every_sample_period(dt):
    # tau dv/dt = -v - v_k from v0: v(t) = (v0 + v_k) e^(-t/tau) - v_k until v = 0
    v0 = 0.3
    b = bot(dt=dt)
    b.velocity = np.array([v0, 0.0])
    t_stop = TAU * np.log(1 + v0 / VK)
    for k in range(1, int(np.ceil(t_stop / dt)) + 3):
        b.command_velocity(0.0, 0.0)
        exact = max((v0 + VK) * np.exp(-k * dt / TAU) - VK, 0.0)
        assert b.velocity[0] == pytest.approx(exact, abs=1e-12)
    assert b.stuck


def test_coast_distance_converges_to_continuous_value():
    v0 = 0.3
    d_exact = TAU * (v0 - VK * np.log(1 + v0 / VK))
    errs = []
    for dt in (0.1, 0.01, 0.001):
        b = bot(dt=dt)
        b.velocity = np.array([v0, 0.0])
        run(b, (0.0, 0.0), 2.0)
        errs.append(abs(b.position[0] - d_exact))
    assert errs[2] < v0 * 0.001              # explicit position update, error ~ v0 dt / 2
    assert errs[0] > errs[1] > errs[2]


@pytest.mark.parametrize("dt", [0.1, 0.01])
def test_breakaway_level_does_not_depend_on_sample_period(dt):
    assert run(bot(dt=dt), (VS, 0.0), 1.0).stuck
    assert not run(bot(dt=dt), (VS * 1.01, 0.0), 1.0).stuck


def test_reversal_with_large_command_does_not_stick():
    b = run(bot(alpha=0.5), (0.3, 0.0), 3.0)
    b.command_velocity(-b.velocity[0], 0.0)   # free update lands exactly on zero, |u| > v_s
    assert not b.stuck
    b.command_velocity(-0.3, 0.0)
    assert b.velocity[0] < 0


def test_velocity_set_by_caller_unsticks():
    b = bot()
    b.velocity = np.array([0.2, 0.0])
    b.command_velocity(0.2, 0.0)
    assert b.velocity[0] > 0


def test_speed_cap_applies_after_friction():
    b = run(bot(vmax=0.3), (1.0, 0.0), 5.0)
    assert np.linalg.norm(b.velocity) == pytest.approx(0.3)


def test_kinetic_defaults_to_static_and_is_validated():
    assert Omnibot(0, 0, stiction_threshold=0.05, stiction_model="coulomb").kinetic_threshold == 0.05
    with pytest.raises(ValueError):
        Omnibot(0, 0, stiction_threshold=0.05, stiction_model="coulomb", kinetic_threshold=0.06)


def test_default_model_is_still_legacy():
    b = Omnibot(0.0, 0.0)
    assert b.stiction_model == "legacy"
    b.command_velocity(0.08, 0.0)          # (1 - 0.7) * 0.08 = 0.024 < 0.025
    assert not b.velocity.any()
    b.command_velocity(0.09, 0.0)          # 0.027 >= 0.025
    assert b.velocity[0] == pytest.approx(0.3 * 0.09)


def test_three_robot_cluster_defaults_to_coulomb():
    import contextlib
    import io
    from src.robot.omni_cluster import OmniCluster
    from src.fields.field_types import AnalyticalField
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "formations", "equilateral_default.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        cl = OmniCluster(cfg, AnalyticalField(lambda x, y: (0.0, 0.0)))
    for r in cl.robots:
        assert r.stiction_model == "coulomb"
        assert r.stiction_threshold == r.kinetic_threshold == 0.025
        assert r.momentum_alpha == pytest.approx(np.exp(-0.1 / 0.28))
        assert r.max_velocity == 0.4


def test_cluster_alpha_follows_sample_period():
    import contextlib
    import io
    from src.robot.omni_cluster import OmniCluster
    from src.fields.field_types import AnalyticalField
    cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "formations", "equilateral_default.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        fast = OmniCluster(cfg, AnalyticalField(lambda x, y: (0.0, 0.0)), timestep=0.01)
        fixed = OmniCluster(cfg, AnalyticalField(lambda x, y: (0.0, 0.0)), momentum_alpha=0.7)
    assert all(r.momentum_alpha == pytest.approx(np.exp(-0.01 / 0.28)) for r in fast.robots)
    assert all(r.timestep == 0.01 for r in fast.robots)
    assert all(r.momentum_alpha == 0.7 for r in fixed.robots)


def test_bare_omnibot_keeps_original_cap():
    # QuadCluster and PentagonCluster build Omnibots without a cap
    assert Omnibot(0.0, 0.0).max_velocity == 0.3
