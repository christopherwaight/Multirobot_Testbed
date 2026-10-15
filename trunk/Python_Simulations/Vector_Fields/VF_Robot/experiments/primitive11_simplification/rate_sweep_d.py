"""Sec. IV-D: the rotation rate at which the D tracker first loses the straddle
before far-saddle contact, from the Fig. 8 start, using traverse_objectivity_demo's
own run and straddle_loss. Also reports where each run captures."""
import os, sys, io, contextlib
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "experiments")); os.chdir(root)
import numpy as np
import traverse_objectivity_demo as T
from src.fields.environments.Double_Gyre import double_gyre_static
from src.fields.environments.Rotating_Frame import make_rotating_frame, pull_back_trajectory
for om in (0.15, 0.17, 0.19, 0.20, 0.205, 0.21, 0.215, 0.22, 0.23, 0.25):
    T.OMEGA_ROT = om
    with contextlib.redirect_stdout(io.StringIO()):
        traj, rob = T.run(make_rotating_frame(double_gyre_static, om), T.logic_c_primitive, rotating=True, return_robots=True)
    times = (np.arange(len(traj)) + 1) * T.DT
    C = pull_back_trajectory(traj, times, om)
    R = np.stack([pull_back_trajectory(rob[:, i, :], times, om) for i in range(rob.shape[1])], axis=1)
    loss = T.straddle_loss(C, R)
    print(f"Omega {om:.3f}: straddle loss {loss}, end ({C[-1,0]:+.3f},{C[-1,1]:+.3f}), max|x| to far saddle {T.ride_deviation(C):.4f}", flush=True)
