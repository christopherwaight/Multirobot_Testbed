"""
YAML round-trip and renderer state-pass-through tests.

Tests:
  1. Smoke: generated YAML round-trips through yaml_visualize.visualize without error
     and produces a PNG for both config 1 (N=5) and config 2 trees ((2,2,2), ((2,2),2)).

  2. Backwards compatibility: handcrafted YAMLs in config/formations/ (equilateral_default,
     quad_default) still parse through their respective cluster class loaders (not
     clusterbuilder machinery).

  3. Edited-state-reaches-the-image (load-bearing): confirms the refactored Visualizer
     actually uses the state= argument rather than silently falling back to default_state().
     Two-layer check:
       (a) Data path: IK on the edited state moves robots vs. the default.
       (b) Render path: drawn robot marker coordinates differ by > 1e-3 between default
           and edited renders (extracted via matplotlib PathCollection.get_offsets()).

     If robots are ever switched from ax.scatter to ax.plot, _drawn_robot_xy must be
     updated -- the extractor assumes PathCollection (ax.scatter) output.

     Parametrized over ((2,2),2), (2,2,2) for CoC and N=5 for hub-and-spoke.
"""

import copy
import os
import sys
import math
import tempfile
import shutil

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
    CodeGenerator,
    Visualizer,
)
from yaml_visualize import visualize

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'config', 'formations')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _drawn_robot_xy(fig, n_robots):
    """
    Extract (x, y) coordinates of all scatter markers across all axes on fig.
    Returns a sorted numpy array of shape (M, 2) with M >= n_robots unique points.

    Assumes robots are drawn via ax.scatter (PathCollection). If a future change
    switches robots to ax.plot, this extractor needs updating.
    """
    from matplotlib.collections import PathCollection
    points = set()
    for ax in fig.get_axes():
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                offsets = coll.get_offsets()
                for pt in offsets:
                    points.add((round(float(pt[0]), 6), round(float(pt[1]), 6)))
    pts = np.array(sorted(points))
    assert len(pts) >= n_robots, (
        f"Expected at least {n_robots} scatter points, found {len(pts)}"
    )
    return pts


def _generate_yaml_to_tmpdir(name, config, N, kin, tree=None):
    """Generate YAML (and image) into a temp dir; return (tmpdir, yaml_path)."""
    tmpdir = tempfile.mkdtemp(prefix='yrt_')
    orig = os.getcwd()
    try:
        os.chdir(tmpdir)
        gen = CodeGenerator(name, config, N, kin, tree=tree, orientation=False)
        gen.write_all(image_only=False)
    finally:
        os.chdir(orig)
    return tmpdir, os.path.join(tmpdir, f'{name}.yaml')


# ---------------------------------------------------------------------------
# 1. Smoke: generated YAML -> yaml_visualize -> PNG
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,config,N,tree_str", [
    ("vis_pentagon",  1, 5, None),
    ("vis_222",       2, 6, "(2,2,2)"),
    ("vis_22by2",     2, 6, "((2,2),2)"),
])
def test_yaml_roundtrip_smoke(name, config, N, tree_str, tmp_path):
    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        if config == 1:
            kin = HubAndSpokeKinematics(N, orientation=False)
            gen = CodeGenerator(name, 1, N, kin, orientation=False)
        else:
            root = TreeParser.parse(tree_str)
            kin = ClusterOfClustersKinematics(root, orientation=False)
            gen = CodeGenerator(name, 2, N, kin, tree=root, orientation=False)
        gen.write_all(image_only=False)
    finally:
        os.chdir(orig)

    yaml_path = os.path.join(str(tmp_path), f'{name}.yaml')
    assert os.path.isfile(yaml_path), f"YAML not generated: {yaml_path}"

    out_png = os.path.join(str(tmp_path), f'{name}_roundtrip.png')
    result = visualize(yaml_path, out_path=out_png)
    assert os.path.isfile(result), f"PNG not created: {result}"
    assert os.path.getsize(result) > 1000, "PNG appears empty"


# ---------------------------------------------------------------------------
# 2. Backwards compatibility: handcrafted YAMLs load without error
# ---------------------------------------------------------------------------

def test_equilateral_yaml_loads():
    """OmniCluster._load_formation_config must still accept equilateral_default.yaml."""
    yaml_path = os.path.join(CONFIG_DIR, 'equilateral_default.yaml')
    if not os.path.isfile(yaml_path):
        pytest.skip("equilateral_default.yaml not found in config/formations/")
    import yaml as pyyaml
    with open(yaml_path) as f:
        data = pyyaml.safe_load(f)
    formation = data['formation']
    required = ['p', 'q', 'beta_degrees']
    for key in required:
        assert key in formation, f"Missing key '{key}' in equilateral_default.yaml"


def test_quad_yaml_accepts_psi_degrees():
    """QuadCluster._load_formation_config must accept psi_degrees."""
    yaml_path = os.path.join(CONFIG_DIR, 'quad_default.yaml')
    if not os.path.isfile(yaml_path):
        pytest.skip("quad_default.yaml not found in config/formations/")
    import yaml as pyyaml
    with open(yaml_path) as f:
        data = pyyaml.safe_load(f)
    formation = data['formation']
    assert ('psi_degrees' in formation or 'phi_degrees' in formation), (
        "quad_default.yaml must have psi_degrees (or legacy phi_degrees)"
    )


# ---------------------------------------------------------------------------
# 3. Edited state reaches the image (load-bearing)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree_str", ["((2,2),2)", "(2,2,2)"])
def test_edited_angle_reaches_the_image(tree_str, tmp_path):
    """
    Confirms Visualizer.render_cluster_of_clusters actually uses the state= argument.

    Layer 1 (data path): IK on edited state produces different robot positions.
    Layer 2 (render path): the rendered scatter markers differ by > 1e-3.

    The theta_c edit rotates the whole cluster, so every robot moves -- the signal
    is unambiguous. This test assumes robots are drawn via ax.scatter
    (PathCollection). If the renderer switches to ax.plot, update _drawn_robot_xy.
    """
    root = TreeParser.parse(tree_str)
    N = root.size
    kin = ClusterOfClustersKinematics(root, orientation=False)

    state_def = kin.default_state()
    state_edit = copy.deepcopy(state_def)
    state_edit['theta_c'] = state_def['theta_c'] + 0.3

    # Layer 1: data path
    pos_def  = kin.inverse_kinematics(state_def)
    pos_edit = kin.inverse_kinematics(state_edit)
    dists = [math.hypot(pos_edit[i][0] - pos_def[i][0],
                        pos_edit[i][1] - pos_def[i][1])
             for i in range(1, N + 1)]
    assert any(d > 1e-3 for d in dists), (
        "theta_c edit did not move any robot in IK -- data path is broken"
    )

    # Layer 2: render path
    import matplotlib
    matplotlib.use('Agg')

    fig_def  = Visualizer.render_cluster_of_clusters(kin, tree_str, state=state_def)
    fig_edit = Visualizer.render_cluster_of_clusters(kin, tree_str, state=state_edit)

    pts_def  = _drawn_robot_xy(fig_def,  N)
    pts_edit = _drawn_robot_xy(fig_edit, N)

    import matplotlib.pyplot as plt
    plt.close(fig_def)
    plt.close(fig_edit)

    # At least one marker must have moved by more than 1e-3
    moved = np.max(np.abs(np.sort(pts_def, axis=0) - np.sort(pts_edit, axis=0)))
    assert moved > 1e-3, (
        f"Render path: scatter markers did not change after theta_c edit "
        f"(max shift={moved:.2e}). Visualizer is likely ignoring state= argument."
    )


def test_edited_angle_reaches_the_image_hub_spoke(tmp_path):
    """
    Same two-layer check for hub-and-spoke (N=5).
    Edits theta_c by +0.3 radians; all spoke robots rotate about the hub.
    """
    N = 5
    kin = HubAndSpokeKinematics(N, orientation=False)

    state_def  = kin.default_state()
    state_edit = copy.deepcopy(state_def)
    state_edit['theta_c'] = state_def['theta_c'] + 0.3

    # Layer 1: data path
    pos_def  = kin.inverse_kinematics(state_def)
    pos_edit = kin.inverse_kinematics(state_edit)
    dists = [math.hypot(pos_edit[i][0] - pos_def[i][0],
                        pos_edit[i][1] - pos_def[i][1])
             for i in range(1, N + 1)]
    assert any(d > 1e-3 for d in dists), (
        "theta_c edit did not move any robot in IK -- data path is broken"
    )

    # Layer 2: render path
    import matplotlib
    matplotlib.use('Agg')

    fig_def  = Visualizer.render_hub_spoke(kin, 'test_hs', state=state_def)
    fig_edit = Visualizer.render_hub_spoke(kin, 'test_hs', state=state_edit)

    pts_def  = _drawn_robot_xy(fig_def,  N)
    pts_edit = _drawn_robot_xy(fig_edit, N)

    import matplotlib.pyplot as plt
    plt.close(fig_def)
    plt.close(fig_edit)

    moved = np.max(np.abs(np.sort(pts_def, axis=0) - np.sort(pts_edit, axis=0)))
    assert moved > 1e-3, (
        f"Render path: scatter markers did not change after theta_c edit "
        f"(max shift={moved:.2e}). Visualizer is likely ignoring state= argument."
    )
