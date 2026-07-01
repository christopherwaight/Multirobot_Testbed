"""Subprocess smoke tests for the clusterbuilder CLI launcher."""

import subprocess
import sys
from pathlib import Path

CLUSTER_BUILDER_DIR = Path(__file__).parent.parent
CLUSTERBUILDER_PY = CLUSTER_BUILDER_DIR / 'clusterbuilder.py'


def test_cli_image_only(tmp_path):
    cmd = [sys.executable, str(CLUSTERBUILDER_PY),
           '6', '2', 'smoke', 'image_only']
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'smoke_visualization.png').exists()


def test_cli_full(tmp_path):
    cmd = [sys.executable, str(CLUSTERBUILDER_PY),
           '6', '2', 'smoke', 'full', '(2,2,2)']
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for stem in ('smoke_visualization.png', 'smoke.yaml',
                 'smoke_forward_kinematics.py', 'smoke_inverse_kinematics.py',
                 'smoke_inverse_jacobian.py', 'smoke_forward_jacobian.py',
                 'smoke_cluster.py'):
        assert (tmp_path / stem).exists(), f"missing: {stem}"


def test_cli_hub_spoke_full(tmp_path):
    cmd = [sys.executable, str(CLUSTERBUILDER_PY),
           '4', '1', 'hs_smoke', 'full']
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'hs_smoke_visualization.png').exists()
    assert (tmp_path / 'hs_smoke.yaml').exists()
