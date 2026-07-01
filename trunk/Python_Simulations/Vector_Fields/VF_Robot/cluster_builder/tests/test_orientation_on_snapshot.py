"""
Regression tests for orientation=on mode: generated files must be byte-for-byte
identical to committed snapshots on every subsequent run.

On first run (when snapshots do not exist), the test auto-generates and saves
them. Subsequent runs diff against those snapshots. If a test fails, a code
change altered the orientation=on emitter, which requires deliberate review.

Cases:
  - hexstar_on_snap: hub-and-spoke N=6, orientation=True
  - my_star_on_snap: cluster-of-clusters (2,2,2), orientation=True

These mirror the orientation=off regression cases in test_orientation_off_regression.py.
The cluster file is excluded for orientation=on because CodeGenerator emits a
RuntimeError guard inside it (by design) and it would snapshot a stub.
"""

import os
import sys
import shutil
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
    CodeGenerator,
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), 'snapshots')

HEXSTAR_ON_FILES = [
    ('_inverse_jacobian.py',   'hexstar_on_snap_inverse_jacobian.py'),
    ('_forward_kinematics.py', 'hexstar_on_snap_forward_kinematics.py'),
    ('_inverse_kinematics.py', 'hexstar_on_snap_inverse_kinematics.py'),
]

MY_STAR_ON_FILES = [
    ('_inverse_jacobian.py',   'my_star_on_snap_inverse_jacobian.py'),
    ('_forward_kinematics.py', 'my_star_on_snap_forward_kinematics.py'),
    ('_inverse_kinematics.py', 'my_star_on_snap_inverse_kinematics.py'),
]


def _generate_to_tmpdir(name, config, N, kin, tree=None, orientation=True):
    tmpdir = tempfile.mkdtemp(prefix='cbtest_on_')
    orig_dir = os.getcwd()
    try:
        os.chdir(tmpdir)
        gen = CodeGenerator(name, config, N, kin, tree=tree, orientation=orientation)
        gen.write_all(image_only=False)
    finally:
        os.chdir(orig_dir)
    return tmpdir


def _read_file(path):
    with open(path, 'r') as f:
        return f.read()


def _compare_or_create_snapshots(tmpdir, name, file_pairs):
    """
    Compare generated files against snapshots. If a snapshot does not exist,
    create it (first-run bootstrap). Fail if snapshots exist but differ.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    mismatches = []
    for suffix, snap_name in file_pairs:
        generated_path = os.path.join(tmpdir, name + suffix)
        snapshot_path  = os.path.join(SNAPSHOT_DIR, snap_name)

        assert os.path.isfile(generated_path), f"Generated file not found: {generated_path}"
        gen_text = _read_file(generated_path)

        if not os.path.isfile(snapshot_path):
            # First run: save snapshot
            with open(snapshot_path, 'w') as f:
                f.write(gen_text)
            print(f"  [snapshot created] {snap_name}")
            continue

        snap_text = _read_file(snapshot_path)
        if gen_text != snap_text:
            gen_lines  = gen_text.splitlines()
            snap_lines = snap_text.splitlines()
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(snap_lines, gen_lines)) if a != b),
                min(len(snap_lines), len(gen_lines))
            )
            mismatches.append(
                f"\n{suffix}: first diff at line {first_diff + 1}\n"
                f"  snapshot:  {snap_lines[first_diff] if first_diff < len(snap_lines) else '<missing>'!r}\n"
                f"  generated: {gen_lines[first_diff] if first_diff < len(gen_lines) else '<missing>'!r}"
            )

    if mismatches:
        pytest.fail("Generated files differ from snapshots:" + "".join(mismatches))


class TestHexstarOnRegression:
    """Hub-and-spoke N=6, orientation=True."""

    def setup_method(self):
        self.kin = HubAndSpokeKinematics(6, orientation=True)
        self.tmpdir = _generate_to_tmpdir('hexstar_on_snap', 1, 6, self.kin, orientation=True)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_inverse_jacobian(self):
        _compare_or_create_snapshots(self.tmpdir, 'hexstar_on_snap', [HEXSTAR_ON_FILES[0]])

    def test_forward_kinematics(self):
        _compare_or_create_snapshots(self.tmpdir, 'hexstar_on_snap', [HEXSTAR_ON_FILES[1]])

    def test_inverse_kinematics(self):
        _compare_or_create_snapshots(self.tmpdir, 'hexstar_on_snap', [HEXSTAR_ON_FILES[2]])


class TestMyStarOnRegression:
    """Cluster-of-clusters (2,2,2), orientation=True."""

    def setup_method(self):
        root = TreeParser.parse('(2,2,2)')
        self.kin = ClusterOfClustersKinematics(root, orientation=True)
        self.tmpdir = _generate_to_tmpdir('my_star_on_snap', 2, 6, self.kin,
                                          tree=root, orientation=True)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_inverse_jacobian(self):
        _compare_or_create_snapshots(self.tmpdir, 'my_star_on_snap', [MY_STAR_ON_FILES[0]])

    def test_forward_kinematics(self):
        _compare_or_create_snapshots(self.tmpdir, 'my_star_on_snap', [MY_STAR_ON_FILES[1]])

    def test_inverse_kinematics(self):
        _compare_or_create_snapshots(self.tmpdir, 'my_star_on_snap', [MY_STAR_ON_FILES[2]])
