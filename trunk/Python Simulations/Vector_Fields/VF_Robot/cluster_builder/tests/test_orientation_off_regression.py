"""
Regression tests: with orientation=off, generated files must be byte-for-byte
identical to the committed snapshots for hexstar_snap (config 1, hub-and-spoke)
and my_star_snap (config 2, cluster-of-clusters).

If a test fails here, it means a code change altered the generated output for the
orientation=off path, which is forbidden by the hard constraint:
  "with orientation=off, config 1 and config 2 must be byte-for-byte identical to today."
"""

import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
    CodeGenerator,
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), 'snapshots')

# Files to compare for each case; maps suffix -> snapshot filename stem
HEXSTAR_FILES = [
    ('_inverse_jacobian.py',   'hexstar_snap_inverse_jacobian.py'),
    ('_forward_kinematics.py', 'hexstar_snap_forward_kinematics.py'),
    ('_inverse_kinematics.py', 'hexstar_snap_inverse_kinematics.py'),
    ('_cluster.py',            'hexstar_snap_cluster.py'),
]

MY_STAR_FILES = [
    ('_inverse_jacobian.py',   'my_star_snap_inverse_jacobian.py'),
    ('_forward_kinematics.py', 'my_star_snap_forward_kinematics.py'),
    ('_inverse_kinematics.py', 'my_star_snap_inverse_kinematics.py'),
    ('_cluster.py',            'my_star_snap_cluster.py'),
]


def _generate_to_tmpdir(name, config, N, kin, tree=None):
    """Run CodeGenerator.write_all (full mode) into a temp directory."""
    tmpdir = tempfile.mkdtemp(prefix='cbtest_')
    orig_dir = os.getcwd()
    try:
        os.chdir(tmpdir)
        gen = CodeGenerator(name, config, N, kin, tree=tree, orientation=False)
        gen.write_all(image_only=False)
    finally:
        os.chdir(orig_dir)
    return tmpdir


def _read_file(path):
    with open(path, 'r') as f:
        return f.read()


def _compare_generated_to_snapshot(tmpdir, name, file_pairs):
    """Assert each generated file matches the snapshot byte-for-byte."""
    mismatches = []
    for suffix, snap_name in file_pairs:
        generated_path = os.path.join(tmpdir, name + suffix)
        snapshot_path  = os.path.join(SNAPSHOT_DIR, snap_name)

        assert os.path.isfile(generated_path), f"Generated file not found: {generated_path}"
        assert os.path.isfile(snapshot_path),  f"Snapshot not found: {snapshot_path}"

        gen_text  = _read_file(generated_path)
        snap_text = _read_file(snapshot_path)

        if gen_text != snap_text:
            # Build a helpful diff summary
            gen_lines  = gen_text.splitlines()
            snap_lines = snap_text.splitlines()
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(snap_lines, gen_lines)) if a != b),
                min(len(snap_lines), len(gen_lines))
            )
            mismatches.append(
                f"\n{suffix}: first diff at line {first_diff + 1}\n"
                f"  snapshot: {snap_lines[first_diff] if first_diff < len(snap_lines) else '<missing>'!r}\n"
                f"  generated: {gen_lines[first_diff] if first_diff < len(gen_lines) else '<missing>'!r}"
            )

    if mismatches:
        pytest.fail("Generated files differ from snapshots:" + "".join(mismatches))


class TestHexstarRegression:
    """Config 1 (hub-and-spoke, N=6) regression against hexstar_snap snapshots."""

    def setup_method(self):
        self.kin = HubAndSpokeKinematics(6, orientation=False)
        self.tmpdir = _generate_to_tmpdir('hexstar_snap', 1, 6, self.kin)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_inverse_jacobian(self):
        _compare_generated_to_snapshot(self.tmpdir, 'hexstar_snap',
                                        [HEXSTAR_FILES[0]])

    def test_forward_kinematics(self):
        _compare_generated_to_snapshot(self.tmpdir, 'hexstar_snap',
                                        [HEXSTAR_FILES[1]])

    def test_inverse_kinematics(self):
        _compare_generated_to_snapshot(self.tmpdir, 'hexstar_snap',
                                        [HEXSTAR_FILES[2]])

    def test_cluster_file(self):
        _compare_generated_to_snapshot(self.tmpdir, 'hexstar_snap',
                                        [HEXSTAR_FILES[3]])


class TestMyStarRegression:
    """Config 2 (cluster-of-clusters, my_star tree) regression against my_star_snap snapshots."""

    def setup_method(self):
        # my_star uses a (2,2,2) triple-of-pairs tree (3 children, each a pair)
        root = TreeParser.parse('(2,2,2)')
        self.kin = ClusterOfClustersKinematics(root, orientation=False)
        self.tmpdir = _generate_to_tmpdir('my_star_snap', 2, 6, self.kin, tree=root)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_inverse_jacobian(self):
        _compare_generated_to_snapshot(self.tmpdir, 'my_star_snap',
                                        [MY_STAR_FILES[0]])

    def test_forward_kinematics(self):
        _compare_generated_to_snapshot(self.tmpdir, 'my_star_snap',
                                        [MY_STAR_FILES[1]])

    def test_inverse_kinematics(self):
        _compare_generated_to_snapshot(self.tmpdir, 'my_star_snap',
                                        [MY_STAR_FILES[2]])

    def test_cluster_file(self):
        _compare_generated_to_snapshot(self.tmpdir, 'my_star_snap',
                                        [MY_STAR_FILES[3]])
