"""Unit tests for CodeGenerator.generate_* methods."""

import yaml
import pytest
from clusterbuilder import (
    ClusterOfClustersKinematics, HubAndSpokeKinematics,
    TreeParser, CodeGenerator,
)


def _make_coc_gen(tree_str='(2,2,2)', n=6):
    root = TreeParser.parse(tree_str)
    kin = ClusterOfClustersKinematics(root)
    return CodeGenerator('test_unit', 2, n, kin, tree=root)


def _make_hs_gen(n=4):
    kin = HubAndSpokeKinematics(n)
    return CodeGenerator('test_hs', 1, n, kin)


class TestCoCGenerators:

    def setup_method(self):
        self.gen = _make_coc_gen()

    def test_generate_yaml_parses(self):
        content = self.gen.generate_yaml()
        data = yaml.safe_load(content)
        assert 'formation' in data
        f = data['formation']
        assert f['num_robots'] == 6
        assert f['cluster_config'] == 2
        assert 'config_tree' in f

    def test_generate_forward_kinematics_syntax(self):
        code = self.gen.generate_forward_kinematics()
        compile(code, '<test_fk>', 'exec')

    def test_generate_inverse_kinematics_syntax(self):
        code = self.gen.generate_inverse_kinematics()
        compile(code, '<test_ik>', 'exec')

    def test_generate_inverse_jacobian_syntax(self):
        code = self.gen.generate_inverse_jacobian()
        compile(code, '<test_j_inv>', 'exec')

    def test_generate_forward_jacobian_syntax(self):
        code = self.gen.generate_forward_jacobian()
        compile(code, '<test_j_fwd>', 'exec')

    def test_generate_cluster_file_syntax(self):
        code = self.gen.generate_cluster_file()
        compile(code, '<test_cluster>', 'exec')


class TestHubSpokeGenerators:

    def setup_method(self):
        self.gen = _make_hs_gen()

    def test_generate_yaml_parses(self):
        content = self.gen.generate_yaml()
        data = yaml.safe_load(content)
        assert 'formation' in data
        f = data['formation']
        assert f['num_robots'] == 4
        assert f['cluster_config'] == 1

    def test_generate_forward_kinematics_syntax(self):
        code = self.gen.generate_forward_kinematics()
        compile(code, '<hs_fk>', 'exec')

    def test_generate_inverse_kinematics_syntax(self):
        code = self.gen.generate_inverse_kinematics()
        compile(code, '<hs_ik>', 'exec')

    def test_generate_inverse_jacobian_syntax(self):
        code = self.gen.generate_inverse_jacobian()
        compile(code, '<hs_j_inv>', 'exec')

    def test_generate_forward_jacobian_syntax(self):
        code = self.gen.generate_forward_jacobian()
        compile(code, '<hs_j_fwd>', 'exec')

    def test_generate_cluster_file_syntax(self):
        code = self.gen.generate_cluster_file()
        compile(code, '<hs_cluster>', 'exec')
