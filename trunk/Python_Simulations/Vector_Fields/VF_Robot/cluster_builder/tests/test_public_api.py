"""Smoke test: all public names re-exported from the clusterbuilder package."""


def test_public_api_imports():
    from clusterbuilder import (
        HubAndSpokeKinematics, ClusterOfClustersKinematics,
        PairBlock, TripleBlock, TreeNode, TreeParser,
        Visualizer, CodeGenerator, ClusterBuilderError, SympyBackend,
    )
    for cls in (HubAndSpokeKinematics, ClusterOfClustersKinematics,
                PairBlock, TripleBlock, TreeNode, TreeParser,
                Visualizer, CodeGenerator):
        assert isinstance(cls, type), f"{cls.__name__} is not a class"
    assert issubclass(ClusterBuilderError, Exception)
    # SympyBackend is a singleton instance, not a class
    assert hasattr(SympyBackend, 'cos') and hasattr(SympyBackend, 'sin')
