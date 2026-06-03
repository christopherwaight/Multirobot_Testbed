"""Shared helpers for cluster_builder tests."""
import sys
import os

# Add cluster_builder root to path so clusterbuilder is importable directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
