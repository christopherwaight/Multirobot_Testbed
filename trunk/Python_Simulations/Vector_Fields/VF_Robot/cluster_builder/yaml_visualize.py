"""
yaml_visualize.py -- render a formation image from a clusterbuilder-generated YAML.

Usage:
    python yaml_visualize.py <path-to-yaml> [--out <path>.png]

For hub-and-spoke YAMLs (type: hub_and_spoke):
  Reads num_robots and the state variables (x_h, y_h, theta_c, r_*, gamma_*),
  constructs HubAndSpokeKinematics, and renders via Visualizer.render_hub_spoke.

For cluster-of-clusters YAMLs (type: cluster_of_clusters):
  Reads num_robots and, if present, config_tree.
  If config_tree is missing, falls back to TreeParser.auto_build(N) with a warning
  (only round-trips correctly for auto-built trees).
  Reads the state variables and renders via Visualizer.render_cluster_of_clusters.

The --out argument sets the output PNG path. Default: <yaml-basename>_visualization.png
in the same directory as the YAML.
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# clusterbuilder lives in the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clusterbuilder import (
    HubAndSpokeKinematics,
    ClusterOfClustersKinematics,
    TreeParser,
    Visualizer,
)


def _load_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def _state_from_formation(formation, state_vars):
    """Extract state dict from YAML formation block, warn on missing keys."""
    state = {}
    for v in state_vars:
        if v in formation:
            state[v] = float(formation[v])
        else:
            warnings.warn(
                f"State variable '{v}' missing from YAML; using default_state() value.",
                stacklevel=3
            )
    return state


def visualize(yaml_path, out_path=None):
    """
    Load the YAML at yaml_path, build kinematics, and render to out_path.
    Returns the output file path.
    """
    data = _load_yaml(yaml_path)
    formation = data.get('formation', data)

    fmt = formation.get('type', '')
    N = int(formation.get('num_robots', 0))
    if N < 2:
        sys.exit(f"Error: num_robots must be >= 2 (got {N}).")

    if out_path is None:
        base = os.path.splitext(os.path.basename(yaml_path))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                                f'{base}_visualization.png')

    if fmt == 'hub_and_spoke':
        kin = HubAndSpokeKinematics(N, orientation=False)
        # Build state from YAML, fall back to default for any missing key
        default = kin.default_state()
        state = {**default, **_state_from_formation(formation, kin.state_vars)}
        fig = Visualizer.render_hub_spoke(kin, os.path.splitext(
            os.path.basename(yaml_path))[0], state=state)

    elif fmt == 'cluster_of_clusters':
        config_tree_str = formation.get('config_tree', None)
        if config_tree_str is None:
            warnings.warn(
                "YAML missing 'config_tree' field; falling back to TreeParser.auto_build. "
                "This only round-trips correctly for auto-built trees.",
                stacklevel=2
            )
            root = TreeParser.auto_build(N)
        else:
            root = TreeParser.parse(str(config_tree_str))
            TreeParser.validate(root, N)

        kin = ClusterOfClustersKinematics(root, orientation=False)
        default = kin.default_state()
        state = {**default, **_state_from_formation(formation, kin.state_vars)}
        fig = Visualizer.render_cluster_of_clusters(
            kin, os.path.splitext(os.path.basename(yaml_path))[0], state=state)

    else:
        sys.exit(f"Error: unknown formation type '{fmt}'. Expected 'hub_and_spoke' or "
                 f"'cluster_of_clusters'.")

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        prog='yaml_visualize.py',
        description='Render a formation PNG from a clusterbuilder YAML.'
    )
    parser.add_argument('yaml_path', help='Path to the YAML file')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: <basename>_visualization.png)')
    args = parser.parse_args()

    if not os.path.isfile(args.yaml_path):
        sys.exit(f"Error: file not found: {args.yaml_path}")

    visualize(args.yaml_path, args.out)


if __name__ == '__main__':
    main()
