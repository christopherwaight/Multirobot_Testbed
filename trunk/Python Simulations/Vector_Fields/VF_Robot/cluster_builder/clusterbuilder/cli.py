"""Command-line interface for clusterbuilder."""

import argparse
import sys

from .tree import TreeParser
from .hub_spoke import HubAndSpokeKinematics
from .coc import ClusterOfClustersKinematics
from .codegen import CodeGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        prog='clusterbuilder.py',
        description='Generate kinematics and cluster files for N-robot formations.'
    )
    parser.add_argument('num_robots',     type=int,
                        help='Number of robots')
    parser.add_argument('cluster_config', type=int, choices=[1, 2],
                        help='1=hub-and-spoke, 2=cluster-of-clusters')
    parser.add_argument('cluster_name',   type=str,
                        help='Name stem for output files (valid Python identifier)')
    parser.add_argument('run_mode',       type=str, choices=['image_only', 'full'],
                        help='image_only or full (7 files)')
    parser.add_argument('config_tree',    type=str, nargs='?', default=None,
                        help='Optional tree spec e.g. "((2,2),(2,2))" or "(2,2,2)"')
    parser.add_argument('--orientation', choices=['on', 'off'], default='off',
                        help='on: add per-robot heading phi_i (state 3N); off: position only (default)')
    parser.add_argument('--symbolic', action='store_true', default=False,
                        help='also emit symbolic inverse Jacobian (.txt and .tex) via sympy')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_robots < 2:
        print("Error: num_robots must be >= 2")
        sys.exit(1)
    if not args.cluster_name.isidentifier():
        print(f"Error: cluster_name '{args.cluster_name}' is not a valid Python identifier")
        sys.exit(1)
    if args.config_tree and args.cluster_config == 1:
        print("Warning: config_tree is ignored for cluster_config=1 (hub-and-spoke)")
        args.config_tree = None

    orientation = (args.orientation == 'on')
    print(f"\nClusterBuilder: {args.num_robots} robots, config={args.cluster_config}, "
          f"name='{args.cluster_name}', mode={args.run_mode}, orientation={args.orientation}")

    if args.cluster_config == 1:
        print("Building hub-and-spoke formation...")
        kin = HubAndSpokeKinematics(args.num_robots, orientation=orientation)
        gen = CodeGenerator(args.cluster_name, 1, args.num_robots, kin,
                            orientation=orientation, symbolic=args.symbolic)

    else:
        if args.config_tree:
            print(f"Parsing config_tree: {args.config_tree}")
            root = TreeParser.parse(args.config_tree)
        else:
            print("Auto-building tree...")
            root = TreeParser.auto_build(args.num_robots)
            TreeParser._assign_ids_and_indices(root)

        TreeParser.validate(root, args.num_robots)
        print("Tree structure:")
        print(TreeParser.describe(root))

        kin = ClusterOfClustersKinematics(root, orientation=orientation)
        print(f"State variables ({len(kin.state_vars)}): {kin.state_vars}")
        gen = CodeGenerator(args.cluster_name, 2, args.num_robots, kin, tree=root,
                            orientation=orientation, symbolic=args.symbolic)

    print(f"\nGenerating output files...")
    gen.write_all(image_only=(args.run_mode == 'image_only'))
    print("\nDone.")
