"""Tree data structure and parser for cluster-of-clusters configurations."""

import ast
from dataclasses import dataclass, field
from typing import List

from .errors import ClusterBuilderError


@dataclass
class TreeNode:
    size: int                              # total robots in this subtree
    children: List['TreeNode']             # empty = leaf
    robot_indices: List[int] = field(default_factory=list)  # 1-based, assigned post-parse
    node_id: int = 0                       # unique id assigned post-parse

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def arity(self):
        return len(self.children)


class TreeParser:

    @staticmethod
    def parse(tree_str: str) -> TreeNode:
        """Parse a config_tree string like '((2,2),(2,2))' or '(2,2,2)'."""
        s = tree_str.strip()
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            raise ClusterBuilderError(f"Could not parse config_tree '{s}': {e}")
        root = TreeParser._build_tree(parsed)
        TreeParser._assign_ids_and_indices(root)
        return root

    @staticmethod
    def _build_tree(obj) -> TreeNode:
        if isinstance(obj, int):
            if obj not in (1, 2, 3):
                raise ClusterBuilderError(
                    f"Leaf value {obj} is invalid — only 1, 2, 3 are allowed.")
            return TreeNode(size=obj, children=[])
        elif isinstance(obj, (tuple, list)):
            if len(obj) < 2:
                raise ClusterBuilderError(
                    "Each internal node must have at least 2 children.")
            if len(obj) > 3:
                raise ClusterBuilderError(
                    f"Each internal node can have at most 3 children, got {len(obj)}.")
            children = [TreeParser._build_tree(c) for c in obj]
            size = sum(c.size for c in children)
            return TreeNode(size=size, children=children)
        else:
            raise ClusterBuilderError(f"Unexpected token type: {type(obj)}")

    @staticmethod
    def auto_build(n: int, _top_level: bool = True) -> TreeNode:
        """
        Recursively split n robots into a tree with 2-or-3-child internal nodes.
        Prefers ternary splits (3 equal children) when n is divisible by 3,
        otherwise binary split (ceiling left, floor right).
        """
        if n in (1, 2, 3):
            node = TreeNode(size=n, children=[])
        elif n % 3 == 0:
            chunk = n // 3
            children = [TreeParser.auto_build(chunk, _top_level=False) for _ in range(3)]
            node = TreeNode(size=n, children=children)
        else:
            left_n = (n + 1) // 2
            right_n = n - left_n
            node = TreeNode(size=n, children=[
                TreeParser.auto_build(left_n, _top_level=False),
                TreeParser.auto_build(right_n, _top_level=False)
            ])
        if _top_level:
            TreeParser._assign_ids_and_indices(node)
        return node

    @staticmethod
    def _assign_ids_and_indices(root: TreeNode):
        """Assign node_id (BFS order) and robot_indices (DFS left-to-right, 1-based)."""
        queue = [root]
        nid = 1
        while queue:
            node = queue.pop(0)
            node.node_id = nid
            nid += 1
            for c in node.children:
                queue.append(c)

        counter = [1]
        def _dfs(node):
            if node.is_leaf:
                node.robot_indices = list(range(counter[0], counter[0] + node.size))
                counter[0] += node.size
            else:
                for c in node.children:
                    _dfs(c)
                node.robot_indices = []
                for c in node.children:
                    node.robot_indices.extend(c.robot_indices)
        _dfs(root)

    @staticmethod
    def validate(root: TreeNode, num_robots: int):
        if root.size != num_robots:
            raise ClusterBuilderError(
                f"Tree specifies {root.size} robots but num_robots={num_robots} was given.")
        if root.is_leaf:
            raise ClusterBuilderError(
                f"Single-leaf root tree is not permitted. "
                f"A leaf-only tree (e.g. '(3)') has no root-level pose variables "
                f"(x_c, y_c, theta_c) and IK will fail. Wrap it in a parent node, "
                f"e.g. '(3,3)' or '((3,2),2)'.")

    @staticmethod
    def describe(root: TreeNode, indent=0) -> str:
        prefix = "  " * indent
        if root.is_leaf:
            return f"{prefix}Leaf(size={root.size}, robots={root.robot_indices})"
        lines = [f"{prefix}Node(size={root.size}, id={root.node_id}, arity={root.arity})"]
        for c in root.children:
            lines.append(TreeParser.describe(c, indent + 1))
        return "\n".join(lines)

    @staticmethod
    def to_string(node: TreeNode) -> str:
        """Serialize a TreeNode back to a config_tree string, e.g. '((2,2),(2,2))'."""
        if node.is_leaf:
            return str(node.size)
        children_str = ','.join(TreeParser.to_string(c) for c in node.children)
        return f'({children_str})'
