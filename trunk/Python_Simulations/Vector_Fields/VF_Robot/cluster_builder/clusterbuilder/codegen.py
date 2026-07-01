"""Code generator: emits FK, IK, Jacobian, YAML, and cluster files."""

import sys
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")

from .tree import TreeNode, TreeParser
from .visualizer import Visualizer


class CodeGenerator:

    def __init__(self, cluster_name: str, config: int, N: int,
                 kin, tree: Optional[TreeNode] = None,
                 orientation: bool = False, symbolic: bool = False):
        self.name = cluster_name
        self.config = config
        self.N = N
        self.kin = kin
        self.tree = tree
        self.orientation = orientation
        self.symbolic = symbolic

    # -- YAML ----------------------------------------------------------------

    def generate_yaml(self) -> str:
        kin = self.kin
        state = kin.default_state()
        if self.config == 1:
            data = {
                'formation': {
                    'type': 'hub_and_spoke',
                    'num_robots': self.N,
                    'cluster_config': 1,
                    **{k: round(v, 6) for k, v in state.items()},
                    'position_gain': 1.0,
                    'angle_gain': 0.1,
                }
            }
        else:
            formation_dict = {
                'type': 'cluster_of_clusters',
                'num_robots': self.N,
                'cluster_config': 2,
            }
            if self.tree is not None:
                formation_dict['config_tree'] = TreeParser.to_string(self.tree)
            formation_dict.update({k: round(v, 6) for k, v in state.items()})
            formation_dict['position_gain'] = 1.0
            formation_dict['angle_gain'] = 0.1
            data = {'formation': formation_dict}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # -- Forward kinematics --------------------------------------------------

    def generate_forward_kinematics(self) -> str:
        kin = self.kin
        N = self.N
        args = ', '.join(f'x{i}, y{i}' for i in range(1, N + 1))

        if self.config == 1:
            lines = [
                f'"""Forward kinematics for {self.name} hub-and-spoke ({N} robots).',
                f'State vars: {kin.state_vars}',
                '"""',
                'import math',
                '',
                f'def forward_kinematics({args}):',
                f'    """Hub = robot {N}. Spokes = robots 1..{N-1}."""',
                f'    x_h, y_h = x{N}, y{N}',
                '    betas = {}',
            ]
            for i in range(1, N):
                lines += [
                    f'    dx_{i} = x{i} - x_h',
                    f'    dy_{i} = y{i} - y_h',
                    f'    r_{i} = math.hypot(dx_{i}, dy_{i})',
                    f'    betas[{i}] = math.atan2(dy_{i}, dx_{i})',
                ]
            lines += ['    theta_c = betas[1]']
            for i in range(2, N):
                lines += [
                    f'    gamma_{i} = (betas[{i}] - betas[1] + math.pi) % (2*math.pi) - math.pi',
                ]
            ret_items = ['x_h', 'y_h', 'theta_c'] + \
                        [f'r_{i}' for i in range(1, N)] + \
                        [f'gamma_{i}' for i in range(2, N)]
            ret_dict = '{' + ', '.join(f"'{k}': {k}" for k in ret_items) + '}'
            lines += [f'    return {ret_dict}']
        else:
            lines = [
                f'"""Forward kinematics for {self.name} cluster-of-clusters ({N} robots)."""',
                'import math',
                '',
                f'def forward_kinematics({args}):',
                f'    positions = ' + '{' +
                ', '.join(f'{i}: (x{i}, y{i})' for i in range(1, N+1)) + '}',
                '    state = {}',
                '    _fk_impl(positions, state)',
                '    return state',
                '',
                '# Internal implementation (generated)',
            ]
            lines += self._gen_fk_impl()

        return '\n'.join(lines)

    def _gen_fk_impl(self) -> List[str]:
        """Generate the _fk_impl function body for cluster-of-clusters."""
        lines = ['def _fk_impl(positions, state):']
        self._gen_fk_node(self.kin.root, lines, is_root=True)
        lines.append('    return state')
        return lines

    def _gen_fk_node(self, node, lines, is_root=False, indent=1):
        pad = '    ' * indent
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                lines.append(f'{pad}cx_{nid}, cy_{nid} = positions[{ri}]')
            elif node.size == 2:
                i1, i2 = node.robot_indices
                lines += [
                    f'{pad}xa_{nid}, ya_{nid} = positions[{i1}]',
                    f'{pad}xb_{nid}, yb_{nid} = positions[{i2}]',
                    f'{pad}cx_{nid} = (xa_{nid} + xb_{nid}) / 2',
                    f'{pad}cy_{nid} = (ya_{nid} + yb_{nid}) / 2',
                    f'{pad}state["theta_{nid}"] = math.atan2(yb_{nid} - ya_{nid}, xb_{nid} - xa_{nid})',
                    f'{pad}state["L_{nid}"] = math.hypot(xb_{nid} - xa_{nid}, yb_{nid} - ya_{nid})',
                ]
            else:
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}xa_{nid}, ya_{nid} = positions[{i1}]',
                    f'{pad}xb_{nid}, yb_{nid} = positions[{i2}]',
                    f'{pad}xc_{nid}, yc_{nid} = positions[{i3}]',
                    f'{pad}p_{nid} = math.hypot(xb_{nid}-xa_{nid}, yb_{nid}-ya_{nid})',
                    f'{pad}q_{nid} = math.hypot(xc_{nid}-xb_{nid}, yc_{nid}-yb_{nid})',
                    f'{pad}r_{nid} = math.hypot(xa_{nid}-xc_{nid}, ya_{nid}-yc_{nid})',
                    f'{pad}cos_beta_{nid} = max(-1.0, min(1.0, (p_{nid}**2+q_{nid}**2-r_{nid}**2)/(2*p_{nid}*q_{nid}+1e-10)))',
                    f'{pad}state["p_{nid}"] = p_{nid}',
                    f'{pad}state["beta_{nid}"] = math.acos(cos_beta_{nid})',
                    f'{pad}state["q_{nid}"] = q_{nid}',
                    f'{pad}state["theta_{nid}"] = math.atan2(ya_{nid}-yb_{nid}, xa_{nid}-xb_{nid})',
                    f'{pad}cx_{nid} = (xa_{nid}+xb_{nid}+xc_{nid})/3',
                    f'{pad}cy_{nid} = (ya_{nid}+yb_{nid}+yc_{nid})/3',
                ]
        else:
            for child in node.children:
                self._gen_fk_node(child, lines, indent=indent)
            if node.arity == 2:
                c0, c1 = node.children
                lines += [
                    f'{pad}cx_{nid} = (cx_{c0.node_id} + cx_{c1.node_id}) / 2',
                    f'{pad}cy_{nid} = (cy_{c0.node_id} + cy_{c1.node_id}) / 2',
                    f'{pad}state["L_{nid}"] = math.hypot(cx_{c1.node_id}-cx_{c0.node_id}, cy_{c1.node_id}-cy_{c0.node_id})',
                ]
                theta_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines.append(f'{pad}state[{theta_key}] = math.atan2(cy_{c1.node_id}-cy_{c0.node_id}, cx_{c1.node_id}-cx_{c0.node_id})')
            else:
                c0, c1, c2 = node.children
                lines += [
                    f'{pad}xa_m{nid}, ya_m{nid} = cx_{c0.node_id}, cy_{c0.node_id}',
                    f'{pad}xb_m{nid}, yb_m{nid} = cx_{c1.node_id}, cy_{c1.node_id}',
                    f'{pad}xc_m{nid}, yc_m{nid} = cx_{c2.node_id}, cy_{c2.node_id}',
                    f'{pad}p_m{nid} = math.hypot(xb_m{nid}-xa_m{nid}, yb_m{nid}-ya_m{nid})',
                    f'{pad}q_m{nid} = math.hypot(xc_m{nid}-xb_m{nid}, yc_m{nid}-yb_m{nid})',
                    f'{pad}r_m{nid} = math.hypot(xa_m{nid}-xc_m{nid}, ya_m{nid}-yc_m{nid})',
                    f'{pad}cb_m{nid} = max(-1.0,min(1.0,(p_m{nid}**2+q_m{nid}**2-r_m{nid}**2)/(2*p_m{nid}*q_m{nid}+1e-10)))',
                    f'{pad}state["p_{nid}"] = p_m{nid}',
                    f'{pad}state["beta_{nid}"] = math.acos(cb_m{nid})',
                    f'{pad}state["q_{nid}"] = q_m{nid}',
                    f'{pad}cx_{nid} = (xa_m{nid}+xb_m{nid}+xc_m{nid})/3',
                    f'{pad}cy_{nid} = (ya_m{nid}+yb_m{nid}+yc_m{nid})/3',
                ]
                theta_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines.append(f'{pad}state[{theta_key}] = math.atan2(ya_m{nid}-yb_m{nid}, xa_m{nid}-xb_m{nid})')
            if is_root:
                lines += [
                    f'{pad}state["x_c"] = cx_{nid}',
                    f'{pad}state["y_c"] = cy_{nid}',
                ]

    # -- Inverse kinematics --------------------------------------------------

    def generate_inverse_kinematics(self) -> str:
        N = self.N
        kin = self.kin
        if self.config == 1:
            lines = [
                f'"""Inverse kinematics for {self.name} hub-and-spoke ({N} robots)."""',
                'import math',
                '',
                'def inverse_kinematics(state):',
                '    """state: dict -> {robot_idx: (x, y)}"""',
                '    x_h = state["x_h"]',
                '    y_h = state["y_h"]',
                '    theta_c = state["theta_c"]',
                f'    positions = {{{N}: (x_h, y_h)}}',
            ]
            for i in range(1, N):
                g = '0.0' if i == 1 else f'state["gamma_{i}"]'
                lines += [
                    f'    r_{i} = state["r_{i}"]',
                    f'    gamma_{i} = {g}',
                    f'    angle_{i} = theta_c + gamma_{i}',
                    f'    positions[{i}] = (x_h + r_{i} * math.cos(angle_{i}), y_h + r_{i} * math.sin(angle_{i}))',
                ]
            lines.append('    return positions')
        else:
            lines = [
                f'"""Inverse kinematics for {self.name} cluster-of-clusters ({N} robots)."""',
                'import math',
                '',
                'def inverse_kinematics(state):',
                '    """state: dict -> {robot_idx: (x, y)}"""',
                '    positions = {}',
                '    _ik_impl(state["x_c"], state["y_c"], state["theta_c"], state, positions)',
                '    return positions',
                '',
            ]
            lines += self._gen_ik_impl()
        return '\n'.join(lines)

    def _gen_ik_impl(self) -> List[str]:
        lines = ['def _ik_impl(cx, cy, theta, state, positions):']
        self._gen_ik_node(self.kin.root, lines, is_root=True)
        return lines

    def _gen_ik_node(self, node, lines, is_root=False, indent=1):
        pad = '    ' * indent
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                lines.append(f'{pad}positions[{ri}] = (cx, cy)')
            elif node.size == 2:
                i1, i2 = node.robot_indices
                lines += [
                    f'{pad}_L = state["L_{nid}"]',
                    f'{pad}_th = state["theta_{nid}"]',
                    f'{pad}_ct, _st = math.cos(_th), math.sin(_th)',
                    f'{pad}positions[{i1}] = (cx - (_L/2)*_ct, cy - (_L/2)*_st)',
                    f'{pad}positions[{i2}] = (cx + (_L/2)*_ct, cy + (_L/2)*_st)',
                ]
            else:
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}_p = state["p_{nid}"]; _beta = state["beta_{nid}"]; _q = state["q_{nid}"]',
                    f'{pad}_th = state["theta_{nid}"]',
                    f'{pad}# SAS inverse kinematics inline',
                    f'{pad}_x2l, _y2l = 0.0, 0.0',
                    f'{pad}_x1l, _y1l = _p, 0.0',
                    f'{pad}_x3l = _q * math.cos(_beta); _y3l = _q * math.sin(_beta)',
                    f'{pad}_cxl = (_x1l+_x2l+_x3l)/3; _cyl = (_y1l+_y2l+_y3l)/3',
                    f'{pad}_x1l -= _cxl; _y1l -= _cyl; _x2l -= _cxl; _y2l -= _cyl; _x3l -= _cxl; _y3l -= _cyl',
                    f'{pad}_ct, _st = math.cos(_th), math.sin(_th)',
                    f'{pad}positions[{i1}] = (_ct*_x1l - _st*_y1l + cx, _st*_x1l + _ct*_y1l + cy)',
                    f'{pad}positions[{i2}] = (_ct*_x2l - _st*_y2l + cx, _st*_x2l + _ct*_y2l + cy)',
                    f'{pad}positions[{i3}] = (_ct*_x3l - _st*_y3l + cx, _st*_x3l + _ct*_y3l + cy)',
                ]
        else:
            if node.arity == 2:
                c0, c1 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state[{th_key}]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}cx_{c0.node_id} = cx - (_L_{nid}/2)*_ct_{nid}',
                    f'{pad}cy_{c0.node_id} = cy - (_L_{nid}/2)*_st_{nid}',
                    f'{pad}cx_{c1.node_id} = cx + (_L_{nid}/2)*_ct_{nid}',
                    f'{pad}cy_{c1.node_id} = cy + (_L_{nid}/2)*_st_{nid}',
                ]
                self._gen_ik_node_with_centroid(c0, lines, f'cx_{c0.node_id}', f'cy_{c0.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c1, lines, f'cx_{c1.node_id}', f'cy_{c1.node_id}', f'_th_{nid}', indent)
            else:
                c0, c1, c2 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}_p_{nid} = state["p_{nid}"]; _b_{nid} = state["beta_{nid}"]; _q_{nid} = state["q_{nid}"]',
                    f'{pad}_th_{nid} = state[{th_key}]',
                    f'{pad}_x2l_{nid}, _y2l_{nid} = 0.0, 0.0',
                    f'{pad}_x1l_{nid}, _y1l_{nid} = _p_{nid}, 0.0',
                    f'{pad}_x3l_{nid} = _q_{nid}*math.cos(_b_{nid}); _y3l_{nid} = _q_{nid}*math.sin(_b_{nid})',
                    f'{pad}_cxl_{nid} = (_x1l_{nid}+_x2l_{nid}+_x3l_{nid})/3',
                    f'{pad}_cyl_{nid} = (_y1l_{nid}+_y2l_{nid}+_y3l_{nid})/3',
                    f'{pad}_x1l_{nid} -= _cxl_{nid}; _y1l_{nid} -= _cyl_{nid}',
                    f'{pad}_x2l_{nid} -= _cxl_{nid}; _y2l_{nid} -= _cyl_{nid}',
                    f'{pad}_x3l_{nid} -= _cxl_{nid}; _y3l_{nid} -= _cyl_{nid}',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}cx_{c0.node_id} = _ct_{nid}*_x1l_{nid} - _st_{nid}*_y1l_{nid} + cx',
                    f'{pad}cy_{c0.node_id} = _st_{nid}*_x1l_{nid} + _ct_{nid}*_y1l_{nid} + cy',
                    f'{pad}cx_{c1.node_id} = _ct_{nid}*_x2l_{nid} - _st_{nid}*_y2l_{nid} + cx',
                    f'{pad}cy_{c1.node_id} = _st_{nid}*_x2l_{nid} + _ct_{nid}*_y2l_{nid} + cy',
                    f'{pad}cx_{c2.node_id} = _ct_{nid}*_x3l_{nid} - _st_{nid}*_y3l_{nid} + cx',
                    f'{pad}cy_{c2.node_id} = _st_{nid}*_x3l_{nid} + _ct_{nid}*_y3l_{nid} + cy',
                ]
                self._gen_ik_node_with_centroid(c0, lines, f'cx_{c0.node_id}', f'cy_{c0.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c1, lines, f'cx_{c1.node_id}', f'cy_{c1.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c2, lines, f'cx_{c2.node_id}', f'cy_{c2.node_id}', f'_th_{nid}', indent)

    def _gen_ik_node_with_centroid(self, node, lines, cx_var, cy_var, th_var, indent):
        """Recursively generate IK, substituting the centroid variable names."""
        pad = '    ' * indent
        nid = node.node_id
        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                lines.append(f'{pad}positions[{ri}] = ({cx_var}, {cy_var})')
            elif node.size == 2:
                i1, i2 = node.robot_indices
                lines += [
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}positions[{i1}] = ({cx_var} - (_L_{nid}/2)*_ct_{nid}, {cy_var} - (_L_{nid}/2)*_st_{nid})',
                    f'{pad}positions[{i2}] = ({cx_var} + (_L_{nid}/2)*_ct_{nid}, {cy_var} + (_L_{nid}/2)*_st_{nid})',
                ]
            else:
                i1, i2, i3 = node.robot_indices
                lines += [
                    f'{pad}_p_{nid} = state["p_{nid}"]; _b_{nid} = state["beta_{nid}"]; _q_{nid} = state["q_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_x2l_{nid}, _y2l_{nid} = 0.0, 0.0',
                    f'{pad}_x1l_{nid}, _y1l_{nid} = _p_{nid}, 0.0',
                    f'{pad}_x3l_{nid} = _q_{nid}*math.cos(_b_{nid}); _y3l_{nid} = _q_{nid}*math.sin(_b_{nid})',
                    f'{pad}_cxl_{nid} = (_x1l_{nid}+_x2l_{nid}+_x3l_{nid})/3',
                    f'{pad}_cyl_{nid} = (_y1l_{nid}+_y2l_{nid}+_y3l_{nid})/3',
                    f'{pad}_x1l_{nid} -= _cxl_{nid}; _y1l_{nid} -= _cyl_{nid}',
                    f'{pad}_x2l_{nid} -= _cxl_{nid}; _y2l_{nid} -= _cyl_{nid}',
                    f'{pad}_x3l_{nid} -= _cxl_{nid}; _y3l_{nid} -= _cyl_{nid}',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}positions[{i1}] = (_ct_{nid}*_x1l_{nid} - _st_{nid}*_y1l_{nid} + {cx_var}, _st_{nid}*_x1l_{nid} + _ct_{nid}*_y1l_{nid} + {cy_var})',
                    f'{pad}positions[{i2}] = (_ct_{nid}*_x2l_{nid} - _st_{nid}*_y2l_{nid} + {cx_var}, _st_{nid}*_x2l_{nid} + _ct_{nid}*_y2l_{nid} + {cy_var})',
                    f'{pad}positions[{i3}] = (_ct_{nid}*_x3l_{nid} - _st_{nid}*_y3l_{nid} + {cx_var}, _st_{nid}*_x3l_{nid} + _ct_{nid}*_y3l_{nid} + {cy_var})',
                ]
        else:
            # Non-leaf: expand inline using cx_var/cy_var as the parent centroid.
            if node.arity == 2:
                c0, c1 = node.children
                lines += [
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}cx_{c0.node_id} = {cx_var} - (_L_{nid}/2)*_ct_{nid}',
                    f'{pad}cy_{c0.node_id} = {cy_var} - (_L_{nid}/2)*_st_{nid}',
                    f'{pad}cx_{c1.node_id} = {cx_var} + (_L_{nid}/2)*_ct_{nid}',
                    f'{pad}cy_{c1.node_id} = {cy_var} + (_L_{nid}/2)*_st_{nid}',
                ]
                self._gen_ik_node_with_centroid(c0, lines, f'cx_{c0.node_id}', f'cy_{c0.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c1, lines, f'cx_{c1.node_id}', f'cy_{c1.node_id}', f'_th_{nid}', indent)
            else:
                c0, c1, c2 = node.children
                lines += [
                    f'{pad}_p_{nid} = state["p_{nid}"]; _b_{nid} = state["beta_{nid}"]; _q_{nid} = state["q_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_x2l_{nid}, _y2l_{nid} = 0.0, 0.0',
                    f'{pad}_x1l_{nid}, _y1l_{nid} = _p_{nid}, 0.0',
                    f'{pad}_x3l_{nid} = _q_{nid}*math.cos(_b_{nid}); _y3l_{nid} = _q_{nid}*math.sin(_b_{nid})',
                    f'{pad}_cxl_{nid} = (_x1l_{nid}+_x2l_{nid}+_x3l_{nid})/3',
                    f'{pad}_cyl_{nid} = (_y1l_{nid}+_y2l_{nid}+_y3l_{nid})/3',
                    f'{pad}_x1l_{nid} -= _cxl_{nid}; _y1l_{nid} -= _cyl_{nid}',
                    f'{pad}_x2l_{nid} -= _cxl_{nid}; _y2l_{nid} -= _cyl_{nid}',
                    f'{pad}_x3l_{nid} -= _cxl_{nid}; _y3l_{nid} -= _cyl_{nid}',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}cx_{c0.node_id} = _ct_{nid}*_x1l_{nid} - _st_{nid}*_y1l_{nid} + {cx_var}',
                    f'{pad}cy_{c0.node_id} = _st_{nid}*_x1l_{nid} + _ct_{nid}*_y1l_{nid} + {cy_var}',
                    f'{pad}cx_{c1.node_id} = _ct_{nid}*_x2l_{nid} - _st_{nid}*_y2l_{nid} + {cx_var}',
                    f'{pad}cy_{c1.node_id} = _st_{nid}*_x2l_{nid} + _ct_{nid}*_y2l_{nid} + {cy_var}',
                    f'{pad}cx_{c2.node_id} = _ct_{nid}*_x3l_{nid} - _st_{nid}*_y3l_{nid} + {cx_var}',
                    f'{pad}cy_{c2.node_id} = _st_{nid}*_x3l_{nid} + _ct_{nid}*_y3l_{nid} + {cy_var}',
                ]
                self._gen_ik_node_with_centroid(c0, lines, f'cx_{c0.node_id}', f'cy_{c0.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c1, lines, f'cx_{c1.node_id}', f'cy_{c1.node_id}', f'_th_{nid}', indent)
                self._gen_ik_node_with_centroid(c2, lines, f'cx_{c2.node_id}', f'cy_{c2.node_id}', f'_th_{nid}', indent)

    # -- Inverse Jacobian ----------------------------------------------------

    def generate_inverse_jacobian(self) -> str:
        N = self.N
        kin = self.kin
        if self.orientation and hasattr(kin, 'theta_ref_var') and not kin.theta_ref_var:
            kin.default_state()
        state_vars = kin.state_vars
        n_rows = 3 * N if self.orientation else 2 * N
        row_doc = (f'[x1,y1,...,x{N},y{N}, theta_1,...,theta_{N}]'
                   if self.orientation else f'[x1,y1, x2,y2, ..., x{N},y{N}]')
        lines = [
            f'"""Analytical inverse Jacobian for {self.name} ({N} robots).',
            f'Maps q_dot ({len(state_vars)} state vars) to r_dot ({n_rows} robot coords).',
            f'State var order: {state_vars}',
            f'Robot row order: {row_doc}',
            '"""',
            'import math',
            'import numpy as np',
            '',
            'STATE_VARS = ' + repr(state_vars),
            '',
            f'def inverse_jacobian(state):',
            f'    """Returns ({n_rows} x {len(state_vars)}) numpy array."""',
            f'    J = np.zeros(({n_rows}, {len(state_vars)}))',
        ]
        if self.config == 1:
            lines += self._gen_hub_spoke_j_inv_body(indent=1)
            if self.orientation:
                lines += self._gen_orientation_heading_rows(indent=1)
            lines.append('    return J')
        else:
            lines += [
                '    # Analytical Jacobian: numerically evaluated at current state',
                '    # using the closed-form composition of pair/SAS blocks.',
                '    # This calls the runtime kinematics objects.',
                '    _fill_jacobian(state, J)',
            ]
            if self.orientation:
                lines += self._gen_orientation_heading_rows(indent=1)
            lines.append('    return J')
            lines += self._gen_coc_j_inv_fill()
        return '\n'.join(lines)

    def _gen_orientation_heading_rows(self, indent=1) -> List[str]:
        """Emit the N heading rows for orientation=on into the generated inverse_jacobian."""
        N = self.N
        pad = '    ' * indent
        kin = self.kin
        lines = [
            f'{pad}# Heading rows: theta_i = theta_ref(i) + phi_i',
            f'{pad}_col = {{v: i for i, v in enumerate(STATE_VARS)}}',
        ]
        for i in range(1, N + 1):
            heading_row = 2 * N + (i - 1)
            ref_key = kin.theta_ref_var.get(i, 'theta_c')
            lines += [
                f'{pad}J[{heading_row}, _col["phi_{i}"]] = 1.0',
                f'{pad}J[{heading_row}, _col["{ref_key}"]] += 1.0',
            ]
        return lines

    def _gen_hub_spoke_j_inv_body(self, indent=1) -> List[str]:
        N = self.N
        pad = '    ' * indent
        lines = [
            f'{pad}col = {{v: i for i, v in enumerate(STATE_VARS)}}',
            f'{pad}theta_c = state["theta_c"]',
        ]
        for i in range(1, N):
            rx = 2 * (i - 1)
            ry = rx + 1
            lines += [
                f'{pad}r_{i} = state["r_{i}"]',
                f'{pad}g_{i} = ' + ('0.0' if i == 1 else f'state["gamma_{i}"]'),
                f'{pad}alpha_{i} = theta_c + g_{i}',
                f'{pad}J[{rx}, col["x_h"]] = 1.0',
                f'{pad}J[{ry}, col["y_h"]] = 1.0',
                f'{pad}J[{rx}, col["theta_c"]] += -r_{i} * math.sin(alpha_{i})',
                f'{pad}J[{ry}, col["theta_c"]] +=  r_{i} * math.cos(alpha_{i})',
                f'{pad}J[{rx}, col["r_{i}"]] = math.cos(alpha_{i})',
                f'{pad}J[{ry}, col["r_{i}"]] = math.sin(alpha_{i})',
            ]
            if i >= 2:
                lines += [
                    f'{pad}J[{rx}, col["gamma_{i}"]] = -r_{i} * math.sin(alpha_{i})',
                    f'{pad}J[{ry}, col["gamma_{i}"]] =  r_{i} * math.cos(alpha_{i})',
                ]
        hub_rx = 2 * (N - 1)
        lines += [
            f'{pad}J[{hub_rx}, col["x_h"]] = 1.0',
            f'{pad}J[{hub_rx+1}, col["y_h"]] = 1.0',
        ]
        return lines

    def _gen_coc_j_inv_fill(self) -> List[str]:
        """Generate a _fill_jacobian function that uses PairBlock/TripleBlock inline."""
        lines = [
            '',
            'def _fill_jacobian(state, J):',
            '    """Fill J using analytical composition of pair/SAS blocks."""',
            '    col = {v: i for i, v in enumerate(STATE_VARS)}',
        ]
        self._gen_jac_fill_node(self.kin.root, lines, is_root=True, indent=1)
        return lines

    def _gen_jac_fill_node(self, node, lines, is_root=False, indent=1):
        pad = '    ' * indent
        nid = node.node_id

        if node.is_leaf:
            if node.size == 1:
                ri = node.robot_indices[0]
                rx, ry = 2*(ri-1), 2*(ri-1)+1
                lines += [
                    f'{pad}# Leaf size-1: robot {ri} tracks centroid',
                    f'{pad}J[{rx}, col["x_c"]] += 1.0',
                    f'{pad}J[{ry}, col["y_c"]] += 1.0',
                ]
            elif node.size == 2:
                i1, i2 = node.robot_indices
                r1x, r1y = 2*(i1-1), 2*(i1-1)+1
                r2x, r2y = 2*(i2-1), 2*(i2-1)+1
                lines += [
                    f'{pad}# Leaf pair: robots {i1},{i2}',
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state["theta_{nid}"]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}_h_{nid} = _L_{nid} / 2',
                    f'{pad}# Centroid columns (from parent propagation)',
                    f'{pad}J[{r1x}, col["x_c"]] += 1.0; J[{r1y}, col["y_c"]] += 1.0',
                    f'{pad}J[{r2x}, col["x_c"]] += 1.0; J[{r2y}, col["y_c"]] += 1.0',
                    f'{pad}# Shape: theta_{nid}',
                    f'{pad}J[{r1x}, col["theta_{nid}"]] +=  _h_{nid}*_st_{nid}',
                    f'{pad}J[{r1y}, col["theta_{nid}"]] += -_h_{nid}*_ct_{nid}',
                    f'{pad}J[{r2x}, col["theta_{nid}"]] += -_h_{nid}*_st_{nid}',
                    f'{pad}J[{r2y}, col["theta_{nid}"]] +=  _h_{nid}*_ct_{nid}',
                    f'{pad}# Shape: L_{nid}',
                    f'{pad}J[{r1x}, col["L_{nid}"]] += -0.5*_ct_{nid}',
                    f'{pad}J[{r1y}, col["L_{nid}"]] += -0.5*_st_{nid}',
                    f'{pad}J[{r2x}, col["L_{nid}"]] +=  0.5*_ct_{nid}',
                    f'{pad}J[{r2y}, col["L_{nid}"]] +=  0.5*_st_{nid}',
                ]
            else:  # size == 3
                i1, i2, i3 = node.robot_indices
                r1x, r1y = 2*(i1-1), 2*(i1-1)+1
                r2x, r2y = 2*(i2-1), 2*(i2-1)+1
                r3x, r3y = 2*(i3-1), 2*(i3-1)+1
                lines += [
                    f'{pad}# Leaf SAS-3: robots {i1},{i2},{i3}',
                    f'{pad}_p_{nid}=state["p_{nid}"]; _b_{nid}=state["beta_{nid}"]; _q_{nid}=state["q_{nid}"]',
                    f'{pad}_th_{nid}=state["theta_{nid}"]',
                    f'{pad}# Centroid columns: all robots in this leaf translate with x_c/y_c',
                    f'{pad}J[{r1x}, col["x_c"]] += 1.0; J[{r1y}, col["y_c"]] += 1.0',
                    f'{pad}J[{r2x}, col["x_c"]] += 1.0; J[{r2y}, col["y_c"]] += 1.0',
                    f'{pad}J[{r3x}, col["x_c"]] += 1.0; J[{r3y}, col["y_c"]] += 1.0',
                ]
                self._emit_sas_jac_inline(node, lines, pad, f'_th_{nid}', f'_p_{nid}', f'_b_{nid}', f'_q_{nid}', nid, is_root=False)
        else:
            for child in node.children:
                self._gen_jac_fill_node(child, lines, is_root=False, indent=indent)
            if node.arity == 2:
                c0, c1 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}# Internal pair node {nid}: children {c0.node_id},{c1.node_id}',
                    f'{pad}_L_{nid} = state["L_{nid}"]',
                    f'{pad}_th_{nid} = state[{th_key}]',
                    f'{pad}_ct_{nid}, _st_{nid} = math.cos(_th_{nid}), math.sin(_th_{nid})',
                    f'{pad}_h_{nid} = _L_{nid} / 2',
                ]
                for child, sign_th_x, sign_th_y, sign_L in zip(
                    [c0, c1], ['+', '-'], ['-', '+'], ['-', '+']
                ):
                    for ri in child.robot_indices:
                        rx, ry = 2*(ri-1), 2*(ri-1)+1
                        lines += [
                            f'{pad}# Robot {ri} (child {child.node_id}) theta/L contribution',
                            f'{pad}J[{rx}, col[{th_key}]] += {sign_th_x}_h_{nid}*_st_{nid}',
                            f'{pad}J[{ry}, col[{th_key}]] += {sign_th_y}_h_{nid}*_ct_{nid}',
                            f'{pad}J[{rx}, col["L_{nid}"]] += {sign_L}0.5*_ct_{nid}',
                            f'{pad}J[{ry}, col["L_{nid}"]] += {sign_L}0.5*_st_{nid}',
                        ]
            else:
                c0, c1, c2 = node.children
                th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
                lines += [
                    f'{pad}# Internal SAS-3 meta-node {nid}',
                    f'{pad}_p_{nid}=state["p_{nid}"]; _b_{nid}=state["beta_{nid}"]; _q_{nid}=state["q_{nid}"]',
                    f'{pad}_th_{nid}=state[{th_key}]',
                ]
                self._emit_sas_jac_inline(node, lines, pad, f'_th_{nid}', f'_p_{nid}', f'_b_{nid}', f'_q_{nid}', nid, is_root=is_root, meta=True)

    def _emit_sas_jac_inline(self, node, lines, pad, th_var, p_var, b_var, q_var, nid, is_root=False, meta=False):
        """Emit the SAS Jacobian fill inline: shape columns p, beta, q and theta.

        Centroid columns (x_c, y_c) are NOT written here. Pair-leaf and size-1-leaf
        emitters already write += 1.0 to the centroid columns for every robot
        beneath them, which is the chain-rule shortcut for the special case where
        each child's centroid equals the parent's view of that child's position.
        Adding the same contribution again at this meta-level would double-count.
        """
        th_key = '"theta_c"' if is_root else f'"theta_{nid}"'
        lines += [
            f'{pad}_ct_{nid} = math.cos({th_var})',
            f'{pad}_st_{nid} = math.sin({th_var})',
            f'{pad}_sb_{nid} = math.sin({b_var})',
            f'{pad}_cb_{nid} = math.cos({b_var})',
            f'{pad}_x1l_{nid} = {p_var}',
            f'{pad}_y1l_{nid} = 0.0',
            f'{pad}_x2l_{nid} = 0.0',
            f'{pad}_y2l_{nid} = 0.0',
            f'{pad}_x3l_{nid} = {q_var}*_cb_{nid}',
            f'{pad}_y3l_{nid} = {q_var}*_sb_{nid}',
            f'{pad}_cxl_{nid} = (_x1l_{nid} + _x2l_{nid} + _x3l_{nid})/3.0',
            f'{pad}_cyl_{nid} = (_y1l_{nid} + _y2l_{nid} + _y3l_{nid})/3.0',
        ]

        dlx_dp = ['2.0/3.0', '-1.0/3.0', '-1.0/3.0']
        dly_dp = ['0.0', '0.0', '0.0']
        dlx_db = [f'{q_var}*_sb_{nid}/3.0',
                  f'{q_var}*_sb_{nid}/3.0',
                  f'(-2.0/3.0)*{q_var}*_sb_{nid}']
        dly_db = [f'-{q_var}*_cb_{nid}/3.0',
                  f'-{q_var}*_cb_{nid}/3.0',
                  f'(2.0/3.0)*{q_var}*_cb_{nid}']
        dlx_dq = [f'-_cb_{nid}/3.0',
                  f'-_cb_{nid}/3.0',
                  f'(2.0/3.0)*_cb_{nid}']
        dly_dq = [f'-_sb_{nid}/3.0',
                  f'-_sb_{nid}/3.0',
                  f'(2.0/3.0)*_sb_{nid}']

        lx_expr = [f'(_x1l_{nid} - _cxl_{nid})',
                   f'(_x2l_{nid} - _cxl_{nid})',
                   f'(_x3l_{nid} - _cxl_{nid})']
        ly_expr = [f'(_y1l_{nid} - _cyl_{nid})',
                   f'(_y2l_{nid} - _cyl_{nid})',
                   f'(_y3l_{nid} - _cyl_{nid})']

        if meta:
            child_robot_lists = [list(c.robot_indices) for c in node.children]
        else:
            child_robot_lists = [[ri] for ri in node.robot_indices]

        for i, robots_under_child in enumerate(child_robot_lists):
            for ri in robots_under_child:
                rx, ry = 2*(ri-1), 2*(ri-1)+1
                lines += [
                    f'{pad}J[{rx}, col[{th_key}]] += -_st_{nid}*{lx_expr[i]} - _ct_{nid}*{ly_expr[i]}',
                    f'{pad}J[{ry}, col[{th_key}]] +=  _ct_{nid}*{lx_expr[i]} - _st_{nid}*{ly_expr[i]}',
                    f'{pad}J[{rx}, col["p_{nid}"]] += _ct_{nid}*({dlx_dp[i]}) - _st_{nid}*({dly_dp[i]})',
                    f'{pad}J[{ry}, col["p_{nid}"]] += _st_{nid}*({dlx_dp[i]}) + _ct_{nid}*({dly_dp[i]})',
                    f'{pad}J[{rx}, col["beta_{nid}"]] += _ct_{nid}*({dlx_db[i]}) - _st_{nid}*({dly_db[i]})',
                    f'{pad}J[{ry}, col["beta_{nid}"]] += _st_{nid}*({dlx_db[i]}) + _ct_{nid}*({dly_db[i]})',
                    f'{pad}J[{rx}, col["q_{nid}"]] += _ct_{nid}*({dlx_dq[i]}) - _st_{nid}*({dly_dq[i]})',
                    f'{pad}J[{ry}, col["q_{nid}"]] += _st_{nid}*({dlx_dq[i]}) + _ct_{nid}*({dly_dq[i]})',
                ]

    # -- Forward Jacobian ----------------------------------------------------

    def generate_forward_jacobian(self) -> str:
        N = self.N
        sv = self.kin.state_vars
        return '\n'.join([
            f'"""Forward Jacobian for {self.name} ({N} robots).',
            f'Inverse of the inverse Jacobian. Maps r_dot -> q_dot.',
            '"""',
            'import numpy as np',
            f'from {self.name}_inverse_jacobian import inverse_jacobian',
            '',
            'def forward_jacobian(state):',
            f'    """Returns ({len(sv)} x {2*N}) numpy array."""',
            '    J_inv = inverse_jacobian(state)',
            '    try:',
            '        return np.linalg.inv(J_inv)',
            '    except np.linalg.LinAlgError:',
            '        return np.linalg.pinv(J_inv)',
        ])

    # -- Cluster file --------------------------------------------------------

    def generate_cluster_file(self) -> str:
        name = self.name
        Name = ''.join(w.capitalize() for w in name.split('_'))
        N = self.N
        kin = self.kin
        state_vars = kin.state_vars
        default_state = kin.default_state()

        shape_vars = [v for v in state_vars if v not in ('x_c', 'y_c', 'theta_c', 'x_h', 'y_h')]

        if self.config == 1:
            load_lines = [
                "        formation = config['formation']",
                "        self.N = formation['num_robots']",
            ] + [f"        self.desired_{v} = formation['{v}']"
                 for v in state_vars if v not in ('x_h', 'y_h', 'theta_c')] + [
                "        self.position_gain = formation.get('position_gain', 1.0)",
                "        self.angle_gain = formation.get('angle_gain', 0.1)",
            ]
        else:
            load_lines = [
                "        formation = config['formation']",
                "        self.N = formation['num_robots']",
            ] + [f"        self.desired_{v} = formation['{v}']"
                 for v in state_vars if v not in ('x_c', 'y_c', 'theta_c')] + [
                "        self.position_gain = formation.get('position_gain', 1.0)",
                "        self.angle_gain = formation.get('angle_gain', 0.1)",
            ]

        if self.config == 1:
            desired_state = (
                "{'x_h': self.robots[self.N-1].get_position()[0],\n"
                "             'y_h': self.robots[self.N-1].get_position()[1],\n"
                "             'theta_c': current_formation['theta_c']," +
                "".join(f"\n             'r_{i}': self.desired_r_{i}," for i in range(1, N)) +
                "".join(f"\n             'gamma_{i}': self.desired_gamma_{i}," for i in range(2, N)) +
                "}"
            )
        else:
            desired_state_items = (
                "{'x_c': current_formation['x_c'], 'y_c': current_formation['y_c'], 'theta_c': current_formation['theta_c']," +
                "".join(f"\n             '{v}': self.desired_{v}," for v in shape_vars) +
                "}"
            )

        orientation_guard = ''
        if self.orientation:
            orientation_guard = f'''
    # ------------------------------------------------------------------
    # ORIENTATION MODE NOTE
    # ------------------------------------------------------------------
    # This cluster was generated with --orientation on.
    # The FK, IK, and inverse Jacobian are fully correct and verified
    # via finite-difference tests.  The simulation controller below
    # (move / command_velocity) is NOT wired for per-robot heading
    # control: Omnibot accepts (vx, vy) only, not (vx, vy, omega).
    # Instantiating this class raises RuntimeError to prevent silent
    # incorrect results.  Use the kinematics and Jacobian files directly
    # for analysis, or extend the controller before enabling simulation.
    # ------------------------------------------------------------------
    ORIENTATION_ENABLED = True
'''
        code = f'''"""
{Name}Cluster — generated by clusterbuilder.py
Config: {"hub-and-spoke" if self.config == 1 else "cluster-of-clusters"}, {N} robots
State variables: {state_vars}
{"" if not self.orientation else chr(10) + "NOTE: Generated with --orientation on.  FK/IK/Jacobian are complete." + chr(10) + "The simulation controller is NOT wired for per-robot heading control." + chr(10) + "See ORIENTATION_ENABLED guard in the class body." + chr(10)}"""
import os
import yaml
import numpy as np
from src.robot.omnibot import Omnibot
from {name}_forward_kinematics import forward_kinematics
from {name}_inverse_kinematics import inverse_kinematics
from {name}_inverse_jacobian import inverse_jacobian, STATE_VARS


class {Name}Cluster:
{orientation_guard}
    def __init__(self, formation_config_path, field, timestep=0.1, momentum_alpha=0.7):
        {"raise RuntimeError(" + repr("Orientation mode is enabled. The controller simulation is not wired for per-robot heading (Omnibot accepts vx, vy only). FK, IK, and J^-1 are verified and usable directly from the kinematics files.") + ")" if self.orientation else ""}
        self.field = field
        self.timestep = timestep
        self.momentum_alpha = momentum_alpha
        self._load_formation_config(formation_config_path)
        self._initialize_robots()
        self.center_history = []
        self.robot_history = []
        self.velocity_history = []

    def _load_formation_config(self, config_path):
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
{chr(10).join(load_lines)}

    def _initialize_robots(self):
        import math, random
        default = {repr(default_state)}
        default['x_c' if 'x_c' in default else 'x_h'] = random.uniform(-0.5, 0.5)
        default['y_c' if 'y_c' in default else 'y_h'] = random.uniform(-0.5, 0.5)
        positions = inverse_kinematics(default)
        self.robots = [Omnibot(positions[i+1][0], positions[i+1][1],
                               self.timestep, self.momentum_alpha)
                       for i in range({N})]

    def get_robot_positions(self):
        return {{i+1: tuple(self.robots[i].get_position()) for i in range({N})}}

    def get_current_formation(self):
        pos = self.get_robot_positions()
        coords = []
        for i in range(1, {N}+1):
            coords += list(pos[i])
        return forward_kinematics(*coords)

    def get_centroid(self):
        f = self.get_current_formation()
        xk = 'x_h' if 'x_h' in f else 'x_c'
        yk = 'y_h' if 'y_h' in f else 'y_c'
        return np.array([f[xk], f[yk]])

    def sample_field_at_robots(self):
        return [robot.sample_field(self.field) for robot in self.robots]

    def move(self, control_primitive):
        current = self.get_current_formation()
        vx_c, vy_c = control_primitive(self)

        # Formation errors (proportional control)
        shape_vel = np.zeros(len(STATE_VARS))
        col = {{v: i for i, v in enumerate(STATE_VARS)}}
        xk = 'x_h' if 'x_h' in current else 'x_c'
        yk = 'y_h' if 'y_h' in current else 'y_c'
        shape_vel[col[xk]] = vx_c
        shape_vel[col[yk]] = vy_c
        shape_vel[col.get('theta_c', -1)] = 0.0  # no spin by default
'''
        if self.config == 1:
            code += f'''
        for i in range(1, {N}):
            r_key = f'r_{{i}}'
            if r_key in col:
                shape_vel[col[r_key]] = self.position_gain * (
                    getattr(self, f'desired_r_{{i}}') - current.get(r_key, 0))
        for i in range(2, {N}):
            g_key = f'gamma_{{i}}'
            if g_key in col:
                err = getattr(self, f'desired_gamma_{{i}}') - current.get(g_key, 0)
                err = (err + 3.14159) % (2*3.14159) - 3.14159
                shape_vel[col[g_key]] = self.angle_gain * err
'''
        else:
            code += f'''
        desired_shape = {{{', '.join(f'"{v}": self.desired_{v}' for v in shape_vars)}}}
        for v, desired in desired_shape.items():
            if v in col:
                err = desired - current.get(v, 0)
                gain = self.angle_gain if 'theta' in v or 'beta' in v else self.position_gain
                shape_vel[col[v]] = gain * err
'''
        code += f'''
        J_inv = inverse_jacobian(current)
        robot_vel = J_inv @ shape_vel

        for i, robot in enumerate(self.robots):
            robot.command_velocity(robot_vel[2*i], robot_vel[2*i+1])

        centroid = self.get_centroid()
        self.center_history.append(centroid.copy())
        pos = self.get_robot_positions()
        self.robot_history.append(np.array([list(pos[i+1]) for i in range({N})]))

    def reset(self, x_c=None, y_c=None):
        import math
        default = {repr(default_state)}
        default['x_c' if 'x_c' in default else 'x_h'] = x_c or 0.0
        default['y_c' if 'y_c' in default else 'y_h'] = y_c or 0.0
        positions = inverse_kinematics(default)
        for i, robot in enumerate(self.robots):
            xi, yi = positions[i+1]
            robot.set_position(xi, yi)
            robot.velocity = np.array([0.0, 0.0])
        self.center_history = []
        self.robot_history = []

    def get_center_history(self):
        return np.array(self.center_history) if self.center_history else np.array([])

    def get_robot_history(self):
        return np.array(self.robot_history) if self.robot_history else np.array([])

    def plot(self, ax):
        import matplotlib.pyplot as plt
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
                  'pink', 'gray', 'olive', 'cyan']
        pos = self.get_robot_positions()
        for i in range(1, {N}+1):
            xi, yi = pos[i]
            ax.scatter(xi, yi, color=colors[(i-1) % len(colors)], s=100, zorder=5)

    def plot_center(self, ax):
        c = self.get_centroid()
        ax.scatter(c[0], c[1], color='black', s=80, marker='x', zorder=5)

    def __repr__(self):
        c = self.get_centroid()
        return f"{Name}Cluster(centroid={{c}}, N={N})"
'''
        return code

    # -- Symbolic Jacobian emitter -------------------------------------------

    def generate_symbolic_jacobian(self):
        """
        Compute the symbolic inverse Jacobian and return (txt_content, tex_content).
        Requires sympy. Uses fresh Symbol per state variable.
        """
        try:
            import sympy
        except ImportError:
            sys.exit("sympy is required for --symbolic: pip install sympy")

        J_sym = self.kin.inverse_jacobian_symbolic()
        txt = sympy.pretty(J_sym, use_unicode=True)
        tex = sympy.latex(J_sym)
        header = (f"# Symbolic inverse Jacobian for {self.name}\n"
                  f"# State vars: {self.kin.state_vars}\n"
                  f"# Shape: {J_sym.shape[0]} x {J_sym.shape[1]}\n\n")
        return header + txt, tex

    # -- Write all files -----------------------------------------------------

    def write_all(self, image_only=False):
        img_file = f'{self.name}_visualization.png'
        if self.config == 1:
            fig = Visualizer.render_hub_spoke(self.kin, self.name)
        else:
            fig = Visualizer.render_cluster_of_clusters(self.kin, self.name)
        fig.savefig(img_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {img_file}")

        if image_only:
            return

        files = {
            f'{self.name}_forward_kinematics.py':  self.generate_forward_kinematics(),
            f'{self.name}_inverse_kinematics.py':  self.generate_inverse_kinematics(),
            f'{self.name}_forward_jacobian.py':    self.generate_forward_jacobian(),
            f'{self.name}_inverse_jacobian.py':    self.generate_inverse_jacobian(),
            f'{self.name}.yaml':                   self.generate_yaml(),
            f'{self.name}_cluster.py':             self.generate_cluster_file(),
        }
        for fname, content in files.items():
            with open(fname, 'w') as f:
                f.write(content)
            print(f"  Saved: {fname}")

        if self.symbolic:
            txt_content, tex_content = self.generate_symbolic_jacobian()
            txt_file = f'{self.name}_inverse_jacobian_symbolic.txt'
            tex_file = f'{self.name}_inverse_jacobian_symbolic.tex'
            with open(txt_file, 'w') as f:
                f.write(txt_content)
            print(f"  Saved: {txt_file}")
            with open(tex_file, 'w') as f:
                f.write(tex_content)
            print(f"  Saved: {tex_file}")
