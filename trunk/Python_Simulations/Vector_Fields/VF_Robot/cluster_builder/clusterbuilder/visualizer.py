"""Visualizer for hub-and-spoke and cluster-of-clusters formations."""

import math
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

from .hub_spoke import HubAndSpokeKinematics
from .coc import ClusterOfClustersKinematics

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#f781bf', '#999999',
          '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']


class Visualizer:

    @staticmethod
    def render_hub_spoke(kin: HubAndSpokeKinematics, cluster_name: str,
                         state=None) -> plt.Figure:
        if state is None:
            state = kin.default_state()
        positions = kin.inverse_kinematics(state)
        N = kin.N

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{cluster_name}: {N}-robot Hub-and-Spoke Formation", fontsize=14, fontweight='bold')

        x_h, y_h = positions[N][0], positions[N][1]

        # Draw spokes
        for i in range(1, N):
            xi, yi = positions[i][0], positions[i][1]
            ax.plot([x_h, xi], [y_h, yi], color='gray', lw=1.5, zorder=1)
            # Label r_i
            mx, my = (x_h + xi) / 2, (y_h + yi) / 2
            ax.annotate(f'r_{i}={state[f"r_{i}"]:.2f}', (mx, my),
                        fontsize=7, ha='center', va='bottom',
                        color='#555555',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7, ec='none'))

        # Draw spokes as circles
        for i in range(1, N):
            xi, yi = positions[i][0], positions[i][1]
            ax.scatter(xi, yi, s=200, color=COLORS[i % len(COLORS)], zorder=3,
                       edgecolors='black', linewidths=1.2)
            ax.text(xi, yi + 0.04, f'R{i}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Draw hub as square
        ax.scatter(x_h, y_h, s=300, color='black', marker='s', zorder=3)
        ax.text(x_h, y_h + 0.04, f'Hub (R{N})', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='black')

        # Draw theta_c arc
        r_arc = state['r_1'] * 0.35
        arc = Arc((x_h, y_h), 2 * r_arc, 2 * r_arc,
                  angle=0, theta1=0, theta2=math.degrees(state['theta_c']),
                  color='navy', lw=1.5)
        ax.add_patch(arc)
        tc_angle = state['theta_c'] / 2
        ax.text(x_h + r_arc * 1.1 * math.cos(tc_angle),
                y_h + r_arc * 1.1 * math.sin(tc_angle),
                'θ_c', fontsize=9, color='navy')

        # Draw gamma arcs
        for i in range(2, N):
            g_i = state[f'gamma_{i}']
            r_garc = state[f'r_{i}'] * 0.25
            arc_g = Arc((x_h, y_h), 2 * r_garc, 2 * r_garc,
                        angle=0,
                        theta1=math.degrees(state['theta_c']),
                        theta2=math.degrees(state['theta_c'] + g_i),
                        color='darkred', lw=1.2, linestyle='dashed')
            ax.add_patch(arc_g)
            mid_angle = state['theta_c'] + g_i / 2
            ax.text(x_h + r_garc * 1.3 * math.cos(mid_angle),
                    y_h + r_garc * 1.3 * math.sin(mid_angle),
                    f'γ_{i}', fontsize=8, color='darkred')

        # State vector text box
        sv_lines = ['State vector q:',
                    f'  x_h, y_h, θ_c'] + \
                   [f'  r_{i}' for i in range(1, N)] + \
                   [f'  γ_{i}' for i in range(2, N)]
        ax.text(0.02, 0.98, '\n'.join(sv_lines), transform=ax.transAxes,
                fontsize=8, va='top', family='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='gray'))

        ax.autoscale_view()
        fig.tight_layout()
        return fig

    @staticmethod
    def render_cluster_of_clusters(kin: ClusterOfClustersKinematics,
                                   cluster_name: str,
                                   state=None) -> plt.Figure:
        if state is None:
            state = kin.default_state()
        positions = kin.inverse_kinematics(state)
        root = kin.root

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{cluster_name}: {kin.N}-robot Cluster-of-Clusters Formation",
                     fontsize=14, fontweight='bold')

        # Assign a color per leaf
        leaf_colors = {}
        leaf_idx = [0]
        def assign_colors(node):
            if node.is_leaf:
                for ri in node.robot_indices:
                    leaf_colors[ri] = COLORS[leaf_idx[0] % len(COLORS)]
                leaf_idx[0] += 1
            else:
                for c in node.children:
                    assign_colors(c)
        assign_colors(root)

        # Collect centroids per node (bottom-up)
        centroids = {}
        def collect_centroids(node):
            if node.is_leaf:
                xs = [positions[ri][0] for ri in node.robot_indices]
                ys = [positions[ri][1] for ri in node.robot_indices]
                centroids[node.node_id] = (sum(xs)/len(xs), sum(ys)/len(ys))
            else:
                for c in node.children:
                    collect_centroids(c)
                xs = [centroids[c.node_id][0] for c in node.children]
                ys = [centroids[c.node_id][1] for c in node.children]
                centroids[node.node_id] = (sum(xs)/len(xs), sum(ys)/len(ys))
        collect_centroids(root)

        # Draw internal-node dashed lines (centroid-to-centroid)
        def draw_internal(node):
            if not node.is_leaf:
                cx, cy = centroids[node.node_id]
                nid = node.node_id
                for child in node.children:
                    ccx, ccy = centroids[child.node_id]
                    ax.plot([cx, ccx], [cy, ccy], '--', color='#777777',
                            lw=1.2, zorder=1)
                    # Label L or shape param on dashed line
                    mx, my = (cx + ccx) / 2, (cy + ccy) / 2
                    if node.arity == 2:
                        lbl = f"L_{nid}={state.get(f'L_{nid}', 0):.2f}"
                    else:
                        lbl = f"SAS_{nid}"
                    ax.annotate(lbl, (mx, my), fontsize=7, ha='center',
                                color='#555555',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          alpha=0.7, ec='none'))
                # Mark centroid
                ax.scatter(cx, cy, marker='x', s=100, color='black',
                           linewidths=2, zorder=4)
                ax.text(cx + 0.02, cy + 0.02, f'C{nid}', fontsize=7,
                        color='black', fontweight='bold')
                for child in node.children:
                    draw_internal(child)
        draw_internal(root)

        # Draw leaf intra-cluster solid lines
        def draw_leaf_edges(node):
            if node.is_leaf and node.size > 1:
                nid = node.node_id
                robots = node.robot_indices
                color = leaf_colors[robots[0]]
                if node.size == 2:
                    xa, ya = positions[robots[0]][:2]
                    xb, yb = positions[robots[1]][:2]
                    ax.plot([xa, xb], [ya, yb], '-', color=color, lw=2, zorder=2)
                    mx, my = (xa + xb) / 2, (ya + yb) / 2
                    ax.annotate(f"L_{nid}", (mx, my), fontsize=7, color=color,
                                ha='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          alpha=0.8, ec='none'))
                elif node.size == 3:
                    pts = [positions[r] for r in robots]
                    for a, b in [(0,1),(1,2),(0,2)]:
                        ax.plot([pts[a][0], pts[b][0]],
                                [pts[a][1], pts[b][1]], '-', color=color, lw=2, zorder=2)
                    cx_l = sum(p[0] for p in pts) / 3
                    cy_l = sum(p[1] for p in pts) / 3
                    ax.annotate(f"p_{nid},β_{nid},q_{nid}", (cx_l, cy_l),
                                fontsize=7, color=color, ha='center',
                                bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                          alpha=0.8, ec='none'))
            else:
                for c in node.children:
                    draw_leaf_edges(c)
        draw_leaf_edges(root)

        # Draw robots
        for ri, pos in positions.items():
            xi, yi = pos[0], pos[1]
            color = leaf_colors.get(ri, 'gray')
            ax.scatter(xi, yi, s=200, color=color, zorder=5,
                       edgecolors='black', linewidths=1.2)
            ax.text(xi, yi + 0.03, f'R{ri}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # State variable list
        sv_lines = ['State variables:'] + [f'  {v}' for v in kin.state_vars]
        ax.text(0.02, 0.98, '\n'.join(sv_lines), transform=ax.transAxes,
                fontsize=7, va='top', family='monospace',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='gray'))

        ax.autoscale_view()
        fig.tight_layout()
        return fig
