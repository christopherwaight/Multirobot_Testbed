"""
Explain why dual Jacobian directions cancel at saddle points.
"""

import numpy as np
import sys
import io
import contextlib
import matplotlib.pyplot as plt

sys.path.append('.')

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalField
from src.fields.environments.Saddle import saddle1
import src.control.quad_primitives as qcp


def analyze_saddle_geometry():
    """
    Show what a saddle point looks like and why triangles see opposite things.
    """
    print("="*80)
    print("SADDLE POINT GEOMETRY ANALYSIS")
    print("="*80)
    print()

    # Sample the saddle field at various points
    print("Saddle field at key points:")
    print("-"*80)

    test_points = [
        (0.3, 0.0, "Right on X-axis"),
        (-0.3, 0.0, "Left on X-axis"),
        (0.0, 0.3, "Up on Y-axis"),
        (0.0, -0.3, "Down on Y-axis"),
        (0.2, 0.2, "Diagonal +/+"),
        (-0.2, -0.2, "Diagonal -/-"),
    ]

    for x, y, label in test_points:
        u, v = saddle1(x, y)
        magnitude = np.sqrt(u**2 + v**2)
        angle_deg = np.arctan2(v, u) * 180 / np.pi
        print(f"{label:20s} ({x:+.1f}, {y:+.1f}): field=({u:+.3f}, {v:+.3f})  "
              f"mag={magnitude:.3f}  angle={angle_deg:+6.1f}°")

    print()
    print("="*80)
    print("DUAL JACOBIAN ESTIMATION AT SADDLE")
    print("="*80)
    print()

    # Create a cluster near saddle center
    field = AnalyticalField(saddle1)
    with contextlib.redirect_stdout(io.StringIO()):
        cluster = QuadCluster('config/formations/quad_square_default.yaml', field)

    # Reset to a position near saddle (not exactly at it)
    cluster.reset(x_c=0.1, y_c=0.1)

    # Get robot positions
    x1, y1, x2, y2, x3, y3, x4, y4 = cluster.get_robot_positions()

    print(f"Cluster centroid: (0.1, 0.1)")
    print(f"Robot positions:")
    print(f"  R1: ({x1:+.3f}, {y1:+.3f})")
    print(f"  R2: ({x2:+.3f}, {y2:+.3f})")
    print(f"  R3: ({x3:+.3f}, {y3:+.3f})")
    print(f"  R4: ({x4:+.3f}, {y4:+.3f})")
    print()

    # Analyze top triangle (1,2,3)
    print("TOP TRIANGLE (robots 1, 2, 3):")
    print("-"*80)
    curl_top, div_top, _, _, _, _ = qcp.calculate_jacobian_top(cluster)
    print(f"  Curl: {curl_top:.6f}")
    print(f"  Divergence: {div_top:+.6f}")

    if div_top > 0:
        print(f"  → POSITIVE divergence = EXPANDING (source-like)")
        print(f"  → Direction estimate: AGAINST flow (toward where it's expanding from)")
    else:
        print(f"  → NEGATIVE divergence = CONTRACTING (sink-like)")
        print(f"  → Direction estimate: WITH flow (toward where it's contracting to)")

    dir1 = qcp.calculate_center_direction_top(cluster)
    if dir1 is not None:
        print(f"  Direction vector: ({dir1[0]:+.3f}, {dir1[1]:+.3f})")
    else:
        print(f"  Direction vector: None (stagnation)")

    print()

    # Analyze bottom triangle (2,3,4)
    print("BOTTOM TRIANGLE (robots 2, 3, 4):")
    print("-"*80)
    curl_bot, div_bot, _, _, _, _ = qcp.calculate_jacobian_bottom(cluster)
    print(f"  Curl: {curl_bot:.6f}")
    print(f"  Divergence: {div_bot:+.6f}")

    if div_bot > 0:
        print(f"  → POSITIVE divergence = EXPANDING (source-like)")
        print(f"  → Direction estimate: AGAINST flow (toward where it's expanding from)")
    else:
        print(f"  → NEGATIVE divergence = CONTRACTING (sink-like)")
        print(f"  → Direction estimate: WITH flow (toward where it's contracting to)")

    dir2 = qcp.calculate_center_direction_bottom(cluster)
    if dir2 is not None:
        print(f"  Direction vector: ({dir2[0]:+.3f}, {dir2[1]:+.3f})")
    else:
        print(f"  Direction vector: None (stagnation)")

    print()
    print("="*80)
    print("WHY THEY CANCEL")
    print("="*80)
    print()

    if dir1 is not None and dir2 is not None:
        combined = (dir1 + dir2) / 2
        dot_product = np.dot(dir1, dir2)
        angle = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi

        print(f"Direction 1: ({dir1[0]:+.3f}, {dir1[1]:+.3f})")
        print(f"Direction 2: ({dir2[0]:+.3f}, {dir2[1]:+.3f})")
        print(f"Dot product: {dot_product:+.3f}")
        print(f"Angle between: {angle:.1f}°")
        print(f"Combined (avg): ({combined[0]:+.6f}, {combined[1]:+.6f})")
        print(f"Combined magnitude: {np.linalg.norm(combined):.10f}")
        print()

        if angle > 170:
            print("✗ OPPOSITE DIRECTIONS (angle ≈ 180°)")
            print("  The two triangles see opposite divergence signs!")
            print("  One sees expansion, other sees contraction")
            print("  → Directions cancel → Magnitude ≈ 0 → Division by zero → NaN")
        elif angle > 100:
            print("⚠ NEARLY OPPOSITE (angle > 100°)")
            print("  The triangles disagree significantly")
            print("  → Combined direction is weak → May cause instability")
        else:
            print("✓ Directions generally agree")

    print()
    print("="*80)
    print("THE ROOT CAUSE")
    print("="*80)
    print()
    print("At a saddle point:")
    print("  • Flow EXPANDS along one principal axis (e.g., x-axis)")
    print("  • Flow CONTRACTS along perpendicular axis (e.g., y-axis)")
    print()
    print("When quad formation straddles both axes:")
    print("  • Top triangle samples mostly EXPANSION → estimates 'center is there →'")
    print("  • Bottom triangle samples mostly CONTRACTION → estimates 'center is there ←'")
    print()
    print("Result:")
    print("  • Direction vectors point OPPOSITE ways")
    print("  • Average = (→) + (←) = 0")
    print("  • Normalizing zero vector: 0/0 = NaN")
    print()


if __name__ == "__main__":
    analyze_saddle_geometry()
