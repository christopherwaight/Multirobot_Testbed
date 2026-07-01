#!/usr/bin/env python3
"""
Quick test of the Interactive Cluster Builder Tool
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Check for required packages first
required_packages = ['sympy', 'numpy', 'matplotlib', 'networkx']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("❌ Missing required packages:")
    for package in missing_packages:
        print(f"   - {package}")
    print("\nInstall them with:")
    print(f"   pip install {' '.join(missing_packages)}")
    print("\nOr install all requirements:")
    print("   pip install -r cluster_builder_interactive/requirements.txt")
    sys.exit(1)

try:
    # Test imports
    from cluster_builder_interactive import presets
    from cluster_builder_interactive.jacobian_calculator import JacobianCalculator
    print("✓ Imports successful")

    # Test preset loading
    system, info = presets.two_robot_midpoint()
    print(f"✓ Loaded preset: {info['name']}")

    # Test Jacobian computation
    calc = JacobianCalculator(system)
    J_inv = calc.compute_inverse_jacobian()
    print(f"✓ Computed inverse Jacobian: {J_inv.shape}")

    # Test that it's the expected size
    assert J_inv.shape == (6, 4), f"Expected (6,4) but got {J_inv.shape}"
    print("✓ Matrix dimensions correct")

    print("\n✅ All tests passed! The tool is ready to use.")
    print("\nRun the interactive tool with:")
    print("  python cluster_builder_interactive/main.py")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)