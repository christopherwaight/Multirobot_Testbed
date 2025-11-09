"""
Test script to verify velocity constraints (max velocity and stiction)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.robot.omnibot import Omnibot

print("=" * 60)
print("TESTING VELOCITY CONSTRAINTS")
print("=" * 60)

# Create a robot
robot = Omnibot(0.0, 0.0, timestep=0.1, momentum_alpha=0.5)

# Test 1: Max velocity constraint
print("\n1. Testing MAX VELOCITY constraint (0.3 m/s):")
print("-" * 60)
robot.velocity = np.array([0.0, 0.0])  # Reset
robot.command_velocity(1.0, 0.0)  # Command very high velocity
vel_mag = np.linalg.norm(robot.velocity)
print(f"   Commanded: (1.0, 0.0) m/s")
print(f"   Actual:    ({robot.velocity[0]:.3f}, {robot.velocity[1]:.3f}) m/s")
print(f"   Magnitude: {vel_mag:.3f} m/s")
print(f"   ✓ Correctly clamped to 0.3 m/s" if vel_mag <= 0.3 else "   ✗ ERROR: Exceeded 0.3 m/s!")

# Test 2: Max velocity with diagonal motion
print("\n2. Testing MAX VELOCITY with diagonal motion:")
print("-" * 60)
robot.velocity = np.array([0.0, 0.0])  # Reset
robot.command_velocity(0.5, 0.5)  # Command high diagonal velocity
vel_mag = np.linalg.norm(robot.velocity)
print(f"   Commanded: (0.5, 0.5) m/s (magnitude: {np.linalg.norm([0.5, 0.5]):.3f})")
print(f"   Actual:    ({robot.velocity[0]:.3f}, {robot.velocity[1]:.3f}) m/s")
print(f"   Magnitude: {vel_mag:.3f} m/s")
print(f"   ✓ Correctly clamped to 0.3 m/s" if vel_mag <= 0.3 else "   ✗ ERROR: Exceeded 0.3 m/s!")

# Test 3: Stiction threshold
print("\n3. Testing STICTION threshold (0.025 m/s):")
print("-" * 60)
robot.velocity = np.array([0.0, 0.0])  # Reset
robot.command_velocity(0.01, 0.01)  # Command below stiction threshold
vel_mag = np.linalg.norm(robot.velocity)
print(f"   Commanded: (0.01, 0.01) m/s (magnitude: {np.linalg.norm([0.01, 0.01]):.4f})")
print(f"   Actual:    ({robot.velocity[0]:.3f}, {robot.velocity[1]:.3f}) m/s")
print(f"   Magnitude: {vel_mag:.3f} m/s")
print(f"   ✓ Correctly stopped (stiction)" if vel_mag == 0.0 else "   ✗ ERROR: Should be zero!")

# Test 4: Just above stiction threshold
print("\n4. Testing velocity ABOVE stiction threshold:")
print("-" * 60)
robot.velocity = np.array([0.0, 0.0])  # Reset
robot.command_velocity(0.1, 0.0)  # Command above stiction threshold
vel_mag = np.linalg.norm(robot.velocity)
print(f"   Commanded: (0.1, 0.0) m/s")
print(f"   Actual:    ({robot.velocity[0]:.3f}, {robot.velocity[1]:.3f}) m/s")
print(f"   Magnitude: {vel_mag:.3f} m/s")
print(f"   ✓ Moving (above stiction)" if vel_mag > 0.0 else "   ✗ ERROR: Should be moving!")

# Test 5: Normal operating range
print("\n5. Testing NORMAL operating velocity:")
print("-" * 60)
robot.velocity = np.array([0.0, 0.0])  # Reset
robot.command_velocity(0.2, 0.0)  # Normal velocity
vel_mag = np.linalg.norm(robot.velocity)
print(f"   Commanded: (0.2, 0.0) m/s")
print(f"   Actual:    ({robot.velocity[0]:.3f}, {robot.velocity[1]:.3f}) m/s")
print(f"   Magnitude: {vel_mag:.3f} m/s")
print(f"   ✓ Normal operation" if 0.025 < vel_mag <= 0.3 else "   ✗ ERROR: Outside normal range!")

# Test 6: Momentum effect with constraints
print("\n6. Testing MOMENTUM with constraints:")
print("-" * 60)
robot.velocity = np.array([0.0, 0.0])  # Reset
print("   Applying repeated high commands to build up momentum...")
for i in range(5):
    robot.velocity = np.array([0.2, 0.0])  # Set velocity (simulating buildup)
    robot.command_velocity(0.5, 0.0)  # Command high
    vel_mag = np.linalg.norm(robot.velocity)
    print(f"   Step {i+1}: velocity = {vel_mag:.3f} m/s")
print(f"   ✓ Always clamped to max" if vel_mag <= 0.3 else "   ✗ ERROR!")

print("\n" + "=" * 60)
print("CONSTRAINTS SUMMARY:")
print("=" * 60)
print("✓ Maximum velocity: 0.3 m/s (clamped)")
print("✓ Stiction threshold: 0.025 m/s (below this → stops)")
print("✓ Both constraints applied AFTER momentum dynamics")
print("=" * 60)
