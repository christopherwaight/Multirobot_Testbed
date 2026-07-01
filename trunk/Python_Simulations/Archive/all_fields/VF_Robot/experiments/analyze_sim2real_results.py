"""
Comprehensive analysis of sim2real comparison results.

Analyzes which field approximation (NN or RBF) better matches reality,
how they compare to analytical, and other insights.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sim2real_center_tracking_results/sim2real_center_tracking_results.csv")

df = pd.read_csv(csv_path)

print("=" * 80)
print("SIM-TO-REAL COMPARISON ANALYSIS")
print("=" * 80)
print(f"Total configurations analyzed: {len(df)}")
print(f"  0XX (3-robot): {len(df[df['series'] == '0XX'])}")
print(f"  1XX (4-robot-square): {len(df[df['series'] == '1XX'])}")
print(f"  2XX (4-robot-advanced): {len(df[df['series'] == '2XX'])}")
print()

# ============================================================================
# QUESTION 1: Does NN or RBF match reality more?
# ============================================================================
print("=" * 80)
print("Q1: WHICH SIMULATION MATCHES REALITY BEST?")
print("=" * 80)
print()

# Calculate absolute errors vs real robot
df['analytical_abs_error'] = np.abs(df['analytical_vs_real'])
df['nn_abs_error'] = np.abs(df['nn_vs_real'])
df['rbf_abs_error'] = np.abs(df['rbf_vs_real'])

# Overall statistics
print("Average Absolute Radius Error vs Real Robot (all configs):")
print(f"  Analytical:  {df['analytical_abs_error'].mean():.4f} ± {df['analytical_abs_error'].std():.4f} m")
print(f"  Neural Net:  {df['nn_abs_error'].mean():.4f} ± {df['nn_abs_error'].std():.4f} m")
print(f"  RBF:         {df['rbf_abs_error'].mean():.4f} ± {df['rbf_abs_error'].std():.4f} m")
print()

# Winner overall
errors = {
    'Analytical': df['analytical_abs_error'].mean(),
    'Neural Net': df['nn_abs_error'].mean(),
    'RBF': df['rbf_abs_error'].mean()
}
winner = min(errors, key=errors.get)
print(f"WINNER (Overall): {winner} with {errors[winner]:.4f}m average error")
print()

# By robot configuration
print("By Robot Configuration:")
for series in ['0XX', '1XX', '2XX']:
    subset = df[df['series'] == series]
    if len(subset) > 0:
        robot_type = subset.iloc[0]['robot_type']
        print(f"\n  {series} ({robot_type}):")
        print(f"    Analytical:  {subset['analytical_abs_error'].mean():.4f} ± {subset['analytical_abs_error'].std():.4f} m")
        print(f"    Neural Net:  {subset['nn_abs_error'].mean():.4f} ± {subset['nn_abs_error'].std():.4f} m")
        print(f"    RBF:         {subset['rbf_abs_error'].mean():.4f} ± {subset['rbf_abs_error'].std():.4f} m")

        errors_series = {
            'Analytical': subset['analytical_abs_error'].mean(),
            'Neural Net': subset['nn_abs_error'].mean(),
            'RBF': subset['rbf_abs_error'].mean()
        }
        winner_series = min(errors_series, key=errors_series.get)
        print(f"    WINNER: {winner_series}")

# ============================================================================
# QUESTION 2: How does real-to-analytical compare to NN/RBF approximations?
# ============================================================================
print("\n" + "=" * 80)
print("Q2: SIM-TO-REAL GAP vs APPROXIMATION ERROR")
print("=" * 80)
print()

print("Comparing the 'reality gap' to ML approximation errors:")
print()

# Analytical vs Real (reality gap)
reality_gap = df['analytical_abs_error'].mean()
print(f"Reality Gap (Analytical→Real):        {reality_gap:.4f} m")
print(f"  This is how far real robots deviate from perfect simulation")
print()

# Approximation errors
nn_approx_error = np.abs(df['nn_vs_analytical']).mean()
rbf_approx_error = np.abs(df['rbf_vs_analytical']).mean()
print(f"NN Approximation Error (NN→Analytical):   {nn_approx_error:.4f} m")
print(f"RBF Approximation Error (RBF→Analytical): {rbf_approx_error:.4f} m")
print()

print("Key Insight:")
if nn_approx_error < reality_gap and rbf_approx_error < reality_gap:
    print("  ✓ Both ML approximations are MORE accurate than the sim-to-real gap!")
    print("    This means field approximation error < hardware/model mismatch")
elif nn_approx_error > reality_gap or rbf_approx_error > reality_gap:
    print("  ✗ ML approximation errors EXCEED the sim-to-real gap")
    print("    Field learning is the limiting factor, not hardware")
else:
    print("  ~ ML approximation errors are similar to sim-to-real gap")
print()

# Net effect
print("Net Effect on Sim-to-Real Accuracy:")
analytical_to_real = df['analytical_abs_error'].mean()
nn_to_real = df['nn_abs_error'].mean()
rbf_to_real = df['rbf_abs_error'].mean()

print(f"  Using Analytical field: {analytical_to_real:.4f}m error vs real")
print(f"  Using NN field:         {nn_to_real:.4f}m error vs real")
print(f"  Using RBF field:        {rbf_to_real:.4f}m error vs real")
print()

if nn_to_real < analytical_to_real:
    improvement = analytical_to_real - nn_to_real
    print(f"  ✓ NN IMPROVES prediction by {improvement:.4f}m!")
    print("    (Learned field compensates for unmodeled dynamics)")
elif nn_to_real > analytical_to_real:
    degradation = nn_to_real - analytical_to_real
    print(f"  ✗ NN DEGRADES prediction by {degradation:.4f}m")
    print("    (Approximation error hurts more than it helps)")

if rbf_to_real < analytical_to_real:
    improvement = analytical_to_real - rbf_to_real
    print(f"  ✓ RBF IMPROVES prediction by {improvement:.4f}m!")
elif rbf_to_real > analytical_to_real:
    degradation = rbf_to_real - analytical_to_real
    print(f"  ✗ RBF DEGRADES prediction by {degradation:.4f}m")

# ============================================================================
# QUESTION 3: Other insights
# ============================================================================
print("\n" + "=" * 80)
print("Q3: ADDITIONAL INSIGHTS")
print("=" * 80)
print()

# Insight 1: Scaling with radius
print("INSIGHT 1: How does error scale with target radius?")
print()
for series in ['0XX', '1XX', '2XX']:
    subset = df[df['series'] == series].sort_values('desired_radius')
    if len(subset) > 0:
        robot_type = subset.iloc[0]['robot_type']
        print(f"  {series} ({robot_type}):")

        # Correlation between radius and error
        corr_ana = subset['desired_radius'].corr(subset['analytical_abs_error'])
        corr_nn = subset['desired_radius'].corr(subset['nn_abs_error'])
        corr_rbf = subset['desired_radius'].corr(subset['rbf_abs_error'])

        print(f"    Correlation (radius vs error):")
        print(f"      Analytical: {corr_ana:+.3f}")
        print(f"      NN:         {corr_nn:+.3f}")
        print(f"      RBF:        {corr_rbf:+.3f}")

        if corr_ana < -0.3:
            print(f"    → Error DECREASES with radius (easier to orbit larger circles)")
        elif corr_ana > 0.3:
            print(f"    → Error INCREASES with radius (harder to maintain large orbits)")
        else:
            print(f"    → Error relatively independent of radius")
print()

# Insight 2: Center estimation quality
print("INSIGHT 2: Center Estimation Accuracy")
print()
print("How well do robots estimate the true center at (0,0)?")
print()

for series in ['0XX', '1XX', '2XX']:
    subset = df[df['series'] == series]
    if len(subset) > 0:
        robot_type = subset.iloc[0]['robot_type']
        print(f"  {series} ({robot_type}):")
        print(f"    Analytical: {subset['analytical_avg_center_error'].mean():.4f} ± {subset['analytical_std_center_error'].mean():.4f} m")
        print(f"    NN:         {subset['nn_avg_center_error'].mean():.4f} ± {subset['nn_std_center_error'].mean():.4f} m")
        print(f"    RBF:        {subset['rbf_avg_center_error'].mean():.4f} ± {subset['rbf_std_center_error'].mean():.4f} m")
        print(f"    Real:       {subset['real_avg_center_error'].mean():.4f} ± {subset['real_std_center_error'].mean():.4f} m")
print()

# Insight 3: Estimation stability (std of estimates over time)
print("INSIGHT 3: Estimation Stability (variation in center estimates)")
print()
print("Lower std = more stable/consistent estimation")
print()

for series in ['0XX', '1XX', '2XX']:
    subset = df[df['series'] == series]
    if len(subset) > 0:
        robot_type = subset.iloc[0]['robot_type']
        print(f"  {series} ({robot_type}):")

        # Average std_x and std_y
        ana_std = np.sqrt(subset['analytical_std_x_est'].mean()**2 + subset['analytical_std_y_est'].mean()**2)
        nn_std = np.sqrt(subset['nn_std_x_est'].mean()**2 + subset['nn_std_y_est'].mean()**2)
        rbf_std = np.sqrt(subset['rbf_std_x_est'].mean()**2 + subset['rbf_std_y_est'].mean()**2)
        real_std = np.sqrt(subset['real_std_x_est'].mean()**2 + subset['real_std_y_est'].mean()**2)

        print(f"    Analytical: {ana_std:.4f} m")
        print(f"    NN:         {nn_std:.4f} m")
        print(f"    RBF:        {rbf_std:.4f} m")
        print(f"    Real:       {real_std:.4f} m")

        if nn_std < ana_std:
            print(f"    → NN estimates MORE stable than analytical")
        if rbf_std < ana_std:
            print(f"    → RBF estimates MORE stable than analytical")
print()

# Insight 4: Which configuration works best?
print("INSIGHT 4: Which robot configuration performs best?")
print()

config_performance = {}
for series in ['0XX', '1XX', '2XX']:
    subset = df[df['series'] == series]
    if len(subset) > 0:
        robot_type = subset.iloc[0]['robot_type']
        avg_real_error = subset['real_radius_error'].abs().mean()
        config_performance[robot_type] = avg_real_error
        print(f"  {robot_type:20s}: {avg_real_error:.4f}m average radius error")

best_config = min(config_performance, key=config_performance.get)
print(f"\n  BEST CONFIGURATION: {best_config}")
print()

# Insight 5: Outliers
print("INSIGHT 5: Configurations with largest errors")
print()

# Find top 3 worst performers
df['max_error'] = df[['analytical_abs_error', 'nn_abs_error', 'rbf_abs_error']].max(axis=1)
worst_configs = df.nlargest(3, 'max_error')[['config_name', 'robot_type', 'desired_radius',
                                              'analytical_abs_error', 'nn_abs_error', 'rbf_abs_error']]

print("  Top 3 worst predictions:")
for idx, row in worst_configs.iterrows():
    print(f"    {row['config_name']}: r={row['desired_radius']:.2f}m, errors: Ana={row['analytical_abs_error']:.3f}, NN={row['nn_abs_error']:.3f}, RBF={row['rbf_abs_error']:.3f}")
print()

# Summary recommendations
print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

print("1. BEST FIELD APPROXIMATION:")
if errors['Neural Net'] < errors['RBF'] and errors['Neural Net'] < errors['Analytical']:
    print("   → Use NEURAL NETWORK for best real-world prediction")
elif errors['RBF'] < errors['Neural Net'] and errors['RBF'] < errors['Analytical']:
    print("   → Use RBF INTERPOLATOR for best real-world prediction")
else:
    print("   → Use ANALYTICAL field (ML approximations don't help)")
print()

print("2. FIELD LEARNING QUALITY:")
if nn_approx_error < 0.02 and rbf_approx_error < 0.02:
    print("   → Excellent: ML models accurately approximate analytical field")
elif nn_approx_error < 0.05 and rbf_approx_error < 0.05:
    print("   → Good: ML models reasonably approximate analytical field")
else:
    print("   → Poor: ML models struggle to approximate analytical field")
    print("   → Consider retraining with more data or better architecture")
print()

print("3. SIM-TO-REAL GAP:")
if reality_gap < 0.05:
    print("   → Excellent: Simulation closely matches reality")
elif reality_gap < 0.10:
    print("   → Good: Reasonable sim-to-real transfer")
else:
    print("   → Significant gap: Consider calibrating model parameters")
    print("   → Possible causes: friction, sensor noise, field disturbances")
print()

print("4. ROBOT CONFIGURATION:")
print(f"   → {best_config} performs best overall")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)