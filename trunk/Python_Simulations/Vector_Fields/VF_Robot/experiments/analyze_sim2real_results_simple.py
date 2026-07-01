"""
Comprehensive analysis of sim2real comparison results (no pandas dependency).
"""
import numpy as np
import csv
import os

# Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "sim2real_center_tracking_results/sim2real_center_tracking_results.csv")

# Read CSV manually
data = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key, value in row.items():
            if key not in data:
                data[key] = []
            try:
                data[key].append(float(value))
            except ValueError:
                data[key].append(value)

# Convert to numpy arrays for numerical columns
def get_subset(data, series_filter):
    """Get subset of data matching series filter."""
    indices = [i for i, s in enumerate(data['series']) if s == series_filter]
    subset = {}
    for key in data:
        subset[key] = [data[key][i] for i in indices]
    return subset

print("=" * 80)
print("SIM-TO-REAL COMPARISON ANALYSIS")
print("=" * 80)
print(f"Total configurations analyzed: {len(data['config_name'])}")

series_counts = {}
for s in data['series']:
    series_counts[s] = series_counts.get(s, 0) + 1
for series, count in sorted(series_counts.items()):
    print(f"  {series}: {count}")
print()

# ============================================================================
# QUESTION 1: Does NN or RBF match reality more?
# ============================================================================
print("=" * 80)
print("Q1: WHICH SIMULATION MATCHES REALITY BEST?")
print("=" * 80)
print()

# Calculate absolute errors
analytical_abs_error = [abs(x) for x in data['analytical_vs_real']]
nn_abs_error = [abs(x) for x in data['nn_vs_real']]
rbf_abs_error = [abs(x) for x in data['rbf_vs_real']]

# Overall statistics
print("Average Absolute Radius Error vs Real Robot (all configs):")
print(f"  Analytical:  {np.mean(analytical_abs_error):.4f} ± {np.std(analytical_abs_error):.4f} m")
print(f"  Neural Net:  {np.mean(nn_abs_error):.4f} ± {np.std(nn_abs_error):.4f} m")
print(f"  RBF:         {np.mean(rbf_abs_error):.4f} ± {np.std(rbf_abs_error):.4f} m")
print()

# Winner
errors = {
    'Analytical': np.mean(analytical_abs_error),
    'Neural Net': np.mean(nn_abs_error),
    'RBF': np.mean(rbf_abs_error)
}
winner = min(errors, key=errors.get)
print(f"WINNER (Overall): {winner} with {errors[winner]:.4f}m average error")
print()

# By robot configuration
print("By Robot Configuration:")
for series in ['0XX', '1XX', '2XX']:
    subset = get_subset(data, series)
    if len(subset['series']) > 0:
        robot_type = subset['robot_type'][0]
        ana_err = [abs(x) for x in subset['analytical_vs_real']]
        nn_err = [abs(x) for x in subset['nn_vs_real']]
        rbf_err = [abs(x) for x in subset['rbf_vs_real']]

        print(f"\n  {series} ({robot_type}):")
        print(f"    Analytical:  {np.mean(ana_err):.4f} ± {np.std(ana_err):.4f} m")
        print(f"    Neural Net:  {np.mean(nn_err):.4f} ± {np.std(nn_err):.4f} m")
        print(f"    RBF:         {np.mean(rbf_err):.4f} ± {np.std(rbf_err):.4f} m")

        errors_series = {
            'Analytical': np.mean(ana_err),
            'Neural Net': np.mean(nn_err),
            'RBF': np.mean(rbf_err)
        }
        winner_series = min(errors_series, key=errors_series.get)
        print(f"    WINNER: {winner_series} ({errors_series[winner_series]:.4f}m)")

# ============================================================================
# QUESTION 2: Reality gap vs approximation error
# ============================================================================
print("\n" + "=" * 80)
print("Q2: SIM-TO-REAL GAP vs APPROXIMATION ERROR")
print("=" * 80)
print()

print("Comparing the 'reality gap' to ML approximation errors:")
print()

# Reality gap (analytical to real)
reality_gap = np.mean(analytical_abs_error)
print(f"Reality Gap (Analytical→Real):        {reality_gap:.4f} m")
print(f"  This is how far real robots deviate from perfect simulation")
print()

# Approximation errors (ML to analytical)
nn_approx_error = np.mean([abs(x) for x in data['nn_vs_analytical']])
rbf_approx_error = np.mean([abs(x) for x in data['rbf_vs_analytical']])
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
analytical_to_real = np.mean(analytical_abs_error)
nn_to_real = np.mean(nn_abs_error)
rbf_to_real = np.mean(rbf_abs_error)

print(f"  Using Analytical field: {analytical_to_real:.4f}m error vs real")
print(f"  Using NN field:         {nn_to_real:.4f}m error vs real")
print(f"  Using RBF field:        {rbf_to_real:.4f}m error vs real")
print()

if nn_to_real < analytical_to_real:
    improvement = analytical_to_real - nn_to_real
    print(f"  ✓ NN IMPROVES prediction by {improvement:.4f}m ({improvement/analytical_to_real*100:.1f}%)!")
    print("    (Learned field compensates for unmodeled dynamics)")
elif nn_to_real > analytical_to_real:
    degradation = nn_to_real - analytical_to_real
    print(f"  ✗ NN DEGRADES prediction by {degradation:.4f}m ({degradation/analytical_to_real*100:.1f}%)")
    print("    (Approximation error hurts more than it helps)")

if rbf_to_real < analytical_to_real:
    improvement = analytical_to_real - rbf_to_real
    print(f"  ✓ RBF IMPROVES prediction by {improvement:.4f}m ({improvement/analytical_to_real*100:.1f}%)!")
elif rbf_to_real > analytical_to_real:
    degradation = rbf_to_real - analytical_to_real
    print(f"  ✗ RBF DEGRADES prediction by {degradation:.4f}m ({degradation/analytical_to_real*100:.1f}%)")
print()

# ============================================================================
# QUESTION 3: Other insights
# ============================================================================
print("=" * 80)
print("Q3: ADDITIONAL INSIGHTS")
print("=" * 80)
print()

# Insight 1: Radius scaling
print("INSIGHT 1: How does error scale with target radius?")
print()
for series in ['0XX', '1XX', '2XX']:
    subset = get_subset(data, series)
    if len(subset['series']) > 0:
        robot_type = subset['robot_type'][0]
        print(f"  {series} ({robot_type}):")

        # Manual correlation calculation
        radius = np.array(subset['desired_radius'])
        ana_err_arr = np.array([abs(x) for x in subset['analytical_vs_real']])
        nn_err_arr = np.array([abs(x) for x in subset['nn_vs_real']])
        rbf_err_arr = np.array([abs(x) for x in subset['rbf_vs_real']])

        corr_ana = np.corrcoef(radius, ana_err_arr)[0, 1]
        corr_nn = np.corrcoef(radius, nn_err_arr)[0, 1]
        corr_rbf = np.corrcoef(radius, rbf_err_arr)[0, 1]

        print(f"    Correlation (radius vs error):")
        print(f"      Analytical: {corr_ana:+.3f}")
        print(f"      NN:         {corr_nn:+.3f}")
        print(f"      RBF:        {corr_rbf:+.3f}")

        if corr_ana < -0.3:
            print(f"    → Error DECREASES with radius (easier at large radii)")
        elif corr_ana > 0.3:
            print(f"    → Error INCREASES with radius (harder at large radii)")
        else:
            print(f"    → Error relatively independent of radius")
print()

# Insight 2: Center estimation
print("INSIGHT 2: Center Estimation Accuracy")
print()
print("Average distance of estimated center from true center (0,0):")
print()

for series in ['0XX', '1XX', '2XX']:
    subset = get_subset(data, series)
    if len(subset['series']) > 0:
        robot_type = subset['robot_type'][0]
        print(f"  {series} ({robot_type}):")
        print(f"    Analytical: {np.mean(subset['analytical_avg_center_error']):.4f} m")
        print(f"    NN:         {np.mean(subset['nn_avg_center_error']):.4f} m")
        print(f"    RBF:        {np.mean(subset['rbf_avg_center_error']):.4f} m")
        print(f"    Real:       {np.mean(subset['real_avg_center_error']):.4f} m")
print()

# Insight 3: Which config works best?
print("INSIGHT 3: Which robot configuration performs best?")
print()

for series in ['0XX', '1XX', '2XX']:
    subset = get_subset(data, series)
    if len(subset['series']) > 0:
        robot_type = subset['robot_type'][0]
        real_err = np.mean([abs(x) for x in subset['real_radius_error']])
        print(f"  {robot_type:25s}: {real_err:.4f}m average |radius error|")
print()

# Insight 4: Stability comparison
print("INSIGHT 4: Estimation Stability (std of center estimates over time)")
print()
print("Lower values = more stable/consistent estimation")
print()

for series in ['0XX', '1XX', '2XX']:
    subset = get_subset(data, series)
    if len(subset['series']) > 0:
        robot_type = subset['robot_type'][0]
        print(f"  {series} ({robot_type}):")

        # Combined std_x and std_y
        ana_std = np.mean(subset['analytical_std_center_error'])
        nn_std = np.mean(subset['nn_std_center_error'])
        rbf_std = np.mean(subset['rbf_std_center_error'])
        real_std = np.mean(subset['real_std_center_error'])

        print(f"    Analytical: {ana_std:.4f} m")
        print(f"    NN:         {nn_std:.4f} m")
        print(f"    RBF:        {rbf_std:.4f} m")
        print(f"    Real:       {real_std:.4f} m")

        if nn_std < ana_std:
            print(f"    → NN estimates are MORE stable than analytical")
        if rbf_std < ana_std:
            print(f"    → RBF estimates are MORE stable than analytical")
print()

# Summary
print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

print("1. BEST FIELD FOR REAL-WORLD PREDICTION:")
best_field = min(errors, key=errors.get)
print(f"   → Use {best_field.upper()} (error: {errors[best_field]:.4f}m)")
print()

print("2. FIELD LEARNING QUALITY:")
avg_ml_error = (nn_approx_error + rbf_approx_error) / 2
if avg_ml_error < 0.02:
    print("   ✓ Excellent: ML models accurately approximate analytical field")
elif avg_ml_error < 0.05:
    print("   ✓ Good: ML models reasonably approximate analytical field")
else:
    print("   ✗ Poor: ML models struggle - consider retraining")
print(f"   Average ML approximation error: {avg_ml_error:.4f}m")
print()

print("3. SIM-TO-REAL GAP:")
if reality_gap < 0.05:
    print("   ✓ Excellent: Simulation closely matches reality")
elif reality_gap < 0.10:
    print("   ✓ Good: Reasonable sim-to-real transfer")
else:
    print("   ⚠ Significant gap: Consider model calibration")
print(f"   Reality gap: {reality_gap:.4f}m")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)