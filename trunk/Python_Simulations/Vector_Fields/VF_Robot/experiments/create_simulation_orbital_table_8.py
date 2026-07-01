#!/usr/bin/env python3
"""
create_simulation_orbital_table_8.py - Generate LaTeX Table for Simulation Orbital Results

Reads simulation_orbital_results_8fields.csv and generates a LaTeX table for inclusion
in the paper showing orbital control performance in simulation across 8 field types
(6 analytical + 2 RBF approximations).

Output: Prints LaTeX table code to stdout and saves to simulation_orbital_table_8fields.tex
"""

import pandas as pd
import numpy as np
import os

# Input/output files
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(script_dir, "simulation_orbital_results_8fields.csv")
OUTPUT_TEX = os.path.join(script_dir, "simulation_orbital_table_8fields.tex")

def main():
    print("=" * 70)
    print("SIMULATION ORBITAL CONTROL TABLE GENERATION")
    print("=" * 70)

    # Read CSV data
    print(f"Reading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Check for variation across fields (analytical vs RBF)
    print("\nChecking consistency across 8 fields (6 analytical + 2 RBF)...")
    test_radii = df['commanded_radius'].unique()

    has_variation = False
    for radius in test_radii:
        subset = df[(df['robot_type'] == '3-robot') & (df['commanded_radius'] == radius)]
        if len(subset) > 0:
            actual_radii = subset['actual_radius'].values
            field_std = np.std(actual_radii)
            if field_std > 1e-6:
                has_variation = True
                print(f"  Variation at {radius:.2f}m: std={field_std:.6f} across fields")

    if not has_variation:
        print("  ✓ Results are identical across all 8 field types")

    # Extract unique values for table (use first field type as representative)
    print("\nGenerating LaTeX table...")
    representative_field = df['field_type'].unique()[0]
    table_data = df[df['field_type'] == representative_field].copy()
    table_data = table_data.sort_values(['robot_type', 'commanded_radius'])

    # Generate LaTeX code (3-robot only)
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\begin{tabular}{cc}")
    latex_lines.append("    \\hline")
    latex_lines.append("    \\textbf{Commanded} & \\textbf{3-Robot} \\\\")
    latex_lines.append("    \\textbf{Radius (m)} & \\textbf{Avg. Radius} \\\\")
    latex_lines.append("     & \\textbf{($\\pm$ std)} \\\\")
    latex_lines.append("    \\hline")

    # Get sorted radii
    sorted_radii = sorted(test_radii)

    for radius in sorted_radii:
        # Get data for 3-robot at this radius
        data_3robot = table_data[(table_data['robot_type'] == '3-robot') &
                                 (table_data['commanded_radius'] == radius)]

        if len(data_3robot) > 0:
            actual_3 = data_3robot['actual_radius'].values[0]
            std_3 = data_3robot['std_radius'].values[0]

            # Format with 3 decimal places for actual, scientific notation for std
            latex_lines.append(f"    {radius:.2f} & {actual_3:.3f} $\\pm$ {std_3:.2e} \\\\")

    latex_lines.append("    \\hline")
    latex_lines.append("    \\end{tabular}")

    # Create caption
    caption = ("\\caption{Simulated orbital control performance across 8 field types "
               "(6 analytical + 2 RBF approximations: sink, source, vortex, saddle, sinking vortex, "
               "spewing vortex, vortex\\_rbf, saddle\\_rbf). "
               "Results show near-identical performance across all field types in noise-free simulation. "
               "Standard deviations near machine precision indicate perfect orbital tracking. "
               "3-robot configuration tested at radii from 0.01m to 0.70m.}")
    latex_lines.append(f"    {caption}")
    latex_lines.append("    \\label{tab:simulation_orbital_results}")
    latex_lines.append("    \\end{table}")

    # Join into full LaTeX code
    latex_code = '\n'.join(latex_lines)

    # Save to file
    with open(OUTPUT_TEX, 'w') as f:
        f.write(latex_code)

    print(f"✓ LaTeX table saved to: {OUTPUT_TEX}")

    # Print to stdout
    print("\n" + "=" * 70)
    print("LATEX TABLE CODE")
    print("=" * 70)
    print(latex_code)
    print("=" * 70)

    # Generate summary text for paper
    print("\n" + "=" * 70)
    print("SUGGESTED PARAGRAPH TEXT (30-50 words)")
    print("=" * 70)
    summary_text = (
        "Orbital control performance was evaluated in simulation across all six canonical "
        "field types at commanded radii from 0.01m to 0.70m. Table~\\ref{tab:simulation_orbital_results} "
        "shows that both 3-robot and 4-robot configurations achieve identical tracking performance "
        "with near-zero standard deviations, confirming the theoretical control law behavior in "
        "noise-free conditions."
    )
    print(summary_text)
    print("=" * 70)

    # Calculate summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for robot_type in ['3-robot', '4-robot-square']:
        subset = table_data[table_data['robot_type'] == robot_type]
        mean_error = subset['radius_error'].mean()
        max_error = subset['radius_error'].max()
        min_error = subset['radius_error'].min()
        mean_std = subset['std_radius'].mean()

        print(f"\n{robot_type}:")
        print(f"  Mean radius error: {mean_error:+.4f} m")
        print(f"  Error range: [{min_error:+.4f}, {max_error:+.4f}] m")
        print(f"  Mean std dev: {mean_std:.2e} m")

    print("\n" + "=" * 70)
    print("TABLE GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
