"""
Saddle Point Fine-Tuning Experiment - Multi-Field & Stability Analysis

This experiment:
1. Fine-tunes parameters around the best results from coarse tuning
2. Tests on MULTIPLE fields (bimodal_gaussian and hyperbolic_saddle)
3. Tracks FORMATION STABILITY (theta_c oscillation at convergence)
4. Tests more starting positions for comprehensive coverage
5. ~5000 trials total (4x the previous experiment)

Success criteria:
- Position: within 0.5m of origin
- Stability: low theta_c oscillation at end (smooth convergence)
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

from src.robot.quad_cluster import QuadCluster
from src.fields.field_types import AnalyticalScalarField
from src.fields.environments.Scalar_Fields import bimodal_gaussian, hyperbolic_saddle
import src.control.quad_primitives as qcp


def compute_theta_stability(theta_history, window=100):
    """
    Measure formation orientation stability.
    
    Args:
        theta_history: list of theta_c values over time
        window: last N values to analyze
    
    Returns:
        {
            'mean_theta': average theta in window
            'std_theta': standard deviation (lower = more stable)
            'max_oscillation': max change between consecutive steps
            'is_stable': bool (std < 0.1 rad ~ 5.7 degrees)
        }
    """
    if len(theta_history) < window:
        window = len(theta_history)
    
    recent = np.array(theta_history[-window:])
    
    # Unwrap angles to handle 0/2π discontinuity
    unwrapped = np.unwrap(recent)
    
    return {
        'mean_theta': np.mean(unwrapped),
        'std_theta': np.std(unwrapped),
        'max_oscillation': np.max(np.abs(np.diff(unwrapped))),
        'is_stable': np.std(unwrapped) < 0.1
    }


def run_single_trial(start_pos, field_func, field_name, translation_gain, rotation_gain, 
                     max_speed, max_omega, sim_steps=3000):
    """
    Run a single trial of saddle point navigation.
    
    Returns:
        {
            'final_pos': (x, y),
            'final_distance': distance from origin,
            'success': bool (within 0.5m),
            'field_name': name of field tested,
            'theta_stability': stability metrics,
            'translation_gain', 'rotation_gain', 'max_speed': parameters
            'start_pos': starting position,
            'initial_theta': initial formation orientation
        }
    """
    # Create field
    field = AnalyticalScalarField(field_func)
    
    # Create cluster at starting position
    cluster = QuadCluster("config/formations/quad_square.yaml", field)
    cluster.reset(x_c=start_pos[0], y_c=start_pos[1])
    
    # Capture initial formation orientation
    initial_formation = cluster.get_current_formation()
    initial_theta = initial_formation.get('theta_c', 0)
    
    # Create wrapped control primitive with parameters
    def scalar_newton_tuned(cl):
        return qcp.scalar_newton_with_rotation(
            cl, 
            translation_gain=translation_gain,
            rotation_gain=rotation_gain,
            max_speed=max_speed,
            max_omega=max_omega
        )
    
    trajectory = [np.array(start_pos)]
    theta_history = [initial_theta]
    
    # Run simulation
    for step in range(sim_steps):
        try:
            # Apply command (move cluster)
            cluster.move(scalar_newton_tuned)
            
            # Store position and orientation
            cx, cy = cluster.get_centroid()
            trajectory.append(np.array([cx, cy]))
            
            # Track theta_c
            formation = cluster.get_current_formation()
            theta_c = formation.get('theta_c', 0)
            theta_history.append(theta_c)
            
        except Exception as e:
            break
    
    # Get final position
    cx, cy = cluster.get_centroid()
    final_pos = np.array([cx, cy])
    final_distance = np.linalg.norm(final_pos)
    success = final_distance < 0.5
    
    # Analyze theta stability
    theta_stability = compute_theta_stability(theta_history)
    
    return {
        'final_pos': final_pos,
        'final_distance': final_distance,
        'success': success,
        'field_name': field_name,
        'theta_stability': theta_stability,
        'trajectory': np.array(trajectory),
        'theta_history': np.array(theta_history),
        'start_pos': start_pos,
        'initial_theta': initial_theta,
        'translation_gain': translation_gain,
        'rotation_gain': rotation_gain,
        'max_speed': max_speed,
    }


def run_fine_tuning_experiment():
    """
    Run fine-tuning experiment with multiple fields and stability tracking.
    """
    print("=" * 80)
    print("SADDLE POINT FINE-TUNING EXPERIMENT - MULTI-FIELD & STABILITY")
    print("=" * 80)
    
    # Fine-tune around best results: trans_gain=1.5, rot_gain around 0.1-0.3
    translation_gains = [1.2, 1.5, 1.8]
    rotation_gains = [0.05, 0.15, 0.25, 0.35]  # Finer resolution around optimum
    max_speeds = [0.20, 0.25, 0.30]
    
    # More comprehensive starting positions
    start_distances = [0.5, 1.0, 1.5, 2.0]
    start_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16 directions (more than before)
    
    start_positions = []
    for distance in start_distances:
        for angle in start_angles:
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            start_positions.append((x, y))
    
    # Multiple fields to test
    fields = [
        (bimodal_gaussian, "bimodal_gaussian"),
        (hyperbolic_saddle, "hyperbolic_saddle"),
    ]
    
    print(f"\nExperiment Configuration:")
    print(f"  Fields tested: {len(fields)} (bimodal, saddle)")
    print(f"  Starting positions: {len(start_positions)} (4 distances × 16 angles)")
    print(f"  Parameter combinations: {len(translation_gains)} × {len(rotation_gains)} × {len(max_speeds)} = {len(translation_gains)*len(rotation_gains)*len(max_speeds)}")
    print(f"  Total trials: {len(start_positions) * len(translation_gains) * len(rotation_gains) * len(max_speeds) * len(fields)}")
    print(f"  Simulation steps per trial: 3000")
    print(f"  Estimated runtime: 2+ hours\n")
    
    results = []
    total_trials = 0
    successful_trials = 0
    stable_convergence = 0
    
    start_time = time.time()
    
    # Sweep through parameters
    param_combos = list(product(
        translation_gains,
        rotation_gains,
        max_speeds
    ))
    
    print(f"Starting trials...")
    
    for field_func, field_name in fields:
        for param_idx, (tg, rg, ms) in enumerate(param_combos):
            for pos_idx, start_pos in enumerate(start_positions):
                total_trials += 1
                
                # Run trial
                result = run_single_trial(
                    start_pos, field_func, field_name, tg, rg, ms, max_omega=0.1, sim_steps=3000
                )
                results.append(result)
                
                if result['success']:
                    successful_trials += 1
                    if result['theta_stability']['is_stable']:
                        stable_convergence += 1
                
                # Progress update every 100 trials
                if total_trials % 100 == 0:
                    elapsed = time.time() - start_time
                    success_rate = successful_trials / total_trials * 100
                    stable_rate = stable_convergence / max(1, successful_trials) * 100
                    print(f"  Trial {total_trials}: {success_rate:.1f}% success | {stable_rate:.1f}% stable | Elapsed: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    final_success_rate = successful_trials / total_trials * 100
    stable_rate = stable_convergence / max(1, successful_trials) * 100
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total trials: {total_trials}")
    print(f"Successful trials: {successful_trials}")
    print(f"Overall success rate: {final_success_rate:.1f}%")
    print(f"Stable convergence rate (of successful): {stable_rate:.1f}%")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Analyze results
    analyze_results(results, fields, total_time, project_root)
    
    return results


def analyze_results(results, fields, total_time, project_root):
    """
    Analyze results including stability metrics.
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    # 1. Results by field
    print("\n1. SUCCESS RATE BY FIELD:")
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        successes = sum(1 for r in field_results if r['success'])
        rate = successes / len(field_results) * 100
        print(f"  {field_name}: {rate:.1f}% ({successes}/{len(field_results)})")
    
    # 2. Best parameter combinations for each field
    print("\n2. BEST PARAMETERS BY FIELD:")
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        param_success = {}
        for result in field_results:
            key = (result['translation_gain'], result['rotation_gain'], result['max_speed'])
            if key not in param_success:
                param_success[key] = {'success': 0, 'stable': 0, 'total': 0}
            param_success[key]['total'] += 1
            if result['success']:
                param_success[key]['success'] += 1
                if result['theta_stability']['is_stable']:
                    param_success[key]['stable'] += 1
        
        sorted_params = sorted(
            param_success.items(),
            key=lambda x: x[1]['success'] / x[1]['total'],
            reverse=True
        )
        
        print(f"\n  {field_name}:")
        for i, (params, counts) in enumerate(sorted_params[:3]):
            tg, rg, ms = params
            success_rate = counts['success'] / counts['total'] * 100
            stable_rate = counts['stable'] / max(1, counts['success']) * 100
            print(f"    {i+1}. TG={tg}, RG={rg}, MS={ms}: {success_rate:.1f}% success, {stable_rate:.1f}% stable")
    
    # 3. Stability analysis
    print("\n3. STABILITY ANALYSIS (Formation Orientation):")
    successful = [r for r in results if r['success']]
    stable = [r for r in successful if r['theta_stability']['is_stable']]
    
    print(f"  Successful trials with stable theta_c: {len(stable)}/{len(successful)} ({len(stable)/len(successful)*100:.1f}%)")
    
    if successful:
        std_theta_values = [r['theta_stability']['std_theta'] for r in successful]
        print(f"  Average std(theta) in final 100 steps: {np.mean(std_theta_values):.4f} rad ({np.degrees(np.mean(std_theta_values)):.2f}°)")
        print(f"  Median std(theta): {np.median(std_theta_values):.4f} rad ({np.degrees(np.median(std_theta_values)):.2f}°)")
        print(f"  Max std(theta): {np.max(std_theta_values):.4f} rad ({np.degrees(np.max(std_theta_values)):.2f}°)")
    
    # 4. Create comparison plots and basin heatmaps
    create_comparison_plots(results, fields, project_root)
    
    # 5. Save detailed results log
    save_results_log(results, fields, total_time, project_root)


def save_results_log(results, fields, total_time, project_root):
    """
    Save detailed results and recommendations to a log file.
    """
    log_path = os.path.join(project_root, 'saddle_point_fine_tuning_results.log')
    
    with open(log_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SADDLE POINT FINE-TUNING EXPERIMENT - DETAILED RESULTS LOG\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Experiment Runtime: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"Total Trials: {len(results)}\n")
        f.write(f"Fields Tested: {', '.join([name for _, name in fields])}\n\n")
        
        # Overall statistics
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        total_success = sum(1 for r in results if r['success'])
        f.write(f"Total Successful: {total_success}/{len(results)} ({total_success/len(results)*100:.1f}%)\n\n")
        
        # Per-field analysis
        f.write("="*80 + "\n")
        f.write("RESULTS BY FIELD\n")
        f.write("="*80 + "\n\n")
        
        for field_func, field_name in fields:
            field_results = [r for r in results if r['field_name'] == field_name]
            field_success = sum(1 for r in field_results if r['success'])
            field_success_rate = field_success / len(field_results) * 100
            
            f.write(f"\n{field_name.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Success Rate: {field_success_rate:.1f}% ({field_success}/{len(field_results)})\n\n")
            
            # Best parameters for this field
            param_stats = {}
            for result in field_results:
                key = (result['translation_gain'], result['rotation_gain'], result['max_speed'])
                if key not in param_stats:
                    param_stats[key] = {
                        'success': 0, 'stable': 0, 'total': 0, 
                        'avg_distance': 0, 'std_theta_values': []
                    }
                param_stats[key]['total'] += 1
                if result['success']:
                    param_stats[key]['success'] += 1
                    if result['theta_stability']['is_stable']:
                        param_stats[key]['stable'] += 1
                    param_stats[key]['avg_distance'] += result['final_distance']
                    param_stats[key]['std_theta_values'].append(result['theta_stability']['std_theta'])
            
            # Sort by success rate
            sorted_params = sorted(
                param_stats.items(),
                key=lambda x: x[1]['success'] / x[1]['total'],
                reverse=True
            )
            
            f.write("Top 5 Parameter Combinations:\n")
            for i, (params, stats) in enumerate(sorted_params[:5]):
                tg, rg, ms = params
                success_rate = stats['success'] / stats['total'] * 100
                stable_rate = stats['stable'] / max(1, stats['success']) * 100
                avg_final_dist = stats['avg_distance'] / max(1, stats['success'])
                
                f.write(f"\n  {i+1}. TransGain={tg}, RotGain={rg}, MaxSpeed={ms}\n")
                f.write(f"     Success: {success_rate:.1f}% ({stats['success']}/{stats['total']})\n")
                f.write(f"     Stable: {stable_rate:.1f}% (θ_c oscillation < 0.1 rad)\n")
                f.write(f"     Avg Final Distance: {avg_final_dist:.4f}m\n")
                if stats['std_theta_values']:
                    f.write(f"     Avg Formation Stability: {np.mean(stats['std_theta_values']):.4f} rad ({np.degrees(np.mean(stats['std_theta_values'])):.2f}°)\n")
            
            # Stability analysis
            f.write(f"\nFormation Stability (θ_c) Analysis:\n")
            successful = [r for r in field_results if r['success']]
            if successful:
                std_thetas = [r['theta_stability']['std_theta'] for r in successful]
                stable_count = sum(1 for r in successful if r['theta_stability']['is_stable'])
                f.write(f"  Stable Convergence: {stable_count}/{len(successful)} ({stable_count/len(successful)*100:.1f}%)\n")
                f.write(f"  Mean θ_c oscillation: {np.mean(std_thetas):.4f} rad ({np.degrees(np.mean(std_thetas)):.2f}°)\n")
                f.write(f"  Median θ_c oscillation: {np.median(std_thetas):.4f} rad ({np.degrees(np.median(std_thetas)):.2f}°)\n")
                f.write(f"  Max θ_c oscillation: {np.max(std_thetas):.4f} rad ({np.degrees(np.max(std_thetas)):.2f}°)\n")
        
        # Recommendations
        f.write("\n\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS FOR DEPLOYMENT\n")
        f.write("="*80 + "\n\n")
        
        # Find best overall parameters considering both fields
        all_param_stats = {}
        for result in results:
            key = (result['translation_gain'], result['rotation_gain'], result['max_speed'])
            if key not in all_param_stats:
                all_param_stats[key] = {'success': 0, 'stable': 0, 'total': 0}
            all_param_stats[key]['total'] += 1
            if result['success']:
                all_param_stats[key]['success'] += 1
                if result['theta_stability']['is_stable']:
                    all_param_stats[key]['stable'] += 1
        
        sorted_all = sorted(
            all_param_stats.items(),
            key=lambda x: (x[1]['success'] / x[1]['total'], x[1]['stable'] / max(1, x[1]['success'])),
            reverse=True
        )
        
        best_params = sorted_all[0][0]
        best_stats = sorted_all[0][1]
        
        f.write("BEST OVERALL PARAMETERS (across both fields):\n\n")
        f.write(f"  Translation Gain: {best_params[0]}\n")
        f.write(f"  Rotation Gain: {best_params[1]}\n")
        f.write(f"  Max Speed: {best_params[2]} m/s\n")
        f.write(f"\n  Expected Success Rate: {best_stats['success']/best_stats['total']*100:.1f}%\n")
        f.write(f"  Stable Convergence Rate: {best_stats['stable']/max(1, best_stats['success'])*100:.1f}%\n\n")
        
        f.write("ANALYSIS BY CRITERIA:\n\n")
        
        # Best for success rate
        f.write("1. IF PRIORITY = Maximum Success Rate:\n")
        best_success = sorted_all[0][0]
        f.write(f"   Use: TransGain={best_success[0]}, RotGain={best_success[1]}, MaxSpeed={best_success[2]}\n\n")
        
        # Best for stability
        sorted_by_stability = sorted(
            all_param_stats.items(),
            key=lambda x: (x[1]['stable'] / max(1, x[1]['success']), x[1]['success'] / x[1]['total']),
            reverse=True
        )
        best_stable = sorted_by_stability[0][0]
        f.write("2. IF PRIORITY = Formation Stability (smooth θ_c):\n")
        f.write(f"   Use: TransGain={best_stable[0]}, RotGain={best_stable[1]}, MaxSpeed={best_stable[2]}\n\n")
        
        # Best balanced
        f.write("3. IF PRIORITY = Balance of Success & Stability:\n")
        sorted_balanced = sorted(
            all_param_stats.items(),
            key=lambda x: (x[1]['success']/x[1]['total'] * 0.7 + x[1]['stable']/max(1, x[1]['success']) * 0.3),
            reverse=True
        )
        best_balanced = sorted_balanced[0][0]
        f.write(f"   Use: TransGain={best_balanced[0]}, RotGain={best_balanced[1]}, MaxSpeed={best_balanced[2]}\n\n")
        
        # Distance analysis
        f.write("\nSTARTING POSITION ANALYSIS:\n\n")
        for dist in [0.5, 1.0, 1.5, 2.0]:
            dist_results = [r for r in results if abs(np.linalg.norm(r['start_pos']) - dist) < 0.15]
            if dist_results:
                dist_success = sum(1 for r in dist_results if r['success'])
                dist_rate = dist_success / len(dist_results) * 100
                f.write(f"Starting at {dist}m: {dist_rate:.1f}% success rate\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Saved detailed results log to: saddle_point_fine_tuning_results.log")




def create_basin_heatmaps(results, fields, project_root):
    """
    Create 2D heatmaps of starting position convergence success.
    Green = successful convergence, Red = failed.
    Shows the "basin of convergence" visually.
    """
    print("\n5. CREATING BASIN OF CONVERGENCE HEATMAPS...")
    
    for field_func, field_name in fields:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Create grid for heatmap
        x_range = np.linspace(-2.5, 2.5, 100)
        y_range = np.linspace(-2.5, 2.5, 100)
        grid = np.zeros((len(y_range), len(x_range)))
        
        # Get field results
        field_results = [r for r in results if r['field_name'] == field_name]
        
        # Fill grid: 1 = success, 0 = failure
        for result in field_results:
            x, y = result['start_pos']
            # Find closest grid point
            ix = np.argmin(np.abs(x_range - x))
            iy = np.argmin(np.abs(y_range - y))
            grid[iy, ix] = 1.0 if result['success'] else 0.0
        
        # Create heatmap
        im = ax.imshow(grid, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', 
                       cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
        
        # Overlay tested positions
        successful = [r for r in field_results if r['success']]
        failed = [r for r in field_results if not r['success']]
        
        if successful:
            sx = [r['start_pos'][0] for r in successful]
            sy = [r['start_pos'][1] for r in successful]
            ax.scatter(sx, sy, c='green', s=20, alpha=0.3, marker='.', label='Success')
        
        if failed:
            fx = [r['start_pos'][0] for r in failed]
            fy = [r['start_pos'][1] for r in failed]
            ax.scatter(fx, fy, c='red', s=20, alpha=0.3, marker='.', label='Failed')
        
        # Mark origin (saddle point)
        ax.plot(0, 0, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Target (0,0)')
        
        # Mark success radius
        circle = plt.Circle((0, 0), 0.5, fill=False, edgecolor='blue', linestyle='--', linewidth=2, label='Success zone (0.5m)')
        ax.add_patch(circle)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'Basin of Convergence - {field_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.colorbar(im, ax=ax, label='Success (1) vs Failure (0)')
        
        output_path = os.path.join(project_root, f'basin_of_convergence_{field_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: basin_of_convergence_{field_name}.png")
        
        plt.close()


def create_comparison_plots(results, fields, project_root):
    """
    Create comparison plots for multiple fields and stability.
    """
    print("\n4. CREATING COMPARISON PLOTS...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Success rate comparison
    ax1 = plt.subplot(2, 3, 1)
    field_names = [name for _, name in fields]
    success_rates = []
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        rate = sum(1 for r in field_results if r['success']) / len(field_results) * 100
        success_rates.append(rate)
    
    ax1.bar(field_names, success_rates, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate by Field')
    ax1.set_ylim([0, 100])
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Plot 2: Stability comparison
    ax2 = plt.subplot(2, 3, 2)
    stable_rates = []
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        successful = [r for r in field_results if r['success']]
        if successful:
            rate = sum(1 for r in successful if r['theta_stability']['is_stable']) / len(successful) * 100
        else:
            rate = 0
        stable_rates.append(rate)
    
    ax2.bar(field_names, stable_rates, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('Stable Convergence (%)')
    ax2.set_title('Formation Stability by Field')
    ax2.set_ylim([0, 100])
    for i, v in enumerate(stable_rates):
        ax2.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Plot 3: Final distance distribution by field
    ax3 = plt.subplot(2, 3, 3)
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        distances = [r['final_distance'] for r in field_results]
        ax3.hist(distances, bins=40, alpha=0.6, label=field_name)
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Success threshold')
    ax3.set_xlabel('Final Distance (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Distance Distribution')
    ax3.legend()
    ax3.set_yscale('log')
    
    # Plot 4: Theta oscillation by field
    ax4 = plt.subplot(2, 3, 4)
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        successful = [r for r in field_results if r['success']]
        std_thetas = [r['theta_stability']['std_theta'] for r in successful]
        ax4.hist(std_thetas, bins=30, alpha=0.6, label=field_name)
    ax4.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Stability threshold')
    ax4.set_xlabel('Std(theta) in final 100 steps (rad)')
    ax4.set_ylabel('Count')
    ax4.set_title('Formation Orientation Stability')
    ax4.legend()
    ax4.set_yscale('log')
    
    # Plot 5: Success vs Stability scatter
    ax5 = plt.subplot(2, 3, 5)
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        successful = [r for r in field_results if r['success']]
        trans_gains = [r['translation_gain'] for r in successful]
        std_thetas = [r['theta_stability']['std_theta'] for r in successful]
        ax5.scatter(trans_gains, std_thetas, alpha=0.5, label=field_name)
    ax5.axhline(0.1, color='red', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Translation Gain')
    ax5.set_ylabel('Std(theta)')
    ax5.set_title('Stability vs Translation Gain')
    ax5.legend()
    ax5.set_yscale('log')
    
    # Plot 6: By distance
    ax6 = plt.subplot(2, 3, 6)
    distances = [0.5, 1.0, 1.5, 2.0]
    for field_func, field_name in fields:
        field_results = [r for r in results if r['field_name'] == field_name]
        success_by_dist = []
        for dist in distances:
            subset = [r for r in field_results if abs(np.linalg.norm(r['start_pos']) - dist) < 0.1]
            if subset:
                rate = sum(1 for r in subset if r['success']) / len(subset) * 100
            else:
                rate = 0
            success_by_dist.append(rate)
        ax6.plot(distances, success_by_dist, marker='o', label=field_name)
    ax6.set_xlabel('Starting Distance (m)')
    ax6.set_ylabel('Success Rate (%)')
    ax6.set_title('Success Rate by Starting Distance')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Basin of convergence heatmaps
    output_path = os.path.join(project_root, 'saddle_point_fine_tuning_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    
    plt.close()
    

    # Create basin of convergence heatmaps
    create_basin_heatmaps(results, fields, project_root)
    for field_func, field_name in fields:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Create grid for heatmap
        x_range = np.linspace(-2.5, 2.5, 100)
        y_range = np.linspace(-2.5, 2.5, 100)
        grid = np.zeros((len(y_range), len(x_range)))
        
        # Get field results
        field_results = [r for r in results if r['field_name'] == field_name]
        
        # Fill grid: 1 = success, 0 = failure
        for result in field_results:
            x, y = result['start_pos']
            # Find closest grid point
            ix = np.argmin(np.abs(x_range - x))
            iy = np.argmin(np.abs(y_range - y))
            grid[iy, ix] = 1.0 if result['success'] else 0.0
        
        # Create heatmap
        im = ax.imshow(grid, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', 
                       cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
        
        # Overlay tested positions
        successful = [r for r in field_results if r['success']]
        failed = [r for r in field_results if not r['success']]
        
        if successful:
            sx = [r['start_pos'][0] for r in successful]
            sy = [r['start_pos'][1] for r in successful]
            ax.scatter(sx, sy, c='green', s=20, alpha=0.3, marker='.', label='Success')
        
        if failed:
            fx = [r['start_pos'][0] for r in failed]
            fy = [r['start_pos'][1] for r in failed]
            ax.scatter(fx, fy, c='red', s=20, alpha=0.3, marker='.', label='Failed')
        
        # Mark origin (saddle point)
        ax.plot(0, 0, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Target (0,0)')
        
        # Mark success radius
        circle = plt.Circle((0, 0), 0.5, fill=False, edgecolor='blue', linestyle='--', linewidth=2, label='Success zone (0.5m)')
        ax.add_patch(circle)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'Basin of Convergence - {field_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.colorbar(im, ax=ax, label='Success (1) vs Failure (0)')
        
        output_path = os.path.join(project_root, f'basin_of_convergence_{field_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: basin_of_convergence_{field_name}.png")
        
        plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SADDLE POINT FINE-TUNING EXPERIMENT")
    print("=" * 80)
    print("\nTo run in background with logging:")
    print("  cd trunk/Python\\ Simulations/Vector_Fields/VF_Robot")
    print("  ./venv/bin/python3 experiments/saddle_point_fine_tuning_multifield.py > fine_tuning.log 2>&1 &")
    print("\nMonitor:")
    print("  tail -f fine_tuning.log")
    print("\n" + "=" * 80 + "\n")
    
    results = run_fine_tuning_experiment()
    
    print("\n" + "=" * 80)
    print("Results saved to: saddle_point_fine_tuning_results.png")
    print("=" * 80)
