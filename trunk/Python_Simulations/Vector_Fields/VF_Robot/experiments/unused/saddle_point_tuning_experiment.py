"""
Saddle Point Newton's Method - Comprehensive Tuning Experiment

This experiment systematically tests scalar_newton_with_rotation across:
- Multiple starting positions (radial sweep from origin)
- Different tuning parameters (translation_gain, rotation_gain, max_speed)

Output: Heatmaps and recommendations for optimal starting conditions and tuning
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
from src.fields.environments.Scalar_Fields import bimodal_gaussian
import src.control.quad_primitives as qcp


def run_single_trial(start_pos, translation_gain, rotation_gain, 
                     max_speed, max_omega, sim_steps=3000):
    """
    Run a single trial of saddle point navigation.
    
    Args:
        start_pos: (x, y) starting position
        translation_gain, rotation_gain, max_speed, max_omega: Control parameters
        sim_steps: Number of simulation steps
    
    Returns:
        {
            'final_pos': (x, y),
            'final_distance': distance from origin,
            'success': bool (within 0.5m),
            'trajectory': list of positions,
            'gradients': list of gradient magnitudes
        }
    """
    # Create field
    field = AnalyticalScalarField(bimodal_gaussian)
    
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
    gradients = []
    
    # Run simulation
    for step in range(sim_steps):
        try:
            # Get control command
            vx_c, vy_c, omega_c = scalar_newton_tuned(cluster)
            
            # Store gradient info
            try:
                H, gradient = qcp.estimate_hessian_from_scalar_readings(cluster)
                gradients.append(np.linalg.norm(gradient))
            except:
                pass
            
            # Apply command (move cluster)
            cluster.move(scalar_newton_tuned)
            
            # Store position
            cx, cy = cluster.get_centroid()
            trajectory.append(np.array([cx, cy]))
            
        except Exception as e:
            # If something fails, return what we have
            break
    
    # Get final position
    cx, cy = cluster.get_centroid()
    final_pos = np.array([cx, cy])
    final_distance = np.linalg.norm(final_pos)
    success = final_distance < 0.5
    
    return {
        'final_pos': final_pos,
        'final_distance': final_distance,
        'success': success,
        'trajectory': np.array(trajectory),
        'gradients': np.array(gradients),
        'start_pos': start_pos,
        'initial_theta': initial_theta,
        'translation_gain': translation_gain,
        'rotation_gain': rotation_gain,
        'max_speed': max_speed,
    }


def run_comprehensive_experiment():
    """
    Run comprehensive tuning experiment.
    """
    print("=" * 80)
    print("SADDLE POINT NEWTON'S METHOD - COMPREHENSIVE TUNING EXPERIMENT")
    print("=" * 80)
    
    # Define parameter ranges
    start_distances = [0.5, 1.0, 1.5, 2.0]  # Distance from origin
    start_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  # 12 directions
    
    translation_gains = [0.5, 1.0, 1.5]
    rotation_gains = [0.1, 0.3, 0.5]
    max_speeds = [0.15, 0.25, 0.35]
    
    # Generate all starting positions
    start_positions = []
    for distance in start_distances:
        for angle in start_angles:
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            start_positions.append((x, y))
    
    print(f"\nExperiment Configuration:")
    print(f"  Starting positions: {len(start_positions)} (4 distances × 12 angles)")
    print(f"  Parameter combinations: {len(translation_gains)} × {len(rotation_gains)} × {len(max_speeds)} = {len(translation_gains)*len(rotation_gains)*len(max_speeds)}")
    print(f"  Total trials: {len(start_positions) * len(translation_gains) * len(rotation_gains) * len(max_speeds)}")
    print(f"  Simulation steps per trial: 3000")
    print(f"  Success criterion: final distance < 0.5m from origin\n")
    
    # Collect all results
    results = []
    total_trials = 0
    successful_trials = 0
    
    start_time = time.time()
    
    # Sweep through parameters
    param_combos = list(product(
        translation_gains,
        rotation_gains,
        max_speeds
    ))
    
    print(f"Starting trials...")
    
    for param_idx, (tg, rg, ms) in enumerate(param_combos):
        for pos_idx, start_pos in enumerate(start_positions):
            total_trials += 1
            
            # Run trial
            result = run_single_trial(
                start_pos, tg, rg, ms, max_omega=0.1, sim_steps=3000
            )
            results.append(result)
            
            if result['success']:
                successful_trials += 1
            
            # Progress update every 50 trials
            if total_trials % 50 == 0:
                elapsed = time.time() - start_time
                success_rate = successful_trials / total_trials * 100
                print(f"  Trial {total_trials}: {success_rate:.1f}% success | Elapsed: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    final_success_rate = successful_trials / total_trials * 100
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total trials: {total_trials}")
    print(f"Successful trials: {successful_trials}")
    print(f"Overall success rate: {final_success_rate:.1f}%")
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Analyze results
    analyze_results(results, param_combos, start_positions)
    
    return results


def analyze_results(results, param_combos, start_positions):
    """
    Analyze results and provide tuning recommendations.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    # 1. Best parameter combination overall
    print("\n1. BEST PARAMETER COMBINATIONS (by success rate):")
    param_success = {}
    for result in results:
        key = (result['translation_gain'], result['rotation_gain'], result['max_speed'])
        if key not in param_success:
            param_success[key] = {'success': 0, 'total': 0}
        param_success[key]['total'] += 1
        param_success[key]['success'] += 1 if result['success'] else 0
    
    # Sort by success rate
    sorted_params = sorted(
        param_success.items(),
        key=lambda x: x[1]['success'] / x[1]['total'],
        reverse=True
    )
    
    for i, (params, counts) in enumerate(sorted_params[:5]):
        tg, rg, ms = params
        rate = counts['success'] / counts['total'] * 100
        print(f"  {i+1}. trans_gain={tg}, rot_gain={rg}, max_speed={ms}: {rate:.1f}% ({counts['success']}/{counts['total']})")
    
    # 2. Best starting positions
    print("\n2. BEST STARTING POSITIONS (by success rate):")
    pos_success = {}
    for result in results:
        key = (round(result['start_pos'][0], 2), round(result['start_pos'][1], 2))
        if key not in pos_success:
            pos_success[key] = {'success': 0, 'total': 0}
        pos_success[key]['total'] += 1
        pos_success[key]['success'] += 1 if result['success'] else 0
    
    sorted_positions = sorted(
        pos_success.items(),
        key=lambda x: x[1]['success'] / x[1]['total'],
        reverse=True
    )
    
    for i, (pos, counts) in enumerate(sorted_positions[:10]):
        rate = counts['success'] / counts['total'] * 100
        distance = np.sqrt(pos[0]**2 + pos[1]**2)
        print(f"  {i+1}. pos=({pos[0]:6.2f}, {pos[1]:6.2f}) dist={distance:.2f}m: {rate:.1f}% ({counts['success']}/{counts['total']})")
    
    # 3. Distance-based analysis
    print("\n3. SUCCESS RATE BY STARTING DISTANCE:")
    distance_success = {}
    for result in results:
        dist = np.linalg.norm(result['start_pos'])
        key = round(dist, 1)
        if key not in distance_success:
            distance_success[key] = {'success': 0, 'total': 0}
        distance_success[key]['total'] += 1
        distance_success[key]['success'] += 1 if result['success'] else 0
    
    for dist in sorted(distance_success.keys()):
        counts = distance_success[dist]
        rate = counts['success'] / counts['total'] * 100
        print(f"  Distance {dist:.1f}m: {rate:.1f}% ({counts['success']}/{counts['total']})")
    
    # 4. Initial orientation analysis
    print("\n4. SUCCESS RATE BY INITIAL ORIENTATION:")
    orient_success = {}
    for result in results:
        # Quantize orientation to nearest 22.5 degrees for grouping
        orient_deg = np.degrees(result['initial_theta']) % 360
        key = round(orient_deg / 5.0) * 5.0  # Group by 5° increments
        if key not in orient_success:
            orient_success[key] = {'success': 0, 'total': 0}
        orient_success[key]['total'] += 1
        orient_success[key]['success'] += 1 if result['success'] else 0
    
    for orient in sorted(orient_success.keys()):
        counts = orient_success[orient]
        rate = counts['success'] / counts['total'] * 100
        print(f"  θ={orient:6.1f}°: {rate:.1f}% ({counts['success']}/{counts['total']})")
    
    # 5. Failure analysis
    print("\n5. FAILURE ANALYSIS:")
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        avg_final_distance = np.mean([r['final_distance'] for r in failed_results])
        print(f"  Failed trials: {len(failed_results)}")
        print(f"  Average final distance (failed): {avg_final_distance:.3f}m")
        print(f"  Median final distance (failed): {np.median([r['final_distance'] for r in failed_results]):.3f}m")
        print(f"  Max final distance (failed): {np.max([r['final_distance'] for r in failed_results]):.3f}m")
    
    # 5. Create heatmaps
    create_heatmaps(results)


def create_heatmaps(results):
    """
    Create heatmaps showing success rate across parameter space.
    """
    print("\n6. CREATING HEATMAPS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Saddle Point Navigation - Success Rate Heatmaps', fontsize=16, fontweight='bold')
    
    # Heatmap 1: Translation Gain vs Rotation Gain (averaged over max_speed)
    tg_values = sorted(set(r['translation_gain'] for r in results))
    rg_values = sorted(set(r['rotation_gain'] for r in results))
    
    heatmap1 = np.zeros((len(rg_values), len(tg_values)))
    for i, rg in enumerate(rg_values):
        for j, tg in enumerate(tg_values):
            subset = [r for r in results if r['translation_gain'] == tg and r['rotation_gain'] == rg]
            if subset:
                rate = sum(1 for r in subset if r['success']) / len(subset) * 100
                heatmap1[i, j] = rate
    
    im1 = axes[0, 0].imshow(heatmap1, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0, 0].set_xticks(range(len(tg_values)))
    axes[0, 0].set_yticks(range(len(rg_values)))
    axes[0, 0].set_xticklabels([f'{x:.1f}' for x in tg_values])
    axes[0, 0].set_yticklabels([f'{x:.1f}' for x in rg_values])
    axes[0, 0].set_xlabel('Translation Gain')
    axes[0, 0].set_ylabel('Rotation Gain')
    axes[0, 0].set_title('Success Rate: Trans Gain vs Rot Gain')
    plt.colorbar(im1, ax=axes[0, 0], label='Success %')
    
    # Heatmap 2: Translation Gain vs Max Speed
    ms_values = sorted(set(r['max_speed'] for r in results))
    heatmap2 = np.zeros((len(ms_values), len(tg_values)))
    for i, ms in enumerate(ms_values):
        for j, tg in enumerate(tg_values):
            subset = [r for r in results if r['translation_gain'] == tg and r['max_speed'] == ms]
            if subset:
                rate = sum(1 for r in subset if r['success']) / len(subset) * 100
                heatmap2[i, j] = rate
    
    im2 = axes[0, 1].imshow(heatmap2, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[0, 1].set_xticks(range(len(tg_values)))
    axes[0, 1].set_yticks(range(len(ms_values)))
    axes[0, 1].set_xticklabels([f'{x:.1f}' for x in tg_values])
    axes[0, 1].set_yticklabels([f'{x:.2f}' for x in ms_values])
    axes[0, 1].set_xlabel('Translation Gain')
    axes[0, 1].set_ylabel('Max Speed (m/s)')
    axes[0, 1].set_title('Success Rate: Trans Gain vs Max Speed')
    plt.colorbar(im2, ax=axes[0, 1], label='Success %')
    
    # Heatmap 3: Rotation Gain vs Max Speed
    heatmap3 = np.zeros((len(ms_values), len(rg_values)))
    for i, ms in enumerate(ms_values):
        for j, rg in enumerate(rg_values):
            subset = [r for r in results if r['rotation_gain'] == rg and r['max_speed'] == ms]
            if subset:
                rate = sum(1 for r in subset if r['success']) / len(subset) * 100
                heatmap3[i, j] = rate
    
    im3 = axes[1, 0].imshow(heatmap3, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[1, 0].set_xticks(range(len(rg_values)))
    axes[1, 0].set_yticks(range(len(ms_values)))
    axes[1, 0].set_xticklabels([f'{x:.1f}' for x in rg_values])
    axes[1, 0].set_yticklabels([f'{x:.2f}' for x in ms_values])
    axes[1, 0].set_xlabel('Rotation Gain')
    axes[1, 0].set_ylabel('Max Speed (m/s)')
    axes[1, 0].set_title('Success Rate: Rot Gain vs Max Speed')
    plt.colorbar(im3, ax=axes[1, 0], label='Success %')
    
    # Plot 4: Distribution of final distances
    successful_distances = [r['final_distance'] for r in results if r['success']]
    failed_distances = [r['final_distance'] for r in results if not r['success']]
    
    axes[1, 1].hist(successful_distances, bins=30, alpha=0.6, label=f'Successful (n={len(successful_distances)})', color='green')
    axes[1, 1].hist(failed_distances, bins=30, alpha=0.6, label=f'Failed (n={len(failed_distances)})', color='red')
    axes[1, 1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Success threshold (0.5m)')
    axes[1, 1].set_xlabel('Final Distance from Origin (m)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Final Distances')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('saddle_point_tuning_heatmaps.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: saddle_point_tuning_heatmaps.png")
    
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SADDLE POINT TUNING EXPERIMENT")
    print("=" * 80)
    print("\nTo run this experiment in background with logging:")
    print("  cd trunk/Python\\ Simulations/Vector_Fields/VF_Robot")
    print("  ./venv/bin/python3 experiments/saddle_point_tuning_experiment.py > saddle_tuning.log 2>&1 &")
    print("\nMonitor progress:")
    print("  tail -f saddle_tuning.log")
    print("\n" + "=" * 80 + "\n")
    
    results = run_comprehensive_experiment()
    
    print("\n" + "=" * 80)
    print("Results saved. Check saddle_point_tuning_heatmaps.png for visualizations.")
    print("=" * 80)
