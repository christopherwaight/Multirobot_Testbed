import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.robot.omni_cluster import OmniCluster
from src.fields.field_types import AnalyticalField
import src.control.primitives as ocp
from src.simulation.runner import execute_omni_simulation

# Import the existing environments that have built-in time dependence
from src.fields.environments.Vortex import vortex1
from src.fields.environments.Double_Gyre import double_gyre_static

def run_experiment_b():
    print("Running RA-L Experiment B: Time-varying environments (using built-in mechanics)")
    
    formation_config = "config/formations/equilateral_default.yaml"
    
    # 1. Wobbling Vortex (Center orbits over time)
    print("Testing Wobbling Vortex...")
    field_vortex = AnalyticalField(vortex1)
    
    # Use built-in wobble mechanics: eps > 0 causes center to orbit
    field_vortex.config = {"eps": 0.15, "omega": 0.628} 
    # Tell runner to step time
    field_vortex.step = lambda dt: setattr(field_vortex, "t", getattr(field_vortex, "t", 0.0) + dt)
    
    cluster_vortex = OmniCluster(formation_config, field_vortex)
    
    fig_vortex = plt.figure(figsize=(8, 8))
    execute_omni_simulation(
        cluster_vortex, 
        ocp.critical_point_plane_fitting, 
        "Wobbling Vortex (Convergence)", 
        sim_time=400
    )
    fig_vortex.savefig("ral_wobbling_vortex_convergence.png", bbox_inches='tight')
    plt.close(fig_vortex)
    print("Saved Wobbling Vortex plot.")

    # 2. Time-Dependent Double Gyre (Shadden-Lekien-Marsden formulation used by Hsieh)
    print("Testing Time-Dependent Double Gyre...")
    field_gyre = AnalyticalField(double_gyre_static)
    # eps > 0 activates the time-varying Shadden formulation
    field_gyre.config = {"A": 0.1, "eps": 0.25, "omega": 0.628}
    field_gyre.step = lambda dt: setattr(field_gyre, "t", getattr(field_gyre, "t", 0.0) + dt)
    
    # Start the cluster near one of the gyre centers or saddles
    cluster_gyre = OmniCluster(formation_config, field_gyre)
    # Override initial position to put them near a moving saddle or center
    offset = np.array([0.1, 0.4])
    for robot in cluster_gyre.robots:
        # Translate the entire cluster
        robot.position = robot.position + offset

    fig_gyre = plt.figure(figsize=(8, 8))
    execute_omni_simulation(
        cluster_gyre, 
        ocp.critical_point_plane_fitting, 
        "Time-Dependent Double Gyre (Convergence to Saddle)", 
        sim_time=400
    )
    fig_gyre.savefig("ral_dynamic_double_gyre_convergence.png", bbox_inches='tight')
    plt.close(fig_gyre)
    print("Saved Dynamic Double Gyre plot.")
    
if __name__ == "__main__":
    run_experiment_b()
