import numpy as np
import matplotlib.pyplot as plt

## Non Uniform Stream
def find_sink_mag(cluster):  # Moves in the Y direction, x is constant.
    _,Ny,_ = cluster.normal_angle()
    if np.sign(Ny)>0:
        cluster.cluster_centre[1] += cluster.step_size
    else:
        cluster.cluster_centre[1] -= cluster.step_size
    return cluster.cluster_centre

def fast_flow_aligned(cluster): #Check if this works if robots are in a straight line. (divide/0 problems)
    _,Ny,_ = cluster.normal_angle()
    if np.sign(Ny)>0:
        cluster.cluster_centre[1] += cluster.step_size
    else:
        cluster.cluster_centre[1] -= cluster.step_size
    return cluster.cluster_centre    