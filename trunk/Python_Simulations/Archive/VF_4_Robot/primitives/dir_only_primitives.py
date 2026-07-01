import numpy as np
import matplotlib.pyplot as plt

## Direction Only Techniques



def ride_separatrix_dipole_dir(cluster): #
    _, angles = cluster.bot_compvec()
    avg_angle = np.mean(angles)
    cnt =0
    for ang in angles:
        if ang > avg_angle:
            cnt +=1
    
    if cnt ==2:
        best_angle = np.min(angles)
    else:
        best_angle = np.max(angles)
            
    cluster.cluster_centre[0] += cluster.step_size*np.cos( best_angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin( best_angle)
    
    return cluster.cluster_centre

def find_source_dipole_dir(cluster): #
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle - np.pi/2
    else:
        angle = avg_angle - np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre




def find_source_sinkvortex_dir(cluster): #
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle #- np.pi/2
    else:
        angle = avg_angle #- np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre



def orbit_source_sprstr_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = -np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle + np.pi/2
    else:
        angle = avg_angle + np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre


def find_source_sprstr_dir(cluster): #
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle - np.pi
    else:
        angle = avg_angle - np.pi

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre


def find_source_rankine_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = np.mean(angles)
    # Need to come up with -ve or positive based on some cue.
    cluster.cluster_centre[0] += cluster.step_size*np.cos( -np.pi/2 + avg_angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin( -np.pi/2 + avg_angle)
    return cluster.cluster_centre

def orbit_source_rankine_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = np.mean(angles)
    
    cluster.cluster_centre[0] += cluster.step_size*np.cos( avg_angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin( avg_angle)
    return cluster.cluster_centre

def orbit_source_sinkvortex_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle #+ np.pi/2
    else:
        angle = avg_angle #+ np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre





def orbit_source_vortex_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = -avg_angle #- np.pi/2
    else:
        angle = avg_angle #- np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre



def find_source_vortex_dir(cluster): # Looks like it works well!
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle - np.pi/2
    else:
        angle = avg_angle - np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre



def orbit_source_dipole_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = np.median(angles)
    
    # Determine the most common sign
    num_positive = sum(1 for idx in angles if idx > 0)
    num_negative = sum(1 for idx in angles if idx < 0)
    
    if num_negative >= num_positive:
        angle = avg_angle + np.pi/2
    else:
        angle = avg_angle + np.pi/2

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    
    return cluster.cluster_centre

def find_source_spring_dir(cluster):
    _, angles = cluster.bot_compvec()
    #print('angles',angles)
    avg_angle = np.mean(angles)
    #print('angles',angles, 'mean angle', avg_angle, 'plus pi ', (avg_angle + np.pi))
    
    cluster.cluster_centre[0] += cluster.step_size*np.cos(avg_angle + np.pi) # Make positive for sink
    cluster.cluster_centre[1] += cluster.step_size*np.sin(avg_angle + np.pi)
    return cluster.cluster_centre


def orbit_source_spring_dir(cluster):
    _, angles = cluster.bot_compvec()
    avg_angle = np.mean(angles)
    
    cluster.cluster_centre[0] += cluster.step_size*np.cos( np.pi/2 + avg_angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin( np.pi/2 + avg_angle)
    return cluster.cluster_centre