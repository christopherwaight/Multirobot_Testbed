import numpy as np
import matplotlib.pyplot as plt

## Magnitude Only Techniques


# Corotoating Vortices

def find_source_corvor_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi)
    return cluster.cluster_centre

def find_kmag_corvor_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi/2)
    return cluster.cluster_centre

def find_stagpt_corvor_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi)
    return cluster.cluster_centre


# Rankine Vortex Policies

def find_source_rankine_mag(cluster): # Doesn't work
    _, _, angle = cluster.normal_angle()  
    angle += np.pi  
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    return cluster.cluster_centre

def orbit_source_rankine_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi/2)
    return cluster.cluster_centre

def find_ridge_rankine_mag(cluster):   # Finds and rides the ridge

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi)
    return cluster.cluster_centre


# Opposing Vortex Policies

def find_source_oppvor_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi)
    return cluster.cluster_centre

def find_kmag_oppvor_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi/2)
    return cluster.cluster_centre

def find_stagpt_oppvor_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi)
    return cluster.cluster_centre


## This finds the Stagnation Point of the flow
def find_source_spring_mag(cluster):
    _, _, angle = cluster.normal_angle()
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle + np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi )
    return cluster.cluster_centre

def find_source_spring_mag2(cluster): # this one is for the saddle in scalars
    Nx, Ny, angle = cluster.normal_angle()
    cluster.cluster_centre[0] -= cluster.step_size*Nx
    cluster.cluster_centre[1] += cluster.step_size*Ny
    return cluster.cluster_centre

def orbit_source_spring_mag(cluster):
    _, _, angle = cluster.normal_angle()
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle + np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi/2)
    return cluster.cluster_centre



def orbit_source_vortex_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi/2)
    return cluster.cluster_centre


# Dipole Policies

def orbit_source_dipole_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi/2)
    return cluster.cluster_centre


## Sinking Vortex Policies

def orbit_source_sinkvortex_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle - np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle - np.pi/2)
    return cluster.cluster_centre

    ''' Spring in Stream Policies '''

def find_source_sprstr_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi)
    return cluster.cluster_centre

def find_stagpt_sprstr_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    angle = angle - np.pi/2
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)# - np.pi
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)# - np.pi
    return cluster.cluster_centre

''' Spring in Stream Policies '''

def find_source_vorstr_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi)
    return cluster.cluster_centre

def find_stagpt_vorstr_mag(cluster):   

    _, _, angle = cluster.normal_angle()

    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)# - np.pi/2)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)# - np.pi/2)
    return cluster.cluster_centre

def find_source_sinkvortex_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi)
    return cluster.cluster_centre



def find_source_vortex_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle)
    return cluster.cluster_centre

def find_source_dipole_mag(cluster):
    _, _, angle = cluster.normal_angle()  
    #angle += np.pi  #Uncomment for Free vortex
    cluster.cluster_centre[0] += cluster.step_size*np.cos(angle+ np.pi)
    cluster.cluster_centre[1] += cluster.step_size*np.sin(angle + np.pi)
    return cluster.cluster_centre