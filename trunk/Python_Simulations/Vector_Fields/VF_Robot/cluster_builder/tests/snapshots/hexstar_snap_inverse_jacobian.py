"""Analytical inverse Jacobian for hexstar_snap (6 robots).
Maps q_dot (12 state vars) to r_dot (12 robot coords).
State var order: ['x_h', 'y_h', 'theta_c', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5']
Robot row order: [x1,y1, x2,y2, ..., x6,y6]
"""
import math
import numpy as np

STATE_VARS = ['x_h', 'y_h', 'theta_c', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5']

def inverse_jacobian(state):
    """Returns (12 x 12) numpy array."""
    J = np.zeros((12, 12))
    col = {v: i for i, v in enumerate(STATE_VARS)}
    theta_c = state["theta_c"]
    r_1 = state["r_1"]
    g_1 = 0.0
    alpha_1 = theta_c + g_1
    J[0, col["x_h"]] = 1.0
    J[1, col["y_h"]] = 1.0
    J[0, col["theta_c"]] += -r_1 * math.sin(alpha_1)
    J[1, col["theta_c"]] +=  r_1 * math.cos(alpha_1)
    J[0, col["r_1"]] = math.cos(alpha_1)
    J[1, col["r_1"]] = math.sin(alpha_1)
    r_2 = state["r_2"]
    g_2 = state["gamma_2"]
    alpha_2 = theta_c + g_2
    J[2, col["x_h"]] = 1.0
    J[3, col["y_h"]] = 1.0
    J[2, col["theta_c"]] += -r_2 * math.sin(alpha_2)
    J[3, col["theta_c"]] +=  r_2 * math.cos(alpha_2)
    J[2, col["r_2"]] = math.cos(alpha_2)
    J[3, col["r_2"]] = math.sin(alpha_2)
    J[2, col["gamma_2"]] = -r_2 * math.sin(alpha_2)
    J[3, col["gamma_2"]] =  r_2 * math.cos(alpha_2)
    r_3 = state["r_3"]
    g_3 = state["gamma_3"]
    alpha_3 = theta_c + g_3
    J[4, col["x_h"]] = 1.0
    J[5, col["y_h"]] = 1.0
    J[4, col["theta_c"]] += -r_3 * math.sin(alpha_3)
    J[5, col["theta_c"]] +=  r_3 * math.cos(alpha_3)
    J[4, col["r_3"]] = math.cos(alpha_3)
    J[5, col["r_3"]] = math.sin(alpha_3)
    J[4, col["gamma_3"]] = -r_3 * math.sin(alpha_3)
    J[5, col["gamma_3"]] =  r_3 * math.cos(alpha_3)
    r_4 = state["r_4"]
    g_4 = state["gamma_4"]
    alpha_4 = theta_c + g_4
    J[6, col["x_h"]] = 1.0
    J[7, col["y_h"]] = 1.0
    J[6, col["theta_c"]] += -r_4 * math.sin(alpha_4)
    J[7, col["theta_c"]] +=  r_4 * math.cos(alpha_4)
    J[6, col["r_4"]] = math.cos(alpha_4)
    J[7, col["r_4"]] = math.sin(alpha_4)
    J[6, col["gamma_4"]] = -r_4 * math.sin(alpha_4)
    J[7, col["gamma_4"]] =  r_4 * math.cos(alpha_4)
    r_5 = state["r_5"]
    g_5 = state["gamma_5"]
    alpha_5 = theta_c + g_5
    J[8, col["x_h"]] = 1.0
    J[9, col["y_h"]] = 1.0
    J[8, col["theta_c"]] += -r_5 * math.sin(alpha_5)
    J[9, col["theta_c"]] +=  r_5 * math.cos(alpha_5)
    J[8, col["r_5"]] = math.cos(alpha_5)
    J[9, col["r_5"]] = math.sin(alpha_5)
    J[8, col["gamma_5"]] = -r_5 * math.sin(alpha_5)
    J[9, col["gamma_5"]] =  r_5 * math.cos(alpha_5)
    J[10, col["x_h"]] = 1.0
    J[11, col["y_h"]] = 1.0
    return J