"""Analytical inverse Jacobian for my_star_snap (6 robots).
Maps q_dot (12 state vars) to r_dot (12 robot coords).
State var order: ['x_c', 'y_c', 'theta_c', 'p_1', 'beta_1', 'q_1', 'L_2', 'theta_2', 'L_3', 'theta_3', 'L_4', 'theta_4']
Robot row order: [x1,y1, x2,y2, ..., x6,y6]
"""
import math
import numpy as np

STATE_VARS = ['x_c', 'y_c', 'theta_c', 'p_1', 'beta_1', 'q_1', 'L_2', 'theta_2', 'L_3', 'theta_3', 'L_4', 'theta_4']

def inverse_jacobian(state):
    """Returns (12 x 12) numpy array."""
    J = np.zeros((12, 12))
    # Analytical Jacobian: numerically evaluated at current state
    # using the closed-form composition of pair/SAS blocks.
    # This calls the runtime kinematics objects.
    _fill_jacobian(state, J)
    return J

def _fill_jacobian(state, J):
    """Fill J using analytical composition of pair/SAS blocks."""
    col = {v: i for i, v in enumerate(STATE_VARS)}
    # Leaf pair: robots 1,2
    _L_2 = state["L_2"]
    _th_2 = state["theta_2"]
    _ct_2, _st_2 = math.cos(_th_2), math.sin(_th_2)
    _h_2 = _L_2 / 2
    # Centroid columns (from parent propagation)
    J[0, col["x_c"]] += 1.0; J[1, col["y_c"]] += 1.0
    J[2, col["x_c"]] += 1.0; J[3, col["y_c"]] += 1.0
    # Shape: theta_2
    J[0, col["theta_2"]] +=  _h_2*_st_2
    J[1, col["theta_2"]] += -_h_2*_ct_2
    J[2, col["theta_2"]] += -_h_2*_st_2
    J[3, col["theta_2"]] +=  _h_2*_ct_2
    # Shape: L_2
    J[0, col["L_2"]] += -0.5*_ct_2
    J[1, col["L_2"]] += -0.5*_st_2
    J[2, col["L_2"]] +=  0.5*_ct_2
    J[3, col["L_2"]] +=  0.5*_st_2
    # Leaf pair: robots 3,4
    _L_3 = state["L_3"]
    _th_3 = state["theta_3"]
    _ct_3, _st_3 = math.cos(_th_3), math.sin(_th_3)
    _h_3 = _L_3 / 2
    # Centroid columns (from parent propagation)
    J[4, col["x_c"]] += 1.0; J[5, col["y_c"]] += 1.0
    J[6, col["x_c"]] += 1.0; J[7, col["y_c"]] += 1.0
    # Shape: theta_3
    J[4, col["theta_3"]] +=  _h_3*_st_3
    J[5, col["theta_3"]] += -_h_3*_ct_3
    J[6, col["theta_3"]] += -_h_3*_st_3
    J[7, col["theta_3"]] +=  _h_3*_ct_3
    # Shape: L_3
    J[4, col["L_3"]] += -0.5*_ct_3
    J[5, col["L_3"]] += -0.5*_st_3
    J[6, col["L_3"]] +=  0.5*_ct_3
    J[7, col["L_3"]] +=  0.5*_st_3
    # Leaf pair: robots 5,6
    _L_4 = state["L_4"]
    _th_4 = state["theta_4"]
    _ct_4, _st_4 = math.cos(_th_4), math.sin(_th_4)
    _h_4 = _L_4 / 2
    # Centroid columns (from parent propagation)
    J[8, col["x_c"]] += 1.0; J[9, col["y_c"]] += 1.0
    J[10, col["x_c"]] += 1.0; J[11, col["y_c"]] += 1.0
    # Shape: theta_4
    J[8, col["theta_4"]] +=  _h_4*_st_4
    J[9, col["theta_4"]] += -_h_4*_ct_4
    J[10, col["theta_4"]] += -_h_4*_st_4
    J[11, col["theta_4"]] +=  _h_4*_ct_4
    # Shape: L_4
    J[8, col["L_4"]] += -0.5*_ct_4
    J[9, col["L_4"]] += -0.5*_st_4
    J[10, col["L_4"]] +=  0.5*_ct_4
    J[11, col["L_4"]] +=  0.5*_st_4
    # Internal SAS-3 meta-node 1
    _p_1=state["p_1"]; _b_1=state["beta_1"]; _q_1=state["q_1"]
    _th_1=state["theta_c"]
    _ct_1 = math.cos(_th_1)
    _st_1 = math.sin(_th_1)
    _sb_1 = math.sin(_b_1)
    _cb_1 = math.cos(_b_1)
    _x1l_1 = _p_1
    _y1l_1 = 0.0
    _x2l_1 = 0.0
    _y2l_1 = 0.0
    _x3l_1 = _q_1*_cb_1
    _y3l_1 = _q_1*_sb_1
    _cxl_1 = (_x1l_1 + _x2l_1 + _x3l_1)/3.0
    _cyl_1 = (_y1l_1 + _y2l_1 + _y3l_1)/3.0
    J[0, col["theta_c"]] += -_st_1*(_x1l_1 - _cxl_1) - _ct_1*(_y1l_1 - _cyl_1)
    J[1, col["theta_c"]] +=  _ct_1*(_x1l_1 - _cxl_1) - _st_1*(_y1l_1 - _cyl_1)
    J[0, col["p_1"]] += _ct_1*(2.0/3.0) - _st_1*(0.0)
    J[1, col["p_1"]] += _st_1*(2.0/3.0) + _ct_1*(0.0)
    J[0, col["beta_1"]] += _ct_1*(_q_1*_sb_1/3.0) - _st_1*(-_q_1*_cb_1/3.0)
    J[1, col["beta_1"]] += _st_1*(_q_1*_sb_1/3.0) + _ct_1*(-_q_1*_cb_1/3.0)
    J[0, col["q_1"]] += _ct_1*(-_cb_1/3.0) - _st_1*(-_sb_1/3.0)
    J[1, col["q_1"]] += _st_1*(-_cb_1/3.0) + _ct_1*(-_sb_1/3.0)
    J[2, col["theta_c"]] += -_st_1*(_x1l_1 - _cxl_1) - _ct_1*(_y1l_1 - _cyl_1)
    J[3, col["theta_c"]] +=  _ct_1*(_x1l_1 - _cxl_1) - _st_1*(_y1l_1 - _cyl_1)
    J[2, col["p_1"]] += _ct_1*(2.0/3.0) - _st_1*(0.0)
    J[3, col["p_1"]] += _st_1*(2.0/3.0) + _ct_1*(0.0)
    J[2, col["beta_1"]] += _ct_1*(_q_1*_sb_1/3.0) - _st_1*(-_q_1*_cb_1/3.0)
    J[3, col["beta_1"]] += _st_1*(_q_1*_sb_1/3.0) + _ct_1*(-_q_1*_cb_1/3.0)
    J[2, col["q_1"]] += _ct_1*(-_cb_1/3.0) - _st_1*(-_sb_1/3.0)
    J[3, col["q_1"]] += _st_1*(-_cb_1/3.0) + _ct_1*(-_sb_1/3.0)
    J[4, col["theta_c"]] += -_st_1*(_x2l_1 - _cxl_1) - _ct_1*(_y2l_1 - _cyl_1)
    J[5, col["theta_c"]] +=  _ct_1*(_x2l_1 - _cxl_1) - _st_1*(_y2l_1 - _cyl_1)
    J[4, col["p_1"]] += _ct_1*(-1.0/3.0) - _st_1*(0.0)
    J[5, col["p_1"]] += _st_1*(-1.0/3.0) + _ct_1*(0.0)
    J[4, col["beta_1"]] += _ct_1*(_q_1*_sb_1/3.0) - _st_1*(-_q_1*_cb_1/3.0)
    J[5, col["beta_1"]] += _st_1*(_q_1*_sb_1/3.0) + _ct_1*(-_q_1*_cb_1/3.0)
    J[4, col["q_1"]] += _ct_1*(-_cb_1/3.0) - _st_1*(-_sb_1/3.0)
    J[5, col["q_1"]] += _st_1*(-_cb_1/3.0) + _ct_1*(-_sb_1/3.0)
    J[6, col["theta_c"]] += -_st_1*(_x2l_1 - _cxl_1) - _ct_1*(_y2l_1 - _cyl_1)
    J[7, col["theta_c"]] +=  _ct_1*(_x2l_1 - _cxl_1) - _st_1*(_y2l_1 - _cyl_1)
    J[6, col["p_1"]] += _ct_1*(-1.0/3.0) - _st_1*(0.0)
    J[7, col["p_1"]] += _st_1*(-1.0/3.0) + _ct_1*(0.0)
    J[6, col["beta_1"]] += _ct_1*(_q_1*_sb_1/3.0) - _st_1*(-_q_1*_cb_1/3.0)
    J[7, col["beta_1"]] += _st_1*(_q_1*_sb_1/3.0) + _ct_1*(-_q_1*_cb_1/3.0)
    J[6, col["q_1"]] += _ct_1*(-_cb_1/3.0) - _st_1*(-_sb_1/3.0)
    J[7, col["q_1"]] += _st_1*(-_cb_1/3.0) + _ct_1*(-_sb_1/3.0)
    J[8, col["theta_c"]] += -_st_1*(_x3l_1 - _cxl_1) - _ct_1*(_y3l_1 - _cyl_1)
    J[9, col["theta_c"]] +=  _ct_1*(_x3l_1 - _cxl_1) - _st_1*(_y3l_1 - _cyl_1)
    J[8, col["p_1"]] += _ct_1*(-1.0/3.0) - _st_1*(0.0)
    J[9, col["p_1"]] += _st_1*(-1.0/3.0) + _ct_1*(0.0)
    J[8, col["beta_1"]] += _ct_1*((-2.0/3.0)*_q_1*_sb_1) - _st_1*((2.0/3.0)*_q_1*_cb_1)
    J[9, col["beta_1"]] += _st_1*((-2.0/3.0)*_q_1*_sb_1) + _ct_1*((2.0/3.0)*_q_1*_cb_1)
    J[8, col["q_1"]] += _ct_1*((2.0/3.0)*_cb_1) - _st_1*((2.0/3.0)*_sb_1)
    J[9, col["q_1"]] += _st_1*((2.0/3.0)*_cb_1) + _ct_1*((2.0/3.0)*_sb_1)
    J[10, col["theta_c"]] += -_st_1*(_x3l_1 - _cxl_1) - _ct_1*(_y3l_1 - _cyl_1)
    J[11, col["theta_c"]] +=  _ct_1*(_x3l_1 - _cxl_1) - _st_1*(_y3l_1 - _cyl_1)
    J[10, col["p_1"]] += _ct_1*(-1.0/3.0) - _st_1*(0.0)
    J[11, col["p_1"]] += _st_1*(-1.0/3.0) + _ct_1*(0.0)
    J[10, col["beta_1"]] += _ct_1*((-2.0/3.0)*_q_1*_sb_1) - _st_1*((2.0/3.0)*_q_1*_cb_1)
    J[11, col["beta_1"]] += _st_1*((-2.0/3.0)*_q_1*_sb_1) + _ct_1*((2.0/3.0)*_q_1*_cb_1)
    J[10, col["q_1"]] += _ct_1*((2.0/3.0)*_cb_1) - _st_1*((2.0/3.0)*_sb_1)
    J[11, col["q_1"]] += _st_1*((2.0/3.0)*_cb_1) + _ct_1*((2.0/3.0)*_sb_1)