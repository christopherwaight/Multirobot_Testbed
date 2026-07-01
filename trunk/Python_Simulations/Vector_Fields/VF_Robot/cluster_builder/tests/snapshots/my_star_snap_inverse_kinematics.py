"""Inverse kinematics for my_star_snap cluster-of-clusters (6 robots)."""
import math

def inverse_kinematics(state):
    """state: dict -> {robot_idx: (x, y)}"""
    positions = {}
    _ik_impl(state["x_c"], state["y_c"], state["theta_c"], state, positions)
    return positions

def _ik_impl(cx, cy, theta, state, positions):
    _p_1 = state["p_1"]; _b_1 = state["beta_1"]; _q_1 = state["q_1"]
    _th_1 = state["theta_c"]
    _x2l_1, _y2l_1 = 0.0, 0.0
    _x1l_1, _y1l_1 = _p_1, 0.0
    _x3l_1 = _q_1*math.cos(_b_1); _y3l_1 = _q_1*math.sin(_b_1)
    _cxl_1 = (_x1l_1+_x2l_1+_x3l_1)/3
    _cyl_1 = (_y1l_1+_y2l_1+_y3l_1)/3
    _x1l_1 -= _cxl_1; _y1l_1 -= _cyl_1
    _x2l_1 -= _cxl_1; _y2l_1 -= _cyl_1
    _x3l_1 -= _cxl_1; _y3l_1 -= _cyl_1
    _ct_1, _st_1 = math.cos(_th_1), math.sin(_th_1)
    cx_2 = _ct_1*_x1l_1 - _st_1*_y1l_1 + cx
    cy_2 = _st_1*_x1l_1 + _ct_1*_y1l_1 + cy
    cx_3 = _ct_1*_x2l_1 - _st_1*_y2l_1 + cx
    cy_3 = _st_1*_x2l_1 + _ct_1*_y2l_1 + cy
    cx_4 = _ct_1*_x3l_1 - _st_1*_y3l_1 + cx
    cy_4 = _st_1*_x3l_1 + _ct_1*_y3l_1 + cy
    _L_2 = state["L_2"]
    _th_2 = state["theta_2"]
    _ct_2, _st_2 = math.cos(_th_2), math.sin(_th_2)
    positions[1] = (cx_2 - (_L_2/2)*_ct_2, cy_2 - (_L_2/2)*_st_2)
    positions[2] = (cx_2 + (_L_2/2)*_ct_2, cy_2 + (_L_2/2)*_st_2)
    _L_3 = state["L_3"]
    _th_3 = state["theta_3"]
    _ct_3, _st_3 = math.cos(_th_3), math.sin(_th_3)
    positions[3] = (cx_3 - (_L_3/2)*_ct_3, cy_3 - (_L_3/2)*_st_3)
    positions[4] = (cx_3 + (_L_3/2)*_ct_3, cy_3 + (_L_3/2)*_st_3)
    _L_4 = state["L_4"]
    _th_4 = state["theta_4"]
    _ct_4, _st_4 = math.cos(_th_4), math.sin(_th_4)
    positions[5] = (cx_4 - (_L_4/2)*_ct_4, cy_4 - (_L_4/2)*_st_4)
    positions[6] = (cx_4 + (_L_4/2)*_ct_4, cy_4 + (_L_4/2)*_st_4)