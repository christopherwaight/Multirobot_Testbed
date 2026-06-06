"""Inverse kinematics for hexstar_on_snap hub-and-spoke (6 robots)."""
import math

def inverse_kinematics(state):
    """state: dict -> {robot_idx: (x, y)}"""
    x_h = state["x_h"]
    y_h = state["y_h"]
    theta_c = state["theta_c"]
    positions = {6: (x_h, y_h)}
    r_1 = state["r_1"]
    gamma_1 = 0.0
    angle_1 = theta_c + gamma_1
    positions[1] = (x_h + r_1 * math.cos(angle_1), y_h + r_1 * math.sin(angle_1))
    r_2 = state["r_2"]
    gamma_2 = state["gamma_2"]
    angle_2 = theta_c + gamma_2
    positions[2] = (x_h + r_2 * math.cos(angle_2), y_h + r_2 * math.sin(angle_2))
    r_3 = state["r_3"]
    gamma_3 = state["gamma_3"]
    angle_3 = theta_c + gamma_3
    positions[3] = (x_h + r_3 * math.cos(angle_3), y_h + r_3 * math.sin(angle_3))
    r_4 = state["r_4"]
    gamma_4 = state["gamma_4"]
    angle_4 = theta_c + gamma_4
    positions[4] = (x_h + r_4 * math.cos(angle_4), y_h + r_4 * math.sin(angle_4))
    r_5 = state["r_5"]
    gamma_5 = state["gamma_5"]
    angle_5 = theta_c + gamma_5
    positions[5] = (x_h + r_5 * math.cos(angle_5), y_h + r_5 * math.sin(angle_5))
    return positions