"""Forward kinematics for hexstar_snap hub-and-spoke (6 robots).
State vars: ['x_h', 'y_h', 'theta_c', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5']
"""
import math

def forward_kinematics(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
    """Hub = robot 6. Spokes = robots 1..5."""
    x_h, y_h = x6, y6
    betas = {}
    dx_1 = x1 - x_h
    dy_1 = y1 - y_h
    r_1 = math.hypot(dx_1, dy_1)
    betas[1] = math.atan2(dy_1, dx_1)
    dx_2 = x2 - x_h
    dy_2 = y2 - y_h
    r_2 = math.hypot(dx_2, dy_2)
    betas[2] = math.atan2(dy_2, dx_2)
    dx_3 = x3 - x_h
    dy_3 = y3 - y_h
    r_3 = math.hypot(dx_3, dy_3)
    betas[3] = math.atan2(dy_3, dx_3)
    dx_4 = x4 - x_h
    dy_4 = y4 - y_h
    r_4 = math.hypot(dx_4, dy_4)
    betas[4] = math.atan2(dy_4, dx_4)
    dx_5 = x5 - x_h
    dy_5 = y5 - y_h
    r_5 = math.hypot(dx_5, dy_5)
    betas[5] = math.atan2(dy_5, dx_5)
    theta_c = betas[1]
    gamma_2 = (betas[2] - betas[1] + math.pi) % (2*math.pi) - math.pi
    gamma_3 = (betas[3] - betas[1] + math.pi) % (2*math.pi) - math.pi
    gamma_4 = (betas[4] - betas[1] + math.pi) % (2*math.pi) - math.pi
    gamma_5 = (betas[5] - betas[1] + math.pi) % (2*math.pi) - math.pi
    return {'x_h': x_h, 'y_h': y_h, 'theta_c': theta_c, 'r_1': r_1, 'r_2': r_2, 'r_3': r_3, 'r_4': r_4, 'r_5': r_5, 'gamma_2': gamma_2, 'gamma_3': gamma_3, 'gamma_4': gamma_4, 'gamma_5': gamma_5}