"""Forward kinematics for my_star_snap cluster-of-clusters (6 robots)."""
import math

def forward_kinematics(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
    positions = {1: (x1, y1), 2: (x2, y2), 3: (x3, y3), 4: (x4, y4), 5: (x5, y5), 6: (x6, y6)}
    state = {}
    _fk_impl(positions, state)
    return state

# Internal implementation (generated)
def _fk_impl(positions, state):
    xa_2, ya_2 = positions[1]
    xb_2, yb_2 = positions[2]
    cx_2 = (xa_2 + xb_2) / 2
    cy_2 = (ya_2 + yb_2) / 2
    state["theta_2"] = math.atan2(yb_2 - ya_2, xb_2 - xa_2)
    state["L_2"] = math.hypot(xb_2 - xa_2, yb_2 - ya_2)
    xa_3, ya_3 = positions[3]
    xb_3, yb_3 = positions[4]
    cx_3 = (xa_3 + xb_3) / 2
    cy_3 = (ya_3 + yb_3) / 2
    state["theta_3"] = math.atan2(yb_3 - ya_3, xb_3 - xa_3)
    state["L_3"] = math.hypot(xb_3 - xa_3, yb_3 - ya_3)
    xa_4, ya_4 = positions[5]
    xb_4, yb_4 = positions[6]
    cx_4 = (xa_4 + xb_4) / 2
    cy_4 = (ya_4 + yb_4) / 2
    state["theta_4"] = math.atan2(yb_4 - ya_4, xb_4 - xa_4)
    state["L_4"] = math.hypot(xb_4 - xa_4, yb_4 - ya_4)
    xa_m1, ya_m1 = cx_2, cy_2
    xb_m1, yb_m1 = cx_3, cy_3
    xc_m1, yc_m1 = cx_4, cy_4
    p_m1 = math.hypot(xb_m1-xa_m1, yb_m1-ya_m1)
    q_m1 = math.hypot(xc_m1-xb_m1, yc_m1-yb_m1)
    r_m1 = math.hypot(xa_m1-xc_m1, ya_m1-yc_m1)
    cb_m1 = max(-1.0,min(1.0,(p_m1**2+q_m1**2-r_m1**2)/(2*p_m1*q_m1+1e-10)))
    state["p_1"] = p_m1
    state["beta_1"] = math.acos(cb_m1)
    state["q_1"] = q_m1
    cx_1 = (xa_m1+xb_m1+xc_m1)/3
    cy_1 = (ya_m1+yb_m1+yc_m1)/3
    state["theta_c"] = math.atan2(yb_m1-ya_m1, xb_m1-xa_m1)
    state["x_c"] = cx_1
    state["y_c"] = cy_1
    return state