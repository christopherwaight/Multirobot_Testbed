"""
Kinematics for 6-robot pentagon cluster-of-clusters.

Formation structure:
  - 3 pairs of robots: pair A (robots 1,2), pair B (robots 3,4), pair C (robots 5,6)
  - Pair midpoints form a triangle (SAS: p_1, beta_1, q_1)
  - Centroid is the mean of the three midpoints

State variables (12 total):
  x_c, y_c       -- cluster centroid
  theta_c        -- triangle heading (angle from midpoint A to midpoint B)
  p_1, beta_1, q_1 -- SAS triangle: side A-B, angle at B, side B-C
  L_2, theta_2   -- pair A length and orientation
  L_3, theta_3   -- pair B length and orientation
  L_4, theta_4   -- pair C length and orientation

Robot index convention (1-based):
  robots 1,2 -> pair A
  robots 3,4 -> pair B
  robots 5,6 -> pair C

Math adapted from clusterbuilder-generated my_pentagon_*.py files.
"""
import math
import numpy as np


# -----------------------------------------------------------------------
# Forward kinematics
# -----------------------------------------------------------------------

def forward_kinematics(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
    """
    Compute cluster state from 6 robot positions.

    Returns dict with keys:
      x_c, y_c, theta_c, p_1, beta_1, q_1,
      L_2, theta_2, L_3, theta_3, L_4, theta_4
    """
    # Pair midpoints
    cx_A = (x1 + x2) / 2.0
    cy_A = (y1 + y2) / 2.0
    cx_B = (x3 + x4) / 2.0
    cy_B = (y3 + y4) / 2.0
    cx_C = (x5 + x6) / 2.0
    cy_C = (y5 + y6) / 2.0

    # Pair parameters
    L_2 = math.hypot(x2 - x1, y2 - y1)
    theta_2 = math.atan2(y2 - y1, x2 - x1)
    L_3 = math.hypot(x4 - x3, y4 - y3)
    theta_3 = math.atan2(y4 - y3, x4 - x3)
    L_4 = math.hypot(x6 - x5, y6 - y5)
    theta_4 = math.atan2(y6 - y5, x6 - x5)

    # Triangle SAS parameters from midpoints
    p_1 = math.hypot(cx_B - cx_A, cy_B - cy_A)
    q_1 = math.hypot(cx_C - cx_B, cy_C - cy_B)
    r_1 = math.hypot(cx_A - cx_C, cy_A - cy_C)

    # Angle at midpoint B via law of cosines
    denom = 2.0 * p_1 * q_1 + 1e-10
    cos_beta = max(-1.0, min(1.0, (p_1**2 + q_1**2 - r_1**2) / denom))
    beta_1 = math.acos(cos_beta)

    # Centroid of triangle
    x_c = (cx_A + cx_B + cx_C) / 3.0
    y_c = (cy_A + cy_B + cy_C) / 3.0

    # Heading = angle from midpoint A to midpoint B
    theta_c = math.atan2(cy_B - cy_A, cx_B - cx_A)

    return {
        'x_c': x_c, 'y_c': y_c, 'theta_c': theta_c,
        'p_1': p_1, 'beta_1': beta_1, 'q_1': q_1,
        'L_2': L_2, 'theta_2': theta_2,
        'L_3': L_3, 'theta_3': theta_3,
        'L_4': L_4, 'theta_4': theta_4,
    }


# -----------------------------------------------------------------------
# Inverse kinematics
# -----------------------------------------------------------------------

def inverse_kinematics(x_c, y_c, theta_c, p_1, beta_1, q_1,
                       L_2, theta_2, L_3, theta_3, L_4, theta_4):
    """
    Compute 6 robot positions from cluster state.

    Returns:
      (x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6)
    """
    # Build triangle midpoints in local frame (SAS with midpoint B at origin)
    x2l = 0.0; y2l = 0.0
    x1l = p_1; y1l = 0.0
    x3l = q_1 * math.cos(beta_1)
    y3l = q_1 * math.sin(beta_1)

    # Shift so local centroid is at origin
    cxl = (x1l + x2l + x3l) / 3.0
    cyl = (y1l + y2l + y3l) / 3.0
    x1l -= cxl; y1l -= cyl
    x2l -= cxl; y2l -= cyl
    x3l -= cxl; y3l -= cyl

    # Rotate by theta_c and translate to (x_c, y_c)
    ct = math.cos(theta_c)
    st = math.sin(theta_c)

    cx_A = ct * x1l - st * y1l + x_c
    cy_A = st * x1l + ct * y1l + y_c
    cx_B = ct * x2l - st * y2l + x_c
    cy_B = st * x2l + ct * y2l + y_c
    cx_C = ct * x3l - st * y3l + x_c
    cy_C = st * x3l + ct * y3l + y_c

    # Expand each pair around its midpoint
    h2 = L_2 / 2.0
    ct2 = math.cos(theta_2); st2 = math.sin(theta_2)
    x1 = cx_A - h2 * ct2;  y1 = cy_A - h2 * st2
    x2 = cx_A + h2 * ct2;  y2 = cy_A + h2 * st2

    h3 = L_3 / 2.0
    ct3 = math.cos(theta_3); st3 = math.sin(theta_3)
    x3 = cx_B - h3 * ct3;  y3 = cy_B - h3 * st3
    x4 = cx_B + h3 * ct3;  y4 = cy_B + h3 * st3

    h4 = L_4 / 2.0
    ct4 = math.cos(theta_4); st4 = math.sin(theta_4)
    x5 = cx_C - h4 * ct4;  y5 = cy_C - h4 * st4
    x6 = cx_C + h4 * ct4;  y6 = cy_C + h4 * st4

    return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6


# -----------------------------------------------------------------------
# Inverse Jacobian (12x12)
# -----------------------------------------------------------------------

# State variable order for Jacobian columns
STATE_VARS = [
    'x_c', 'y_c', 'theta_c',
    'p_1', 'beta_1', 'q_1',
    'L_2', 'theta_2',
    'L_3', 'theta_3',
    'L_4', 'theta_4',
]

# Robot coordinate row order: [x1,y1, x2,y2, x3,y3, x4,y4, x5,y5, x6,y6]


def compute_inverse_jacobian(state):
    """
    Compute the 12x12 inverse Jacobian mapping state velocities to robot velocities.

    Args:
        state: dict with all 12 state variables

    Returns:
        J: (12, 12) numpy array  s.t.  J @ q_dot = r_dot
           q_dot is state velocity vector (STATE_VARS order)
           r_dot is [vx1,vy1, vx2,vy2, ..., vx6,vy6]
    """
    J = np.zeros((12, 12))
    col = {v: i for i, v in enumerate(STATE_VARS)}

    p_1    = state['p_1']
    beta_1 = state['beta_1']
    q_1    = state['q_1']
    theta_c = state['theta_c']
    L_2 = state['L_2']; theta_2 = state['theta_2']
    L_3 = state['L_3']; theta_3 = state['theta_3']
    L_4 = state['L_4']; theta_4 = state['theta_4']

    ct = math.cos(theta_c)
    st = math.sin(theta_c)

    # Local midpoint positions (same computation as IK, before rotation)
    x1l_raw = p_1;   y1l_raw = 0.0
    x2l_raw = 0.0;   y2l_raw = 0.0
    x3l_raw = q_1 * math.cos(beta_1)
    y3l_raw = q_1 * math.sin(beta_1)

    cxl = (x1l_raw + x2l_raw + x3l_raw) / 3.0
    cyl = (y1l_raw + y2l_raw + y3l_raw) / 3.0

    lx = [x1l_raw - cxl, x2l_raw - cxl, x3l_raw - cxl]
    ly = [y1l_raw - cyl, y2l_raw - cyl, y3l_raw - cyl]

    pair_L     = [L_2, L_3, L_4]
    pair_theta = [theta_2, theta_3, theta_4]
    pair_L_key     = ['L_2', 'L_3', 'L_4']
    pair_theta_key = ['theta_2', 'theta_3', 'theta_4']

    sb     = math.sin(beta_1)
    cb_val = math.cos(beta_1)

    # d(local midpoint)/d(shape vars) for each pair k in {0,1,2}
    # Derived analytically from the SAS local-frame construction.
    dlx_dp = [2.0/3,           -1.0/3,         -1.0/3        ]
    dly_dp = [0.0,              0.0,             0.0          ]
    dlx_db = [q_1*sb/3,        q_1*sb/3,       -2.0/3*q_1*sb ]
    dly_db = [-q_1*cb_val/3,  -q_1*cb_val/3,   2.0/3*q_1*cb_val]
    dlx_dq = [-cb_val/3,      -cb_val/3,        2.0/3*cb_val  ]
    dly_dq = [-sb/3,           -sb/3,            2.0/3*sb     ]

    for k in range(3):
        row_a = 4 * k      # robot 2k+1: rows row_a, row_a+1
        row_b = 4 * k + 2  # robot 2k+2: rows row_b, row_b+1

        lxk = lx[k]; lyk = ly[k]
        Lk  = pair_L[k]
        thk = pair_theta[k]
        ctk = math.cos(thk); stk = math.sin(thk)
        hk  = Lk / 2.0

        # d/d(x_c), d/d(y_c): both robots in pair shift equally
        J[row_a,     col['x_c']] = 1.0
        J[row_a + 1, col['y_c']] = 1.0
        J[row_b,     col['x_c']] = 1.0
        J[row_b + 1, col['y_c']] = 1.0

        # d/d(theta_c): midpoint rotates around cluster centroid
        dmx_dth = -st * lxk - ct * lyk
        dmy_dth =  ct * lxk - st * lyk
        J[row_a,     col['theta_c']] = dmx_dth
        J[row_a + 1, col['theta_c']] = dmy_dth
        J[row_b,     col['theta_c']] = dmx_dth
        J[row_b + 1, col['theta_c']] = dmy_dth

        # d/d(pair_theta_k): only affects the spread within the pair
        J[row_a,     col[pair_theta_key[k]]] =  hk * stk
        J[row_a + 1, col[pair_theta_key[k]]] = -hk * ctk
        J[row_b,     col[pair_theta_key[k]]] = -hk * stk
        J[row_b + 1, col[pair_theta_key[k]]] =  hk * ctk

        # d/d(L_k): pair spread changes symmetrically
        J[row_a,     col[pair_L_key[k]]] = -0.5 * ctk
        J[row_a + 1, col[pair_L_key[k]]] = -0.5 * stk
        J[row_b,     col[pair_L_key[k]]] =  0.5 * ctk
        J[row_b + 1, col[pair_L_key[k]]] =  0.5 * stk

        # d/d(SAS shape vars p_1, beta_1, q_1): move midpoints via rotation
        for shape_key, dlx, dly in [
            ('p_1',    dlx_dp[k], dly_dp[k]),
            ('beta_1', dlx_db[k], dly_db[k]),
            ('q_1',    dlx_dq[k], dly_dq[k]),
        ]:
            dmx = ct * dlx - st * dly
            dmy = st * dlx + ct * dly
            J[row_a,     col[shape_key]] += dmx
            J[row_a + 1, col[shape_key]] += dmy
            J[row_b,     col[shape_key]] += dmx
            J[row_b + 1, col[shape_key]] += dmy

    return J


def shape_velocities_to_robot_velocities(J, q_dot):
    """
    Convert state velocity vector to robot velocity vector.

    Args:
        J: (12, 12) inverse Jacobian from compute_inverse_jacobian
        q_dot: (12,) array in STATE_VARS order

    Returns:
        r_dot: (12,) array [vx1,vy1, vx2,vy2, ..., vx6,vy6]
    """
    return J @ q_dot
