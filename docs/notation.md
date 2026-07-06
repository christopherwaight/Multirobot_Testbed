# Notation and Terminology

Symbols used in Paper_Draft_4A.tex (vector field paper) and the associated notebooks.
Use these exactly. Do not introduce alternative notation.

The two hard rules (the alpha collision and bold p vs scalar p) live inline in
CLAUDE.md because they are anti-error directives, not lookups. This file is the full
symbol reference.

## Field and Position

| Symbol | Meaning |
|--------|---------|
| p (bold) | Position vector in the plane, p = (x, y) |
| v(p) (bold) | Vector field at position p, output is (u, v) |
| p* (bold) | Critical point: location where v(p*) = 0 |
| p_c (bold) | Cluster centroid: mean of robot positions |
| p_i (bold) | Position of robot i, p_i = (x_i, y_i) |
| (u_i, v_i) | Field measurement (scalar components) by robot i |

## Estimation Framework

| Symbol | Meaning |
|--------|---------|
| A (bold) | Formation matrix (3x3): rows are [x_i, y_i, 1] |
| theta_u, theta_v (bold) | Plane fit coefficient vectors: [a, b, c]^T and [d, e, f]^T |
| J (bold) | Estimated Jacobian matrix: [[a, b], [d, e]], 2x2 |
| h (bold, lowercase) | Offset vector: [c, f]^T (constant terms from plane fits) |
| det(J) | Determinant of J; must be nonzero for a unique critical point estimate |
| lambda | Eigenvalue(s) of J; used to classify critical point type |
| alpha +/- i*omega | Complex eigenvalue pair for spiral types; alpha here is the eigenvalue real part (see the alpha collision rule in CLAUDE.md) |

## Critical Point Types (Table I in paper)

| Field Type | Critical Point Name | Eigenvalue Signature |
|------------|--------------------|--------------------|
| Vortex | Center | purely imaginary: +/- i*omega |
| Sinking Vortex | Stable Spiral | complex with alpha < 0 |
| Sink | Stable Node | both real negative |
| Source | Unstable Node | both real positive |
| Spewing Vortex | Unstable Spiral | complex with alpha > 0 |
| Saddle | Saddle Point | real with opposite signs |

## Control Laws

| Symbol | Meaning |
|--------|---------|
| p_c_dot (bold) | Commanded centroid velocity |
| k | Proportional gain for attraction control (scalar, positive) |
| r (bold) | Radial vector from centroid to critical point: r = p* - p_c |
| r_hat (bold) | Unit radial vector: r / ||r|| |
| r_hat_perp (bold) | Tangent unit vector: r_hat rotated -90 degrees |
| r_d | Desired orbital radius (scalar) |
| k_t | Tangential gain; sign determines orbit direction |
| k_r | Radial gain (positive); drives convergence to r_d |

## Robot Dynamics

| Symbol | Meaning |
|--------|---------|
| tau | Actuator time constant; measured at 0.3 s on hardware |
| alpha | Momentum coefficient: alpha = exp(-dt/tau). alpha here is the momentum coefficient (see the alpha collision rule in CLAUDE.md). |
| dt | Control period: 0.1 s (10 Hz) |
| v_des (bold) | Commanded velocity input to robot |
| v[k] (bold) | Robot velocity at discrete step k |
| v_max | Maximum robot speed: 0.3 m/s |
| v_stiction | Minimum speed to overcome stiction: 0.05 m/s (hardware value; the 3-robot Python sim default is 0.025, see AUDIT_REPORT_4A.md item E2) |

## Formation (SAS Parameterization, 3-robot)

| Symbol | Meaning |
|--------|---------|
| p (scalar) | Distance from robot 1 to robot 2 (see the bold-p vs scalar-p hard rule in CLAUDE.md) |
| q (scalar) | Distance from robot 2 to robot 3 |
| beta | Interior angle at robot 2 |
| x_c, y_c | Centroid position in global frame |
| theta_c | Cluster heading angle |

Cluster space state vector: (x_c, y_c, theta_c, p, q, beta), six variables.
