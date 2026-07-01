# Scalar Fields vs Vector Fields: Quick Comparison

**For users coming from vector field background**

---

## Field Types

### Vector Field
```python
def vortex(x, y):
    u = -y / (x**2 + y**2)  # x-component
    v = x / (x**2 + y**2)   # y-component
    return u, v  # Direction + magnitude
```
- **Output**: 2D vector `(u, v)` - flow direction and speed
- **Example**: Wind velocity, water current
- **Visualization**: Arrow field (quiver plot)

### Scalar Field
```python
def bimodal_saddle(x, y):
    z = log(exp(G1) + exp(G2))  # Single value
    return z  # Height/potential
```
- **Output**: Single value `z` - elevation or potential
- **Example**: Temperature, elevation, energy landscape
- **Visualization**: Contour plot, surface plot

---

## What Robots Measure

### Vector Field
```python
u, v = field.get_value(x, y)
# Directly measure flow direction and magnitude
```
- Robots sense local flow vector
- Can follow streamlines
- No derivative needed for basic navigation

### Scalar Field
```python
z = field.get_value(x, y)
# Measure only height/potential
# Must compute gradient to get direction
```
- Robots sense only field value (like altimeter)
- Must estimate gradient from nearby samples
- Need Hessian for second-order methods

---

## Derivatives

### Vector Field: Jacobian (First-Order)
```
J = [∂u/∂x  ∂u/∂y]  ← How u changes with position
    [∂v/∂x  ∂v/∂y]  ← How v changes with position
```
- 2×2 matrix of first derivatives
- Describes local flow deformation
- Curl = ∂v/∂x - ∂u/∂y (rotation)
- Divergence = ∂u/∂x + ∂v/∂y (expansion)

### Scalar Field: Gradient + Hessian (Second-Order)
```
∇z = [∂z/∂x]  ← Steepest ascent direction
     [∂z/∂y]

H = [∂²z/∂x²   ∂²z/∂x∂y]  ← Curvature
    [∂²z/∂y∂x  ∂²z/∂y²  ]
```
- **Gradient**: Direction of steepest ascent (first derivative)
- **Hessian**: 2×2 matrix of second derivatives
- Hessian describes local curvature/topology

---

## Critical Points

### Vector Field
**Goal**: Find where flow is zero: `u = 0, v = 0`

Types (from Jacobian eigenvalues):
- **Sink**: Both λ < 0 (stable, inward flow)
- **Source**: Both λ > 0 (unstable, outward flow)
- **Vortex**: Complex λ (rotational flow)
- **Saddle**: Opposite-sign λ (mixed stability)

### Scalar Field
**Goal**: Find where gradient is zero: `∇z = 0`

Types (from Hessian eigenvalues):
- **Minimum**: Both λ > 0 (bowl shape)
- **Maximum**: Both λ < 0 (peak shape)
- **Saddle**: Opposite-sign λ (mountain pass)

---

## Optimization Methods

### Vector Field: Gradient-Based
```python
# Move opposite to flow (to find sink)
vx_c = -u
vy_c = -v
```
- First-order method
- Works for stable equilibria (sinks)
- Fails on unstable equilibria (sources, saddles)

### Scalar Field: Newton's Method
```python
# Newton step uses Hessian inverse
Δp = -H⁻¹ ∇z
```
- Second-order method
- Can converge to saddles (unstable!)
- Uses curvature information
- Faster convergence than gradient descent

---

## Why Newton vs Gradient Descent?

### Gradient Descent
```python
direction = -∇z  # Move down gradient
```
- **Works for**: Minima (stable)
- **Fails for**: Maxima and saddles (unstable)
- **Problem**: Saddle points are repulsive along one direction
- First-order method sees only slope, not curvature

### Newton's Method
```python
direction = -H⁻¹ ∇z  # Use curvature info
```
- **Works for**: All critical points (min, max, saddle)
- **Advantage**: Hessian "corrects" for instability
- **Key insight**: Eigenvalue sign tells us which directions are stable/unstable
- Second-order method accounts for topology

---

## Formation Role

### Vector Field Navigation
- **Triangle formation**: Samples field at 3 points
- **Estimates**: Jacobian matrix via plane fitting
- **Uses**: Direction to critical point from cross-products
- **Output**: Move toward estimated center

### Scalar Field Navigation
- **Same formation**: Samples z at 3 points (or 4 for quad)
- **Estimates**:
  - Gradient ∇z via plane fitting
  - Hessian H via C(4,3) gradient combinations (4-robot)
- **Uses**: Newton step Δp = -H⁻¹∇z
- **Output**: Move in Newton direction

---

## Key Mathematical Difference

### Vector Field Primitive
```python
def critical_point_finder(cluster):
    # Sample field at robots
    readings = cluster.sample_field_at_robots()  # [(u1,v1), (u2,v2), (u3,v3)]

    # Estimate Jacobian
    J = calculate_jacobian(cluster)

    # Direction to center from vector field properties
    direction = compute_direction_to_center(J, readings)

    return direction  # First-order method
```

### Scalar Field Primitive
```python
def newton_saddle_finder(cluster):
    # Sample field at robots
    z_values = [field.get_value(x, y) for x, y in positions]  # [z1, z2, z3, z4]

    # Estimate gradient at center
    ∇z = estimate_gradient(cluster)

    # Estimate Hessian from 4 gradients (if 4-robot)
    H = estimate_hessian(cluster)

    # Newton step
    Δp = -H⁻¹ @ ∇z

    return Δp  # Second-order method
```

---

## Saddle Point Uniqueness

### Vector Field Saddle
- Flow converges along one axis, diverges along another
- **Robots can detect** via curl and divergence
- Streamlines show topology directly

### Scalar Field Saddle
- Minimum along one axis, maximum along another (mountain pass)
- **Gradient descent fails** - repelled along unstable direction
- **Newton's method succeeds** - uses Hessian to handle instability
- Contour lines show topology

---

## Quick Cheat Sheet

| Aspect | Vector Field | Scalar Field |
|--------|--------------|--------------|
| **Output** | (u, v) vector | z value |
| **1st Derivative** | Jacobian J (2×2) | Gradient ∇z (2×1) |
| **2nd Derivative** | N/A (rarely used) | Hessian H (2×2) |
| **Critical Point** | u=0, v=0 | ∇z=0 |
| **Classification** | J eigenvalues | H eigenvalues |
| **Method** | Gradient-based | Newton's method |
| **Order** | First-order | Second-order |
| **Saddles** | Detectable | Need Hessian |

---

## When to Use Each

### Use Vector Field Navigation When:
- Field naturally represents flow (fluid, wind, traffic)
- Goal is to find stable equilibria (sinks)
- First-order methods suffice
- Field gives direction information directly

### Use Scalar Field Navigation When:
- Field represents potential/energy (temperature, elevation)
- Goal includes unstable critical points (saddles, maxima)
- Second-order convergence needed
- Field gives only magnitude (must compute gradient)

---

## Example: Finding a Saddle

### Vector Field Approach (FAILS)
```python
# Try to follow field backward
vx_c = -u  # Move opposite to flow
vy_c = -v

# Problem: Saddle is unstable in one direction
# Gradient pushes away from saddle → diverges
```

### Scalar Field Approach (SUCCEEDS)
```python
# Newton's method
grad = compute_gradient(field, x, y)
hess = compute_hessian(field, x, y)
step = -np.linalg.inv(hess) @ grad

# Hessian accounts for mixed stability
# Converges to saddle despite instability
```

---

## Code Architecture Parallel

Both use similar structure:

```
Vector:                          Scalar:
├── VectorField                  ├── ScalarField
├── VF_bases.py                  ├── SF_bases.py
├── environments/                ├── scalar_environments/
│   ├── Vortex.py               │   ├── Saddle.py
│   └── Sink.py                 │   └── (others)
├── primitives.py                ├── scalar_quad_primitives.py
└── kinematics.py                └── scalar_utils.py (NEW!)
```

Key addition for scalar: **`scalar_utils.py`** with gradient/Hessian/Newton functions

---

## Practical Tips

1. **Coming from vector fields?**
   - Scalar fields are "simpler" (single value vs vector)
   - But require derivatives (gradient, Hessian)
   - Newton's method is more complex than gradient following

2. **Debugging scalar code:**
   - Check gradient norm → should approach 0 at critical point
   - Check Hessian eigenvalues → opposite signs = saddle
   - Visualize contours, not arrows

3. **Formation control:**
   - Identical to vector field case
   - Same kinematics (p, q, β for 3-robot)
   - Same momentum dynamics

4. **Experiment workflow:**
   - Use `visualize_field.py` to see topology
   - Use `demo_scalar_fields.py` to compare fields
   - Use `compare_rotation_methods.py` to test convergence

---

**Bottom line**: If you understand vector field Jacobians, scalar field Hessians are similar but operate on gradients instead of flow components. Newton's method is the "smart" way to handle unstable equilibria.
