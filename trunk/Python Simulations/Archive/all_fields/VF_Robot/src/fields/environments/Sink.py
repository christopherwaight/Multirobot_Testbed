import numpy as np
import matplotlib.pyplot as plt

#5. Define the Vector Field by summing different bases together.


def sink1(x, y):
    # Define sink center
    x_sink = 0
    y_sink = 0
    
    # Calculate distance from center
    x_centre = x - x_sink
    y_centre = y - y_sink
    
    # Calculate radial distance
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Create inward flow, strength increases as you get closer to center
    strength = -1.0  # Negative for sink (positive would make a source)
    v_r = strength * r  # Radial velocity
    
    # Convert to cartesian coordinates
    u = v_r * x_centre / r
    v = v_r * y_centre / r
    
    # Clip to prevent extreme values
    u = np.clip(u, -100, 100)
    v = np.clip(v, -100, 100)
    
    return u, v


def sink2(x, y):
    # Define sink center
    x_sink = 0
    y_sink = 0
    
    # Calculate distance from center
    x_centre = x - x_sink
    y_centre = y - y_sink
    
    # Calculate radial distance
    r = np.sqrt(x_centre**2 + y_centre**2)
    r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
    # Create unit vectors pointing toward the center
    # Normalize the direction vector to have constant magnitude
    strength = -1.0  # Negative for sink (positive would make a source)
    
    # Create unit vectors in the radial direction
    u = strength * x_centre / r
    v = strength * y_centre / r
    
    return u, v


def sink3(x, y):

    center_x = 0
    center_y = 0
    # Calculate radius and angle
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Adjusting field components to create a "spinning plate" effect
    u = - np.sin(theta) * r**2
    v =   np.cos(theta) * r**2

    # Calculating A sink
    x_centre = x - center_x
    y_centre = y - center_y
    r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

    # Sink
    u =  -0.15*x_centre / r2**2
    v =  -0.15*y_centre / r2**2

    return u, v

# def sink3(x, y, eps=0.2):
#     """
#     Smooth analogue of (sink2 - sink1):
#       F(x,y) ≈ (1 - 1/r) * (x, y), but with r smoothed by eps.
#     Behavior:
#       - For eps < 1: inward near the origin (Jacobian negative), 
#         zero ring at r* ≈ sqrt(1 - eps^2), outward beyond that.
#     """
#     x_c, y_c = x, y
#     r = np.sqrt(x_c**2 + y_c**2)
#     r_eps = np.sqrt(r**2 + eps**2)          # smooth |r|
    
#     coeff = 1.0 - 1.0 / r_eps               # ~ 1 - 1/r
#     u = coeff * x_c
#     v = coeff * y_c
#     return u, v



# def source1(x, y):
#     # Define sink center
#     x_sink = 0
#     y_sink = 0
    
#     # Calculate distance from center
#     x_centre = x - x_sink
#     y_centre = y - y_sink
    
#     # Calculate radial distance
#     r = np.sqrt(x_centre**2 + y_centre**2)
#     r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
#     # Create inward flow, strength increases as you get closer to center
#     strength = -1.0  # Negative for sink (positive would make a source)
#     v_r = strength * r  # Radial velocity
    
#     # Convert to cartesian coordinates
#     u = -v_r * x_centre / r
#     v = -v_r * y_centre / r
    
#     # Clip to prevent extreme values
#     u = np.clip(u, -100, 100)
#     v = np.clip(v, -100, 100)
    
#     return u, v


# def source2(x, y):
#     # Define sink center
#     x_sink = 0
#     y_sink = 0
    
#     # Calculate distance from center
#     x_centre = x - x_sink
#     y_centre = y - y_sink
    
#     # Calculate radial distance
#     r = np.sqrt(x_centre**2 + y_centre**2)
#     r = np.where(r == 0, 1e-6, r)  # Avoid division by zero
    
#     # Create unit vectors pointing toward the center
#     # Normalize the direction vector to have constant magnitude
#     strength = 1.0  # Negative for sink (positive would make a source)
    
#     # Create unit vectors in the radial direction
#     u = strength * x_centre / r
#     v = strength * y_centre / r
    
#     return u, v



# def source3(x, y):

#     center_x = 0
#     center_y = 0
#     # Calculate radius and angle
#     r = np.sqrt((x - center_x)**2 + (y - center_y)**2) + 1e-10  # small epsilon to prevent divide-by-zero
#     theta = np.arctan2(y - center_y, x - center_x)
    
#     # Adjusting field components to create a "spinning plate" effect
#     u = - np.sin(theta) * r**2
#     v =   np.cos(theta) * r**2

#     # Calculating A sink
#     x_centre = x - center_x
#     y_centre = y - center_y
#     r2 = np.sqrt(x_centre**2 + y_centre**2) + 5*1e-3

#     # Sink
#     u = 0.15*x_centre / r2**2
#     v = 0.15*y_centre / r2**2

#     return u, v


## Uncomment to pplot
# Create a grid of points
# x = np.linspace(-5, 5, 20)
# y = np.linspace(-5, 5, 20)
# X, Y = np.meshgrid(x, y)

# # Calculate the vector field
# U, V = source3(X, Y)

# # Create the plot
# fig, ax = plt.subplots(figsize=(10, 8))

# # Plot the vector field using quiver
# ax.quiver(X, Y, U, V, color='blue', alpha=0.6)

# # Add streamlines for better visualization
# ax.streamplot(X, Y, U, V, color='red', linewidth=1, density=1.5)

# # Mark the sink center
# ax.plot(0, 0, 'ko', markersize=8, label='Sink Center')

# # Set equal aspect ratio and add grid
# ax.set_aspect('equal')
# ax.grid(True, alpha=0.3)

# # Labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Sink Vector Field')
# ax.legend()

# # Set axis limits
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)

# plt.show()


def roi_finder2_simple(
    cluster,
    mode="auto",          # "auto" | "radial" | "vortex"
    tiny=1e-12, frac=0.5,
    verbose=True, log_vectors=False, log_path="/mnt/data/roi_finder2_simple.log"
):
    """Super-simple k-type finder: pick radial or vortex and use the textbook shortcut."""
    def _emit(msg):
        if verbose: print(msg)
        if log_path:
            try:
                with open(log_path, "a") as f: f.write(msg + "\n")
            except Exception:
                pass

    def _env_eval(xy):
        x, y = map(float, xy)
        for attr in ("environment_function", "env_fn", "field_fn"):
            fn = getattr(cluster, attr, None)
            if callable(fn): return np.array(fn(x, y), float)
        for meth in ("field_at", "eval_field", "evaluate_field"):
            m = getattr(cluster, meth, None)
            if callable(m): return np.array(m(x, y), float)
        R = np.array(cluster.bot_readings(), float)
        return R.mean(axis=0)

    def _calc_J(center_xy):
        try:
            curl, div, du_dx, du_dy, dv_dx, dv_dy = calculate_jacobian(cluster)
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            return J, float(div), float(curl)
        except Exception:
            x, y = map(float, center_xy)
            base = max(1.0, np.hypot(x, y)); h = 1e-4 * base
            u_x, v_x = _env_eval((x + h, y)); u_X, v_X = _env_eval((x - h, y))
            u_y, v_y = _env_eval((x, y + h)); u_Y, v_Y = _env_eval((x, y - h))
            du_dx = (u_x - u_X)/(2*h); du_dy = (u_y - u_Y)/(2*h)
            dv_dx = (v_x - v_X)/(2*h); dv_dy = (v_y - v_Y)/(2*h)
            J = np.array([[du_dx, du_dy],[dv_dx, dv_dy]], float)
            return J, float(du_dx + dv_dy), float(dv_dx - du_dy)

    def _rot_cw(z): return np.array([ z[1], -z[0] ], float)

    c = np.asarray(cluster.cluster_centre, float)
    v = np.array(cluster.bot_readings(), float).sum(axis=0)
    V = float(np.hypot(*v))
    if V < tiny:
        _emit(f"[ROI2s] stagnation |v|≈0 at {c.tolist()} — no move"); return cluster.cluster_centre

    J, div, curl = _calc_J(c)
    t = v / (V + tiny)
    s = J @ v; s_perp = s - (t @ s) * t     # optional inward cue for vortex

    # choose simple model
    if mode == "radial":
        choose = "radial"
    elif mode == "vortex":
        choose = "vortex"
    else:
        # auto: whichever invariant is larger in V-units
        dbar = abs(div)/(V + tiny); wbar = abs(curl)/(V + tiny)
        choose = "radial" if dbar >= wbar else "vortex"

    if choose == "radial":
        r_hat = V / (abs(div) + tiny)
        dir_c = -np.sign(div if div != 0 else 1.0) * t
        tag = "radial: r=V/|div|, dir=-sign(div)·t"
    else:
        r_hat = V / (abs(curl) + tiny)
        sp_n = float(np.linalg.norm(s_perp))
        if sp_n > 1e-12:
            dir_c = -s_perp / sp_n
            tag = "vortex: r=V/|ω|, dir=-ŝ⊥"
        else:
            dir_c = -np.sign(curl if curl != 0 else 1.0) * _rot_cw(t)
            tag = "vortex: r=V/|ω|, dir≈-sign(ω)·R_cw·t"

    h = float(getattr(cluster, "step_size", 0.1))
    h_eff = min(h, frac * r_hat) if np.isfinite(r_hat) and r_hat > tiny else h

    if log_vectors:
        _emit(f"[ROI2s] v={v.tolist()} |v|={V:.4f}  J=[[{J[0,0]:+.3f},{J[0,1]:+.3f}],[{J[1,0]:+.3f},{J[1,1]:+.3f}]]")
        _emit(f"         div={div:+.4f}  ω={curl:+.4f}  s⊥={s_perp.tolist()}")
    _emit(f"[ROI2s] {tag}  r̂={r_hat:.3f}  h_eff={h_eff:.3f}")

    cluster.cluster_centre[0] += h_eff * dir_c[0]
    cluster.cluster_centre[1] += h_eff * dir_c[1]
    _emit(f"[ROI2s] → new_pos={cluster.cluster_centre.tolist()}")
    return cluster.cluster_centre
