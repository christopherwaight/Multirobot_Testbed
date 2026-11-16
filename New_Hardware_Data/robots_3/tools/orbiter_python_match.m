function [xc_velocity, yc_velocity] = Orbiter(xc_actual, yc_actual, x_critical_point, y_critical_point, clockwise, desired_radius)
% Orbit the estimated critical point at radius desired_radius (Jacobian_primitive/MATLAB Function).
% Same STRUCTURE as the Python orbiter, critical_point_orbiter_plane_fitting in
% from_mac/VF_Robot/VF_Robot/src/control/primitives.py (paper Eq. 8/11):
%   v = k_t * tangent + k_r * (r - r_d) * r_hat,  radial speed clipped to +-max_radial_speed
%   r_hat = unit vector from the centroid toward the critical point (length 1), tangent = r_hat turned 90 deg,
%   so the tangential speed is always k_t; the radial speed grows with |r - r_d| (capped).
% clockwise = +1 gives Python's direction (tangent = [r_hat_y, -r_hat_x]); -1 reverses it.
% GAINS chosen by the user 2026-11-12: k_t 0.15, k_r 1 (the old Simulink values; slower orbit holds
% the radius better, strong radial pull). Python uses k_t 0.2, k_r 0.5 -> re-run the Python sims with
% k_t 0.15, k_r 1 for comparison (README_hardware.md 5.4). Used by BOTH the 3-robot and 4-robot models.
% History: before 2026-11-12 orbit_speed 0.15, K_p 1, radius ERROR clipped +-5; 2026-11-12 (briefly)
% Python gains k_t 0.2, k_r 0.5.

k_t = 0.15;              % tangential speed, m/s        [Python base_orbital_speed = 0.2]
k_r = 1;                 % radial gain, 1/s             [Python radial_gain = 0.5]
max_radial_speed = 0.3;  % radial speed clip, m/s       [Python max_radial_speed = 0.3]

dx = x_critical_point - xc_actual;
dy = y_critical_point - yc_actual;
current_radius = sqrt(dx^2 + dy^2);
if current_radius < 1e-6
    % On top of the critical point: direction undefined. Python picks a random direction;
    % here a fixed one, so runs stay reproducible.
    unit_x = 1; unit_y = 0; current_radius = 1e-6;
else
    unit_x = dx / current_radius;
    unit_y = dy / current_radius;
end

orbit_x = clockwise * unit_y;    % tangent
orbit_y = -clockwise * unit_x;

radial_speed = k_r * (current_radius - desired_radius);   % > 0 when too far -> move inward
radial_speed = max(min(radial_speed, max_radial_speed), -max_radial_speed);

xc_velocity = k_t * orbit_x + radial_speed * unit_x;
yc_velocity = k_t * orbit_y + radial_speed * unit_y;
end
