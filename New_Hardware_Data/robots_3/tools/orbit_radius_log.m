function y = orbit_radius_log(xc, yc, x_cp, y_cp, desired_radius)
% LOGGING ONLY: the code inside the 'LOG orbit_radius fn' MATLAB Function block in Jacobian_primitive.
% Same radius and radius error as the Orbiter (Jacobian_primitive/MATLAB Function).
% Output (1x2): [current_radius, desired_radius - current_radius]  (error > 0: too close)
r = sqrt((x_cp - xc)^2 + (y_cp - yc)^2);
y = [r, desired_radius - r];
end
