% Assuming your variables are already loaded:
% robot1_pose, robot2_pose, robot3_pose (each Nx3: x, y, theta)
% Hue (timeseries with Nx3 data: one column per robot, values 0-1)
% Saturation (timeseries with Nx3 data: one column per robot, values representing magnitude)

% Extract data from timeseries objects
hue_data = hue.Data;
saturation_data = saturation.Data;

% Get the number of time steps
n_timesteps = size(robot1_pose, 1);

% Rescale hue from [0,1] to [0, 2*pi]
hue_scaled = hue_data * 2 * pi;

% Combine all robot poses into single arrays
all_x = [robot1_pose(:,1); robot2_pose(:,1); robot3_pose(:,1)];
all_y = [robot1_pose(:,2); robot2_pose(:,2); robot3_pose(:,2)];

% Combine hue and saturation for all robots
all_hue = [hue_scaled(:,1); hue_scaled(:,2); hue_scaled(:,3)];
all_saturation = [saturation_data(:,1); saturation_data(:,2); saturation_data(:,3)];

% Calculate vector components using polar to Cartesian conversion
% u = magnitude * cos(angle)
% v = magnitude * sin(angle)
u = all_saturation .* cos(all_hue);
v = all_saturation .* sin(all_hue);

% Create the vector field plot
figure('Name', 'Robot Vector Field', 'Position', [100, 100, 800, 600]);
quiver(all_x, all_y, u, v, 'AutoScale', 'on');

% Add labels and title
xlabel('X Position');
ylabel('Y Position');
title(sprintf('Vector Field for 3 Robots (%d vectors total)', length(all_x)));
grid on;
axis equal;

% Optional: Color-code by robot
% If you want to distinguish between robots visually
figure('Name', 'Robot Vector Field (Color-coded)', 'Position', [950, 100, 800, 600]);
hold on;

% Plot each robot's vectors with different colors
colors = {'r', 'g', 'b'};
robot_names = {'Robot 1', 'Robot 2', 'Robot 3'};

for i = 1:3
    idx_start = (i-1)*n_timesteps + 1;
    idx_end = i*n_timesteps;
    
    quiver(all_x(idx_start:idx_end), all_y(idx_start:idx_end), ...
           u(idx_start:idx_end), v(idx_start:idx_end), ...
           'Color', colors{i}, 'AutoScale', 'on');
end

xlabel('X Position');
ylabel('Y Position');
title(sprintf('Vector Field for 3 Robots (%d vectors total)', length(all_x)));
legend(robot_names, 'Location', 'best');
grid on;
axis equal;
hold off;

% Optional: If vectors are too dense, you can subsample
% Uncomment the following section if you want to plot every nth vector
%{
subsample_rate = 10; % Plot every 10th vector
idx = 1:subsample_rate:length(all_x);

figure('Name', 'Subsampled Vector Field', 'Position', [100, 750, 800, 600]);
quiver(all_x(idx), all_y(idx), u(idx), v(idx), 'AutoScale', 'on');
xlabel('X Position');
ylabel('Y Position');
title(sprintf('Subsampled Vector Field (%d vectors shown)', length(idx)));
grid on;
axis equal;
%}

% Optional: Animate the vector field over time
% Uncomment to see how the field evolves
%{
figure('Name', 'Animated Vector Field', 'Position', [500, 400, 800, 600]);
for t = 1:10:n_timesteps  % Animate every 10th timestep
    clf;
    
    % Get indices for current timestep for all robots
    idx = [t, t + n_timesteps, t + 2*n_timesteps];
    
    % Plot current vectors
    quiver(all_x(idx), all_y(idx), u(idx), v(idx), 'AutoScale', 'on', 'LineWidth', 2);
    
    xlabel('X Position');
    ylabel('Y Position');
    title(sprintf('Vector Field at timestep %d/%d', t, n_timesteps));
    grid on;
    axis equal;
    
    % Set consistent axis limits
    xlim([min(all_x) max(all_x)]);
    ylim([min(all_y) max(all_y)]);
    
    drawnow;
    pause(0.05);  % Adjust pause duration for animation speed
end
%}
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Create streamlines from your vector field
figure('Name', 'Streamlines', 'Position', [100, 100, 900, 700]);

% Create a grid for interpolation
x_range = linspace(min(all_x), max(all_x), 50);
y_range = linspace(min(all_y), max(all_y), 50);
[X_grid, Y_grid] = meshgrid(x_range, y_range);

% Interpolate the vector field onto the grid
U_grid = griddata(all_x, all_y, u, X_grid, Y_grid, 'cubic');
V_grid = griddata(all_x, all_y, v, X_grid, Y_grid, 'cubic');

% Replace NaN values with 0 (for points outside convex hull)
U_grid(isnan(U_grid)) = 0;
V_grid(isnan(V_grid)) = 0;

% Create streamlines
hold on;
% Plot the interpolated vector field (sparse)
quiver(X_grid(1:3:end,1:3:end), Y_grid(1:3:end,1:3:end), ...
       U_grid(1:3:end,1:3:end), V_grid(1:3:end,1:3:end), ...
       'Color', [0.5 0.5 0.5], 'AutoScale', 'on');

% Add streamlines
[startx, starty] = meshgrid(linspace(min(all_x), max(all_x), 15), ...
                            linspace(min(all_y), max(all_y), 15));
streamline(X_grid, Y_grid, U_grid, V_grid, startx(:), starty(:));

% Overlay robot positions
% plot(robot1_pose(:,1), robot1_pose(:,2), 'r-', 'LineWidth', 2);
% plot(robot2_pose(:,1), robot2_pose(:,2), 'g-', 'LineWidth', 2);
% plot(robot3_pose(:,1), robot3_pose(:,2), 'b-', 'LineWidth', 2);

xlabel('X Position');
ylabel('Y Position');
title('Streamlines with Robot Trajectories');
legend({'Vector Field', 'Streamlines', 'Robot 1', 'Robot 2', 'Robot 3'}, ...
       'Location', 'best');
grid on;
axis equal;
hold off;