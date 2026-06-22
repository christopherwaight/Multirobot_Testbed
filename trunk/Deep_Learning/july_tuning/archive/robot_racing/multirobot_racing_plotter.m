%% Simple Multi-robot Racing Plotter
clc; clear all; close all;

% Define the run files and their properties
run_files = {'tidal.mat', 'celeste.mat', 'pacific-blue2.mat', 'redwood.mat'};
colors = [1 0 1; 0 1 1; 0 0 1; 0 1 0]; % RGB colors: magenta, cyan, blue, green
robot_names = {'R1', 'R2', 'R3', 'R4'};
markers = {'s', 'o', 'd', '^'}; % Square, circle, diamond, triangle

% Saturations (modify if different)
saturations = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

% Create main figure
figure('Position', [100, 100, 800, 600]);
hold on;

% Process each robot file
for robot_idx = 1:length(run_files)
    fprintf('Processing %s...\n', run_files{robot_idx});
    
    % Load data
    data = load(run_files{robot_idx});
    
    % Find all pose variables
    field_names = fieldnames(data);
    pose_fields = field_names(startsWith(field_names, 'pose_'));
    pose_fields = sort(pose_fields); % Ensure correct order
    
    % Calculate velocities for each pose
    velocities = zeros(1, length(pose_fields));
    
    for pose_idx = 1:length(pose_fields)
        pose_data = data.(pose_fields{pose_idx});
        
        % Calculate 2D velocity
        dx = diff(pose_data(:,1));
        dy = diff(pose_data(:,2));
        velocity = sqrt(dx.^2 + dy.^2) * 10; % Scale factor
        
        % Skip first few points and smooth
        velocity = velocity(6:end);
        velocity = movmean(velocity, 3);
        
        % Store mean velocity
        velocities(pose_idx) = mean(velocity);
    end
    
    % Match data lengths
    num_points = min(length(saturations), length(velocities));
    x_data = saturations(1:num_points);
    y_data = velocities(1:num_points);
    
    % Plot with markers and lines (single plot call to avoid duplicate legend)
    plot(x_data, y_data, [markers{robot_idx} '-'], 'Color', colors(robot_idx,:), ...
         'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', colors(robot_idx,:), ...
         'DisplayName', robot_names{robot_idx});
    
    % Print velocity values
    fprintf('Robot %s (%s):\n', robot_names{robot_idx}, run_files{robot_idx});
    fprintf('  Saturations: [');
    fprintf('%.1f ', x_data);
    fprintf(']\n');
    fprintf('  Velocities:  [');
    fprintf('%.3f ', y_data);
    fprintf(']\n');
    for pose_idx = 1:length(pose_fields)
        fprintf('  %s: %.3f m/s\n', pose_fields{pose_idx}, velocities(pose_idx));
    end
    fprintf('  Mean: %.3f, Max: %.3f, Min: %.3f m/s\n\n', mean(velocities), max(velocities), min(velocities));
end

% Format plot
xlabel('Saturation');
ylabel('Velocity (m/s)');
title('Robot Velocity vs Saturation');
legend('show', 'Location', 'northwest');
grid on;
hold off;

% Save figure
saveas(gcf, 'robot_velocity_comparison.png');