% Radii Analysis Plotter - Computes cluster center from robot poses
% This script loads multiple radii files and analyzes the cluster radius over time

% Define the radii files to process
radii_files = {'radii001.mat', 'radii01.mat', 'radii02.mat', 'radii03.mat', ...
               'radii04.mat', 'radii05.mat', 'radii06.mat'};

% Display info message at start         
fprintf('Loading and processing %d radii files...\n', length(radii_files));

% Define colors for different files
file_colors = [
    0.8 0.2 0.2;  % Red
    0.2 0.6 0.2;  % Green
    0.2 0.3 0.8;  % Blue
    0.8 0.5 0.2;  % Orange
    0.6 0.2 0.8;  % Purple
    0.8 0.8 0.2;  % Yellow
    0.2 0.8 0.8;  % Cyan
];

% Storage for radius data
all_radius_data = {};
all_time_data = {};
all_cluster_x = {};
all_cluster_y = {};
file_labels = {};

% Process each radii file
for file_idx = 1:length(radii_files)
    current_file = radii_files{file_idx};
    
    if exist(current_file, 'file')
        fprintf('\nLoading file: %s\n', current_file);
        
        % Load data
        data = load(current_file);
        
        % Check if robot poses exist
        if isfield(data, 'robot1_pose') && isfield(data, 'robot2_pose') && ...
           isfield(data, 'robot3_pose') && isfield(data, 'robot4_pose')
            
            % Get number of time steps
            num_steps = size(data.robot1_pose, 1);
            
            % Initialize arrays for cluster center
            x_c_array = zeros(num_steps, 1);
            y_c_array = zeros(num_steps, 1);
            th_c_array = zeros(num_steps, 1);
            
            % Compute cluster center at each time step
            for t = 1:num_steps
                % Extract robot positions and orientations
                x1 = data.robot1_pose(t, 1);
                y1 = data.robot1_pose(t, 2);
                th1 = data.robot1_pose(t, 3);
                
                x2 = data.robot2_pose(t, 1);
                y2 = data.robot2_pose(t, 2);
                th2 = data.robot2_pose(t, 3);
                
                x3 = data.robot3_pose(t, 1);
                y3 = data.robot3_pose(t, 2);
                th3 = data.robot3_pose(t, 3);
                
                x4 = data.robot4_pose(t, 1);
                y4 = data.robot4_pose(t, 2);
                th4 = data.robot4_pose(t, 3);
                
                % Use forward kinematics to compute cluster center
                [~, x_c, y_c, th_c, ~, ~, ~, ~, ~] = Forward_Kinematics(x1, y1, th1, x2, y2, th2, x3, y3, th3, x4, y4, th4);
                
                x_c_array(t) = x_c;
                y_c_array(t) = y_c;
                th_c_array(t) = th_c;
            end
            
            % Calculate radius (Euclidean distance from origin)
            r = sqrt(x_c_array.^2 + y_c_array.^2);
            
            % Store data
            all_radius_data{end+1} = r;
            all_cluster_x{end+1} = x_c_array;
            all_cluster_y{end+1} = y_c_array;
            
            % Generate time vector
            if isfield(data, 'time')
                all_time_data{end+1} = data.time;
            else
                % Assume 1 Hz sampling if no time data
                all_time_data{end+1} = (0:num_steps-1)';
            end
            
            % Store file label
            file_labels{end+1} = strrep(current_file, '.mat', '');
            
            % Calculate and display statistics
            fprintf('Statistics for %s:\n', current_file);
            fprintf('  Number of samples: %d\n', length(r));
            fprintf('  Mean radius: %.4f m\n', mean(r));
            fprintf('  Std deviation: %.4f m\n', std(r));
            fprintf('  Min radius: %.4f m\n', min(r));
            fprintf('  Max radius: %.4f m\n', max(r));
            fprintf('  Initial radius: %.4f m\n', r(1));
            fprintf('  Final radius: %.4f m\n', r(end));
            fprintf('  Total change: %.4f m\n', r(end) - r(1));
            fprintf('  Mean cluster position: (%.4f, %.4f) m\n', mean(x_c_array), mean(y_c_array));
            
        else
            fprintf('Warning: Missing robot pose data in %s\n', current_file);
            fprintf('  Required fields: robot1_pose, robot2_pose, robot3_pose, robot4_pose\n');
        end
    else
        fprintf('Warning: File not found - %s\n', current_file);
    end
end

% Create figure for radius over time plot
figure('Position', [100, 100, 1000, 600]);
hold on;

% Plot all radius trajectories
for i = 1:length(all_radius_data)
    plot(all_time_data{i}, all_radius_data{i}, 'Color', file_colors(i,:), ...
         'LineWidth', 2, 'DisplayName', file_labels{i});
end

title('Cluster Radius Over Time - All Files (Computed from Robot Poses)');
xlabel('Time (s)');
ylabel('Radius (m)');
grid on;
legend('Location', 'best');
hold off;

% Save the plot
saveas(gcf, 'radii_comparison_plot.png');
saveas(gcf, 'radii_comparison_plot.fig');

% Create a summary statistics table
fprintf('\n\n=== SUMMARY STATISTICS TABLE ===\n');
fprintf('%-12s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n', ...
    'File', 'Samples', 'Mean(m)', 'Std(m)', 'Min(m)', 'Max(m)', 'Initial(m)', 'Final(m)');
fprintf('%s\n', repmat('-', 1, 100));

for i = 1:length(all_radius_data)
    r = all_radius_data{i};
    fprintf('%-12s | %8d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
        file_labels{i}, length(r), mean(r), std(r), min(r), max(r), r(1), r(end));
end

% Export radius and cluster data to CSV files
fprintf('\n\nExporting radius and cluster data to CSV files...\n');
for i = 1:length(all_radius_data)
    % Create filename
    csv_filename = sprintf('%s_cluster_data.csv', file_labels{i});
    
    % Prepare data matrix [time, radius, x_center, y_center]
    export_data = [all_time_data{i}, all_radius_data{i}, all_cluster_x{i}, all_cluster_y{i}];
    
    % Write header and data
    fid = fopen(csv_filename, 'w');
    fprintf(fid, 'Time(s),Radius(m),X_center(m),Y_center(m)\n');
    fprintf(fid, '%.6f,%.6f,%.6f,%.6f\n', export_data');
    fclose(fid);
    
    fprintf('Exported: %s\n', csv_filename);
end

% Create figure for trajectory plots in X-Y plane
figure('Position', [100, 700, 900, 700]);
hold on;

% Plot all cluster center trajectories in X-Y plane
for i = 1:length(all_cluster_x)
    % Plot trajectory
    plot(all_cluster_x{i}, all_cluster_y{i}, 'Color', file_colors(i,:), ...
         'LineWidth', 2, 'DisplayName', file_labels{i});
    
    % Mark start and end points
    plot(all_cluster_x{i}(1), all_cluster_y{i}(1), 'o', 'MarkerSize', 8, ...
         'MarkerFaceColor', file_colors(i,:), 'MarkerEdgeColor', 'k', ...
         'HandleVisibility', 'off');
    plot(all_cluster_x{i}(end), all_cluster_y{i}(end), 's', 'MarkerSize', 8, ...
         'MarkerFaceColor', file_colors(i,:), 'MarkerEdgeColor', 'k', ...
         'HandleVisibility', 'off');
end

% Add reference circles
theta = linspace(0, 2*pi, 100);
for r_ref = 0.2:0.2:0.6
    plot(r_ref*cos(theta), r_ref*sin(theta), 'k--', 'LineWidth', 0.5, ...
         'HandleVisibility', 'off');
    text(r_ref, 0, sprintf('r=%.1f', r_ref), 'VerticalAlignment', 'bottom');
end

title('Cluster Center Trajectories in X-Y Plane (Computed from Robot Poses)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
axis equal;
legend('Location', 'best');
hold off;

% Save the trajectory plot
saveas(gcf, 'cluster_trajectory_comparison_plot.png');
saveas(gcf, 'cluster_trajectory_comparison_plot.fig');

% Create a combined data matrix for all runs
fprintf('\n\nCreating combined radius data file...\n');
max_length = max(cellfun(@length, all_radius_data));
combined_data = NaN(max_length, 3*length(all_radius_data) + 1);

% Find the longest time vector for the combined file
[~, longest_idx] = max(cellfun(@length, all_time_data));
combined_data(1:length(all_time_data{longest_idx}), 1) = all_time_data{longest_idx};

% Add radius, x, and y data for each file
col_idx = 2;
for i = 1:length(all_radius_data)
    combined_data(1:length(all_radius_data{i}), col_idx) = all_radius_data{i};
    combined_data(1:length(all_cluster_x{i}), col_idx+1) = all_cluster_x{i};
    combined_data(1:length(all_cluster_y{i}), col_idx+2) = all_cluster_y{i};
    col_idx = col_idx + 3;
end

% Write combined CSV
fid = fopen('combined_cluster_data.csv', 'w');
fprintf(fid, 'Time(s)');
for i = 1:length(file_labels)
    fprintf(fid, ',%s_radius,%s_x,%s_y', file_labels{i}, file_labels{i}, file_labels{i});
end
fprintf(fid, '\n');

% Write data rows
for row = 1:max_length
    fprintf(fid, '%.6f', combined_data(row, 1));
    for col = 2:size(combined_data, 2)
        if isnan(combined_data(row, col))
            fprintf(fid, ',');
        else
            fprintf(fid, ',%.6f', combined_data(row, col));
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);

fprintf('Exported: combined_cluster_data.csv\n');
fprintf('\nAll analysis complete!\n');

% Include the Forward_Kinematics function at the end
function [fwdkin, x_c, y_c, th_c, d1, d2, r1, r2, phi] = Forward_Kinematics(x1, y1, th1, x2, y2, th2, x3, y3, th3, x4, y4, th4)
% Forward_Kinematics: Convert corner coordinates to diagonal parameters
%
% Inputs:
% x1, y1, th1: Position and orientation of corner 1
% x2, y2, th2: Position and orientation of corner 2
% x3, y3, th3: Position and orientation of corner 3
% x4, y4, th4: Position and orientation of corner 4
%
% Outputs:
% fwdkin: [d1, d2, r1, r2, phi, x_c, y_c, th_c] - diagonal parameters
% x_c, y_c: Centroid coordinates
% th_c: Overall rotation angle (theta)
% d1, d2: Diagonal lengths
% r1, r2: Intersection ratios
% theta_1: Same as th_c (for compatibility)
% Centre frame (centroid)
x_c = (x1 + x2 + x3 + x4) / 4;
y_c = (y1 + y2 + y3 + y4) / 4;
% Diagonal lengths
d1 = sqrt((x3 - x1)^2 + (y3 - y1)^2); % Diagonal 1-3
d2 = sqrt((x4 - x2)^2 + (y4 - y2)^2); % Diagonal 2-4
% Overall rotation angle (angle of diagonal 1-3)
th_c = atan2(y3 - y1, x3 - x1);
% Angle between diagonals
phi = -atan2(y4 - y2, x4 - x2) + atan2(y3 - y1, x3 - x1);
% Intersection ratios using line intersection method
% Solving for where diagonals 1-3 and 2-4 intersect
delta = (x3 - x1) * (y2 - y4) + (y3 - y1) * (x4 - x2);
% Standard intersection calculation
r1 = ((x2 - x1) * (y2 - y4) + (y2 - y1) * (x4 - x2)) / delta;
r2 = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / delta;
% Combining all readings into fwdkin vector
fwdkin = [0 0 0 0 0 0 0 0];
fwdkin(1) = d1; % Diagonal 1 length
fwdkin(2) = d2; % Diagonal 2 length
fwdkin(3) = r1; % Intersection ratio 1
fwdkin(4) = r2; % Intersection ratio 2
fwdkin(5) = phi; % Angle between diagonals
fwdkin(6) = x_c; % Centroid x
fwdkin(7) = y_c; % Centroid y
fwdkin(8) = th_c; % Overall rotation angle
end