% Distance from Origin Plotter (4-Robot System)
% This script loads 80 run files and plots the distance of the cluster
% centroid from the origin (0,0) over time for each group of 10 runs
%
% Creates 8 subplots (4 rows x 2 columns) showing average distance
% with standard deviation bands for runs 1-10, 11-20, ..., 71-80
%
% MODIFIED: Cluster center is calculated from 4 robot positions

% Define all 80 run files
run_files = cell(1, 80);
for i = 1:80
    run_files{i} = sprintf('run%d.mat', i);
end

% Display info message at start
fprintf('=== Distance from Origin Analysis (4-Robot System) ===\n');
fprintf('Loading and processing %d run files in groups of 10...\n', length(run_files));

% Define colors for different groups (8 groups total) - same as allrunplotter.m
group_colors = [
    0.8 0.2 0.2;  % Red
    0.2 0.6 0.2;  % Green
    0.2 0.3 0.8;  % Blue
    0.8 0.5 0.2;  % Orange
    0.6 0.2 0.8;  % Purple
    0.8 0.8 0.2;  % Yellow
    0.2 0.8 0.8;  % Cyan
    0.8 0.2 0.6;  % Magenta
];

% Process each group of 10 runs
num_groups = 8;
runs_per_group = 10;

% Time parameters
timestep = 0.1;  % seconds
target_length = 101;  % number of points (0 to 10 seconds)
time_vector = linspace(0, 10, target_length);  % time in seconds

% Scaling factor to correct printer scaling issue
SCALE_FACTOR = 1.6764;  % Converts from printer units to actual meters

% Storage for average distance traces of each group
group_avg_distance = cell(1, num_groups);
group_std_distance = cell(1, num_groups);

% Process each group
for group_idx = 1:num_groups
    fprintf('\n=== Processing Group %d (runs %d-%d) ===\n', group_idx, ...
        (group_idx-1)*runs_per_group + 1, group_idx*runs_per_group);

    % Create data storage for distance traces in this group
    all_distance_traces = [];

    % Get the run indices for this group
    start_idx = (group_idx - 1) * runs_per_group + 1;
    end_idx = group_idx * runs_per_group;

    % Check which files exist in this group
    existing_files = {};
    for i = start_idx:end_idx
        if exist(run_files{i}, 'file')
            existing_files{end+1} = run_files{i};
        else
            fprintf('Warning: File not found - %s\n', run_files{i});
        end
    end

    fprintf('Found %d existing run files in group %d\n', length(existing_files), group_idx);

    if isempty(existing_files)
        fprintf('Warning: No run files found for group %d. Skipping...\n', group_idx);
        continue;
    end

    % Process each run file in this group
    for run_idx = 1:length(existing_files)
        current_file = existing_files{run_idx};
        fprintf('Loading file: %s\n', current_file);

        % Load data
        temp_data = load(current_file);

        % Extract or calculate cluster centroid position
        cluster_pos = [];

        % Try to calculate from 4 robot positions first
        robot1_exists = isfield(temp_data, 'robot1_pose');
        robot2_exists = isfield(temp_data, 'robot2_pose');
        robot3_exists = isfield(temp_data, 'robot3_pose');
        robot4_exists = isfield(temp_data, 'robot4_pose');

        if robot1_exists && robot2_exists && robot3_exists && robot4_exists
            % Calculate cluster center from 4 robot positions
            num_steps = size(temp_data.robot1_pose, 1);
            cluster_pos = zeros(num_steps, 2);

            for t = 1:num_steps
                cluster_pos(t, 1) = (temp_data.robot1_pose(t, 1) + ...
                                    temp_data.robot2_pose(t, 1) + ...
                                    temp_data.robot3_pose(t, 1) + ...
                                    temp_data.robot4_pose(t, 1)) / 4;

                cluster_pos(t, 2) = (temp_data.robot1_pose(t, 2) + ...
                                    temp_data.robot2_pose(t, 2) + ...
                                    temp_data.robot3_pose(t, 2) + ...
                                    temp_data.robot4_pose(t, 2)) / 4;
            end
            fprintf('  Calculated cluster center from 4 robot positions\n');
        elseif isfield(temp_data, 'cluster_pose')
            % Use existing cluster_pose from file
            cluster_pos = temp_data.cluster_pose(:, 1:2);
            fprintf('  Using existing cluster_pose from file\n');
        else
            fprintf('Warning: Could not get cluster position from %s\n', current_file);
            continue;
        end

        if ~isempty(cluster_pos)
            % Calculate distance from origin at each timestep
            x_c = cluster_pos(:, 1);
            y_c = cluster_pos(:, 2);
            distance = sqrt(x_c.^2 + y_c.^2) / SCALE_FACTOR;  % Apply scaling correction

            % Resample to target length (101 points) using same method as allrunplotter
            original_length = length(distance);

            if original_length == target_length
                % No resampling needed
                resampled_distance = distance;
            else
                % Create a parametric representation based on cumulative distance
                % Use time indices as parameter
                param = linspace(0, 1, original_length)';
                new_param = linspace(0, 1, target_length)';
                resampled_distance = interp1(param, distance, new_param, 'pchip');
            end

            % Store resampled distance trace
            all_distance_traces = [all_distance_traces; resampled_distance'];
        end
    end

    % Calculate average and standard deviation for this group
    if ~isempty(all_distance_traces)
        group_avg_distance{group_idx} = mean(all_distance_traces, 1);
        group_std_distance{group_idx} = std(all_distance_traces, 0, 1);
        fprintf('Processed %d distance traces for group %d\n', size(all_distance_traces, 1), group_idx);
    else
        fprintf('Warning: No distance traces for group %d\n', group_idx);
    end
end

%% Create figure with 8 subplots (4 rows x 2 columns)
figure('Position', [100, 100, 1200, 1000]);

for group_idx = 1:num_groups
    % Calculate subplot position (4 rows x 2 columns)
    subplot(4, 2, group_idx);
    hold on;

    % Get color for this group
    color = group_colors(group_idx, :);

    % Check if data exists for this group
    if ~isempty(group_avg_distance{group_idx})
        avg_dist = group_avg_distance{group_idx};
        std_dist = group_std_distance{group_idx};

        % Plot shaded error band (±1 standard deviation)
        upper_bound = avg_dist + std_dist;
        lower_bound = avg_dist - std_dist;

        % Create filled area for standard deviation
        x_fill = [time_vector, fliplr(time_vector)];
        y_fill = [upper_bound, fliplr(lower_bound)];
        fill(x_fill, y_fill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
             'HandleVisibility', 'off');

        % Plot average distance line
        plot(time_vector, avg_dist, 'Color', color, 'LineWidth', 2, ...
             'DisplayName', 'Average Distance');
    else
        % No data for this group
        text(5, 0.5, 'No Data', 'HorizontalAlignment', 'center', ...
             'FontSize', 14, 'Color', [0.5 0.5 0.5]);
    end

    % Configure subplot
    grid on;
    xlabel('Time (s)');
    ylabel('Distance from Origin (m)');
    title(sprintf('Runs %d-%d', (group_idx-1)*runs_per_group + 1, group_idx*runs_per_group), ...
          'FontWeight', 'bold');
    xlim([0 10]);

    % Set reasonable y-axis limits (auto-scale but ensure visibility)
    if ~isempty(group_avg_distance{group_idx})
        y_max = max(avg_dist + std_dist);
        y_min = max(0, min(avg_dist - std_dist));  % Don't go below 0
        ylim([y_min - 0.05*(y_max-y_min), y_max + 0.05*(y_max-y_min)]);
    else
        ylim([0 1]);  % Default range if no data
    end
end

% Add overall title
sgtitle('Distance from Origin vs Time - By Group (4-Robot System)', 'FontSize', 16, 'FontWeight', 'bold');

% Save figures
fprintf('\nSaving figures...\n');
saveas(gcf, 'distance_from_origin_by_group.png');
fprintf('  Saved: distance_from_origin_by_group.png\n');
saveas(gcf, 'distance_from_origin_by_group.fig');
fprintf('  Saved: distance_from_origin_by_group.fig\n');

%% Create second figure - Zoomed in on Runs 21-30 (0-7 seconds only)
fprintf('\nCreating zoomed figure for Runs 21-30 (0-7 seconds)...\n');

figure('Position', [150, 150, 800, 600]);
hold on;

% Get data for group 3 (runs 21-30)
if ~isempty(group_avg_distance{3})
    % Extract data up to 7 seconds (71 points: 0, 0.1, 0.2, ..., 7.0)
    time_7s = time_vector(1:71);
    avg_dist_7s = group_avg_distance{3}(1:71);
    std_dist_7s = group_std_distance{3}(1:71);

    % Get color for group 3
    color = group_colors(3, :);

    % Plot shaded error band (±1 standard deviation)
    upper_bound = avg_dist_7s + std_dist_7s;
    lower_bound = avg_dist_7s - std_dist_7s;

    % Create filled area for standard deviation
    x_fill = [time_7s, fliplr(time_7s)];
    y_fill = [upper_bound, fliplr(lower_bound)];
    fill(x_fill, y_fill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');

    % Plot average distance line
    plot(time_7s, avg_dist_7s, 'Color', color, 'LineWidth', 3, ...
         'DisplayName', 'Average Distance');

    % Configure plot
    grid on;
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('Distance from Origin (m)', 'FontSize', 12);
    title('Distance from Origin vs Time - Runs 21-30 (Zoomed: 0-7s, 4-Robot)', ...
          'FontSize', 14, 'FontWeight', 'bold');
    xlim([0 7]);

    % Set y-axis limits with some padding
    y_max = max(upper_bound);
    y_min = max(0, min(lower_bound));
    ylim([y_min - 0.05*(y_max-y_min), y_max + 0.05*(y_max-y_min)]);

    % Make the plot more readable
    set(gca, 'FontSize', 11);
else
    text(3.5, 0.5, 'No Data for Runs 21-30', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Color', [0.5 0.5 0.5]);
    xlim([0 7]);
    ylim([0 1]);
end

% Save zoomed figure
saveas(gcf, 'distance_from_origin_runs21-30_zoomed.png');
fprintf('  Saved: distance_from_origin_runs21-30_zoomed.png\n');
saveas(gcf, 'distance_from_origin_runs21-30_zoomed.fig');
fprintf('  Saved: distance_from_origin_runs21-30_zoomed.fig\n');

fprintf('\n=== Analysis Complete ===\n');
fprintf('Two figures created:\n');
fprintf('  1. All 8 groups (4x2 subplots, 0-10 seconds)\n');
fprintf('  2. Runs 21-30 zoomed (single plot, 0-7 seconds)\n');
fprintf('Each shows average distance (solid line) ± standard deviation (shaded band).\n');
fprintf('Distance scaled by factor %.4f to correct printer units.\n', SCALE_FACTOR);
fprintf('Cluster center calculated from 4 robot positions.\n');
