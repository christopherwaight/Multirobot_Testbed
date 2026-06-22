% Distance from Origin Plotter
% This script loads 80 run files and plots the distance of the cluster
% centroid from the origin (0,0) over time for each group of 10 runs
%
% Creates 8 subplots (4 rows x 2 columns) showing average distance
% with standard deviation bands for runs 1-10, 11-20, ..., 71-80

% Define all 80 run files
run_files = cell(1, 80);
for i = 1:80
    run_files{i} = sprintf('run%d.mat', i);
end

% Display info message at start
fprintf('=== Distance from Origin Analysis ===\n');
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

% Storage for all individual run traces
all_run_distances = cell(1, 80);  % Store each run's distance trace
run_to_group = zeros(1, 80);      % Map run number to group number

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

        % Extract run number from filename (e.g., "run42.mat" -> 42)
        run_num = sscanf(current_file, 'run%d.mat');

        % Load data
        temp_data = load(current_file);

        % Extract cluster centroid position
        if isfield(temp_data, 'cluster_position')
            cluster_pos = temp_data.cluster_position(:, 1:2);  % Extract x, y columns

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

            % Store individual run data for Figure 4
            all_run_distances{run_num} = resampled_distance';
            run_to_group(run_num) = group_idx;
        else
            fprintf('Warning: No cluster_position field in %s\n', current_file);
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

%% COMMENTED OUT: Create figure with 8 subplots (4 rows x 2 columns)
% figure('Position', [100, 100, 1200, 1000]);
%
% for group_idx = 1:num_groups
%     % Calculate subplot position (4 rows x 2 columns)
%     subplot(4, 2, group_idx);
%     hold on;
%
%     % Get color for this group
%     color = group_colors(group_idx, :);
%
%     % Check if data exists for this group
%     if ~isempty(group_avg_distance{group_idx})
%         avg_dist = group_avg_distance{group_idx};
%         std_dist = group_std_distance{group_idx};
%
%         % Plot shaded error band (±1 standard deviation)
%         upper_bound = avg_dist + std_dist;
%         lower_bound = avg_dist - std_dist;
%
%         % Create filled area for standard deviation
%         x_fill = [time_vector, fliplr(time_vector)];
%         y_fill = [upper_bound, fliplr(lower_bound)];
%         fill(x_fill, y_fill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
%              'HandleVisibility', 'off');
%
%         % Plot average distance line
%         plot(time_vector, avg_dist, 'Color', color, 'LineWidth', 2, ...
%              'DisplayName', 'Average Distance');
%     else
%         % No data for this group
%         text(5, 0.5, 'No Data', 'HorizontalAlignment', 'center', ...
%              'FontSize', 14, 'Color', [0.5 0.5 0.5]);
%     end
%
%     % Configure subplot
%     grid on;
%     xlabel('Time (s)');
%     ylabel('Distance from Origin (m)');
%     title(sprintf('Runs %d-%d', (group_idx-1)*runs_per_group + 1, group_idx*runs_per_group), ...
%           'FontWeight', 'bold');
%     xlim([0 10]);
%
%     % Set reasonable y-axis limits (auto-scale but ensure visibility)
%     if ~isempty(group_avg_distance{group_idx})
%         y_max = max(avg_dist + std_dist);
%         y_min = max(0, min(avg_dist - std_dist));  % Don't go below 0
%         ylim([y_min - 0.05*(y_max-y_min), y_max + 0.05*(y_max-y_min)]);
%     else
%         ylim([0 1]);  % Default range if no data
%     end
% end
%
% % Add overall title
% sgtitle('Distance from Origin vs Time - By Group', 'FontSize', 16, 'FontWeight', 'bold');
%
% % Save figures
% fprintf('\nSaving figures...\n');
% saveas(gcf, 'distance_from_origin_by_group.png');
% fprintf('  Saved: distance_from_origin_by_group.png\n');
% saveas(gcf, 'distance_from_origin_by_group.fig');
% fprintf('  Saved: distance_from_origin_by_group.fig\n');

%% COMMENTED OUT: Create second figure - Zoomed in on Runs 21-30 (0-7 seconds only)
% fprintf('\nCreating zoomed figure for Runs 21-30 (0-7 seconds)...\n');
%
% figure('Position', [150, 150, 800, 600]);
% hold on;
%
% % Get data for group 3 (runs 21-30)
% if ~isempty(group_avg_distance{3})
%     % Extract data up to 7 seconds (71 points: 0, 0.1, 0.2, ..., 7.0)
%     time_7s = time_vector(1:71);
%     avg_dist_7s = group_avg_distance{3}(1:71);
%     std_dist_7s = group_std_distance{3}(1:71);
%
%     % Get color for group 3
%     color = group_colors(3, :);
%
%     % Plot shaded error band (±1 standard deviation)
%     upper_bound = avg_dist_7s + std_dist_7s;
%     lower_bound = avg_dist_7s - std_dist_7s;
%
%     % Create filled area for standard deviation
%     x_fill = [time_7s, fliplr(time_7s)];
%     y_fill = [upper_bound, fliplr(lower_bound)];
%     fill(x_fill, y_fill, color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
%          'HandleVisibility', 'off');
%
%     % Plot average distance line
%     plot(time_7s, avg_dist_7s, 'Color', color, 'LineWidth', 3, ...
%          'DisplayName', 'Average Distance');
%
%     % Configure plot
%     grid on;
%     xlabel('Time (s)', 'FontSize', 12);
%     ylabel('Distance from Origin (m)', 'FontSize', 12);
%     title('Distance from Origin vs Time - Runs 21-30 (Zoomed: 0-7s)', ...
%           'FontSize', 14, 'FontWeight', 'bold');
%     xlim([0 7]);
%
%     % Set y-axis limits with some padding
%     y_max = max(upper_bound);
%     y_min = max(0, min(lower_bound));
%     ylim([y_min - 0.05*(y_max-y_min), y_max + 0.05*(y_max-y_min)]);
%
%     % Make the plot more readable
%     set(gca, 'FontSize', 11);
% else
%     text(3.5, 0.5, 'No Data for Runs 21-30', 'HorizontalAlignment', 'center', ...
%          'FontSize', 14, 'Color', [0.5 0.5 0.5]);
%     xlim([0 7]);
%     ylim([0 1]);
% end
%
% % Save zoomed figure
% saveas(gcf, 'distance_from_origin_runs21-30_zoomed.png');
% fprintf('  Saved: distance_from_origin_runs21-30_zoomed.png\n');
% saveas(gcf, 'distance_from_origin_runs21-30_zoomed.fig');
% fprintf('  Saved: distance_from_origin_runs21-30_zoomed.fig\n');

%% Figure 3: All 8 groups on same graph with std dev bands
fprintf('\nCreating Figure 3: All 8 groups with std dev bands...\n');

figure('Position', [100, 100, 1000, 700]);
hold on;

% Plot each group with std dev band and average line
legend_entries = {};
for group_idx = 1:num_groups
    color = group_colors(group_idx, :);

    if ~isempty(group_avg_distance{group_idx})
        avg_dist = group_avg_distance{group_idx};
        std_dist = group_std_distance{group_idx};

        % Plot shaded error band (±1 standard deviation)
        upper_bound = avg_dist + std_dist;
        lower_bound = avg_dist - std_dist;

        % Create filled area for standard deviation
        x_fill = [time_vector, fliplr(time_vector)];
        y_fill = [upper_bound, fliplr(lower_bound)];
        fill(x_fill, y_fill, color, 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
             'HandleVisibility', 'off');

        % Plot average distance line
        plot(time_vector, avg_dist, 'Color', color, 'LineWidth', 2.5);

        % Create legend entry
        run_start = (group_idx-1)*runs_per_group + 1;
        run_end = group_idx*runs_per_group;
        legend_entries{end+1} = sprintf('Runs %d-%d', run_start, run_end);
    end
end

% Configure plot
grid on;
xlabel('Time (s)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Distance from Origin (m)', 'FontSize', 13, 'FontWeight', 'bold');
title('Distance from Origin vs Time - All Groups', 'FontSize', 15, 'FontWeight', 'bold');
xlim([0 10]);
ylim([0 inf]);  % Auto-scale y-axis but start from 0

% Add legend
legend(legend_entries, 'Location', 'best', 'FontSize', 10);
set(gca, 'FontSize', 11);

% Save figure 3
fprintf('\nSaving Figure 3...\n');
saveas(gcf, 'distance_from_origin_all_groups.png');
fprintf('  Saved: distance_from_origin_all_groups.png\n');
saveas(gcf, 'distance_from_origin_all_groups.fig');
fprintf('  Saved: distance_from_origin_all_groups.fig\n');

%% Figure 4: All individual runs (up to 80 runs)
fprintf('\nCreating Figure 4: All individual runs...\n');

figure('Position', [150, 150, 1000, 700]);
hold on;

% Plot each individual run with color based on its group
num_runs_plotted = 0;
legend_handles = [];
legend_labels = {};

for run_num = 1:80
    if ~isempty(all_run_distances{run_num})
        % Get the group this run belongs to
        group_idx = run_to_group(run_num);
        color = group_colors(group_idx, :);

        % Plot this run's distance trace
        h = plot(time_vector, all_run_distances{run_num}, 'Color', color, ...
                 'LineWidth', 1.5, 'HandleVisibility', 'off');

        num_runs_plotted = num_runs_plotted + 1;

        % Add one legend entry per group (not per run)
        if isempty(legend_handles) || group_idx > length(legend_handles) || isempty(legend_handles(group_idx))
            if group_idx > length(legend_handles)
                legend_handles(group_idx) = h;
            else
                legend_handles(group_idx) = h;
            end
            set(h, 'HandleVisibility', 'on');
            run_start = (group_idx-1)*runs_per_group + 1;
            run_end = group_idx*runs_per_group;
            legend_labels{group_idx} = sprintf('Runs %d-%d', run_start, run_end);
        end
    end
end

% Configure plot
grid on;
xlabel('Time (s)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Distance from Origin (m)', 'FontSize', 13, 'FontWeight', 'bold');
title(sprintf('Distance from Origin vs Time - All %d Individual Runs', num_runs_plotted), ...
      'FontSize', 15, 'FontWeight', 'bold');
xlim([0 10]);
ylim([0 inf]);  % Auto-scale y-axis but start from 0

% Add legend (one entry per group)
if ~isempty(legend_handles)
    legend(legend_handles(legend_handles ~= 0), legend_labels(~cellfun(@isempty, legend_labels)), ...
           'Location', 'best', 'FontSize', 10);
end
set(gca, 'FontSize', 11);

% Save figure 4
fprintf('\nSaving Figure 4...\n');
saveas(gcf, 'distance_from_origin_all_runs.png');
fprintf('  Saved: distance_from_origin_all_runs.png\n');
saveas(gcf, 'distance_from_origin_all_runs.fig');
fprintf('  Saved: distance_from_origin_all_runs.fig\n');

fprintf('\n=== Analysis Complete ===\n');
fprintf('Two new figures created:\n');
fprintf('  Figure 3: All 8 groups on same graph with std dev bands\n');
fprintf('  Figure 4: All %d individual runs plotted (color-coded by group)\n', num_runs_plotted);
fprintf('Distance scaled by factor %.4f to correct printer units.\n', SCALE_FACTOR);
