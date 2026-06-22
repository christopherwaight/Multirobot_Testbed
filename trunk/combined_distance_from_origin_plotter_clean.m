% Combined Distance from Origin Plotter - Clean Version
%
% FIGURE 9 FOR THE PAPER
%
% This script creates figures showing distance from origin over time
% for hardware test runs from two test scenarios:
%   - 3-robot vortex tests (red)
%   - 3-robot saddle tests (blue)
%
% IMPORTANT - SCALE FACTOR NOTE:
% This script reports SCALE-CORRECTED values for theoretical field coordinates.
% The physical printed field map is 66 inches (1.6764m) instead of the
% theoretical 1m domain used in simulations.
%
% All OptiTrack position values are divided by 1.6764 to convert to
% theoretical coordinates. These corrected values are used in Table III.

%% Configuration
% Scaling factor to correct printer scaling issue
SCALE_FACTOR = 1.6764;  % Converts from printer units to actual meters

% Colors for each test type
color_3r_vortex = [0.8 0.2 0.2];  % Red
color_3r_saddle = [0.2 0.3 0.8];  % Blue

% Get script directory to build absolute paths
script_path = fileparts(mfilename('fullpath'));
trunk_dir = script_path;  % Script is in trunk/

% Storage for final positions (for Table verification)
vortex_final_positions = [];
saddle_final_positions = [];

fprintf('=== Combined Distance from Origin Analysis (Figure 9) ===\n');
fprintf('Script location: %s\n', trunk_dir);
fprintf('Loading data from two test scenarios...\n\n');

%% Load 3-Robot Vortex Tests
fprintf('=== Loading 3-Robot Vortex Tests ===\n');
vortex_dir = fullfile(trunk_dir, 'robots_3', 'vortex_tests');
vortex_distances = {};
vortex_time_vectors = {};
vortex_count = 0;

% Load run1.mat through run80.mat
for i = 1:80
    filename = fullfile(vortex_dir, sprintf('run%d.mat', i));
    if exist(filename, 'file')
        try
            temp_data = load(filename);

            % Check for cluster data (could be 'cluster_position' or 'cluster_pose')
            if isfield(temp_data, 'cluster_position')
                cluster_pos = temp_data.cluster_position(:, 1:2);
            elseif isfield(temp_data, 'cluster_pose')
                cluster_pos = temp_data.cluster_pose(:, 1:2);
            else
                continue;  % Skip if neither field exists
            end

            % Calculate distance from origin at each timestep
            x_c = cluster_pos(:, 1);
            y_c = cluster_pos(:, 2);
            distance = sqrt(x_c.^2 + y_c.^2) / SCALE_FACTOR;

            % Store final position for Table verification
            final_pos = [x_c(end), y_c(end)] / SCALE_FACTOR;
            vortex_final_positions = [vortex_final_positions; final_pos];

            % Create time vector for this run (no resampling)
            num_points = length(distance);
            time_vec = (0:(num_points-1)) * 0.1;  % Assuming 0.1s timestep

            % Store the distance trace and time vector
            vortex_distances{end+1} = distance;
            vortex_time_vectors{end+1} = time_vec;
            vortex_count = vortex_count + 1;
        catch ME
            fprintf('  Warning: Could not load %s - %s\n', filename, ME.message);
        end
    end
end
fprintf('Loaded %d vortex test runs\n', vortex_count);

%% Load 3-Robot Saddle Tests
fprintf('\n=== Loading 3-Robot Saddle Tests ===\n');
saddle_dir = fullfile(trunk_dir, 'robots_3', 'saddle_tests');
saddle_distances = {};
saddle_time_vectors = {};
saddle_count = 0;

% Load run1.mat through run80.mat
for i = 1:80
    filename = fullfile(saddle_dir, sprintf('run%d.mat', i));
    if exist(filename, 'file')
        try
            temp_data = load(filename);

            % Saddle tests use 'cluster_pose' instead of 'cluster_position'
            if isfield(temp_data, 'cluster_pose')
                cluster_pos = temp_data.cluster_pose(:, 1:2);
            elseif isfield(temp_data, 'cluster_position')
                cluster_pos = temp_data.cluster_position(:, 1:2);
            else
                continue;  % Skip if neither field exists
            end

            % Calculate distance from origin at each timestep
            x_c = cluster_pos(:, 1);
            y_c = cluster_pos(:, 2);
            distance = sqrt(x_c.^2 + y_c.^2) / SCALE_FACTOR;

            % Store final position for Table verification
            final_pos = [x_c(end), y_c(end)] / SCALE_FACTOR;
            saddle_final_positions = [saddle_final_positions; final_pos];

            % Create time vector for this run (no resampling)
            num_points = length(distance);
            time_vec = (0:(num_points-1)) * 0.1;  % Assuming 0.1s timestep

            % Store the distance trace and time vector
            saddle_distances{end+1} = distance;
            saddle_time_vectors{end+1} = time_vec;
            saddle_count = saddle_count + 1;
        catch ME
            fprintf('  Warning: Could not load %s - %s\n', filename, ME.message);
        end
    end
end
fprintf('Loaded %d saddle test runs\n', saddle_count);

%% Figure 11: Average with Standard Deviation Bands (Overall)
fprintf('\n=== Creating Figure 11: Average Distance with Std Dev Bands ===\n');

% Common time vector for resampling (0 to 10 seconds)
common_time = linspace(0, 10, 101);

% Resample all vortex runs to common time vector
vortex_resampled = [];
if ~isempty(vortex_distances)
    for i = 1:length(vortex_distances)
        resampled = interp1(vortex_time_vectors{i}, vortex_distances{i}, common_time, 'pchip', 'extrap');
        vortex_resampled = [vortex_resampled; resampled];
    end
    vortex_mean = mean(vortex_resampled, 1);
    vortex_std = std(vortex_resampled, 0, 1);
end

% Resample all saddle runs to common time vector
saddle_resampled = [];
if ~isempty(saddle_distances)
    for i = 1:length(saddle_distances)
        resampled = interp1(saddle_time_vectors{i}, saddle_distances{i}, common_time, 'pchip', 'extrap');
        saddle_resampled = [saddle_resampled; resampled];
    end
    saddle_mean = mean(saddle_resampled, 1);
    saddle_std = std(saddle_resampled, 0, 1);
end

% Create Figure 11
figure('Position', [150, 150, 1200, 800]);
hold on;

% Plot 3-robot vortex: mean + std dev band
if ~isempty(vortex_distances)
    upper = vortex_mean + vortex_std;
    lower = vortex_mean - vortex_std;

    % Shaded std dev band
    x_fill = [common_time, fliplr(common_time)];
    y_fill = [upper, fliplr(lower)];
    fill(x_fill, y_fill, color_3r_vortex, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');

    % Mean line
    plot(common_time, vortex_mean, 'Color', color_3r_vortex, 'LineWidth', 3, ...
         'DisplayName', sprintf('3-Robot Vortex (n=%d)', vortex_count));
end

% Plot 3-robot saddle: mean + std dev band
if ~isempty(saddle_distances)
    upper = saddle_mean + saddle_std;
    lower = saddle_mean - saddle_std;

    % Shaded std dev band
    x_fill = [common_time, fliplr(common_time)];
    y_fill = [upper, fliplr(lower)];
    fill(x_fill, y_fill, color_3r_saddle, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');

    % Mean line
    plot(common_time, saddle_mean, 'Color', color_3r_saddle, 'LineWidth', 3, ...
         'DisplayName', sprintf('3-Robot Saddle (n=%d)', saddle_count));
end

% Configure plot
grid on;
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Distance from Origin (m)', 'FontSize', 14, 'FontWeight', 'bold');
title('Distance from Origin vs Time - Mean ± Std Dev', ...
      'FontSize', 16, 'FontWeight', 'bold');
xlim([0 10]);
ylim([0 inf]);

% Add legend
legend('Location', 'best', 'FontSize', 12);
set(gca, 'FontSize', 12);

% Save Figure 11
fprintf('\nSaving Figure 11...\n');
output_png_11 = fullfile(trunk_dir, 'combined_distance_from_origin_mean_std.png');
output_fig_11 = fullfile(trunk_dir, 'combined_distance_from_origin_mean_std.fig');
saveas(gcf, output_png_11);
fprintf('  Saved: %s\n', output_png_11);
saveas(gcf, output_fig_11);
fprintf('  Saved: %s\n', output_fig_11);

%% Figure 12: 16 Groups (8 Vortex + 8 Saddle)
fprintf('\n=== Creating Figure 12: 16 Groups with Std Dev Bands ===\n');

% Define grouping parameters
runs_per_group = 10;
num_groups = 8;

% Color palette for 16 groups (8 vortex + 8 saddle)
vortex_colors = [
    0.8 0.2 0.2;  % Red
    0.9 0.4 0.3;  % Light red
    0.7 0.1 0.1;  % Dark red
    0.9 0.5 0.4;  % Lighter red
    0.6 0.0 0.0;  % Very dark red
    0.9 0.3 0.2;  % Orange-red
    0.8 0.1 0.0;  % Dark orange-red
    0.9 0.6 0.5;  % Light orange-red
];

saddle_colors = [
    0.2 0.3 0.8;  % Blue
    0.3 0.4 0.9;  % Light blue
    0.1 0.2 0.7;  % Dark blue
    0.4 0.5 0.9;  % Lighter blue
    0.0 0.1 0.6;  % Very dark blue
    0.2 0.4 0.9;  % Sky blue
    0.1 0.3 0.8;  % Medium blue
    0.5 0.6 0.9;  % Light sky blue
];

% Process vortex groups
vortex_group_means = cell(1, num_groups);
vortex_group_stds = cell(1, num_groups);

for group_idx = 1:num_groups
    start_idx = (group_idx - 1) * runs_per_group + 1;
    end_idx = group_idx * runs_per_group;

    group_data = [];
    for run_idx = start_idx:end_idx
        if run_idx <= length(vortex_distances)
            resampled = interp1(vortex_time_vectors{run_idx}, vortex_distances{run_idx}, ...
                               common_time, 'pchip', 'extrap');
            group_data = [group_data; resampled];
        end
    end

    if ~isempty(group_data)
        vortex_group_means{group_idx} = mean(group_data, 1);
        vortex_group_stds{group_idx} = std(group_data, 0, 1);
    end
end

% Process saddle groups
saddle_group_means = cell(1, num_groups);
saddle_group_stds = cell(1, num_groups);

for group_idx = 1:num_groups
    start_idx = (group_idx - 1) * runs_per_group + 1;
    end_idx = group_idx * runs_per_group;

    group_data = [];
    for run_idx = start_idx:end_idx
        if run_idx <= length(saddle_distances)
            resampled = interp1(saddle_time_vectors{run_idx}, saddle_distances{run_idx}, ...
                               common_time, 'pchip', 'extrap');
            group_data = [group_data; resampled];
        end
    end

    if ~isempty(group_data)
        saddle_group_means{group_idx} = mean(group_data, 1);
        saddle_group_stds{group_idx} = std(group_data, 0, 1);
    end
end

% Create Figure 12
figure('Position', [200, 200, 1470, 945]);
hold on;

% Plot vortex groups
for group_idx = 1:num_groups
    if ~isempty(vortex_group_means{group_idx})
        mean_trace = vortex_group_means{group_idx};
        std_trace = vortex_group_stds{group_idx};

        upper = mean_trace + std_trace;
        lower = mean_trace - std_trace;

        % Shaded std dev band
        x_fill = [common_time, fliplr(common_time)];
        y_fill = [upper, fliplr(lower)];
        fill(x_fill, y_fill, vortex_colors(group_idx, :), 'FaceAlpha', 0.15, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Mean line
        start_run = (group_idx - 1) * runs_per_group + 1;
        end_run = group_idx * runs_per_group;
        plot(common_time, mean_trace, 'Color', vortex_colors(group_idx, :), ...
             'LineWidth', 2, 'DisplayName', sprintf('Vortex %d-%d', start_run, end_run));
    end
end

% Plot saddle groups
for group_idx = 1:num_groups
    if ~isempty(saddle_group_means{group_idx})
        mean_trace = saddle_group_means{group_idx};
        std_trace = saddle_group_stds{group_idx};

        upper = mean_trace + std_trace;
        lower = mean_trace - std_trace;

        % Shaded std dev band
        x_fill = [common_time, fliplr(common_time)];
        y_fill = [upper, fliplr(lower)];
        fill(x_fill, y_fill, saddle_colors(group_idx, :), 'FaceAlpha', 0.15, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Mean line
        start_run = (group_idx - 1) * runs_per_group + 1;
        end_run = group_idx * runs_per_group;
        plot(common_time, mean_trace, 'Color', saddle_colors(group_idx, :), ...
             'LineWidth', 2, 'DisplayName', sprintf('Saddle %d-%d', start_run, end_run));
    end
end

% Configure plot
grid on;
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Distance from Origin (m)', 'FontSize', 14, 'FontWeight', 'bold');
title('Distance from Origin vs Time - 16 Groups (Mean ± Std Dev)', ...
      'FontSize', 16, 'FontWeight', 'bold');
xlim([0 10]);
ylim([0 inf]);

% Add legend (two columns)
legend('Location', 'eastoutside', 'FontSize', 10, 'NumColumns', 2);
set(gca, 'FontSize', 12);

% Save Figure 12
fprintf('\nSaving Figure 12...\n');
output_png_12 = fullfile(trunk_dir, 'convergence.png');
output_fig_12 = fullfile(trunk_dir, 'convergence.fig');
saveas(gcf, output_png_12);
fprintf('  Saved: %s\n', output_png_12);
saveas(gcf, output_fig_12);
fprintf('  Saved: %s\n', output_fig_12);

%% Summary Statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Figure 11: Overall mean ± standard deviation for each test type\n');
fprintf('Figure 12: 16 groups with mean ± standard deviation\n');
fprintf('Total runs plotted: %d\n', vortex_count + saddle_count);
fprintf('  3-Robot Vortex: %d runs\n', vortex_count);
fprintf('  3-Robot Saddle: %d runs\n', saddle_count);
fprintf('\nDistance scaled by factor %.4f to correct printer units.\n', SCALE_FACTOR);
fprintf('Resampled to common time vector (0-10s, 101 points) for averaging.\n');

fprintf('\n=== Analysis Complete ===\n');
