% Combined Distance from Origin Plotter
%
% FIGURE 10 FOR THE PAPER and TABLE 4
%f
% This script creates Figure 10, which shows distance from origin over time
% for all hardware test runs from three test scenarios:
%   - 3-robot vortex tests (red)
%   - 3-robot saddle tests (blue)
%   - 4-robot tests (black)
%
% Related to Table 4 analysis - shows orbital control and convergence performance
% across all 236+ hardware experiments
%
% IMPORTANT - SCALE FACTOR NOTE:
% This script reports SCALE-CORRECTED values for theoretical field coordinates.
% The physical printed field map is 66 inches (1.6764m) instead of the
% theoretical 1m domain used in simulations.
%
% All OptiTrack position values are divided by 1.6764 to convert to
% theoretical coordinates. These corrected values are used in Table III.
%
% To convert from theoretical coordinates back to raw OptiTrack values:
%   optitrack_value = theoretical_value * 1.6764
%
% For RAW OptiTrack values WITHOUT scale correction, use:
% trunk/robots_3/calculate_resting_point_stats.m

%% Configuration
% Scaling factor to correct printer scaling issue
SCALE_FACTOR = 1.6764;  % Converts from printer units to actual meters

% Colors for each test type
color_3r_vortex = [0.8 0.2 0.2];  % Red
color_3r_saddle = [0.2 0.3 0.8];  % Blue
% color_4r = [0.2 0.2 0.2];         % Black

% Get script directory to build absolute paths
script_path = fileparts(mfilename('fullpath'));
trunk_dir = script_path;  % Script is in trunk/

% Storage for final positions (for Table IV verification)
vortex_final_positions = [];
saddle_final_positions = [];
% fourrobot_final_positions = [];

fprintf('=== Combined Distance from Origin Analysis (Figure 10) ===\n');
fprintf('Script location: %s\n', trunk_dir);
fprintf('Loading data from three test scenarios...\n\n');

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

            % Store final position for Table IV verification
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

            % Store final position for Table IV verification
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

%% Load 4-Robot Tests
% fprintf('\n=== Loading 4-Robot Tests ===\n');
% fourrobot_dir = fullfile(trunk_dir, 'robots_4');
% fourrobot_distances = {};
% fourrobot_time_vectors = {};
% fourrobot_count = 0;
%
% % Load run1.mat through run80.mat
% for i = 1:80
%     filename = fullfile(fourrobot_dir, sprintf('run%d.mat', i));
%     if exist(filename, 'file')
%         try
%             temp_data = load(filename);
%
%             % For 4-robot data, cluster center must be calculated from robot positions
%             if isfield(temp_data, 'robot1_pose') && isfield(temp_data, 'robot2_pose') && ...
%                isfield(temp_data, 'robot3_pose') && isfield(temp_data, 'robot4_pose')
%
%                 num_steps = size(temp_data.robot1_pose, 1);
%                 cluster_center = zeros(num_steps, 2);
%
%                 % Calculate cluster center for each timestep
%                 for t = 1:num_steps
%                     cluster_center(t, 1) = (temp_data.robot1_pose(t, 1) + ...
%                                             temp_data.robot2_pose(t, 1) + ...
%                                             temp_data.robot3_pose(t, 1) + ...
%                                             temp_data.robot4_pose(t, 1)) / 4;
%                     cluster_center(t, 2) = (temp_data.robot1_pose(t, 2) + ...
%                                             temp_data.robot2_pose(t, 2) + ...
%                                             temp_data.robot3_pose(t, 2) + ...
%                                             temp_data.robot4_pose(t, 2)) / 4;
%                 end
%
%                 % Calculate distance from origin
%                 x_c = cluster_center(:, 1);
%                 y_c = cluster_center(:, 2);
%                 distance = sqrt(x_c.^2 + y_c.^2) / SCALE_FACTOR;
%
%                 % Store final position for Table IV verification
%                 final_pos = [x_c(end), y_c(end)] / SCALE_FACTOR;
%                 fourrobot_final_positions = [fourrobot_final_positions; final_pos];
%
%                 % Create time vector for this run (no resampling)
%                 num_points = length(distance);
%                 time_vec = (0:(num_points-1)) * 0.1;  % Assuming 0.1s timestep
%
%                 % Store the distance trace and time vector
%                 fourrobot_distances{end+1} = distance;
%                 fourrobot_time_vectors{end+1} = time_vec;
%                 fourrobot_count = fourrobot_count + 1;
%
%             elseif isfield(temp_data, 'cluster_position') || isfield(temp_data, 'cluster_pose')
%                 % Fallback if cluster data exists without individual robot poses
%                 if isfield(temp_data, 'cluster_position')
%                     cluster_pos = temp_data.cluster_position(:, 1:2);
%                 else
%                     cluster_pos = temp_data.cluster_pose(:, 1:2);
%                 end
%
%                 % Calculate distance from origin
%                 x_c = cluster_pos(:, 1);
%                 y_c = cluster_pos(:, 2);
%                 distance = sqrt(x_c.^2 + y_c.^2) / SCALE_FACTOR;
%
%                 % Store final position for Table IV verification
%                 final_pos = [x_c(end), y_c(end)] / SCALE_FACTOR;
%                 fourrobot_final_positions = [fourrobot_final_positions; final_pos];
%
%                 % Create time vector for this run (no resampling)
%                 num_points = length(distance);
%                 time_vec = (0:(num_points-1)) * 0.1;  % Assuming 0.1s timestep
%
%                 % Store the distance trace and time vector
%                 fourrobot_distances{end+1} = distance;
%                 fourrobot_time_vectors{end+1} = time_vec;
%                 fourrobot_count = fourrobot_count + 1;
%             end
%         catch ME
%             fprintf('  Warning: Could not load %s - %s\n', filename, ME.message);
%         end
%     end
% end
% fprintf('Loaded %d 4-robot test runs\n', fourrobot_count);

%% Create Combined Plot (Figure 10)
fprintf('\n=== Creating Figure 10: Combined Distance from Origin Plot ===\n');

figure('Position', [100, 100, 1200, 800]);
hold on;

% Plot 3-robot vortex tests (red)
for i = 1:length(vortex_distances)
    plot(vortex_time_vectors{i}, vortex_distances{i}, 'Color', color_3r_vortex, ...
         'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% Plot 3-robot saddle tests (blue)
for i = 1:length(saddle_distances)
    plot(saddle_time_vectors{i}, saddle_distances{i}, 'Color', color_3r_saddle, ...
         'LineWidth', 1.2, 'HandleVisibility', 'off');
end

% % Plot 4-robot tests (black)
% for i = 1:length(fourrobot_distances)
%     plot(fourrobot_time_vectors{i}, fourrobot_distances{i}, 'Color', color_4r, ...
%          'LineWidth', 1.2, 'HandleVisibility', 'off');
% end

% Add legend entries (plot one dummy line for each type)
if ~isempty(vortex_distances)
    plot(NaN, NaN, 'Color', color_3r_vortex, 'LineWidth', 3, ...
         'DisplayName', sprintf('3-Robot Vortex (n=%d)', vortex_count));
end
if ~isempty(saddle_distances)
    plot(NaN, NaN, 'Color', color_3r_saddle, 'LineWidth', 3, ...
         'DisplayName', sprintf('3-Robot Saddle (n=%d)', saddle_count));
end
% if ~isempty(fourrobot_distances)
%     plot(NaN, NaN, 'Color', color_4r, 'LineWidth', 3, ...
%          'DisplayName', sprintf('4-Robot Vortex (n=%d)', fourrobot_count));
% end

% Configure plot
grid on;
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Distance from Origin (m)', 'FontSize', 14, 'FontWeight', 'bold');
title('Distance from Origin vs Time - All Test Scenarios', ...
      'FontSize', 16, 'FontWeight', 'bold');
xlim([0 10]);
ylim([0 inf]);  % Auto-scale y-axis but start from 0

% Add legend
legend('Location', 'best', 'FontSize', 12);
set(gca, 'FontSize', 12);

% Save figure
fprintf('\nSaving Figure 10...\n');
output_png = fullfile(trunk_dir, 'combined_distance_from_origin.png');
output_fig = fullfile(trunk_dir, 'combined_distance_from_origin.fig');
saveas(gcf, output_png);
fprintf('  Saved: %s\n', output_png);
saveas(gcf, output_fig);
fprintf('  Saved: %s\n', output_fig);

%% Figure 11: Average with Standard Deviation Bands
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

% % Resample all 4-robot runs to common time vector
% fourrobot_resampled = [];
% if ~isempty(fourrobot_distances)
%     for i = 1:length(fourrobot_distances)
%         resampled = interp1(fourrobot_time_vectors{i}, fourrobot_distances{i}, common_time, 'pchip', 'extrap');
%         fourrobot_resampled = [fourrobot_resampled; resampled];
%     end
%     fourrobot_mean = mean(fourrobot_resampled, 1);
%     fourrobot_std = std(fourrobot_resampled, 0, 1);
% end

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

% % Plot 4-robot: mean + std dev band
% if ~isempty(fourrobot_distances)
%     upper = fourrobot_mean + fourrobot_std;
%     lower = fourrobot_mean - fourrobot_std;
%
%     % Shaded std dev band
%     x_fill = [common_time, fliplr(common_time)];
%     y_fill = [upper, fliplr(lower)];
%     fill(x_fill, y_fill, color_4r, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
%          'HandleVisibility', 'off');
%
%     % Mean line
%     plot(common_time, fourrobot_mean, 'Color', color_4r, 'LineWidth', 3, ...
%          'DisplayName', sprintf('4-Robot Vortex (n=%d)', fourrobot_count));
% end

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

%% Summary Statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Figure 10: All individual runs plotted\n');
fprintf('Figure 11: Mean ± standard deviation for each test type\n');
fprintf('Total runs plotted: %d\n', vortex_count + saddle_count);
fprintf('  3-Robot Vortex: %d runs\n', vortex_count);
fprintf('  3-Robot Saddle: %d runs\n', saddle_count);
% fprintf('  4-Robot Vortex: %d runs\n', fourrobot_count);
fprintf('\nDistance scaled by factor %.4f to correct printer units.\n', SCALE_FACTOR);
fprintf('Figure 10: No resampling applied - using raw data from .mat files.\n');
fprintf('Figure 11: Resampled to common time vector (0-10s, 101 points) for averaging.\n');

%% TABLE IV VERIFICATION
fprintf('\n=== TABLE IV VERIFICATION ===\n\n');
fprintf('Computing convergence statistics from final resting positions...\n\n');

% Expected Table IV values (from paper)
table4_expected = struct();
table4_expected.saddle_3r = struct('x', 0.007, 'y', -0.002, 'bias', 0.008, 'precision', 0.024);
table4_expected.vortex_3r = struct('x', -0.006, 'y', 0.020, 'bias', 0.021, 'precision', 0.014);
% table4_expected.vortex_4r = struct('x', 0.015, 'y', -0.006, 'bias', 0.016, 'precision', 0.031);

% Calculate statistics for each test type
results = struct();

% 3-Robot Saddle
if ~isempty(saddle_final_positions)
    saddle_avg_x = mean(saddle_final_positions(:, 1));
    saddle_avg_y = mean(saddle_final_positions(:, 2));
    saddle_bias = sqrt(saddle_avg_x^2 + saddle_avg_y^2);
    deviations = saddle_final_positions - [saddle_avg_x, saddle_avg_y];
    distances_from_mean = sqrt(deviations(:, 1).^2 + deviations(:, 2).^2);
    saddle_precision = sqrt(mean(distances_from_mean.^2));

    results.saddle_3r = struct('x', saddle_avg_x, 'y', saddle_avg_y, ...
                               'bias', saddle_bias, 'precision', saddle_precision, ...
                               'num_runs', size(saddle_final_positions, 1));
end

% 3-Robot Vortex
if ~isempty(vortex_final_positions)
    vortex_avg_x = mean(vortex_final_positions(:, 1));
    vortex_avg_y = mean(vortex_final_positions(:, 2));
    vortex_bias = sqrt(vortex_avg_x^2 + vortex_avg_y^2);
    deviations = vortex_final_positions - [vortex_avg_x, vortex_avg_y];
    distances_from_mean = sqrt(deviations(:, 1).^2 + deviations(:, 2).^2);
    vortex_precision = sqrt(mean(distances_from_mean.^2));

    results.vortex_3r = struct('x', vortex_avg_x, 'y', vortex_avg_y, ...
                               'bias', vortex_bias, 'precision', vortex_precision, ...
                               'num_runs', size(vortex_final_positions, 1));
end

% % 4-Robot Vortex
% if ~isempty(fourrobot_final_positions)
%     fourrobot_avg_x = mean(fourrobot_final_positions(:, 1));
%     fourrobot_avg_y = mean(fourrobot_final_positions(:, 2));
%     fourrobot_bias = sqrt(fourrobot_avg_x^2 + fourrobot_avg_y^2);
%     deviations = fourrobot_final_positions - [fourrobot_avg_x, fourrobot_avg_y];
%     distances_from_mean = sqrt(deviations(:, 1).^2 + deviations(:, 2).^2);
%     fourrobot_precision = sqrt(mean(distances_from_mean.^2));
%
%     results.vortex_4r = struct('x', fourrobot_avg_x, 'y', fourrobot_avg_y, ...
%                                'bias', fourrobot_bias, 'precision', fourrobot_precision, ...
%                                'num_runs', size(fourrobot_final_positions, 1));
% end

% Display comparison table
fprintf('┌────────────────┬──────────────────────────┬────────────────┬──────────────┐\n');
fprintf('│ Environment    │ Avg Rest Point (m)       │ Sys Bias (m)   │ Precision σ  │\n');
fprintf('├────────────────┼──────────────────────────┼────────────────┼──────────────┤\n');

% Saddle 3-Robot
if isfield(results, 'saddle_3r')
    fprintf('│ Saddle 3R      │                          │                │              │\n');
    fprintf('│   Computed     │ (%+.4f, %+.4f)        │ %.4f         │ %.4f       │\n', ...
        results.saddle_3r.x, results.saddle_3r.y, results.saddle_3r.bias, results.saddle_3r.precision);
    fprintf('│   Table IV     │ (%+.4f, %+.4f)        │ %.4f         │ %.4f       │\n', ...
        table4_expected.saddle_3r.x, table4_expected.saddle_3r.y, ...
        table4_expected.saddle_3r.bias, table4_expected.saddle_3r.precision);
    fprintf('│   Difference   │ (%+.4f, %+.4f)        │ %+.4f        │ %+.4f      │\n', ...
        results.saddle_3r.x - table4_expected.saddle_3r.x, ...
        results.saddle_3r.y - table4_expected.saddle_3r.y, ...
        results.saddle_3r.bias - table4_expected.saddle_3r.bias, ...
        results.saddle_3r.precision - table4_expected.saddle_3r.precision);
    fprintf('├────────────────┼──────────────────────────┼────────────────┼──────────────┤\n');
end

% Vortex 3-Robot
if isfield(results, 'vortex_3r')
    fprintf('│ Vortex 3R      │                          │                │              │\n');
    fprintf('│   Computed     │ (%+.4f, %+.4f)        │ %.4f         │ %.4f       │\n', ...
        results.vortex_3r.x, results.vortex_3r.y, results.vortex_3r.bias, results.vortex_3r.precision);
    fprintf('│   Table IV     │ (%+.4f, %+.4f)        │ %.4f         │ %.4f       │\n', ...
        table4_expected.vortex_3r.x, table4_expected.vortex_3r.y, ...
        table4_expected.vortex_3r.bias, table4_expected.vortex_3r.precision);
    fprintf('│   Difference   │ (%+.4f, %+.4f)        │ %+.4f        │ %+.4f      │\n', ...
        results.vortex_3r.x - table4_expected.vortex_3r.x, ...
        results.vortex_3r.y - table4_expected.vortex_3r.y, ...
        results.vortex_3r.bias - table4_expected.vortex_3r.bias, ...
        results.vortex_3r.precision - table4_expected.vortex_3r.precision);
    % fprintf('├────────────────┼──────────────────────────┼────────────────┼──────────────┤\n');
end

% % Vortex 4-Robot
% if isfield(results, 'vortex_4r')
%     fprintf('│ Vortex 4R      │                          │                │              │\n');
%     fprintf('│   Computed     │ (%+.4f, %+.4f)        │ %.4f         │ %.4f       │\n', ...
%         results.vortex_4r.x, results.vortex_4r.y, results.vortex_4r.bias, results.vortex_4r.precision);
%     fprintf('│   Table IV     │ (%+.4f, %+.4f)        │ %.4f         │ %.4f       │\n', ...
%         table4_expected.vortex_4r.x, table4_expected.vortex_4r.y, ...
%         table4_expected.vortex_4r.bias, table4_expected.vortex_4r.precision);
%     fprintf('│   Difference   │ (%+.4f, %+.4f)        │ %+.4f        │ %+.4f      │\n', ...
%         results.vortex_4r.x - table4_expected.vortex_4r.x, ...
%         results.vortex_4r.y - table4_expected.vortex_4r.y, ...
%         results.vortex_4r.bias - table4_expected.vortex_4r.bias, ...
%         results.vortex_4r.precision - table4_expected.vortex_4r.precision);
% end

fprintf('└────────────────┴──────────────────────────┴────────────────┴──────────────┘\n');

fprintf('\n=== Verification Summary ===\n');
fprintf('All values computed using same formulas as calculate_resting_point_stats.m\n');
fprintf('Differences show: Computed - Table IV\n');
fprintf('Small differences (<0.001m) are likely due to rounding in the paper.\n');

fprintf('\n=== Analysis Complete ===\n');
