% Calculate Average Resting Point and RMSE Statistics
% This script analyzes the final resting positions of robot clusters
% across all experimental runs for both saddle and vortex tests.
%
% IMPORTANT - SCALE FACTOR NOTE:
% This script reports RAW OptiTrack position values WITHOUT scale correction.
% The physical printed field map is 66 inches (1.6764m) instead of the
% theoretical 1m domain used in simulations.
%
% To convert to theoretical coordinates for paper tables:
%   theoretical_value = optitrack_value / 1.6764
%
% For SCALE-CORRECTED values that match theoretical field coordinates,
% use: trunk/combined_distance_from_origin_plotter.m
% That script applies the correction and generates Table III values for the paper.
%
% FIGURE 10 FOR THE PAPER and TABLE 4 - but without scale factor

%% Saddle Tests Analysis
fprintf('===== SADDLE POINT STATISTICS =====\n\n');

% Navigate to saddle_tests directory
saddle_dir = 'saddle_tests';
if ~exist(saddle_dir, 'dir')
    error('saddle_tests directory not found');
end

% Storage for saddle final positions
saddle_final_positions = [];

% Load all 80 run files for saddle tests
fprintf('Loading saddle test runs...\n');
for i = 1:80
    filename = fullfile(saddle_dir, sprintf('run%d.mat', i));

    if exist(filename, 'file')
        data = load(filename);

        % Extract final position from cluster_pose (saddle uses cluster_pose)
        if isfield(data, 'cluster_pose')
            trajectory = data.cluster_pose(:, 1:2);  % X and Y only
            final_pos = trajectory(end, :);           % Last point
            saddle_final_positions = [saddle_final_positions; final_pos];
        else
            fprintf('Warning: cluster_pose not found in %s\n', filename);
        end
    else
        fprintf('Warning: File not found - %s\n', filename);
    end
end

% Calculate statistics for saddle tests
if ~isempty(saddle_final_positions)
    % Average resting point (in meters) - COLUMN 3 in LaTeX Table IV
    saddle_avg_x = mean(saddle_final_positions(:, 1));
    saddle_avg_y = mean(saddle_final_positions(:, 2));

    % Systematic Bias - COLUMN 4 in LaTeX Table IV
    % Distance from origin to average rest point
    saddle_bias = sqrt(saddle_avg_x^2 + saddle_avg_y^2);

    % Precision (σ) - COLUMN 5 in LaTeX Table IV
    % RMS scatter around the average rest point (repeatability)
    deviations_from_mean = saddle_final_positions - [saddle_avg_x, saddle_avg_y];
    distances_from_mean = sqrt(deviations_from_mean(:, 1).^2 + deviations_from_mean(:, 2).^2);
    saddle_precision = sqrt(mean(distances_from_mean.^2));

    % Convert to millimeters
    saddle_avg_x_mm = saddle_avg_x * 1000;
    saddle_avg_y_mm = saddle_avg_y * 1000;
    saddle_bias_mm = saddle_bias * 1000;
    saddle_precision_mm = saddle_precision * 1000;

    % Display results
    fprintf('Number of runs analyzed: %d\n', size(saddle_final_positions, 1));
    fprintf('Average Rest Point: (%.2f, %.2f) mm\n', saddle_avg_x_mm, saddle_avg_y_mm);
    fprintf('Systematic Bias: %.2f mm\n', saddle_bias_mm);
    fprintf('Precision (sigma): %.2f mm\n\n', saddle_precision_mm);
else
    fprintf('Error: No valid data found for saddle tests\n\n');
end

%% Vortex Tests Analysis
fprintf('===== VORTEX STATISTICS =====\n\n');

% Navigate to vortex_tests directory
vortex_dir = 'vortex_tests';
if ~exist(vortex_dir, 'dir')
    error('vortex_tests directory not found');
end

% Storage for vortex final positions
vortex_final_positions = [];

% Load all 80 run files for vortex tests
fprintf('Loading vortex test runs...\n');
for i = 1:80
    filename = fullfile(vortex_dir, sprintf('run%d.mat', i));

    if exist(filename, 'file')
        data = load(filename);

        % Extract final position from cluster_position (vortex uses cluster_position)
        if isfield(data, 'cluster_position')
            trajectory = data.cluster_position(:, 1:2);  % X and Y only
            final_pos = trajectory(end, :);              % Last point
            vortex_final_positions = [vortex_final_positions; final_pos];
        else
            fprintf('Warning: cluster_position not found in %s\n', filename);
        end
    else
        fprintf('Warning: File not found - %s\n', filename);
    end
end

% Calculate statistics for vortex tests
if ~isempty(vortex_final_positions)
    % Average resting point (in meters) - COLUMN 3 in LaTeX Table IV
    vortex_avg_x = mean(vortex_final_positions(:, 1));
    vortex_avg_y = mean(vortex_final_positions(:, 2));

    % Systematic Bias - COLUMN 4 in LaTeX Table IV
    % Distance from origin to average rest point
    vortex_bias = sqrt(vortex_avg_x^2 + vortex_avg_y^2);

    % Precision (σ) - COLUMN 5 in LaTeX Table IV
    % RMS scatter around the average rest point (repeatability)
    deviations_from_mean = vortex_final_positions - [vortex_avg_x, vortex_avg_y];
    distances_from_mean = sqrt(deviations_from_mean(:, 1).^2 + deviations_from_mean(:, 2).^2);
    vortex_precision = sqrt(mean(distances_from_mean.^2));

    % Convert to millimeters
    vortex_avg_x_mm = vortex_avg_x * 1000;
    vortex_avg_y_mm = vortex_avg_y * 1000;
    vortex_bias_mm = vortex_bias * 1000;
    vortex_precision_mm = vortex_precision * 1000;

    % Display results
    fprintf('Number of runs analyzed: %d\n', size(vortex_final_positions, 1));
    fprintf('Average Rest Point: (%.2f, %.2f) mm\n', vortex_avg_x_mm, vortex_avg_y_mm);
    fprintf('Systematic Bias: %.2f mm\n', vortex_bias_mm);
    fprintf('Precision (sigma): %.2f mm\n\n', vortex_precision_mm);
else
    fprintf('Error: No valid data found for vortex tests\n\n');
end

%% Summary Output (for easy copy-paste to LaTeX Table IV)
fprintf('===== SUMMARY FOR LATEX TABLE IV =====\n\n');
fprintf('Format: | Environment | Cluster Size | Avg Rest Point (x,y) | Systematic Bias | Precision (sigma) |\n\n');

if ~isempty(saddle_final_positions)
    fprintf('Saddle Point:\n');
    fprintf('  3-Robot | (%.4f, %.4f) m | %.4f m | %.4f m\n', ...
        saddle_avg_x, saddle_avg_y, saddle_bias, saddle_precision);
    fprintf('  (%.2f, %.2f) mm | %.2f mm | %.2f mm\n\n', ...
        saddle_avg_x_mm, saddle_avg_y_mm, saddle_bias_mm, saddle_precision_mm);
end

if ~isempty(vortex_final_positions)
    fprintf('Vortex (Center):\n');
    fprintf('  3-Robot | (%.4f, %.4f) m | %.4f m | %.4f m\n', ...
        vortex_avg_x, vortex_avg_y, vortex_bias, vortex_precision);
    fprintf('  (%.2f, %.2f) mm | %.2f mm | %.2f mm\n\n', ...
        vortex_avg_x_mm, vortex_avg_y_mm, vortex_bias_mm, vortex_precision_mm);
end
