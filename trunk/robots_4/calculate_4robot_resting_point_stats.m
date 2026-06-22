% Calculate Average Resting Point and RMSE Statistics for 4-Robot Tests
% This script analyzes the final resting positions of 4-robot clusters
% across all experimental runs for vortex tests (run to center).

%% 4-Robot Vortex Tests Analysis
fprintf('===== 4-ROBOT VORTEX CENTER FINDING STATISTICS =====\n\n');

% Storage for final positions
vortex_4robot_final_positions = [];

% Load all 80 run files for 4-robot vortex tests
fprintf('Loading 4-robot vortex test runs...\n');
for i = 1:80
    filename = sprintf('run%d.mat', i);

    if exist(filename, 'file')
        data = load(filename);

        % Try different possible field names for cluster position
        if isfield(data, 'cluster_position')
            trajectory = data.cluster_position(:, 1:2);  % X and Y only
            final_pos = trajectory(end, :);              % Last point
            vortex_4robot_final_positions = [vortex_4robot_final_positions; final_pos];
        elseif isfield(data, 'cluster_pose')
            trajectory = data.cluster_pose(:, 1:2);      % X and Y only
            final_pos = trajectory(end, :);              % Last point
            vortex_4robot_final_positions = [vortex_4robot_final_positions; final_pos];
        elseif isfield(data, 'centroid_position')
            trajectory = data.centroid_position(:, 1:2); % X and Y only
            final_pos = trajectory(end, :);              % Last point
            vortex_4robot_final_positions = [vortex_4robot_final_positions; final_pos];
        else
            fprintf('Warning: No cluster position data found in %s\n', filename);
            % List available fields for debugging
            fields_in_file = fieldnames(data);
            fprintf('  Available fields: %s\n', strjoin(fields_in_file, ', '));
        end
    else
        fprintf('Warning: File not found - %s\n', filename);
    end
end

% Calculate statistics for 4-robot vortex tests
if ~isempty(vortex_4robot_final_positions)
    % Average resting point (in meters) - COLUMN 3 in LaTeX Table IV
    vortex_4robot_avg_x = mean(vortex_4robot_final_positions(:, 1));
    vortex_4robot_avg_y = mean(vortex_4robot_final_positions(:, 2));

    % Systematic Bias - COLUMN 4 in LaTeX Table IV
    % Distance from origin to average rest point
    vortex_4robot_bias = sqrt(vortex_4robot_avg_x^2 + vortex_4robot_avg_y^2);

    % Precision (σ) - COLUMN 5 in LaTeX Table IV
    % RMS scatter around the average rest point (repeatability)
    deviations_from_mean = vortex_4robot_final_positions - [vortex_4robot_avg_x, vortex_4robot_avg_y];
    distances_from_mean = sqrt(deviations_from_mean(:, 1).^2 + deviations_from_mean(:, 2).^2);
    vortex_4robot_precision = sqrt(mean(distances_from_mean.^2));

    % Convert to millimeters
    vortex_4robot_avg_x_mm = vortex_4robot_avg_x * 1000;
    vortex_4robot_avg_y_mm = vortex_4robot_avg_y * 1000;
    vortex_4robot_bias_mm = vortex_4robot_bias * 1000;
    vortex_4robot_precision_mm = vortex_4robot_precision * 1000;

    % Display results
    fprintf('Number of runs analyzed: %d\n', size(vortex_4robot_final_positions, 1));
    fprintf('Average Rest Point: (%.2f, %.2f) mm\n', vortex_4robot_avg_x_mm, vortex_4robot_avg_y_mm);
    fprintf('Systematic Bias: %.2f mm\n', vortex_4robot_bias_mm);
    fprintf('Precision (sigma): %.2f mm\n\n', vortex_4robot_precision_mm);

    % Additional statistics
    fprintf('Standard deviation X: %.2f mm\n', std(vortex_4robot_final_positions(:, 1)) * 1000);
    fprintf('Standard deviation Y: %.2f mm\n', std(vortex_4robot_final_positions(:, 2)) * 1000);
    fprintf('Max distance from mean: %.2f mm\n', max(distances_from_mean) * 1000);
    fprintf('Min distance from mean: %.2f mm\n\n', min(distances_from_mean) * 1000);
else
    fprintf('Error: No valid data found for 4-robot vortex tests\n\n');
end

%% Check if we need to analyze saddle tests too
% Check if there's a saddle_tests directory for 4-robot
if exist('saddle_tests', 'dir')
    fprintf('===== 4-ROBOT SADDLE POINT STATISTICS =====\n\n');

    % Storage for saddle final positions
    saddle_4robot_final_positions = [];

    % Navigate to saddle_tests directory and load files
    cd('saddle_tests');

    fprintf('Loading 4-robot saddle test runs...\n');
    for i = 1:80
        filename = sprintf('run%d.mat', i);

        if exist(filename, 'file')
            data = load(filename);

            % Try different possible field names for cluster position
            if isfield(data, 'cluster_position')
                trajectory = data.cluster_position(:, 1:2);  % X and Y only
                final_pos = trajectory(end, :);              % Last point
                saddle_4robot_final_positions = [saddle_4robot_final_positions; final_pos];
            elseif isfield(data, 'cluster_pose')
                trajectory = data.cluster_pose(:, 1:2);      % X and Y only
                final_pos = trajectory(end, :);              % Last point
                saddle_4robot_final_positions = [saddle_4robot_final_positions; final_pos];
            else
                fprintf('Warning: No cluster position data found in %s\n', filename);
            end
        else
            break;  % Stop if no more files found
        end
    end

    cd('..');  % Return to parent directory

    % Calculate statistics for 4-robot saddle tests
    if ~isempty(saddle_4robot_final_positions)
        % Average resting point (in meters)
        saddle_4robot_avg_x = mean(saddle_4robot_final_positions(:, 1));
        saddle_4robot_avg_y = mean(saddle_4robot_final_positions(:, 2));

        % RMSE from origin (0,0)
        distances_from_origin = sqrt(saddle_4robot_final_positions(:, 1).^2 + ...
                                     saddle_4robot_final_positions(:, 2).^2);
        saddle_4robot_rmse = sqrt(mean(distances_from_origin.^2));

        % Convert to millimeters
        saddle_4robot_avg_x_mm = saddle_4robot_avg_x * 1000;
        saddle_4robot_avg_y_mm = saddle_4robot_avg_y * 1000;
        saddle_4robot_rmse_mm = saddle_4robot_rmse * 1000;

        % Display results
        fprintf('Number of runs analyzed: %d\n', size(saddle_4robot_final_positions, 1));
        fprintf('Average Rest Point: (%.2f, %.2f) mm\n', saddle_4robot_avg_x_mm, saddle_4robot_avg_y_mm);
        fprintf('RMSE from origin: %.2f mm\n\n', saddle_4robot_rmse_mm);
    end
end

%% Summary Output for LaTeX Table IV
fprintf('===== SUMMARY FOR LATEX TABLE IV (4-ROBOT) =====\n\n');
fprintf('Format: | Environment | Cluster Size | Avg Rest Point (x,y) | Systematic Bias | Precision (sigma) |\n\n');

if ~isempty(vortex_4robot_final_positions)
    fprintf('Center (Vortex):\n');
    fprintf('  4-Robot | (%.4f, %.4f) m | %.4f m | %.4f m\n', ...
        vortex_4robot_avg_x, vortex_4robot_avg_y, vortex_4robot_bias, vortex_4robot_precision);
    fprintf('  (%.2f, %.2f) mm | %.2f mm | %.2f mm\n\n', ...
        vortex_4robot_avg_x_mm, vortex_4robot_avg_y_mm, vortex_4robot_bias_mm, vortex_4robot_precision_mm);
end

if exist('saddle_4robot_final_positions', 'var') && ~isempty(saddle_4robot_final_positions)
    fprintf('Saddle Point:\n');
    fprintf('  4-Robot | (%.4f, %.4f) m | %.4f m | %.4f m\n', ...
        saddle_4robot_avg_x, saddle_4robot_avg_y, saddle_4robot_bias, saddle_4robot_precision);
    fprintf('  (%.2f, %.2f) mm | %.2f mm | %.2f mm\n\n', ...
        saddle_4robot_avg_x_mm, saddle_4robot_avg_y_mm, saddle_4robot_bias_mm, saddle_4robot_precision_mm);
end

fprintf('\nNote: These results combine all 4-robot runs.\n');
fprintf('To separate by control primitive (Planar Center vs Dual Jacobian),\n');
fprintf('additional metadata about which runs used which method is needed.\n');

%% Save results to CSV for easy import
if ~isempty(vortex_4robot_final_positions)
    results_table = table(...
        {'Vortex_4Robot'}, ...
        {size(vortex_4robot_final_positions, 1)}, ...
        {vortex_4robot_avg_x_mm}, ...
        {vortex_4robot_avg_y_mm}, ...
        {vortex_4robot_bias_mm}, ...
        {vortex_4robot_precision_mm}, ...
        'VariableNames', {'Environment', 'NumRuns', 'AvgX_mm', 'AvgY_mm', 'Bias_mm', 'Precision_mm'});

    writetable(results_table, '4robot_vortex_stats.csv');
    fprintf('\nResults saved to 4robot_vortex_stats.csv\n');
end