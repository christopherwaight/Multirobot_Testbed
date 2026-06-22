% Export All Orbit Trajectories to CSV
% This script exports cluster trajectories from all orbit*.mat files to CSV format
% for use in Python sim-to-real comparison experiments
%
% File naming convention:
% - 0XX: 3-robot equilateral formation (radius = XX/100 meters)
% - 1XX: 4-robot square formation (radius = XX/100 meters)
% - 2XX: 4-robot advanced formation (radius = XX/100 meters)

fprintf('===========================================================\n');
fprintf('         EXPORT ALL ORBIT TRAJECTORIES TO CSV             \n');
fprintf('===========================================================\n\n');

% Create output directory
output_dir = 'real_robot_trajectories';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
else
    fprintf('Output directory exists: %s\n', output_dir);
end
fprintf('\n');

% Get all orbit*.mat files in current directory
all_files = dir('orbit*.mat');
if isempty(all_files)
    error('No orbit*.mat files found in current directory');
end

fprintf('Found %d orbit files to process\n', length(all_files));
fprintf('-----------------------------------------------------------\n\n');

% Process statistics
total_files = length(all_files);
successful_exports = 0;
failed_exports = 0;
failed_files = {};

% Process each file
for i = 1:length(all_files)
    filename = all_files(i).name;
    [~, basename, ~] = fileparts(filename);

    fprintf('[%d/%d] Processing %s...\n', i, total_files, filename);

    try
        % Load the data file
        data = load(filename);

        % Extract orbit number and determine configuration
        orbit_num = regexp(basename, '\d+', 'match');
        if isempty(orbit_num)
            error('Could not extract orbit number from filename');
        end
        orbit_num = str2double(orbit_num{1});

        % Determine robot configuration
        if orbit_num < 100
            robot_config = '3-robot';
            radius = orbit_num / 100;
            series = '0XX';
        elseif orbit_num < 200
            robot_config = '4-robot-square';
            radius = (orbit_num - 100) / 100;
            series = '1XX';
        else
            robot_config = '4-robot-advanced';
            radius = (orbit_num - 200) / 100;
            series = '2XX';
        end

        fprintf('  Configuration: %s (series %s), Radius: %.2f m\n', ...
                robot_config, series, radius);

        % Extract cluster position data
        positions_extracted = false;

        % Method 1: Use existing cluster_position field
        if isfield(data, 'cluster_position')
            positions = data.cluster_position(:, 1:2);  % Extract x, y (ignore theta)
            fprintf('  Using cluster_position field\n');
            positions_extracted = true;

        % Method 2: Calculate from 3 robot positions
        elseif isfield(data, 'robot1_pose') && isfield(data, 'robot2_pose') && ...
               isfield(data, 'robot3_pose')
            fprintf('  Calculating cluster center from 3 robot positions\n');
            num_steps = size(data.robot1_pose, 1);
            positions = zeros(num_steps, 2);

            for t = 1:num_steps
                positions(t, 1) = (data.robot1_pose(t, 1) + ...
                                  data.robot2_pose(t, 1) + ...
                                  data.robot3_pose(t, 1)) / 3;
                positions(t, 2) = (data.robot1_pose(t, 2) + ...
                                  data.robot2_pose(t, 2) + ...
                                  data.robot3_pose(t, 2)) / 3;
            end
            positions_extracted = true;

        % Method 3: Calculate from 4 robot positions
        elseif isfield(data, 'robot1_pose') && isfield(data, 'robot2_pose') && ...
               isfield(data, 'robot3_pose') && isfield(data, 'robot4_pose')
            fprintf('  Calculating cluster center from 4 robot positions\n');
            num_steps = size(data.robot1_pose, 1);
            positions = zeros(num_steps, 2);

            for t = 1:num_steps
                positions(t, 1) = (data.robot1_pose(t, 1) + ...
                                  data.robot2_pose(t, 1) + ...
                                  data.robot3_pose(t, 1) + ...
                                  data.robot4_pose(t, 1)) / 4;
                positions(t, 2) = (data.robot1_pose(t, 2) + ...
                                  data.robot2_pose(t, 2) + ...
                                  data.robot3_pose(t, 2) + ...
                                  data.robot4_pose(t, 2)) / 4;
            end
            positions_extracted = true;

        % Method 4: Use centroid field if available
        elseif isfield(data, 'centroid')
            positions = data.centroid(:, 1:2);
            fprintf('  Using centroid field\n');
            positions_extracted = true;
        end

        if ~positions_extracted
            error('Could not find cluster position data');
        end

        % Extract x and y coordinates (already in meters)
        x_scaled = positions(:, 1);
        y_scaled = positions(:, 2);

        % Generate time vector (assuming 0.1s timestep)
        timestep = 0.1;  % seconds
        time = (0:length(x_scaled)-1)' * timestep;

        % Calculate statistics
        actual_radius = sqrt(x_scaled.^2 + y_scaled.^2);
        mean_radius = mean(actual_radius);
        std_radius = std(actual_radius);

        fprintf('  Data points: %d, Duration: %.1f s\n', ...
                length(x_scaled), time(end));
        fprintf('  Radius stats: mean=%.3f m, std=%.3f m\n', ...
                mean_radius, std_radius);

        % Prepare export data matrix [time, x, y]
        export_data = [time, x_scaled, y_scaled];

        % Export to CSV
        output_file = fullfile(output_dir, sprintf('%s_trajectory.csv', basename));

        % Write CSV with header
        fid = fopen(output_file, 'w');
        if fid == -1
            error('Could not open output file for writing');
        end

        fprintf(fid, 'time_s,x_m,y_m\n');  % Header row
        fprintf(fid, '%.6f,%.6f,%.6f\n', export_data');  % Data rows
        fclose(fid);

        fprintf('  ✓ Exported to %s\n\n', output_file);
        successful_exports = successful_exports + 1;

    catch ME
        fprintf('  ✗ ERROR: %s\n\n', ME.message);
        failed_exports = failed_exports + 1;
        failed_files{end+1} = filename;
    end
end

% Print summary
fprintf('===========================================================\n');
fprintf('                        SUMMARY                           \n');
fprintf('===========================================================\n');
fprintf('Total files processed: %d\n', total_files);
fprintf('Successful exports: %d\n', successful_exports);
fprintf('Failed exports: %d\n', failed_exports);

if failed_exports > 0
    fprintf('\nFailed files:\n');
    for i = 1:length(failed_files)
        fprintf('  - %s\n', failed_files{i});
    end
end

fprintf('\n===========================================================\n');
fprintf('                     NEXT STEPS                           \n');
fprintf('===========================================================\n');
fprintf('1. Copy the folder "%s" to:\n', output_dir);
fprintf('   trunk/Python Simulations/Vector_Fields/VF_Robot/real_robot_data/\n');
fprintf('2. Run the Python sim2real_comparison_all.py experiment\n');
fprintf('\n');

% Create a summary file with configuration details
summary_file = fullfile(output_dir, 'trajectory_summary.txt');
fid = fopen(summary_file, 'w');
fprintf(fid, 'Orbit Trajectory Export Summary\n');
fprintf(fid, '================================\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now));
fprintf(fid, 'Configuration Key:\n');
fprintf(fid, '0XX series: 3-robot equilateral formation\n');
fprintf(fid, '1XX series: 4-robot square formation (d1=0.3, d2=0.3)\n');
fprintf(fid, '2XX series: 4-robot advanced formation (d1=0.433, d2=0.25)\n\n');
fprintf(fid, 'Files exported: %d\n', successful_exports);
fprintf(fid, 'Files failed: %d\n', failed_exports);
fclose(fid);

fprintf('Summary written to: %s\n', summary_file);