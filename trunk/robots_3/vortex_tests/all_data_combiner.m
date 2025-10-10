% Combine All Vortex Data into Single CSV
% This script reads all run files and orbit files
% and combines them into a single CSV file

clear; clc;

fprintf('=== Starting Vortex Data Combination ===\n');
fprintf('This script will combine data from run files and orbit files\n\n');

%% Define file lists
% Run files (run1.mat to run80.mat and run100.mat to run180.mat)
run_files = cell(1, 161);
idx = 1;

% Add run1 to run80
for i = 1:80
    run_files{idx} = sprintf('run%d.mat', i);
    idx = idx + 1;
end

% Add run100 to run180
for i = 100:180
    run_files{idx} = sprintf('run%d.mat', i);
    idx = idx + 1;
end

% Orbit files
orbit_files = {};

% Add orbit001, orbit010, orbit020, ..., orbit070
orbit_files{end+1} = 'orbit001.mat';
for i = 10:10:70
    orbit_files{end+1} = sprintf('orbit%03d.mat', i);
end

% Add orbit101, orbit110, orbit120, ..., orbit170
orbit_files{end+1} = 'orbit101.mat';
for i = 110:10:170
    orbit_files{end+1} = sprintf('orbit%d.mat', i);
end

%% Initialize data storage
all_data = [];
successful_loads = 0;
failed_files = {};

%% Process run files
fprintf('Processing %d run files...\n', length(run_files));
fprintf('----------------------------------------\n');

for file_idx = 1:length(run_files)
    current_file = run_files{file_idx};
    
    if exist(current_file, 'file')
        try
            % Load the file
            data = load(current_file);
            
            % Check if all required fields exist
            has_robot1 = isfield(data, 'robot1_pose');
            has_robot2 = isfield(data, 'robot2_pose');
            has_robot3 = isfield(data, 'robot3_pose');
            has_hue = isfield(data, 'hue');
            has_saturation = isfield(data, 'saturation');
            
            if has_robot1 && has_robot2 && has_robot3
                % Get the pose data
                robot1_data = data.robot1_pose;
                robot2_data = data.robot2_pose;
                robot3_data = data.robot3_pose;
                
                % Get hue and saturation data if they exist
                hue_data = [];
                sat_data = [];
                
                if has_hue
                    if isa(data.hue, 'timeseries')
                        hue_data = data.hue.Data;  % This should be Nx3 matrix
                    else
                        hue_data = data.hue;
                    end
                end
                
                if has_saturation
                    if isa(data.saturation, 'timeseries')
                        sat_data = data.saturation.Data;  % This should be Nx3 matrix
                    else
                        sat_data = data.saturation;
                    end
                end
                
                % Determine the number of time points
                % Use minimum length to ensure all data aligns
                num_points = min([size(robot1_data, 1), ...
                                 size(robot2_data, 1), ...
                                 size(robot3_data, 1)]);
                
                % Also check hue and saturation lengths if they exist
                if ~isempty(hue_data)
                    num_points = min(num_points, size(hue_data, 1));
                end
                if ~isempty(sat_data)
                    num_points = min(num_points, size(sat_data, 1));
                end
                
                % Process each time point
                for t = 1:num_points
                    % Create row for this time point
                    % Format: [x1, y1, theta1, hue1, sat1, x2, y2, theta2, hue2, sat2, x3, y3, theta3, hue3, sat3]
                    
                    row = zeros(1, 15);
                    
                    % Robot 1 data
                    row(1) = robot1_data(t, 1);     % x1
                    row(2) = robot1_data(t, 2);     % y1
                    if size(robot1_data, 2) >= 3
                        row(3) = robot1_data(t, 3); % theta1
                    end
                    
                    % Hue for robot 1 (column 1 of hue data)
                    if ~isempty(hue_data) && size(hue_data, 2) >= 1
                        row(4) = hue_data(t, 1);    % hue1
                    end
                    
                    % Saturation for robot 1 (column 1 of saturation data)
                    if ~isempty(sat_data) && size(sat_data, 2) >= 1
                        row(5) = sat_data(t, 1);    % sat1
                    end
                    
                    % Robot 2 data
                    row(6) = robot2_data(t, 1);     % x2
                    row(7) = robot2_data(t, 2);     % y2
                    if size(robot2_data, 2) >= 3
                        row(8) = robot2_data(t, 3); % theta2
                    end
                    
                    % Hue for robot 2 (column 2 of hue data)
                    if ~isempty(hue_data) && size(hue_data, 2) >= 2
                        row(9) = hue_data(t, 2);    % hue2
                    end
                    
                    % Saturation for robot 2 (column 2 of saturation data)
                    if ~isempty(sat_data) && size(sat_data, 2) >= 2
                        row(10) = sat_data(t, 2);   % sat2
                    end
                    
                    % Robot 3 data
                    row(11) = robot3_data(t, 1);    % x3
                    row(12) = robot3_data(t, 2);    % y3
                    if size(robot3_data, 2) >= 3
                        row(13) = robot3_data(t, 3); % theta3
                    end
                    
                    % Hue for robot 3 (column 3 of hue data)
                    if ~isempty(hue_data) && size(hue_data, 2) >= 3
                        row(14) = hue_data(t, 3);   % hue3
                    end
                    
                    % Saturation for robot 3 (column 3 of saturation data)
                    if ~isempty(sat_data) && size(sat_data, 2) >= 3
                        row(15) = sat_data(t, 3);   % sat3
                    end
                    
                    % Append to all_data
                    all_data = [all_data; row];
                end
                
                successful_loads = successful_loads + 1;
                fprintf('  ✓ Loaded %s: %d time points added\n', current_file, num_points);
            else
                fprintf('  ⚠ Warning: %s missing required robot pose fields\n', current_file);
                failed_files{end+1} = sprintf('%s (missing robot poses)', current_file);
            end
            
        catch ME
            fprintf('  ✗ Error loading %s: %s\n', current_file, ME.message);
            failed_files{end+1} = sprintf('%s (error: %s)', current_file, ME.message);
        end
    else
        fprintf('  ✗ File not found: %s\n', current_file);
        failed_files{end+1} = sprintf('%s (not found)', current_file);
    end
end

%% Process orbit files
fprintf('\n----------------------------------------\n');
fprintf('Processing %d orbit files...\n', length(orbit_files));
fprintf('----------------------------------------\n');

for file_idx = 1:length(orbit_files)
    current_file = orbit_files{file_idx};
    
    if exist(current_file, 'file')
        try
            % Load the file
            data = load(current_file);
            
            % Check if robot poses exist
            has_robot1 = isfield(data, 'robot1_pose');
            has_robot2 = isfield(data, 'robot2_pose');
            has_robot3 = isfield(data, 'robot3_pose');
            has_hue = isfield(data, 'hue');
            has_saturation = isfield(data, 'saturation');
            
            if has_robot1 && has_robot2 && has_robot3
                % Get robot data
                robot1_data = data.robot1_pose;
                robot2_data = data.robot2_pose;
                robot3_data = data.robot3_pose;
                
                % Get hue and saturation data if they exist
                hue_data = [];
                sat_data = [];
                
                if has_hue
                    if isa(data.hue, 'timeseries')
                        hue_data = data.hue.Data;  % This should be Nx3 matrix
                    else
                        hue_data = data.hue;
                    end
                end
                
                if has_saturation
                    if isa(data.saturation, 'timeseries')
                        sat_data = data.saturation.Data;  % This should be Nx3 matrix
                    else
                        sat_data = data.saturation;
                    end
                end
                
                % Determine the number of time points
                num_points = min([size(robot1_data, 1), ...
                                 size(robot2_data, 1), ...
                                 size(robot3_data, 1)]);
                
                % Also check hue and saturation lengths if they exist
                if ~isempty(hue_data)
                    num_points = min(num_points, size(hue_data, 1));
                end
                if ~isempty(sat_data)
                    num_points = min(num_points, size(sat_data, 1));
                end
                
                % Process each time point
                for t = 1:num_points
                    row = zeros(1, 15);
                    
                    % Robot 1 data
                    row(1) = robot1_data(t, 1);     % x1
                    row(2) = robot1_data(t, 2);     % y1
                    if size(robot1_data, 2) >= 3
                        row(3) = robot1_data(t, 3); % theta1
                    end
                    
                    % Hue for robot 1 (column 1 of hue data)
                    if ~isempty(hue_data) && size(hue_data, 2) >= 1
                        row(4) = hue_data(t, 1);    % hue1
                    end
                    
                    % Saturation for robot 1 (column 1 of saturation data)
                    if ~isempty(sat_data) && size(sat_data, 2) >= 1
                        row(5) = sat_data(t, 1);    % sat1
                    end
                    
                    % Robot 2 data
                    row(6) = robot2_data(t, 1);     % x2
                    row(7) = robot2_data(t, 2);     % y2
                    if size(robot2_data, 2) >= 3
                        row(8) = robot2_data(t, 3); % theta2
                    end
                    
                    % Hue for robot 2 (column 2 of hue data)
                    if ~isempty(hue_data) && size(hue_data, 2) >= 2
                        row(9) = hue_data(t, 2);    % hue2
                    end
                    
                    % Saturation for robot 2 (column 2 of saturation data)
                    if ~isempty(sat_data) && size(sat_data, 2) >= 2
                        row(10) = sat_data(t, 2);   % sat2
                    end
                    
                    % Robot 3 data
                    row(11) = robot3_data(t, 1);    % x3
                    row(12) = robot3_data(t, 2);    % y3
                    if size(robot3_data, 2) >= 3
                        row(13) = robot3_data(t, 3); % theta3
                    end
                    
                    % Hue for robot 3 (column 3 of hue data)
                    if ~isempty(hue_data) && size(hue_data, 2) >= 3
                        row(14) = hue_data(t, 3);   % hue3
                    end
                    
                    % Saturation for robot 3 (column 3 of saturation data)
                    if ~isempty(sat_data) && size(sat_data, 2) >= 3
                        row(15) = sat_data(t, 3);   % sat3
                    end
                    
                    % Append to all_data
                    all_data = [all_data; row];
                end
                
                successful_loads = successful_loads + 1;
                fprintf('  ✓ Loaded %s: %d time points added\n', current_file, num_points);
            else
                % Try to use cluster_pose if robot poses are missing
                if isfield(data, 'cluster_pose')
                    fprintf('  ⚠ Warning: %s has cluster_pose but missing robot poses\n', current_file);
                else
                    fprintf('  ⚠ Warning: %s missing required data\n', current_file);
                end
                failed_files{end+1} = sprintf('%s (missing robot data)', current_file);
            end
            
        catch ME
            fprintf('  ✗ Error loading %s: %s\n', current_file, ME.message);
            failed_files{end+1} = sprintf('%s (error: %s)', current_file, ME.message);
        end
    else
        fprintf('  ✗ File not found: %s\n', current_file);
        failed_files{end+1} = sprintf('%s (not found)', current_file);
    end
end

%% Write to CSV file
fprintf('\n----------------------------------------\n');
fprintf('Writing data to CSV file...\n');
fprintf('----------------------------------------\n');

if ~isempty(all_data)
    % Define output filename
    output_filename = 'all_vortex_data_complete.csv';
    
    % Create header
    header = {'x1', 'y1', 'theta1', 'hue1', 'sat1', ...
              'x2', 'y2', 'theta2', 'hue2', 'sat2', ...
              'x3', 'y3', 'theta3', 'hue3', 'sat3'};
    
    % Write to CSV
    fid = fopen(output_filename, 'w');
    
    % Write header
    fprintf(fid, '%s', header{1});
    for i = 2:length(header)
        fprintf(fid, ',%s', header{i});
    end
    fprintf(fid, '\n');
    
    % Write data
    for row = 1:size(all_data, 1)
        fprintf(fid, '%.6f', all_data(row, 1));
        for col = 2:size(all_data, 2)
            fprintf(fid, ',%.6f', all_data(row, col));
        end
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    
    fprintf('  ✓ Successfully wrote %d rows to %s\n', size(all_data, 1), output_filename);
else
    fprintf('  ✗ No data to write!\n');
end

%% Summary Report
fprintf('\n========================================\n');
fprintf('           SUMMARY REPORT\n');
fprintf('========================================\n');
fprintf('Total files to process:\n');
fprintf('  - Run files: %d\n', length(run_files));
fprintf('  - Orbit files: %d\n', length(orbit_files));
fprintf('  - Total: %d\n', length(run_files) + length(orbit_files));
fprintf('\nSuccessfully loaded: %d\n', successful_loads);
fprintf('Failed to load: %d\n', length(failed_files));

if ~isempty(failed_files)
    fprintf('\nFailed files:\n');
    for i = 1:length(failed_files)
        fprintf('  - %s\n', failed_files{i});
    end
end

if ~isempty(all_data)
    fprintf('\nData Statistics:\n');
    fprintf('  Total rows (time points): %d\n', size(all_data, 1));
    fprintf('  Total columns: %d\n', size(all_data, 2));
    fprintf('  Output file: %s\n', output_filename);
    fprintf('  File size: %.2f MB\n', numel(all_data) * 8 / (1024^2));
    
    % Show data range for each column
    fprintf('\nData ranges:\n');
    for col = 1:15
        col_data = all_data(:, col);
        non_zero = col_data(col_data ~= 0);
        if ~isempty(non_zero)
            fprintf('  %s: [%.4f, %.4f]\n', header{col}, min(non_zero), max(non_zero));
        else
            fprintf('  %s: all zeros\n', header{col});
        end
    end
end

fprintf('\n========================================\n');
fprintf('        Processing Complete!\n');
fprintf('========================================\n');