% Export orbit040.mat Trajectory to CSV
% This script exports the cluster trajectory from orbit040.mat to CSV format
% for use in Python sim-to-real comparison experiments

fprintf('=== Exporting orbit040.mat to CSV ===\n\n');

% Load the data file
data_file = 'orbit040.mat';

if ~exist(data_file, 'file')
    error('File not found: %s\nPlease ensure orbit040.mat exists in the current directory.', data_file);
end

fprintf('Loading data from %s...\n', data_file);
data = load(data_file);

% Extract cluster position data
if isfield(data, 'cluster_position')
    % Use existing cluster_position field
    positions = data.cluster_position(:, 1:2);  % Extract x, y (ignore theta)
    fprintf('  Using cluster_position field from file\n');
elseif isfield(data, 'robot1_pose') && isfield(data, 'robot2_pose') && isfield(data, 'robot3_pose')
    % Calculate cluster center from 3 robot positions
    fprintf('  Calculating cluster center from robot positions...\n');
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
else
    error('Could not find cluster position data in %s', data_file);
end

% Note: Robot was already operating at correct physical scale (0.4m radius)
% No scaling needed - positions are already in actual meters
x_scaled = positions(:, 1);
y_scaled = positions(:, 2);

fprintf('  Number of data points: %d\n', length(x_scaled));
fprintf('  Positions exported without scaling (already in meters)\n');

% Generate time vector (assuming 0.1s timestep)
timestep = 0.1;  % seconds
time = (0:length(x_scaled)-1)' * timestep;

fprintf('  Time range: %.1f to %.1f seconds\n', time(1), time(end));
fprintf('  X range (scaled): %.4f to %.4f m\n', min(x_scaled), max(x_scaled));
fprintf('  Y range (scaled): %.4f to %.4f m\n', min(y_scaled), max(y_scaled));

% Calculate statistics
radius = sqrt(x_scaled.^2 + y_scaled.^2);
fprintf('\n  Radius statistics:\n');
fprintf('    Mean: %.4f m\n', mean(radius));
fprintf('    Std:  %.4f m\n', std(radius));
fprintf('    Min:  %.4f m\n', min(radius));
fprintf('    Max:  %.4f m\n', max(radius));

% Prepare export data matrix [time, x, y]
export_data = [time, x_scaled, y_scaled];

% Export to CSV
output_file = 'orbit040_trajectory.csv';
fprintf('\nExporting to %s...\n', output_file);

% Write CSV with header
fid = fopen(output_file, 'w');
fprintf(fid, 'time_s,x_m,y_m\n');  % Header row
fprintf(fid, '%.6f,%.6f,%.6f\n', export_data');  % Data rows
fclose(fid);

fprintf('  ✓ Export complete!\n');
fprintf('  File saved: %s\n', output_file);
fprintf('  Size: %d rows × 3 columns\n', size(export_data, 1));

fprintf('\n=== Next Steps ===\n');
fprintf('1. Copy %s to:\n', output_file);
fprintf('   trunk/Python Simulations/Vector_Fields/VF_Robot/real_robot_data/\n');
fprintf('2. Run the Python sim2real_comparison.py experiment\n');
fprintf('\n');
