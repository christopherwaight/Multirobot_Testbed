% Vortex Vector Field Analyzer and Visualizer
% This script reads the combined CSV file, reshapes the data,
% rescales hue values, and creates vector field visualizations

clear; clc; close all;

fprintf('=== Vortex Vector Field Analysis ===\n');
fprintf('Loading and processing vortex data...\n\n');

%% Load the CSV file
csv_filename = 'all_vortex_data_possible.csv';

if ~exist(csv_filename, 'file')
    error('File not found: %s\nPlease ensure the CSV file exists in the current directory.', csv_filename);
end

fprintf('Loading data from %s...\n', csv_filename);

% Read the CSV file
data_table = readtable(csv_filename);
data_matrix = table2array(data_table);

fprintf('  ✓ Loaded %d rows x %d columns\n', size(data_matrix, 1), size(data_matrix, 2));

%% Reshape the data from 15xn to 5x3n
% Original format: [x1,y1,theta1,hue1,sat1, x2,y2,theta2,hue2,sat2, x3,y3,theta3,hue3,sat3]
% New format: Each robot's data stacked vertically

n_timepoints = size(data_matrix, 1);
n_robots = 3;

% Initialize reshaped data matrix (5 columns x 3n rows)
% Columns: [x, y, theta, hue, saturation]
reshaped_data = zeros(n_timepoints * n_robots, 5);

fprintf('\nReshaping data from %dx%d to %dx%d...\n', ...
    size(data_matrix, 1), size(data_matrix, 2), ...
    n_timepoints * n_robots, 5);

% Extract and stack data for each robot
for t = 1:n_timepoints
    % Robot 1 data
    reshaped_data((t-1)*3 + 1, :) = [
        data_matrix(t, 1),  % x1
        data_matrix(t, 2),  % y1
        data_matrix(t, 3),  % theta1
        data_matrix(t, 4),  % hue1
        data_matrix(t, 5)   % sat1
    ];
    
    % Robot 2 data
    reshaped_data((t-1)*3 + 2, :) = [
        data_matrix(t, 6),  % x2
        data_matrix(t, 7),  % y2
        data_matrix(t, 8),  % theta2
        data_matrix(t, 9),  % hue2
        data_matrix(t, 10)  % sat2
    ];
    
    % Robot 3 data
    reshaped_data((t-1)*3 + 3, :) = [
        data_matrix(t, 11), % x3
        data_matrix(t, 12), % y3
        data_matrix(t, 13), % theta3
        data_matrix(t, 14), % hue3
        data_matrix(t, 15)  % sat3
    ];
end

fprintf('  ✓ Data reshaped successfully\n');

%% Rescale hue from [0,1] to [0,2π]
fprintf('\nRescaling hue values from [0,1] to [0,2π]...\n');

% Original hue range
original_hue = reshaped_data(:, 4);
hue_min = min(original_hue(original_hue > 0));  % Ignore zeros
hue_max = max(original_hue);

fprintf('  Original hue range: [%.4f, %.4f]\n', hue_min, hue_max);

% Rescale to [0, 2π]
reshaped_data(:, 4) = reshaped_data(:, 4) * 2 * pi;

fprintf('  ✓ Hue rescaled to [0, 2π]\n');

%% Prepare data for visualization
% Extract non-zero data points (to avoid plotting at origin when no data)
valid_idx = (reshaped_data(:, 1) ~= 0) | (reshaped_data(:, 2) ~= 0);
valid_data = reshaped_data(valid_idx, :);

x = valid_data(:, 1);
y = valid_data(:, 2);
theta = valid_data(:, 3);
hue = valid_data(:, 4);
saturation = valid_data(:, 5);

% Calculate vector components from hue (now in radians)
u = cos(hue);
v = sin(hue);

% Alternative: use theta for direction if available
u_theta = cos(theta);
v_theta = sin(theta);

fprintf('\nData statistics:\n');
fprintf('  Valid data points: %d\n', sum(valid_idx));
fprintf('  X range: [%.4f, %.4f]\n', min(x), max(x));
fprintf('  Y range: [%.4f, %.4f]\n', min(y), max(y));
fprintf('  Theta range: [%.4f, %.4f]\n', min(theta), max(theta));
fprintf('  Rescaled hue range: [%.4f, %.4f]\n', min(hue), max(hue));
fprintf('  Saturation range: [%.4f, %.4f]\n', min(saturation), max(saturation));

%% Calculate Expected Vortex Direction and Find Anomalies
fprintf('\nAnalyzing vector field for anomalies...\n');

% For each point, calculate expected vortex direction (clockwise)
% In a vortex, the expected direction is perpendicular to radius vector
radius_angle = atan2(y, x);
expected_angle = radius_angle - pi/2;  % Perpendicular, clockwise

% Normalize angles to [0, 2π]
expected_angle = mod(expected_angle, 2*pi);
actual_angle = mod(hue, 2*pi);

% Calculate angular difference
angle_diff = abs(actual_angle - expected_angle);
angle_diff = min(angle_diff, 2*pi - angle_diff);  % Handle wrap-around

% Identify anomalies (vectors pointing in "wrong" direction)
anomaly_threshold = pi/3;  % 60 degrees off expected
is_anomaly = angle_diff > anomaly_threshold;

fprintf('  Found %d anomalous vectors (%.1f%%)\n', sum(is_anomaly), 100*sum(is_anomaly)/length(is_anomaly));

%% Create Grid-based Averaged Vector Field for Smoother Visualization
fprintf('\nCreating smoothed vector field...\n');

% Create a regular grid
grid_resolution = 40;  % Increase for smoother field
x_edges = linspace(-1, 1, grid_resolution);
y_edges = linspace(-1, 1, grid_resolution);
[X_grid, Y_grid] = meshgrid(x_edges(1:end-1) + diff(x_edges)/2, ...
                            y_edges(1:end-1) + diff(y_edges)/2);

% Bin data and average within each grid cell
U_grid = zeros(size(X_grid));
V_grid = zeros(size(X_grid));
Count_grid = zeros(size(X_grid));
Anomaly_grid = zeros(size(X_grid));

for i = 1:length(x)
    % Find which grid cell this point belongs to
    [~, xi] = min(abs(x_edges - x(i)));
    [~, yi] = min(abs(y_edges - y(i)));
    
    % Ensure we're within bounds
    xi = min(xi, grid_resolution-1);
    yi = min(yi, grid_resolution-1);
    
    % Add to grid (weighted by saturation)
    weight = saturation(i);
    U_grid(yi, xi) = U_grid(yi, xi) + u(i) * weight;
    V_grid(yi, xi) = V_grid(yi, xi) + v(i) * weight;
    Count_grid(yi, xi) = Count_grid(yi, xi) + weight;
    
    if is_anomaly(i)
        Anomaly_grid(yi, xi) = Anomaly_grid(yi, xi) + 1;
    end
end

% Average the grid values
valid_cells = Count_grid > 0;
U_grid(valid_cells) = U_grid(valid_cells) ./ Count_grid(valid_cells);
V_grid(valid_cells) = V_grid(valid_cells) ./ Count_grid(valid_cells);
Anomaly_grid(valid_cells) = Anomaly_grid(valid_cells) ./ Count_grid(valid_cells);

% Apply Gaussian smoothing for even smoother field
sigma = 1.5;
h = fspecial('gaussian', [5 5], sigma);
U_smooth = imfilter(U_grid, h, 'replicate');
V_smooth = imfilter(V_grid, h, 'replicate');

% Normalize vectors for consistent visualization
magnitude = sqrt(U_smooth.^2 + V_smooth.^2);
max_mag = max(magnitude(:));
if max_mag > 0
    U_smooth = U_smooth ./ max_mag;
    V_smooth = V_smooth ./ max_mag;
end

%% Figure 1: Main Vector Field Analysis
figure('Position', [100, 100, 1200, 900]);

% Subplot 1: Smoothed Vector Field with Streamlines
subplot(2, 2, 1);
hold on;
title('Smoothed Vector Field with Streamlines');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Plot smoothed vector field
scale_factor = 0.8;  % Longer arrows
skip = 2;  % Skip every other point for clarity
quiver(X_grid(1:skip:end, 1:skip:end), Y_grid(1:skip:end, 1:skip:end), ...
       U_smooth(1:skip:end, 1:skip:end)*scale_factor, ...
       V_smooth(1:skip:end, 1:skip:end)*scale_factor, ...
       'AutoScale', 'off', 'Color', [0 0.4 0.8], 'LineWidth', 1.5);

% Add streamlines for flow visualization
startx = [-0.8:0.2:0.8, zeros(1,9)];
starty = [zeros(1,9), -0.8:0.2:0.8];
try
    h_stream = streamline(X_grid, Y_grid, U_smooth, V_smooth, startx, starty);
    set(h_stream, 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
catch
    fprintf('Warning: Could not create all streamlines\n');
end

axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 2: Anomaly Detection Visualization
subplot(2, 2, 2);
hold on;
title('Vector Field Anomalies');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Plot background vector field in gray
subsample = 50;
idx_sub = 1:subsample:length(x);
quiver(x(idx_sub), y(idx_sub), u(idx_sub)*0.05, v(idx_sub)*0.05, ...
       'AutoScale', 'off', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);

% Highlight anomalous vectors in red
anomaly_idx = find(is_anomaly);
anomaly_sub = anomaly_idx(1:min(10, length(anomaly_idx)):end);  % Subsample anomalies
if ~isempty(anomaly_sub)
    quiver(x(anomaly_sub), y(anomaly_sub), ...
           u(anomaly_sub)*0.1, v(anomaly_sub)*0.1, ...
           'AutoScale', 'off', 'Color', 'r', 'LineWidth', 2);
end

% Add expected vortex direction for reference
[X_ref, Y_ref] = meshgrid(linspace(-0.8, 0.8, 10));
R_ref = sqrt(X_ref.^2 + Y_ref.^2);
U_ref = Y_ref ./ (R_ref + 0.1) * 0.3;  % Clockwise vortex
V_ref = -X_ref ./ (R_ref + 0.1) * 0.3;
quiver(X_ref, Y_ref, U_ref, V_ref, 'AutoScale', 'off', 'Color', [0 0.5 0], ...
       'LineWidth', 1, 'LineStyle', '--');

legend('Normal', 'Anomalous', 'Expected Vortex', 'Location', 'best');
axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 3: Continuous Flow Lines
subplot(2, 2, 3);
hold on;
title('Continuous Flow Visualization');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Create denser streamlines for continuous appearance
[startx_dense, starty_dense] = meshgrid(-0.9:0.15:0.9, -0.9:0.15:0.9);
% Remove points too close to origin
dist_from_origin = sqrt(startx_dense(:).^2 + starty_dense(:).^2);
valid_starts = dist_from_origin > 0.1;
startx_dense = startx_dense(valid_starts);
starty_dense = starty_dense(valid_starts);

try
    h_dense = streamline(X_grid, Y_grid, U_smooth, V_smooth, ...
                        startx_dense, starty_dense);
    
    % Color streamlines by local magnitude
    for i = 1:length(h_dense)
        if ishandle(h_dense(i))
            xdata = get(h_dense(i), 'XData');
            ydata = get(h_dense(i), 'YData');
            if length(xdata) > 1
                % Calculate local speed along streamline
                speed = sqrt(diff(xdata).^2 + diff(ydata).^2);
                avg_speed = mean(speed);
                color_val = min(avg_speed * 10, 1);
                set(h_dense(i), 'Color', [color_val, 0, 1-color_val], 'LineWidth', 1.2);
            end
        end
    end
catch
    fprintf('Warning: Could not create dense streamlines\n');
end

% Add a few seed points as markers
plot(startx_dense(1:5:end), starty_dense(1:5:end), 'k.', 'MarkerSize', 6);

axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 4: Comparison with Theoretical Vortex
subplot(2, 2, 4);
hold on;
title('Comparison with Theoretical Vortex');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Theoretical vortex field (corrected rotation direction - clockwise)
[X_theory, Y_theory] = meshgrid(linspace(-1, 1, 20), linspace(-1, 1, 20));
r_theory = sqrt(X_theory.^2 + Y_theory.^2);
% Clockwise rotation: v_theta = -r, which gives:
U_theory = Y_theory ./ (r_theory + 0.1) * 0.4;   % y/r normalized
V_theory = -X_theory ./ (r_theory + 0.1) * 0.4;  % -x/r normalized

% Plot theoretical field in gray
quiver(X_theory, Y_theory, U_theory, V_theory, 'AutoScale', 'off', ...
       'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'DisplayName', 'Theoretical');

% Overlay smoothed actual data
skip2 = 3;
quiver(X_grid(1:skip2:end, 1:skip2:end), Y_grid(1:skip2:end, 1:skip2:end), ...
       U_smooth(1:skip2:end, 1:skip2:end)*0.4, ...
       V_smooth(1:skip2:end, 1:skip2:end)*0.4, ...
       'AutoScale', 'off', 'Color', 'b', 'LineWidth', 1.5, 'DisplayName', 'Actual');

legend('Location', 'best');
axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

sgtitle('Vector Field Analysis - Smoothed and Continuous');
saveas(gcf, 'vector_field_analysis_smooth.png');
saveas(gcf, 'vector_field_analysis_smooth.fig');

%% Figure 2: Advanced Analysis
figure('Position', [1350, 100, 1200, 900]);

% Subplot 1: Vector Magnitude Heatmap
subplot(2, 2, 1);
hold on;
title('Vector Field Magnitude');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Calculate magnitude grid
Magnitude_grid = sqrt(U_smooth.^2 + V_smooth.^2);
imagesc(x_edges(1:end-1), y_edges(1:end-1), Magnitude_grid);
axis xy;
colormap(jet);
colorbar;
c = colorbar;
c.Label.String = 'Normalized Magnitude';

% Overlay vectors
quiver(X_grid(1:skip:end, 1:skip:end), Y_grid(1:skip:end, 1:skip:end), ...
       U_smooth(1:skip:end, 1:skip:end)*0.3, ...
       V_smooth(1:skip:end, 1:skip:end)*0.3, ...
       'AutoScale', 'off', 'Color', 'w', 'LineWidth', 1);

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 2: Vorticity Analysis
subplot(2, 2, 2);
hold on;
title('Vorticity (Curl of Vector Field)');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Calculate vorticity (curl)
[~, vorticity] = curl(X_grid, Y_grid, U_smooth, V_smooth);
imagesc(x_edges(1:end-1), y_edges(1:end-1), vorticity);
axis xy;
colormap(redblue);
colorbar;
c = colorbar;
c.Label.String = 'Vorticity';
caxis([-max(abs(vorticity(:))), max(abs(vorticity(:)))]);

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 3: Divergence Analysis
subplot(2, 2, 3);
hold on;
title('Divergence of Vector Field');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Calculate divergence
divergence_field = divergence(X_grid, Y_grid, U_smooth, V_smooth);
imagesc(x_edges(1:end-1), y_edges(1:end-1), divergence_field);
axis xy;
colormap(cool);
colorbar;
c = colorbar;
c.Label.String = 'Divergence';

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 4: Line Integral Convolution (LIC) style visualization
subplot(2, 2, 4);
hold on;
title('Flow Texture Visualization');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Create a texture-like visualization using dense streamlines
% This mimics LIC (Line Integral Convolution)
[startx_lic, starty_lic] = meshgrid(-1:0.05:1, -1:0.05:1);
dist_lic = sqrt(startx_lic(:).^2 + starty_lic(:).^2);
valid_lic = dist_lic < 0.95;  % Keep within circle
startx_lic = startx_lic(valid_lic);
starty_lic = starty_lic(valid_lic);

% Random subset for performance
n_lines = min(500, length(startx_lic));
rand_idx = randperm(length(startx_lic), n_lines);
startx_lic = startx_lic(rand_idx);
starty_lic = starty_lic(rand_idx);

try
    h_lic = streamline(X_grid, Y_grid, U_smooth, V_smooth, ...
                      startx_lic, starty_lic, [0.1, 500]);
    
    for i = 1:length(h_lic)
        if ishandle(h_lic(i))
            set(h_lic(i), 'Color', [0.2 0.2 0.8], 'LineWidth', 0.5, ...
                'Color', [0.2 0.2 0.8 0.3]);  % Semi-transparent
        end
    end
catch
    fprintf('Warning: Could not create LIC visualization\n');
end

axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

sgtitle('Advanced Vector Field Analysis');
saveas(gcf, 'vector_field_advanced.png');
saveas(gcf, 'vector_field_advanced.fig');

%% Export processed data
fprintf('\nExporting processed data...\n');

% Save reshaped and processed data with anomaly flags
output_filename = 'processed_vortex_data_with_analysis.csv';

% Add anomaly column to data
export_data = [reshaped_data(valid_idx, :), is_anomaly];

% Create header
header = {'x', 'y', 'theta', 'hue_rad', 'saturation', 'is_anomaly'};

% Write to CSV
fid = fopen(output_filename, 'w');
fprintf(fid, '%s,%s,%s,%s,%s,%s\n', header{:});

for row = 1:size(export_data, 1)
    fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%d\n', export_data(row, :));
end

fclose(fid);
fprintf('  ✓ Processed data saved to %s\n', output_filename);

%% Summary
fprintf('\n========================================\n');
fprintf('           ANALYSIS COMPLETE\n');
fprintf('========================================\n');
fprintf('Input file: %s\n', csv_filename);
fprintf('Output files:\n');
fprintf('  - vector_field_analysis_smooth.png\n');
fprintf('  - vector_field_advanced.png\n');
fprintf('  - processed_vortex_data_with_analysis.csv\n');
fprintf('\nData summary:\n');
fprintf('  Original: %d timepoints x %d robots\n', n_timepoints, n_robots);
fprintf('  Reshaped: %d rows x 5 columns\n', size(reshaped_data, 1));
fprintf('  Valid points: %d (%.1f%%)\n', sum(valid_idx), 100*sum(valid_idx)/length(valid_idx));
fprintf('  Anomalous vectors: %d (%.1f%%)\n', sum(is_anomaly), 100*sum(is_anomaly)/length(is_anomaly));
fprintf('  Grid resolution: %dx%d\n', grid_resolution-1, grid_resolution-1);
fprintf('========================================\n');

%% Helper function for red-blue colormap
function cmap = redblue(n)
    if nargin < 1
        n = 256;
    end
    red = [1 0 0];
    white = [1 1 1];
    blue = [0 0 1];
    
    cmap = [linspace(blue(1), white(1), n/2)', linspace(blue(2), white(2), n/2)', linspace(blue(3), white(3), n/2)';
            linspace(white(1), red(1), n/2)', linspace(white(2), red(2), n/2)', linspace(white(3), red(3), n/2)'];
end