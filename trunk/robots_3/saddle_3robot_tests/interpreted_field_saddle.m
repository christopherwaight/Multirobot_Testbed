% Vector Field Analyzer and Visualizer
% This script reads the combined CSV file, reshapes the data,
% rescales hue values, and creates vector field visualizations

clear; clc; close all;

fprintf('=== Vector Field Analysis ===\n');
fprintf('Loading and processing vector field data...\n\n');

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

% Keep original hue for visualization
hue_original = reshaped_data(:, 4);

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
hue_norm = hue_original(valid_idx);  % Keep original 0-1 scale for visualization

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

%% Create Grid-based RAW Vector Field (No Smoothing)
fprintf('\nCreating raw vector field grid...\n');

% Create a regular grid
grid_resolution = 40;  % Grid resolution
x_edges = linspace(-1, 1, grid_resolution);
y_edges = linspace(-1, 1, grid_resolution);
[X_grid, Y_grid] = meshgrid(x_edges(1:end-1) + diff(x_edges)/2, ...
                            y_edges(1:end-1) + diff(y_edges)/2);

% Bin data and average within each grid cell - RAW
U_grid_raw = zeros(size(X_grid));
V_grid_raw = zeros(size(X_grid));
Sat_grid_raw = zeros(size(X_grid));
Hue_grid_raw = zeros(size(X_grid));
Count_grid = zeros(size(X_grid));

for i = 1:length(x)
    % Find which grid cell this point belongs to
    [~, xi] = min(abs(x_edges - x(i)));
    [~, yi] = min(abs(y_edges - y(i)));
    
    % Ensure we're within bounds
    xi = min(xi, grid_resolution-1);
    yi = min(yi, grid_resolution-1);
    
    % Add to grid (weighted by saturation for vectors)
    weight = saturation(i);
    U_grid_raw(yi, xi) = U_grid_raw(yi, xi) + u(i) * weight;
    V_grid_raw(yi, xi) = V_grid_raw(yi, xi) + v(i) * weight;
    Sat_grid_raw(yi, xi) = Sat_grid_raw(yi, xi) + saturation(i);
    Hue_grid_raw(yi, xi) = Hue_grid_raw(yi, xi) + hue_norm(i);
    Count_grid(yi, xi) = Count_grid(yi, xi) + 1;
end

% Average the grid values
valid_cells = Count_grid > 0;
U_grid_raw(valid_cells) = U_grid_raw(valid_cells) ./ (Count_grid(valid_cells) .* Sat_grid_raw(valid_cells) ./ Count_grid(valid_cells));
V_grid_raw(valid_cells) = V_grid_raw(valid_cells) ./ (Count_grid(valid_cells) .* Sat_grid_raw(valid_cells) ./ Count_grid(valid_cells));
Sat_grid_raw(valid_cells) = Sat_grid_raw(valid_cells) ./ Count_grid(valid_cells);
Hue_grid_raw(valid_cells) = Hue_grid_raw(valid_cells) ./ Count_grid(valid_cells);

% Set empty cells to NaN for better visualization
U_grid_raw(~valid_cells) = NaN;
V_grid_raw(~valid_cells) = NaN;
Sat_grid_raw(~valid_cells) = NaN;
Hue_grid_raw(~valid_cells) = NaN;

%% Create Smoothed Vector Field
fprintf('Creating smoothed vector field...\n');

% Copy raw grids for smoothing
U_grid = U_grid_raw;
V_grid = V_grid_raw;

% Replace NaN with 0 for filtering
U_grid(isnan(U_grid)) = 0;
V_grid(isnan(V_grid)) = 0;

% Apply Gaussian smoothing for smoother field
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

%% Figure 1: Raw vs Smoothed Vector Fields
figure('Position', [100, 100, 1400, 900]);

% Subplot 1: Raw Vector Field
subplot(2, 3, 1);
hold on;
title('Raw Vector Field');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Plot raw vector field
scale_factor = 0.8;
skip = 2;
valid_arrows = ~isnan(U_grid_raw(1:skip:end, 1:skip:end));
quiver(X_grid(1:skip:end, 1:skip:end), Y_grid(1:skip:end, 1:skip:end), ...
       U_grid_raw(1:skip:end, 1:skip:end)*scale_factor, ...
       V_grid_raw(1:skip:end, 1:skip:end)*scale_factor, ...
       'AutoScale', 'off', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);

axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 2: Smoothed Vector Field
subplot(2, 3, 2);
hold on;
title('Smoothed Vector Field');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Plot smoothed vector field
quiver(X_grid(1:skip:end, 1:skip:end), Y_grid(1:skip:end, 1:skip:end), ...
       U_smooth(1:skip:end, 1:skip:end)*scale_factor, ...
       V_smooth(1:skip:end, 1:skip:end)*scale_factor, ...
       'AutoScale', 'off', 'Color', [0 0.4 0.8], 'LineWidth', 1.5);

axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 3: Streamlines (Smoothed)
subplot(2, 3, 3);
hold on;
title('Flow Streamlines');
xlabel('X Position (m)');
ylabel('Y Position (m)');

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

% Subplot 4: Saturation Grid Mesh
subplot(2, 3, 4);
hold on;
title('Saturation Magnitude Field');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Create mesh plot of saturation
surf(X_grid, Y_grid, Sat_grid_raw, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
view(2);  % Top-down view
colormap(subplot(2,3,4), hot);
colorbar;
c = colorbar;
c.Label.String = 'Saturation';

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 5: Hue Grid Mesh
subplot(2, 3, 5);
hold on;
title('Hue Direction Field (0-1 scale)');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Create mesh plot of hue
surf(X_grid, Y_grid, Hue_grid_raw, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
view(2);  % Top-down view
colormap(subplot(2,3,5), hsv);
colorbar;
c = colorbar;
c.Label.String = 'Hue (0-1)';

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 6: Combined Visualization
subplot(2, 3, 6);
hold on;
title('Combined: Vectors colored by Saturation');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Create quiver plot with color coding by saturation
skip2 = 3;
for i = 1:skip2:size(X_grid, 1)
    for j = 1:skip2:size(X_grid, 2)
        if ~isnan(U_grid_raw(i,j)) && ~isnan(Sat_grid_raw(i,j))
            color_val = Sat_grid_raw(i,j) / max(Sat_grid_raw(:));
            quiver(X_grid(i,j), Y_grid(i,j), ...
                   U_grid_raw(i,j)*scale_factor*0.5, ...
                   V_grid_raw(i,j)*scale_factor*0.5, ...
                   'AutoScale', 'off', 'Color', [color_val, 0, 1-color_val], ...
                   'LineWidth', 1.5);
        end
    end
end

axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

sgtitle('Vector Field Analysis - Raw and Processed');
saveas(gcf, 'vector_field_comparison.png');
saveas(gcf, 'vector_field_comparison.fig');

%% Figure 2: Advanced Analysis with Curl and Divergence
figure('Position', [1550, 100, 1200, 900]);

% Subplot 1: Vorticity (Curl) Analysis
subplot(2, 2, 1);
hold on;
title('Vorticity (Curl of Vector Field)');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Calculate vorticity (curl) - use raw field for more accurate analysis
[~, vorticity] = curl(X_grid, Y_grid, U_grid_raw, V_grid_raw);
vorticity(isnan(vorticity)) = 0;

imagesc(x_edges(1:end-1), y_edges(1:end-1), vorticity);
axis xy;
colormap(subplot(2,2,1), redblue);
colorbar;
c = colorbar;
c.Label.String = 'Vorticity';
if max(abs(vorticity(:))) > 0
    caxis([-max(abs(vorticity(:))), max(abs(vorticity(:)))]);
end

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 2: Divergence Analysis
subplot(2, 2, 2);
hold on;
title('Divergence of Vector Field');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Calculate divergence - use raw field
divergence_field = divergence(X_grid, Y_grid, U_grid_raw, V_grid_raw);
divergence_field(isnan(divergence_field)) = 0;

imagesc(x_edges(1:end-1), y_edges(1:end-1), divergence_field);
axis xy;
colormap(subplot(2,2,2), cool);
colorbar;
c = colorbar;
c.Label.String = 'Divergence';

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 3: Vector Magnitude Heatmap
subplot(2, 2, 3);
hold on;
title('Vector Field Magnitude');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Calculate magnitude grid
Magnitude_grid = sqrt(U_grid_raw.^2 + V_grid_raw.^2);
imagesc(x_edges(1:end-1), y_edges(1:end-1), Magnitude_grid);
axis xy;
colormap(subplot(2,2,3), jet);
colorbar;
c = colorbar;
c.Label.String = 'Magnitude';

% Overlay vectors
quiver(X_grid(1:skip:end, 1:skip:end), Y_grid(1:skip:end, 1:skip:end), ...
       U_grid_raw(1:skip:end, 1:skip:end)*0.3, ...
       V_grid_raw(1:skip:end, 1:skip:end)*0.3, ...
       'AutoScale', 'off', 'Color', 'w', 'LineWidth', 1);

axis equal;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 4: Flow Classification
subplot(2, 2, 4);
hold on;
title('Flow Type Classification');
xlabel('X Position (m)');
ylabel('Y Position (m)');

% Classify flow based on vorticity and divergence
flow_type = zeros(size(X_grid));
for i = 1:size(X_grid, 1)
    for j = 1:size(X_grid, 2)
        if ~isnan(vorticity(i,j)) && ~isnan(divergence_field(i,j))
            vort_val = abs(vorticity(i,j));
            div_val = abs(divergence_field(i,j));
            
            if vort_val > 2*div_val
                flow_type(i,j) = 1;  % Rotational (vortex-like)
            elseif div_val > 2*vort_val
                flow_type(i,j) = 2;  % Source/Sink
            elseif vort_val > 0.1 || div_val > 0.1
                flow_type(i,j) = 3;  % Mixed/Saddle
            else
                flow_type(i,j) = 0;  % Uniform/Low activity
            end
        else
            flow_type(i,j) = NaN;
        end
    end
end

% Custom colormap for flow types
cmap_flow = [0.9 0.9 0.9;  % Gray for uniform
             0.8 0.2 0.2;  % Red for rotational
             0.2 0.2 0.8;  % Blue for source/sink
             0.2 0.8 0.2]; % Green for mixed

imagesc(x_edges(1:end-1), y_edges(1:end-1), flow_type);
axis xy;
colormap(subplot(2,2,4), cmap_flow);
caxis([-0.5 3.5]);
colorbar('Ticks', [0 1 2 3], 'TickLabels', {'Uniform', 'Rotational', 'Source/Sink', 'Mixed'});

axis equal;
xlim([-1 1]);
ylim([-1 1]);

sgtitle('Advanced Vector Field Analysis');
saveas(gcf, 'vector_field_advanced.png');
saveas(gcf, 'vector_field_advanced.fig');

%% Figure 3: Detailed Grid Visualizations
figure('Position', [100, 1050, 1400, 600]);

% Subplot 1: 3D Saturation Surface
subplot(1, 3, 1);
hold on;
surf(X_grid, Y_grid, Sat_grid_raw, 'EdgeColor', 'k', 'EdgeAlpha', 0.2, ...
     'FaceAlpha', 0.9, 'FaceColor', 'interp');
title('Saturation Field (3D Surface)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Saturation');
colormap(subplot(1,3,1), hot);
colorbar;
view(45, 30);
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 2: 3D Hue Surface
subplot(1, 3, 2);
hold on;
surf(X_grid, Y_grid, Hue_grid_raw, 'EdgeColor', 'k', 'EdgeAlpha', 0.2, ...
     'FaceAlpha', 0.9, 'FaceColor', 'interp');
title('Hue Field (3D Surface)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Hue (0-1)');
colormap(subplot(1,3,2), hsv);
colorbar;
view(45, 30);
grid on;
xlim([-1 1]);
ylim([-1 1]);

% Subplot 3: Contour Plot Overlay
subplot(1, 3, 3);
hold on;

% Create filled contour for saturation
contourf(X_grid, Y_grid, Sat_grid_raw, 10, 'LineStyle', 'none');
colormap(subplot(1,3,3), parula);
alpha(0.5);

% Overlay hue contour lines
contour(X_grid, Y_grid, Hue_grid_raw, 8, 'LineColor', 'k', 'LineWidth', 1.5);

% Add vector field on top
quiver(X_grid(1:skip:end, 1:skip:end), Y_grid(1:skip:end, 1:skip:end), ...
       U_grid_raw(1:skip:end, 1:skip:end)*0.4, ...
       V_grid_raw(1:skip:end, 1:skip:end)*0.4, ...
       'AutoScale', 'off', 'Color', 'w', 'LineWidth', 1.5);

title('Combined Contour: Saturation (filled) + Hue (lines)');
xlabel('X Position (m)');
ylabel('Y Position (m)');
colorbar;
c = colorbar;
c.Label.String = 'Saturation';
axis equal;
grid on;
xlim([-1 1]);
ylim([-1 1]);

sgtitle('Detailed Grid Visualizations');
saveas(gcf, 'vector_field_grids.png');
saveas(gcf, 'vector_field_grids.fig');

%% Export processed data
fprintf('\nExporting processed data...\n');

% Save reshaped and processed data
output_filename = 'processed_vector_data.csv';

% Create header
header = {'x', 'y', 'theta', 'hue_rad', 'saturation'};

% Write to CSV
fid = fopen(output_filename, 'w');
fprintf(fid, '%s,%s,%s,%s,%s\n', header{:});

for row = 1:size(reshaped_data, 1)
    fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f\n', reshaped_data(row, :));
end

fclose(fid);
fprintf('  ✓ Processed data saved to %s\n', output_filename);

% Also save grid data
grid_filename = 'vector_field_grids.mat';
save(grid_filename, 'X_grid', 'Y_grid', 'U_grid_raw', 'V_grid_raw', ...
     'U_smooth', 'V_smooth', 'Sat_grid_raw', 'Hue_grid_raw', ...
     'vorticity', 'divergence_field');
fprintf('  ✓ Grid data saved to %s\n', grid_filename);

%% Summary
fprintf('\n========================================\n');
fprintf('           ANALYSIS COMPLETE\n');
fprintf('========================================\n');
fprintf('Input file: %s\n', csv_filename);
fprintf('Output files:\n');
fprintf('  - vector_field_comparison.png\n');
fprintf('  - vector_field_advanced.png\n');
fprintf('  - vector_field_grids.png\n');
fprintf('  - processed_vector_data.csv\n');
fprintf('  - vector_field_grids.mat\n');
fprintf('\nData summary:\n');
fprintf('  Original: %d timepoints x %d robots\n', n_timepoints, n_robots);
fprintf('  Reshaped: %d rows x 5 columns\n', size(reshaped_data, 1));
fprintf('  Valid points: %d (%.1f%%)\n', sum(valid_idx), 100*sum(valid_idx)/length(valid_idx));
fprintf('  Grid resolution: %dx%d\n', grid_resolution-1, grid_resolution-1);
fprintf('\nFlow classification:\n');
if exist('flow_type', 'var')
    fprintf('  Uniform/Low: %d cells\n', sum(flow_type(:) == 0));
    fprintf('  Rotational: %d cells\n', sum(flow_type(:) == 1));
    fprintf('  Source/Sink: %d cells\n', sum(flow_type(:) == 2));
    fprintf('  Mixed/Saddle: %d cells\n', sum(flow_type(:) == 3));
end
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