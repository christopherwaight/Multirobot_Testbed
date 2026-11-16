% Vector Field Analyzer - Accounting for Y-axis flip
clear; clc; close all;

%% Load and reshape data
csv_filename = 'all_vortex_data_possible3.csv';
data_matrix = table2array(readtable(csv_filename));
n_timepoints = size(data_matrix, 1);

% Reshape: stack all 3 robots' data
reshaped_data = zeros(n_timepoints * 3, 5);
for t = 1:n_timepoints
    for robot = 1:3
        col_offset = (robot-1)*5;
        reshaped_data((t-1)*3 + robot, :) = [
            data_matrix(t, col_offset+1),  % x
            data_matrix(t, col_offset+2),  % y  
            data_matrix(t, col_offset+3),  % theta (robot heading)
            data_matrix(t, col_offset+4),  % hue (field direction)
            data_matrix(t, col_offset+5)   % saturation (field magnitude)
        ];
    end
end

%% Process field measurements
valid_idx = (reshaped_data(:, 1) ~= 0) | (reshaped_data(:, 2) ~= 0);
valid_data = reshaped_data(valid_idx, :);

x = valid_data(:, 1);
y = valid_data(:, 2);
hue = valid_data(:, 4);
saturation = valid_data(:, 5);

% Convert hue to field angle
field_angle = hue * 2 * pi;

% Calculate vector components
u = cos(field_angle) .* saturation;
v = sin(field_angle) .* saturation;

% CRITICAL FIX: Account for Y-axis flip (Y = -y transformation)
v = -v;  % Negate v-component due to coordinate flip

%% Create grid
grid_resolution = 30;
x_edges = linspace(-0.5, 0.5, grid_resolution);
y_edges = linspace(-0.5, 0.5, grid_resolution);
[X_grid, Y_grid] = meshgrid(x_edges(1:end-1) + diff(x_edges)/2, ...
                            y_edges(1:end-1) + diff(y_edges)/2);

% Initialize grids
U_grid = zeros(size(X_grid));
V_grid = zeros(size(X_grid));
Sat_grid = zeros(size(X_grid));
Hue_grid = zeros(size(X_grid));
Count_grid = zeros(size(X_grid));

% Bin data into grid
for i = 1:length(x)
    [~, xi] = min(abs(x_edges - x(i)));
    [~, yi] = min(abs(y_edges - y(i)));
    xi = min(xi, grid_resolution-1);
    yi = min(yi, grid_resolution-1);
    
    U_grid(yi, xi) = U_grid(yi, xi) + u(i);
    V_grid(yi, xi) = V_grid(yi, xi) + v(i);
    Sat_grid(yi, xi) = Sat_grid(yi, xi) + saturation(i);
    Hue_grid(yi, xi) = Hue_grid(yi, xi) + hue(i);
    Count_grid(yi, xi) = Count_grid(yi, xi) + 1;
end

% Average
valid_cells = Count_grid > 0;
U_grid(valid_cells) = U_grid(valid_cells) ./ Count_grid(valid_cells);
V_grid(valid_cells) = V_grid(valid_cells) ./ Count_grid(valid_cells);
Sat_grid(valid_cells) = Sat_grid(valid_cells) ./ Count_grid(valid_cells);
Hue_grid(valid_cells) = Hue_grid(valid_cells) ./ Count_grid(valid_cells);
U_grid(~valid_cells) = NaN;
V_grid(~valid_cells) = NaN;

%% Visualizations
figure('Position', [100, 100, 1600, 800]);

% Raw data with vectors
subplot(2, 3, 1);
skip_raw = 50;
idx = 1:skip_raw:length(x);
quiver(x(idx), y(idx), u(idx)*0.03, v(idx)*0.03, ...
       'AutoScale', 'off', 'Color', 'b', 'LineWidth', 1);
hold on;
scatter(x(idx), y(idx), 10, hue(idx), 'filled');
colormap(subplot(2,3,1), hsv);
colorbar;
title('Raw Field Measurements');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on; xlim([-0.5 0.5]); ylim([-0.5 0.5]);

% Gridded vectors
subplot(2, 3, 2);
quiver(X_grid, Y_grid, U_grid*0.03, V_grid*0.03, ...
       'AutoScale', 'off', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
title('Averaged Field');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on; xlim([-0.5 0.5]); ylim([-0.5 0.5]);

% Hue field
subplot(2, 3, 3);
imagesc(x_edges(1:end-1), y_edges(1:end-1), Hue_grid);
axis xy; colormap(subplot(2,3,3), hsv); colorbar;
title('Field Direction (Hue)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; xlim([-0.5 0.5]); ylim([-0.5 0.5]);

% Saturation field
subplot(2, 3, 4);
imagesc(x_edges(1:end-1), y_edges(1:end-1), Sat_grid);
axis xy; colormap(subplot(2,3,4), hot); colorbar;
title('Field Magnitude (Saturation)');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; xlim([-0.5 0.5]); ylim([-0.5 0.5]);

% Streamlines
subplot(2, 3, 5);
U_stream = U_grid; V_stream = V_grid;
U_stream(isnan(U_stream)) = 0; V_stream(isnan(V_stream)) = 0;
[startx, starty] = meshgrid(-0.4:0.1:0.4, -0.4:0.1:0.4);
h_stream = streamline(X_grid, Y_grid, U_stream, V_stream, startx(:), starty(:));
set(h_stream, 'Color', [0.2 0.2 0.8], 'LineWidth', 1);
title('Field Streamlines');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on; xlim([-0.5 0.5]); ylim([-0.5 0.5]);

% Combined visualization
subplot(2, 3, 6);
imagesc(x_edges(1:end-1), y_edges(1:end-1), Sat_grid);
axis xy; colormap(subplot(2,3,6), gray); alpha(0.3); hold on;
for i = 1:size(X_grid, 1)
    for j = 1:size(X_grid, 2)
        if ~isnan(U_grid(i,j))
            c_val = min(max(Sat_grid(i,j), 0), 1);
            quiver(X_grid(i,j), Y_grid(i,j), U_grid(i,j)*0.03, V_grid(i,j)*0.03, ...
                   'AutoScale', 'off', 'Color', [c_val, 0, 1-c_val], 'LineWidth', 1.5);
        end
    end
end
title('Combined Visualization');
xlabel('X (m)'); ylabel('Y (m)');
axis equal; grid on; xlim([-0.5 0.5]); ylim([-0.5 0.5]);

sgtitle('Vector Field Analysis - Corrected for Y-axis Flip');