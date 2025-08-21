% Radius Analysis Plotter
% This script loads multiple radius files and analyzes the cluster radius over time

% Define the radius files to process
radius_files = {'radius001.mat', 'radius01.mat', 'radius02.mat', 'radius03.mat', ...
                'radius04.mat'};



% Display info message at start         
fprintf('Loading and processing %d radius files...\n', length(radius_files));

% Define colors for different files
file_colors = [
    0.8 0.2 0.2;  % Red
    0.2 0.6 0.2;  % Green
    0.2 0.3 0.8;  % Blue
    0.8 0.5 0.2;  % Orange
    0.6 0.2 0.8;  % Purple
    0.8 0.8 0.2;  % Yellow
    0.2 0.8 0.8;  % Cyan
];

% Storage for radius data
all_radius_data = {};
all_time_data = {};
file_labels = {};

% Process each radius file
for file_idx = 1:length(radius_files)
    current_file = radius_files{file_idx};
    
    if exist(current_file, 'file')
        fprintf('\nLoading file: %s\n', current_file);
        
        % Load data
        data = load(current_file);
        
        % Check if cluster_pose exists
        if isfield(data, 'cluster_position')
            % Extract x and y positions
            x_pos = data.cluster_position(:, 1);
            y_pos = data.cluster_position(:, 2);
            
            % Calculate radius (Euclidean distance from origin)
            r = sqrt(x_pos.^2 + y_pos.^2);
            
            % Store radius data
            all_radius_data{end+1} = r;
            
            % Generate time vector (assuming constant sampling)
            % If time data exists in the file, use it instead
            if isfield(data, 'time')
                all_time_data{end+1} = data.time;
            else
                % Assume 1 Hz sampling if no time data
                all_time_data{end+1} = (0:length(r)-1)';
            end
            
            % Store file label
            file_labels{end+1} = strrep(current_file, '.mat', '');
            
            % Calculate and display statistics
            fprintf('Statistics for %s:\n', current_file);
            fprintf('  Number of samples: %d\n', length(r));
            fprintf('  Mean radius: %.4f m\n', mean(r));
            fprintf('  Std deviation: %.4f m\n', std(r));
            fprintf('  Min radius: %.4f m\n', min(r));
            fprintf('  Max radius: %.4f m\n', max(r));
            fprintf('  Initial radius: %.4f m\n', r(1));
            fprintf('  Final radius: %.4f m\n', r(end));
            fprintf('  Total change: %.4f m\n', r(end) - r(1));
            
        else
            fprintf('Warning: cluster_pose not found in %s\n', current_file);
        end
    else
        fprintf('Warning: File not found - %s\n', current_file);
    end
end

% Create figure for radius over time plot
figure('Position', [100, 100, 1000, 600]);
hold on;

% Plot all radius trajectories
for i = 1:length(all_radius_data)
    plot(all_time_data{i}, all_radius_data{i}, 'Color', file_colors(i,:), ...
         'LineWidth', 2, 'DisplayName', file_labels{i});
end

title('Cluster Radius Over Time - All Files');
xlabel('Time (s)');
ylabel('Radius (m)');
grid on;
legend('Location', 'best');
hold off;

% Save the plot
saveas(gcf, 'radius_comparison_plot.png');
saveas(gcf, 'radius_comparison_plot.fig');

% Create a summary statistics table
fprintf('\n\n=== SUMMARY STATISTICS TABLE ===\n');
fprintf('%-12s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n', ...
    'File', 'Samples', 'Mean(m)', 'Std(m)', 'Min(m)', 'Max(m)', 'Initial(m)', 'Final(m)');
fprintf('%s\n', repmat('-', 1, 100));

for i = 1:length(all_radius_data)
    r = all_radius_data{i};
    fprintf('%-12s | %8d | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
        file_labels{i}, length(r), mean(r), std(r), min(r), max(r), r(1), r(end));
end

% Export radius data to CSV files for further analysis
fprintf('\n\nExporting radius data to CSV files...\n');
for i = 1:length(all_radius_data)
    % Create filename
    csv_filename = sprintf('%s_radius_data.csv', file_labels{i});
    
    % Prepare data matrix [time, radius]
    export_data = [all_time_data{i}, all_radius_data{i}];
    
    % Write header and data
    fid = fopen(csv_filename, 'w');
    fprintf(fid, 'Time(s),Radius(m)\n');
    fprintf(fid, '%.6f,%.6f\n', export_data');
    fclose(fid);
    
    fprintf('Exported: %s\n', csv_filename);
end

% Create figure for trajectory plots in X-Y plane
figure('Position', [100, 700, 900, 700]);
hold on;

% Plot all trajectories in X-Y plane
for file_idx = 1:length(radius_files)
    current_file = radius_files{file_idx};
    
    if exist(current_file, 'file')
        data = load(current_file);
        
        if isfield(data, 'cluster_position')
            x_pos = data.cluster_position(:, 1);
            y_pos = data.cluster_position(:, 2);
            
            % Plot trajectory
            plot(x_pos, y_pos, 'Color', file_colors(file_idx,:), ...
                 'LineWidth', 2, 'DisplayName', file_labels{file_idx});
            
            % Mark start and end points
            plot(x_pos(1), y_pos(1), 'o', 'MarkerSize', 8, ...
                 'MarkerFaceColor', file_colors(file_idx,:), 'MarkerEdgeColor', 'k', ...
                 'HandleVisibility', 'off');
            plot(x_pos(end), y_pos(end), 's', 'MarkerSize', 8, ...
                 'MarkerFaceColor', file_colors(file_idx,:), 'MarkerEdgeColor', 'k', ...
                 'HandleVisibility', 'off');
        end
    end
end

% Add reference circles
theta = linspace(0, 2*pi, 100);
for r_ref = 0.2:0.2:1.0
    plot(r_ref*cos(theta), r_ref*sin(theta), 'k--', 'LineWidth', 0.5, ...
         'HandleVisibility', 'off');
    text(r_ref, 0, sprintf('r=%.1f', r_ref), 'VerticalAlignment', 'bottom');
end

title('Cluster Trajectories in X-Y Plane');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
axis equal;
legend('Location', 'best');
hold off;

% Save the trajectory plot
saveas(gcf, 'trajectory_comparison_plot.png');
saveas(gcf, 'trajectory_comparison_plot.fig');

% Create a combined data matrix for all runs
fprintf('\n\nCreating combined radius data file...\n');
max_length = max(cellfun(@length, all_radius_data));
combined_data = NaN(max_length, length(all_radius_data) + 1);

% Find the longest time vector for the combined file
[~, longest_idx] = max(cellfun(@length, all_time_data));
combined_data(1:length(all_time_data{longest_idx}), 1) = all_time_data{longest_idx};

% Add radius data
for i = 1:length(all_radius_data)
    combined_data(1:length(all_radius_data{i}), i+1) = all_radius_data{i};
end

% Write combined CSV
fid = fopen('combined_radius_data.csv', 'w');
fprintf(fid, 'Time(s)');
for i = 1:length(file_labels)
    fprintf(fid, ',%s', file_labels{i});
end
fprintf(fid, '\n');

% Write data rows
for row = 1:max_length
    fprintf(fid, '%.6f', combined_data(row, 1));
    for col = 2:size(combined_data, 2)
        if isnan(combined_data(row, col))
            fprintf(fid, ',');
        else
            fprintf(fid, ',%.6f', combined_data(row, col));
        end
    end
    fprintf(fid, '\n');
end
fclose(fid);

fprintf('Exported: combined_radius_data.csv\n');
fprintf('\nAll analysis complete!\n');