% Distance-Time Plotter
% This script loads multiple run files, calculates the distance from origin vs time,
% and plots the average distance with confidence intervals

% Define the run files to process - modify these filenames as needed
run_files = {'run1.mat', 'run2.mat', 'run3.mat', 'run4.mat', 'run5.mat', ...
             'run6.mat', 'run7.mat', 'run8.mat'};%, 'run9.mat', 'run10.mat'};
             
% Display info message at start         
fprintf('Loading and processing %d run files...\n', length(run_files));

% Define plot color
cluster_color = [0.2 0.2 0.2];  % Dark gray

% Create data storage for distances and times across runs
all_distances = {};
all_times = {};

% Determine which files exist and how many valid runs we have
existing_files = {};
for i = 1:length(run_files)
    if exist(run_files{i}, 'file')
        existing_files{end+1} = run_files{i};
    else
        fprintf('Warning: File not found - %s\n', run_files{i});
    end
end

fprintf('Found %d existing run files to process\n', length(existing_files));
if isempty(existing_files)
    error('No run files found. Please check file paths and names.');
end

% Process each run file
for run_idx = 1:length(existing_files)
    current_file = existing_files{run_idx};
    fprintf('Loading file: %s\n', current_file);
    
    % Load data
    temp_data = load(current_file);
    
    % Determine what fields are available in the data
    field_names = fieldnames(temp_data);
    fprintf('Available fields: %s\n', strjoin(field_names, ', '));
    
    % Find the cluster position data (try different possible field names)
    if isfield(temp_data, 'cluster_pose')
        % Extract x,y coordinates from cluster_pose
        x_pos = temp_data.cluster_pose(:, 1);
        y_pos = temp_data.cluster_pose(:, 2);
    elseif isfield(temp_data, 'xc') && isfield(temp_data, 'yc')
        % Using separate x,y fields
        x_pos = temp_data.xc;
        y_pos = temp_data.yc;
    elseif isfield(temp_data, 'x_c') && isfield(temp_data, 'y_c')
        % Alternative naming with underscore
        x_pos = temp_data.x_c;
        y_pos = temp_data.y_c;
    else
        % Try to find any field with position data
        fprintf('Could not find cluster position data in standard fields\n');
        
        % Look for other possible fields containing positions
        found = false;
        for fn = 1:length(field_names)
            fieldname = field_names{fn};
            fielddata = temp_data.(fieldname);
            
            % Check if this might be a position array
            if isnumeric(fielddata) && (size(fielddata,2) >= 2 || size(fielddata,1) >= 2)
                fprintf('  Found potential position data in field: %s\n', fieldname);
                
                % If it's a position array with format [x,y,...]
                if size(fielddata,2) >= 2
                    x_pos = fielddata(:,1);
                    y_pos = fielddata(:,2);
                    found = true;
                    break;
                % If it's a position array with format [x;y;...]
                elseif size(fielddata,1) >= 2
                    x_pos = fielddata(1,:)';
                    y_pos = fielddata(2,:)';
                    found = true;
                    break;
                end
            end
        end
        
        if ~found
            fprintf('Warning: Could not find any position data in file %s\n', current_file);
            continue;
        end
    end
    
    % Calculate distance from origin
    distances = sqrt(x_pos.^2 + y_pos.^2);
    
    % Create artificial time vector (0.1 second per frame)
    time_vec = (0:(length(distances)-1))' * 0.1;
    
    % Store the results
    all_distances{end+1} = distances;
    all_times{end+1} = time_vec;
    fprintf('Successfully processed run %d with %d data points\n', run_idx, length(distances));
end

% Check if we have data to process
if isempty(all_distances)
    error('No valid distance data found in the run files.');
end

% Find min and max time across all runs for common time axis
min_times = cellfun(@min, all_times);
max_times = cellfun(@max, all_times);
global_min_time = min(min_times);
global_max_time = max(max_times);

% Create common time vector for interpolation
common_time = linspace(global_min_time, global_max_time, 100);

% Interpolate distances to common time vector
interpolated_distances = zeros(length(common_time), length(all_distances));
for i = 1:length(all_distances)
    interpolated_distances(:,i) = interp1(all_times{i}, all_distances{i}, common_time, 'pchip');
end

% Calculate mean and standard deviation
mean_distance = mean(interpolated_distances, 2);
std_distance = std(interpolated_distances, 0, 2);

% Create figure for plotting
figure('Position', [100, 100, 800, 500]);
hold on;

% Plot the average distance
plot(common_time, mean_distance, 'Color', cluster_color, 'LineWidth', 2, 'DisplayName', 'Average Distance');

% Create confidence band (mean ± standard deviation)
x_band = [common_time, fliplr(common_time)];
y_band = [mean_distance + std_distance; flipud(mean_distance - std_distance)]';

% Plot the confidence band with transparency
fill(x_band, y_band, cluster_color, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'Standard Deviation');

% Finalize plot
title('Randomized Starting Location - Distance from Origin Over Time');
xlabel('Time (s)');
ylabel('Distance from Origin (m)');
grid on;
legend('Location', 'best');

% Save figure
saveas(gcf, 'average_distance_vs_time.png');
saveas(gcf, 'average_distance_vs_time.fig');
fprintf('Plot saved as average_distance_vs_time.png and average_distance_vs_time.fig\n');