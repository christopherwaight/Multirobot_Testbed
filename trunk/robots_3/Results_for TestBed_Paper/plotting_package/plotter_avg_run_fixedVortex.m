% Average Trajectory Plotter
% This script loads multiple run files, resamples trajectories to the same length,
% and plots the average trajectory with confidence intervals
% Define the run files to process - modify these filenames as needed
run_files = {'run1.mat', 'run2.mat', 'run3.mat', 'run4.mat', 'run5.mat', ...
             'run6.mat', 'run7.mat', 'run8.mat', 'run9.mat', 'run10.mat'};
% Display info message at start         
fprintf('Loading and processing %d run files...\n', length(run_files));
% Define colors for different entities
cluster_color = [0.2 0.2 0.2];  % Dark gray
robot1_color = [0.8 0.2 0.2];   % Red
robot2_color = [0.2 0.6 0.2];   % Green
robot3_color = [0.2 0.3 0.8];   % Blue
% Create data storage for trajectories across runs
all_cluster_traj = {};
all_robot1_traj = {};
all_robot2_traj = {};
all_robot3_traj = {};
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
    
    % Store trajectories if they exist
    if isfield(temp_data, 'cluster_pose')
        all_cluster_traj{end+1} = temp_data.cluster_pose(:,1:2); % Just keep X and Y
    end
    
    if isfield(temp_data, 'robot1_pose')
        all_robot1_traj{end+1} = temp_data.robot1_pose(:,1:2);
    end
    
    if isfield(temp_data, 'robot2_pose')
        all_robot2_traj{end+1} = temp_data.robot2_pose(:,1:2);
    end
    
    if isfield(temp_data, 'robot3_pose')
        all_robot3_traj{end+1} = temp_data.robot3_pose(:,1:2);
    end
end
% Process and average trajectories
% This function will be defined below
avg_cluster_traj = [];
std_cluster_traj = [];
if ~isempty(all_cluster_traj)
    [avg_cluster_traj, std_cluster_traj] = averageTrajectories(all_cluster_traj);
    fprintf('Processed %d cluster trajectories\n', length(all_cluster_traj));
end
avg_robot1_traj = [];
std_robot1_traj = [];
if ~isempty(all_robot1_traj)
    [avg_robot1_traj, std_robot1_traj] = averageTrajectories(all_robot1_traj);
    fprintf('Processed %d robot1 trajectories\n', length(all_robot1_traj));
end
avg_robot2_traj = [];
std_robot2_traj = [];
if ~isempty(all_robot2_traj)
    [avg_robot2_traj, std_robot2_traj] = averageTrajectories(all_robot2_traj);
    fprintf('Processed %d robot2 trajectories\n', length(all_robot2_traj));
end
avg_robot3_traj = [];
std_robot3_traj = [];
if ~isempty(all_robot3_traj)
    [avg_robot3_traj, std_robot3_traj] = averageTrajectories(all_robot3_traj);
    fprintf('Processed %d robot3 trajectories\n', length(all_robot3_traj));
end
% Create figure for plotting
figure('Position', [100, 100, 900, 700]);
hold on;

% Function to plot trajectory with confidence band
function h = plotTrajectoryWithConfidence(avg_traj, std_traj, color, line_width, alpha_val, name)
    if isempty(avg_traj)
        h = [];
        return;
    end
    
    % Plot the average trajectory
    h = plot(avg_traj(:,1), avg_traj(:,2), 'Color', color, 'LineWidth', line_width, 'DisplayName', name);
    
    % Create confidence band (mean ± standard deviation)
    x_band = [avg_traj(:,1); flipud(avg_traj(:,1))];
    y_band = [avg_traj(:,2) + std_traj(:,2); flipud(avg_traj(:,2) - std_traj(:,2))];
    
    % Plot the confidence band with transparency
    fill(x_band, y_band, color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

% Add fixed vortex quiver plot
[X, Y] = meshgrid(linspace(-1, 1, 15), linspace(-1, 1, 15));
center_x = 0;
center_y = 0;
r = sqrt((X - center_x).^2 + (Y - center_y).^2);
theta = atan2(Y - center_y, X - center_x);
U = r .* sin(theta);
V = -r .* cos(theta);

% Normalize vectors for better visualization
lengths = sqrt(U.^2 + V.^2);
scale_factor = 0.5; % Adjust this for arrow length
U_norm = scale_factor * U ;
V_norm = scale_factor * V ;

% Plot quiver with gray arrows
q = quiver(X, Y, U_norm, V_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
q.HandleVisibility = 'off'; % Hide from legend

% Plot trajectories with confidence bands
h1 = plotTrajectoryWithConfidence(avg_cluster_traj, std_cluster_traj, cluster_color, 3, 0.2, 'Cluster Centre');
h2 = plotTrajectoryWithConfidence(avg_robot1_traj, std_robot1_traj, robot1_color, 2, 0.2, 'Robot 1');
h3 = plotTrajectoryWithConfidence(avg_robot2_traj, std_robot2_traj, robot2_color, 2, 0.2, 'Robot 2');
h4 = plotTrajectoryWithConfidence(avg_robot3_traj, std_robot3_traj, robot3_color, 2, 0.2, 'Robot 3');

% Mark starting and ending points with labels
if ~isempty(avg_cluster_traj)
    % Start point
    s1 = plot(avg_cluster_traj(1,1), avg_cluster_traj(1,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', cluster_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    text(avg_cluster_traj(1,1)+0.05, avg_cluster_traj(1,2)+0.05, 'Start', 'FontWeight', 'bold', 'HandleVisibility', 'off');
    
    % End point
    e1 = plot(avg_cluster_traj(end,1), avg_cluster_traj(end,2), 's', 'MarkerSize', 10, 'MarkerFaceColor', cluster_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    text(avg_cluster_traj(end,1)+0.05, avg_cluster_traj(end,2)+0.05, 'End', 'FontWeight', 'bold', 'HandleVisibility', 'off');
end

if ~isempty(avg_robot1_traj)
    s2 = plot(avg_robot1_traj(1,1), avg_robot1_traj(1,2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', robot1_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    e2 = plot(avg_robot1_traj(end,1), avg_robot1_traj(end,2), 's', 'MarkerSize', 8, 'MarkerFaceColor', robot1_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
if ~isempty(avg_robot2_traj)
    s3 = plot(avg_robot2_traj(1,1), avg_robot2_traj(1,2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', robot2_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    e3 = plot(avg_robot2_traj(end,1), avg_robot2_traj(end,2), 's', 'MarkerSize', 8, 'MarkerFaceColor', robot2_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
if ~isempty(avg_robot3_traj)
    s4 = plot(avg_robot3_traj(1,1), avg_robot3_traj(1,2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', robot3_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    e4 = plot(avg_robot3_traj(end,1), avg_robot3_traj(end,2), 's', 'MarkerSize', 8, 'MarkerFaceColor', robot3_color, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end

% Finalize plot
title('Average Robot Trajectories Across Multiple Runs');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
axis equal;
xlim([-1 1]);
ylim([-1 1]);
legend('Location', 'best');

% Save figure
saveas(gcf, 'average_trajectories.png');
saveas(gcf, 'average_trajectories.fig');
fprintf('Plot saved as average_trajectories.png and average_trajectories.fig\n');

% Define the trajectory averaging function
function [avg_traj, std_traj] = averageTrajectories(trajectories)
    % Find the minimum length among all trajectories
    traj_lengths = cellfun(@(x) size(x, 1), trajectories);
    min_length = min(traj_lengths);
    max_length = max(traj_lengths);
    
    fprintf('  Trajectory lengths vary from %d to %d points\n', min_length, max_length);
    
    % Number of points in the resampled trajectory (use min_length for safety)
    target_length = min_length;
    
    % Storage for resampled trajectories
    resampled_trajectories = cell(1, length(trajectories));
    
    % Resample all trajectories to the target length
    for i = 1:length(trajectories)
        original_traj = trajectories{i};
        original_length = size(original_traj, 1);
        
        if original_length == target_length
            % No resampling needed
            resampled_trajectories{i} = original_traj;
        else
            % Create a parametric representation based on cumulative distance
            % This preserves the spatial distribution better than uniform time sampling
            
            % Calculate cumulative distance along the trajectory
            dist = zeros(original_length, 1);
            for j = 2:original_length
                dist(j) = dist(j-1) + norm(original_traj(j,:) - original_traj(j-1,:));
            end
            
            % Normalize to [0, 1]
            if dist(end) > 0
                param = dist / dist(end);
            else
                param = linspace(0, 1, original_length)';
            end
            
            % Sample points at uniform parameter intervals
            new_param = linspace(0, 1, target_length)';
            resampled_x = interp1(param, original_traj(:,1), new_param, 'pchip');
            resampled_y = interp1(param, original_traj(:,2), new_param, 'pchip');
            
            resampled_trajectories{i} = [resampled_x, resampled_y];
        end
    end
    
    % Convert cell array to 3D array for easier computation
    traj_array = zeros(target_length, 2, length(trajectories));
    for i = 1:length(trajectories)
        traj_array(:,:,i) = resampled_trajectories{i};
    end
    
    % Calculate mean and standard deviation across runs
    avg_traj = mean(traj_array, 3);
    std_traj = std(traj_array, 0, 3);
end