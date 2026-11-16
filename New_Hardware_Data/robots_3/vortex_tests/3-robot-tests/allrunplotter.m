% Average Trajectory Plotter - Modified for 80 runs in groups of 10
% This script loads 80 run files and creates separate average trajectories
% for each group of 10 runs (1-10, 11-20, ..., 71-80)

% Define all 80 run files
run_files = cell(1, 80);
for i = 1:80
    run_files{i} = sprintf('run%d.mat', i);
end

% Display info message at start
fprintf('Loading and processing %d run files in groups of 10...\n', length(run_files));

% Define colors for different entities
cluster_color = [0.2 0.2 0.2];  % Dark gray
robot1_color = [0.8 0.2 0.2];   % Red
robot2_color = [0.2 0.6 0.2];   % Green
robot3_color = [0.2 0.3 0.8];   % Blue

% Define colors for different groups (8 groups total)
group_colors = [
    0.8 0.2 0.2;  % Red
    0.2 0.6 0.2;  % Green
    0.2 0.3 0.8;  % Blue
    0.8 0.5 0.2;  % Orange
    0.6 0.2 0.8;  % Purple
    0.8 0.8 0.2;  % Yellow
    0.2 0.8 0.8;  % Cyan
    0.8 0.2 0.6;  % Magenta
];

% Process each group of 10 runs
num_groups = 8;
runs_per_group = 10;

% Storage for average trajectories of each group
group_avg_cluster = cell(1, num_groups);
group_std_cluster = cell(1, num_groups);
group_avg_robot1 = cell(1, num_groups);
group_std_robot1 = cell(1, num_groups);
group_avg_robot2 = cell(1, num_groups);
group_std_robot2 = cell(1, num_groups);
group_avg_robot3 = cell(1, num_groups);
group_std_robot3 = cell(1, num_groups);

% Process each group
for group_idx = 1:num_groups
    fprintf('\n=== Processing Group %d (runs %d-%d) ===\n', group_idx, ...
        (group_idx-1)*runs_per_group + 1, group_idx*runs_per_group);
    
    % Create data storage for trajectories in this group
    all_cluster_traj = {};
    all_robot1_traj = {};
    all_robot2_traj = {};
    all_robot3_traj = {};
    
    % Get the run indices for this group
    start_idx = (group_idx - 1) * runs_per_group + 1;
    end_idx = group_idx * runs_per_group;
    
    % Check which files exist in this group
    existing_files = {};
    for i = start_idx:end_idx
        if exist(run_files{i}, 'file')
            existing_files{end+1} = run_files{i};
        else
            fprintf('Warning: File not found - %s\n', run_files{i});
        end
    end
    
    fprintf('Found %d existing run files in group %d\n', length(existing_files), group_idx);
    
    if isempty(existing_files)
        fprintf('Warning: No run files found for group %d. Skipping...\n', group_idx);
        continue;
    end
    
    % Process each run file in this group
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
    
    % Process and average trajectories for this group
    if ~isempty(all_cluster_traj)
        [group_avg_cluster{group_idx}, group_std_cluster{group_idx}] = averageTrajectories(all_cluster_traj);
        fprintf('Processed %d cluster trajectories for group %d\n', length(all_cluster_traj), group_idx);
    end
    
    if ~isempty(all_robot1_traj)
        [group_avg_robot1{group_idx}, group_std_robot1{group_idx}] = averageTrajectories(all_robot1_traj);
        fprintf('Processed %d robot1 trajectories for group %d\n', length(all_robot1_traj), group_idx);
    end
    
    if ~isempty(all_robot2_traj)
        [group_avg_robot2{group_idx}, group_std_robot2{group_idx}] = averageTrajectories(all_robot2_traj);
        fprintf('Processed %d robot2 trajectories for group %d\n', length(all_robot2_traj), group_idx);
    end
    
    if ~isempty(all_robot3_traj)
        [group_avg_robot3{group_idx}, group_std_robot3{group_idx}] = averageTrajectories(all_robot3_traj);
        fprintf('Processed %d robot3 trajectories for group %d\n', length(all_robot3_traj), group_idx);
    end
end

% Create separate figures for each entity type
% Figure 1: Cluster trajectories
figure('Position', [100, 100, 900, 700]);
hold on;
title('Average Cluster Centre Trajectories by Group');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
axis equal;
xlim([-0.65 0.65]);
ylim([-0.65 0.65]);

% Add vortex field
[X, Y] = meshgrid(linspace(-1, 1, 15), linspace(-1, 1, 15));
center_x = 0; center_y = 0;
r = sqrt((X - center_x).^2 + (Y - center_y).^2);
theta = atan2(Y - center_y, X - center_x);
U = r .* sin(theta);
V = -r .* cos(theta);
scale_factor = 0.5;
U_norm = scale_factor * U;
V_norm = scale_factor * V;
q = quiver(X, Y, U_norm, V_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
q.HandleVisibility = 'off';

% Plot cluster trajectories for each group
for group_idx = 1:num_groups
    if ~isempty(group_avg_cluster{group_idx})
        plotTrajectoryWithConfidence(group_avg_cluster{group_idx}, group_std_cluster{group_idx}, ...
            group_colors(group_idx,:), 2, 0.15, sprintf('Runs %d-%d', ...
            (group_idx-1)*runs_per_group + 1, group_idx*runs_per_group));
    end
end
legend('Location', 'best');
saveas(gcf, 'average_cluster_trajectories_by_group.png');
saveas(gcf, 'average_cluster_trajectories_by_group.fig');

% Figure 2: All robot trajectories for each group
for group_idx = 1:num_groups
    if ~isempty(group_avg_robot1{group_idx}) || ~isempty(group_avg_robot2{group_idx}) || ~isempty(group_avg_robot3{group_idx})
        figure('Position', [100 + 50*group_idx, 100 + 50*group_idx, 900, 700]);
        hold on;
        title(sprintf('Average Robot Trajectories - Runs %d-%d', ...
            (group_idx-1)*runs_per_group + 1, group_idx*runs_per_group));
        xlabel('X Position (m)');
        ylabel('Y Position (m)');
        grid on;
        axis equal;
        xlim([-1 1]);
        ylim([-1 1]);
        
        % Add vortex field
        q = quiver(X, Y, U_norm, V_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
        q.HandleVisibility = 'off';
        
        % Plot trajectories
        if ~isempty(group_avg_cluster{group_idx})
            plotTrajectoryWithConfidence(group_avg_cluster{group_idx}, group_std_cluster{group_idx}, ...
                cluster_color, 3, 0.2, 'Cluster Centre');
        end
        
        if ~isempty(group_avg_robot1{group_idx})
            plotTrajectoryWithConfidence(group_avg_robot1{group_idx}, group_std_robot1{group_idx}, ...
                robot1_color, 2, 0.2, 'Robot 1');
        end
        
        if ~isempty(group_avg_robot2{group_idx})
            plotTrajectoryWithConfidence(group_avg_robot2{group_idx}, group_std_robot2{group_idx}, ...
                robot2_color, 2, 0.2, 'Robot 2');
        end
        
        if ~isempty(group_avg_robot3{group_idx})
            plotTrajectoryWithConfidence(group_avg_robot3{group_idx}, group_std_robot3{group_idx}, ...
                robot3_color, 2, 0.2, 'Robot 3');
        end
        
        % Mark start and end points
        if ~isempty(group_avg_cluster{group_idx})
            avg_traj = group_avg_cluster{group_idx};
            plot(avg_traj(1,1), avg_traj(1,2), 'o', 'MarkerSize', 10, 'MarkerFaceColor', cluster_color, ...
                'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            text(avg_traj(1,1)+0.05, avg_traj(1,2)+0.05, 'Start', 'FontWeight', 'bold');
            plot(avg_traj(end,1), avg_traj(end,2), 's', 'MarkerSize', 10, 'MarkerFaceColor', cluster_color, ...
                'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            text(avg_traj(end,1)+0.05, avg_traj(end,2)+0.05, 'End', 'FontWeight', 'bold');
        end
        
        legend('Location', 'best');
        saveas(gcf, sprintf('average_trajectories_group_%d.png', group_idx));
        saveas(gcf, sprintf('average_trajectories_group_%d.fig', group_idx));
    end
end

fprintf('\nAll plots saved successfully!\n');

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