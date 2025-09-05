% Multi-Run Trajectory Plotter

%% Plotting the RUNS

run_files = {'run1.mat', 'run2.mat', 'run3.mat', 'run4.mat', 'run5.mat', ...
             'run6.mat', 'run7.mat', 'run8.mat', 'run9.mat', 'run10.mat'};

% Define line styles to distinguish different runs
colors = {'r', 'g', 'b', 'm', 'c', 'y', [0.8 0.4 0], [0.4 0.8 0], [0.8 0 0.8], [0 0.8 0.8]};
line_styles = {'-', '--', ':', '-.'};

% Create a figure with appropriate size
figure('Position', [100, 100, 1000, 800]);
hold on;

% Create empty arrays for legend
legend_entries = {};

% Process each run file
for run_idx = 1:length(run_files)
    
    % Load the current run file
    current_file = run_files{run_idx};  
    fprintf('Processing file: %s\n', current_file);
    
    % Cycle through combinations of colors and line styles
    color_idx = mod(run_idx-1, length(colors)) + 1;
    style_idx = ceil(run_idx/length(colors));
    style_idx = mod(style_idx-1, length(line_styles)) + 1;
    
    current_color = colors{color_idx};
    current_style = line_styles{style_idx};
    line_format = [current_color, current_style];
    
    % Create temporary variables to avoid conflicts between runs
    cluster_pose_temp = [];
    robot1_pose_temp = [];
    robot2_pose_temp = [];
    robot3_pose_temp = [];
    
    % Load data into temporary workspace to avoid variable conflicts
    temp_data = load(current_file);
    
    % Check if expected variables exist in this file
    if isfield(temp_data, 'cluster_pose')
        cluster_pose_temp = temp_data.cluster_pose;
        plot(cluster_pose_temp(:,1), cluster_pose_temp(:,2), line_format, 'LineWidth', 2);
        legend_entries{end+1} = ['Run ' num2str(run_idx) ' - Cluster'];
    end
    
    if isfield(temp_data, 'robot1_pose')
        robot1_pose_temp = temp_data.robot1_pose;
        plot(robot1_pose_temp(:,1), robot1_pose_temp(:,2), line_format, 'LineWidth', 1.5);
        legend_entries{end+1} = ['Run ' num2str(run_idx) ' - Robot 1'];
    end
    
    if isfield(temp_data, 'robot2_pose')
        robot2_pose_temp = temp_data.robot2_pose;
        plot(robot2_pose_temp(:,1), robot2_pose_temp(:,2), line_format, 'LineWidth', 1.5);
        legend_entries{end+1} = ['Run ' num2str(run_idx) ' - Robot 2'];
    end
    
    if isfield(temp_data, 'robot3_pose')
        robot3_pose_temp = temp_data.robot3_pose;
        plot(robot3_pose_temp(:,1), robot3_pose_temp(:,2), line_format, 'LineWidth', 1.5);
        legend_entries{end+1} = ['Run ' num2str(run_idx) ' - Robot 3'];
    end
    
    
    if ~isempty(cluster_pose_temp)
        plot(cluster_pose_temp(1,1), cluster_pose_temp(1,2), [current_color, 'o'], 'MarkerSize', 8, 'MarkerFaceColor', current_color);
    end
    if ~isempty(robot1_pose_temp)
        plot(robot1_pose_temp(1,1), robot1_pose_temp(1,2), [current_color, 'o'], 'MarkerSize', 6, 'MarkerFaceColor', current_color);
    end
    
end

% Add title and labels
title('Robot Trajectories - Multiple Runs Comparison');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
axis equal;

% Create legend - if there are too many entries, consider customizing this
if ~isempty(legend_entries)
    % For many runs, you might want to adjust the legend
    if length(legend_entries) > 20
        % Option 1: Only show a subset of legend entries
        legend(legend_entries(1:20), 'Location', 'eastoutside');
        
        % Option 2: Create a more compact custom legend by run
        % Uncomment the following code to use this approach instead
        %{
        custom_legend_entries = cell(1, length(run_files));
        for i = 1:length(run_files)
            custom_legend_entries{i} = ['Run ' num2str(i)];
        end
        legend(custom_legend_entries, 'Location', 'eastoutside');
        %}
    else
        legend(legend_entries, 'Location', 'eastoutside');
    end
end



%  Save figure
saveas(gcf, 'multi_run_comparison.png');
fprintf('Plotting complete. Figure saved as multi_run_comparison.png\n');

%% Plotting an Average Run