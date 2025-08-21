%%  - Multi-robot Racing Plotter
clc; clear all; close all;
% Define the run files to process
run_files = {'tidal_runs.mat', 'celeste_runs.mat', 'pacific_blue_runs.mat', 'redwood_runs.mat'};
colors = {'magenta', 'cyan', 'blue', 'green'};  % Corresponding colors for plotting
color_names = {'R1', 'R2', 'R3', 'R4'};  % Color names for legend
markers = {'o', 's', 'd', 'g'};  % Different markers for each color

% Figure for the final combined velocity vs saturation plot
figure('Position', [100, 100, 800, 600]);
hold on;

% Process each run file
for file_idx = 1:length(run_files)
    % Load the current run file
    current_file = run_files{file_idx};
    disp(['Processing file: ', current_file]);
    load(current_file);
    
    % Get all variables in the workspace that might be poses
    vars = whos;
    pose_vars = {};
    
    % Find all pose variables in the loaded file
    for i = 1:length(vars)
        if strncmp(vars(i).name, 'pose_', 5) && isnumeric(eval(vars(i).name))
            pose_vars{end+1} = vars(i).name;
        end
    end
    
    % Sort pose variables to ensure correct order
    pose_vars = sort(pose_vars);
    
    % Initialize array to store velocities
    velocities = zeros(1, length(pose_vars));
    
    % Optional: Create a figure for the velocity traces of the current color
    figure('Name', ['Velocity Traces - ', color_names{file_idx}]);
    hold on;
    
    % Process each pose variable
    for i = 1:length(pose_vars)
        % Get the current pose data
        current_pose = eval(pose_vars{i});
        
        % Calculate velocity (using both x and y for 2D velocity)
        diff_pose = sqrt(diff(current_pose(:,1)).^2 + diff(current_pose(:,2)).^2) * 10;
        
        % Moving average to smooth the data
        mov_avg_pts = 10;
        mov_avg_diff = movmean(diff_pose, mov_avg_pts);
        
        % Plot the smoothed velocity trace for this pose
        plot(mov_avg_diff);
        
        % Calculate mean velocity for this pose
        mean_diff = mean(diff_pose);
        velocities(i) = mean_diff;
        
        disp(['Mean velocity for ', pose_vars{i}, ': ', num2str(mean_diff)]);
    end
    
    % Finalize the velocity traces figure
    title(['Velocity Traces - ', color_names{file_idx}]);
    xlabel('Time Steps');
    ylabel('Velocity (m/s)');
    hold off;
    
    % Saturations corresponding to the pose files (assuming they match across all colors)
    % Modify these values if your saturation values are different
    saturations = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    % Ensure the number of saturations matches the number of poses
    % Truncate if necessary
    min_length = min(length(saturations), length(velocities));
    saturations = saturations(1:min_length);
    velocities = velocities(1:min_length);
    
    % Switch back to the combined figure and plot the velocity vs saturation
    figure(1);
    plot(saturations, velocities, [markers{file_idx}, '-'], 'Color', colors{file_idx}, 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', color_names{file_idx});
    
    % Clear variables before loading the next file to avoid conflicts
    clear pose_*
end

% Finalize the combined figure
xlabel('Saturation');
ylabel('Velocity (m/s)');
title('Robot Velocity at Different Saturations for Multiple Colors');
legend('Location', 'northwest');
grid on;
hold off;

% Optional: Save the figure
saveas(gcf, 'multi_color_velocity_comparison.png');