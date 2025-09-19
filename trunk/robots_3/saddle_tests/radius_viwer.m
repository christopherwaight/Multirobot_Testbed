% MATLAB Script: Analyze cluster pose and detected critical points
% This script reads cluster_pose (601x9) and detected_critical_point (601x2)
% Uses only first 500 data points
% Calculates distances and creates five plots

%clear; clc; close all;

%% Load Data
% Assuming your data is in the workspace or you need to load it
% If data is in files, uncomment and modify these lines:
% cluster_pose = load('cluster_pose.mat');
% detected_critical_point = load('detected_critical_point.mat');

% Use only first 500 data points
n_points = 500;

% Extract first two columns (x,y) from cluster_pose (first 500 points)
cluster_xy = cluster_pose(1:n_points, 1:2);

% Extract first 500 points from detected_critical_point
detected_xy = detected_critical_point(1:n_points, :);

%% Calculate Euclidean Distance
% Calculate distance between paired points at each time step
distances = sqrt((cluster_xy(:,1) - detected_xy(:,1)).^2 + ...
                 (cluster_xy(:,2) - detected_xy(:,2)).^2);

% Create time vector for first 500 points
time = 0:(n_points-1);  % or use actual time data if available
% time = (0:(n_points-1)) * dt;  % if you have a specific time step dt

%% Create Plots
figure('Position', [50, 50, 1400, 900]);

% Plot 1: Distance over time
subplot(3,2,1);
plot(time, distances, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Distance');
title('Distance between Cluster Pose and Detected Critical Point over Time');
xlim([min(time), max(time)]);

% Plot 2: Cluster pose x,y over time
subplot(3,2,2);
plot(time, cluster_xy(:,1), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Cluster X');
hold on;
plot(time, cluster_xy(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Cluster Y');
grid on;
xlabel('Time');
ylabel('Position');
title('Cluster Pose (X,Y) over Time');
legend('Location', 'best');
xlim([min(time), max(time)]);

% Plot 3: Detected critical point x,y over time
subplot(3,2,3);
plot(time, detected_xy(:,1), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Detected X');
hold on;
plot(time, detected_xy(:,2), 'm-', 'LineWidth', 1.5, 'DisplayName', 'Detected Y');
grid on;
xlabel('Time');
ylabel('Position');
title('Detected Critical Point (X,Y) over Time');
legend('Location', 'best');
xlim([min(time), max(time)]);

% Plot 4: 2D plot of cluster pose (X vs Y)
subplot(3,2,4);
plot(cluster_xy(:,1), cluster_xy(:,2), 'r-', 'LineWidth', 1.5);
hold on;
plot(cluster_xy(1,1), cluster_xy(1,2), 'go', 'MarkerSize', 10, 'LineWidth', 2); % Start point
plot(cluster_xy(end,1), cluster_xy(end,2), 'rs', 'MarkerSize', 10, 'LineWidth', 2); % End point
grid on;
xlabel('X');
ylabel('Y');
title('Cluster Pose Trajectory (X vs Y)');
legend('Trajectory', 'Start', 'End', 'Location', 'best');
axis equal;

% Plot 5: 2D plot of detected critical point (X vs Y)
subplot(3,2,5);
plot(detected_xy(:,1), detected_xy(:,2), 'b-', 'LineWidth', 1.5);
hold on;
plot(detected_xy(1,1), detected_xy(1,2), 'go', 'MarkerSize', 10, 'LineWidth', 2); % Start point
plot(detected_xy(end,1), detected_xy(end,2), 'bs', 'MarkerSize', 10, 'LineWidth', 2); % End point
grid on;
xlabel('X');
ylabel('Y');
title('Detected Critical Point Trajectory (X vs Y)');
legend('Trajectory', 'Start', 'End', 'Location', 'best');
axis equal;

% Plot 6: Combined 2D plot showing both trajectories
subplot(3,2,6);
plot(cluster_xy(:,1), cluster_xy(:,2), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Cluster Pose');
hold on;
plot(detected_xy(:,1), detected_xy(:,2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Detected Point');
% Plot start and end points
plot(cluster_xy(1,1), cluster_xy(1,2), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Start');
plot(cluster_xy(end,1), cluster_xy(end,2), 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'End (Cluster)');
plot(detected_xy(end,1), detected_xy(end,2), 'bs', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'End (Detected)');
% Draw lines connecting corresponding points at specific intervals
interval = 50; % Connect every 50th point
for i = 1:interval:n_points
    plot([cluster_xy(i,1), detected_xy(i,1)], [cluster_xy(i,2), detected_xy(i,2)], ...
         'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
grid on;
xlabel('X');
ylabel('Y');
title('Combined Trajectories (X vs Y)');
legend('Location', 'best');
axis equal;

% Adjust spacing between subplots
sgtitle('Cluster Pose and Critical Point Analysis (First 500 Points)', 'FontSize', 14, 'FontWeight', 'bold');

%% Display Statistics
fprintf('Analysis for first %d data points:\n', n_points);
fprintf('\nDistance Statistics:\n');
fprintf('  Mean distance: %.4f\n', mean(distances));
fprintf('  Max distance: %.4f\n', max(distances));
fprintf('  Min distance: %.4f\n', min(distances));
fprintf('  Std deviation: %.4f\n', std(distances));

fprintf('\nCluster Pose Range:\n');
fprintf('  X range: [%.4f, %.4f]\n', min(cluster_xy(:,1)), max(cluster_xy(:,1)));
fprintf('  Y range: [%.4f, %.4f]\n', min(cluster_xy(:,2)), max(cluster_xy(:,2)));

fprintf('\nDetected Point Range:\n');
fprintf('  X range: [%.4f, %.4f]\n', min(detected_xy(:,1)), max(detected_xy(:,1)));
fprintf('  Y range: [%.4f, %.4f]\n', min(detected_xy(:,2)), max(detected_xy(:,2)));

%% Optional: Create separate figure windows for each plot
% Uncomment if you prefer separate windows instead of subplots

% figure;
% plot(time, distances, 'b-', 'LineWidth', 1.5);
% grid on;
% xlabel('Time');
% ylabel('Distance');
% title('Distance between Cluster Pose and Detected Critical Point over Time');
% 
% figure;
% plot(cluster_xy(:,1), cluster_xy(:,2), 'r-', 'LineWidth', 1.5);
% hold on;
% plot(cluster_xy(1,1), cluster_xy(1,2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
% plot(cluster_xy(end,1), cluster_xy(end,2), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
% grid on;
% xlabel('X');
% ylabel('Y');
% title('Cluster Pose Trajectory (X vs Y)');
% legend('Trajectory', 'Start', 'End');
% axis equal;