% MATLAB Script: Analyze critical point detection error across different radii
% This script loads radius01.mat through radius06.mat files
% Calculates error metrics and RMSE from origin (0,0)

clear; clc; close all;

%% Initialize storage variables
num_radii = 6;
radii = 1:num_radii;  % Radius values (01 means 1, 02 means 2, etc.)
n_points = 500;  % Use first 500 data points

% Storage for results
mean_distances = zeros(num_radii, 1);
std_distances = zeros(num_radii, 1);
rmse_from_origin = zeros(num_radii, 1);  % RMSE assuming critical point should be at (0,0)

%% Loop through each radius file
fprintf('Analyzing critical point detection across radii...\n');
fprintf('================================================\n\n');

for r = 1:num_radii
    % Construct filename
    filename = sprintf('radius%02d.mat', r);
    
    fprintf('Processing %s (Radius = %d)...\n', filename, r);
    
    try
        % Load data
        data = load(filename);
        
        % Extract cluster pose (x,y) and detected critical point
        cluster_xy = data.cluster_pose(1:n_points, 1:2);
        detected_xy = data.detected_critical_point(1:n_points, :);
        
        % Calculate Euclidean distances between cluster and detected points
        distances = sqrt((cluster_xy(:,1) - detected_xy(:,1)).^2 + ...
                        (cluster_xy(:,2) - detected_xy(:,2)).^2);
        
        % Calculate distance from detected critical point to origin (0,0)
        distances_from_origin = sqrt(detected_xy(:,1).^2 + detected_xy(:,2).^2);
        
        % Calculate error metrics
        mean_distances(r) = mean(distances);
        std_distances(r) = std(distances);
        
        % Calculate RMSE from origin (0,0)
        rmse_from_origin(r) = sqrt(mean(distances_from_origin.^2));
        
        % Print statistics for this radius
        fprintf('  Mean distance (cluster to detected): %.4f ± %.4f\n', mean_distances(r), std_distances(r));
        fprintf('  RMSE from origin (0,0): %.4f\n\n', rmse_from_origin(r));
        
    catch ME
        fprintf('  Error loading %s: %s\n\n', filename, ME.message);
        % Set NaN for missing data
        mean_distances(r) = NaN;
        std_distances(r) = NaN;
        rmse_from_origin(r) = NaN;
    end
end

%% Create visualizations
fprintf('Creating visualizations...\n\n');

% Figure 1: Error metrics vs radius
figure('Position', [100, 100, 1200, 500]);

% Subplot 1: Mean distance with error bars
subplot(1,2,1);
errorbar(radii, mean_distances, std_distances, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
grid on;
xlabel('Radius', 'FontSize', 12);
ylabel('Distance', 'FontSize', 12);
title('Mean Distance between Cluster and Detected Point', 'FontSize', 14);
xlim([0.5, num_radii+0.5]);
set(gca, 'XTick', radii);

% Subplot 2: RMSE from origin (0,0)
subplot(1,2,2);
plot(radii, rmse_from_origin, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
xlabel('Radius', 'FontSize', 12);
ylabel('RMSE from Origin', 'FontSize', 12);
title('RMSE of Detected Point from Origin (0,0)', 'FontSize', 14);
xlim([0.5, num_radii+0.5]);
set(gca, 'XTick', radii);

sgtitle('Critical Point Detection Error Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% Display summary table
fprintf('\nSummary Statistics:\n');
fprintf('===================\n');
fprintf('Radius | Mean Dist ± Std  | RMSE from (0,0)\n');
fprintf('-------|------------------|----------------\n');
for r = 1:num_radii
    fprintf('   %d   | %.4f ± %.4f | %.4f\n', ...
            r, mean_distances(r), std_distances(r), rmse_from_origin(r));
end

%% Calculate and display trends
fprintf('\n\nTrend Analysis:\n');
fprintf('===============\n');

% Check if error decreases with radius
if ~any(isnan(mean_distances))
    correlation_mean = corr(radii', mean_distances);
    correlation_rmse = corr(radii', rmse_from_origin);
    
    fprintf('Correlation between radius and mean distance: %.4f\n', correlation_mean);
    fprintf('Correlation between radius and RMSE from origin: %.4f\n', correlation_rmse);
    
    if correlation_mean < -0.5
        fprintf('Strong negative correlation: Error decreases with radius\n');
    elseif correlation_mean > 0.5
        fprintf('Strong positive correlation: Error increases with radius\n');
    else
        fprintf('Weak correlation: No clear trend\n');
    end
end

% %% Save results to file (optional)
% save('radius_analysis_results.mat', 'radii', 'mean_distances', 'std_distances', 'rmse_from_origin');
% fprintf('\nResults saved to radius_analysis_results.mat\n');