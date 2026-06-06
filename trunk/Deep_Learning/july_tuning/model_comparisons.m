%% HSV Analysis and Neural Network Prediction Script
% This script reads 8 datasets, analyzes HSV values, and uses trained neural networks

%clear all; close all; clc;

%% Define dataset names and parameters
dataset_names = {'celeste_ver', 'tidal_ver', 'pacific_blue_ver', 'redwood_ver', 'redwood_cal', 'celeste_cal', 'tidal_cal', 'pacific_blue_cal'};
dataset_files = {'data/celeste_ver.csv', 'data/tidal_ver.csv', 'data/pacific_blue_ver.csv', 'data/redwood_ver.csv', 'data/redwood_cal.csv', 'data/celeste_cal.csv', 'data/tidal_cal.csv', 'data/pacific_blue_cal.csv'};

%% Load the trained neural networks
fprintf('Loading trained neural networks...\n');
% Get the directory where this script is located
script_dir = fileparts(mfilename('fullpath'));
nn_path = fullfile(script_dir, 'nn_training', 'everything_nets_final.mat');

if exist(nn_path, 'file') == 2
    load(nn_path, 'everything_hue_net', 'everything_sat_net');
else
    error('Cannot find everything_nets_final.mat at: %s', nn_path);
end

%% Initialize storage for all data
all_inputs = cell(8, 1);
all_hue_targets = cell(8, 1);
all_sat_targets = cell(8, 1);
all_hsv_calculated = cell(8, 1);

%% Read and process all datasets
fprintf('\nReading and processing datasets...\n');
for i = 1:8
    fprintf('Processing %s...\n', dataset_names{i});

    % Read data (use full path relative to script location)
    data_path = fullfile(script_dir, dataset_files{i});
    data = readmatrix(data_path);
    
    % Determine number of rows to use
    if i <= 4  % _ver files
        rows_to_use = 70;
    elseif i == 5  % redwood_cal
        rows_to_use = 480;
    else  % other _cal files (celeste_cal, tidal_cal, pacific_blue_cal)
        rows_to_use = 480;  % 20 * 24
    end
    
    % Extract data
    all_inputs{i} = data(1:rows_to_use, 3:6);  % RGBK values
    all_hue_targets{i} = data(1:rows_to_use, 1);  % Hue targets
    all_sat_targets{i} = data(1:rows_to_use, 2);  % Saturation targets
    
    % Normalize RGB values (matching training script normalization)
    rgb_normalized = zeros(rows_to_use, 3);
    rgb_normalized(:,1) = (all_inputs{i}(:,1)-120)/1300;
    rgb_normalized(:,2) = (all_inputs{i}(:,2)-120)/1300;
    rgb_normalized(:,3) = (all_inputs{i}(:,3)-120)/1300;
    rgb_normalized = max(min(rgb_normalized, 1), 0);
    
    % Calculate HSV from normalized RGB
    all_hsv_calculated{i} = rgb2hsv(rgb_normalized);
end

%% Neural Network Predictions (compute before plotting)
fprintf('\n\nMaking predictions using trained neural networks...\n');

% Initialize storage for predictions
all_hue_predictions = cell(8, 1);
all_sat_predictions = cell(8, 1);

for i = 1:8
    fprintf('Predicting for %s...\n', dataset_names{i});
    
    % Prepare inputs (same normalization as training)
    inputs = all_inputs{i};
    inputs(:,1) = (inputs(:,1)-120)/1300;
    inputs(:,2) = (inputs(:,2)-120)/1300;
    inputs(:,3) = (inputs(:,3)-120)/1300;
    inputs(:,4) = (inputs(:,4)-800)/3500;
    inputs = max(min(inputs, 1), 0);
    
    % Transpose and normalize to [-1, 1]
    inputs = inputs';
    inputs_normalized = (inputs*2) - 1;
    
    % Predict Hue
    huePredictionsNormalized = everything_hue_net(inputs_normalized);
    huePredictions_sin = (huePredictionsNormalized(1,:)+1)/2;
    huePredictions_cos = (huePredictionsNormalized(2,:)+1)/2;
    huePredictions = atan2(huePredictions_sin, huePredictions_cos);
    huePredictions(huePredictions < 0) = huePredictions(huePredictions < 0) + 2 * pi;
    huePredictions = huePredictions / (2 * pi);
    all_hue_predictions{i} = huePredictions';
    
    % Predict Saturation
    satPredictionsNormalized = everything_sat_net(inputs_normalized);
    satPredictions = (satPredictionsNormalized+1)/2;
    satPredictions = max(min(satPredictions, 1), 0);
    all_sat_predictions{i} = satPredictions';
end

%% Figure 1: 3D scatter plots of calculated Hue vs target Hue and Sat
figure('Name', 'Calculated Hue vs Target Hue and Saturation', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter3(all_hue_targets{i}, all_sat_targets{i}, all_hsv_calculated{i}(:,1), 20, 'filled');
    xlabel('Hue Target');
    ylabel('Saturation Target');
    zlabel('Calculated Hue');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    view(45, 30);
    colorbar;
end
sgtitle('Figure 1: Calculated Hue (from RGB) vs Target Values');

%% Figure 2: 3D scatter plots of NN predicted Hue vs target Hue and Sat
figure('Name', 'NN Predicted Hue vs Target Hue and Saturation', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter3(all_hue_targets{i}, all_sat_targets{i}, all_hue_predictions{i}, 20, 'filled');
    xlabel('Hue Target');
    ylabel('Saturation Target');
    zlabel('NN Predicted Hue');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    view(45, 30);
    colorbar;
end
sgtitle('Figure 2: Neural Network Predicted Hue vs Target Values');

%% Figure 3: 3D scatter plots of calculated Saturation vs target Hue and Sat
figure('Name', 'Calculated Saturation vs Target Hue and Saturation', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter3(all_hue_targets{i}, all_sat_targets{i}, all_hsv_calculated{i}(:,2), 20, 'filled');
    xlabel('Hue Target');
    ylabel('Saturation Target');
    zlabel('Calculated Saturation');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    view(45, 30);
    colorbar;
end
sgtitle('Figure 3: Calculated Saturation (from RGB) vs Target Values');

%% Figure 4: 3D scatter plots of NN predicted Saturation vs target Hue and Sat
figure('Name', 'NN Predicted Saturation vs Target Hue and Saturation', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter3(all_hue_targets{i}, all_sat_targets{i}, all_sat_predictions{i}, 20, 'filled');
    xlabel('Hue Target');
    ylabel('Saturation Target');
    zlabel('NN Predicted Saturation');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    view(45, 30);
    colorbar;
end
sgtitle('Figure 4: Neural Network Predicted Saturation vs Target Values');

%% Calculate and display performance metrics
fprintf('\n\nPerformance Metrics Comparison:\n');
fprintf('%-20s | NN Hue R² | NN Sat R² | NN Hue RMSE | NN Sat RMSE | Calc Hue R² | Calc Sat R² | Calc Hue RMSE | Calc Sat RMSE\n', 'Dataset');
fprintf('%-20s------------+-----------+-------------+-------------+-------------+-------------+---------------+--------------\n', '--------------------');

for i = 1:8
    % === Neural Network Metrics ===
    % Hue (circular R-squared and RMSE)
    hue_targets_vec = all_hue_targets{i}(:);
    huePredictions_vec = all_hue_predictions{i}(:);
    nn_circular_errors = min(abs(hue_targets_vec - huePredictions_vec), ...
                         1 - abs(hue_targets_vec - huePredictions_vec));
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    nn_circular_SSE_hue = sum(nn_circular_errors.^2);
    nn_R2_hue = 1 - nn_circular_SSE_hue/circular_SST_hue;
    nn_hue_rmse = sqrt(mean(nn_circular_errors.^2));
    
    % Saturation (standard R-squared and RMSE)
    sat_targets_vec = all_sat_targets{i}(:);
    satPredictions_vec = all_sat_predictions{i}(:);
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    nn_SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    nn_R2_sat = 1 - nn_SSE_sat/SST_sat;
    nn_sat_rmse = sqrt(mean((sat_targets_vec - satPredictions_vec).^2));
    
    % === RGB->HSV Calculation Metrics ===
    % Hue (circular R-squared and RMSE)
    calc_hue_vec = all_hsv_calculated{i}(:,1);
    calc_circular_errors = min(abs(hue_targets_vec - calc_hue_vec), ...
                         1 - abs(hue_targets_vec - calc_hue_vec));
    calc_circular_SSE_hue = sum(calc_circular_errors.^2);
    calc_R2_hue = 1 - calc_circular_SSE_hue/circular_SST_hue;
    calc_hue_rmse = sqrt(mean(calc_circular_errors.^2));
    
    % Saturation (standard R-squared and RMSE)
    calc_sat_vec = all_hsv_calculated{i}(:,2);
    calc_SSE_sat = sum((sat_targets_vec - calc_sat_vec).^2);
    calc_R2_sat = 1 - calc_SSE_sat/SST_sat;
    calc_sat_rmse = sqrt(mean((sat_targets_vec - calc_sat_vec).^2));
    
    fprintf('%-20s | %.4f    | %.4f    | %.4f      | %.4f      | %.4f      | %.4f      | %.4f        | %.4f\n', ...
            dataset_names{i}, nn_R2_hue, nn_R2_sat, nn_hue_rmse, nn_sat_rmse, ...
            calc_R2_hue, calc_R2_sat, calc_hue_rmse, calc_sat_rmse);
end

%% Summary statistics
fprintf('\n\nSummary - Average Performance:\n');
fprintf('%-20s | Hue R² | Sat R² | Hue RMSE | Sat RMSE\n', 'Method');
fprintf('%-20s---------+--------+----------+---------\n', '--------------------');

% Calculate averages
nn_hue_r2_vals = zeros(8, 1);
nn_sat_r2_vals = zeros(8, 1);
nn_hue_rmse_vals = zeros(8, 1);
nn_sat_rmse_vals = zeros(8, 1);
calc_hue_r2_vals = zeros(8, 1);
calc_sat_r2_vals = zeros(8, 1);
calc_hue_rmse_vals = zeros(8, 1);
calc_sat_rmse_vals = zeros(8, 1);

for i = 1:8
    % Recalculate for averages (same calculations as above)
    hue_targets_vec = all_hue_targets{i}(:);
    sat_targets_vec = all_sat_targets{i}(:);
    hue_mean = mean(hue_targets_vec);
    
    % NN metrics
    huePredictions_vec = all_hue_predictions{i}(:);
    satPredictions_vec = all_sat_predictions{i}(:);
    nn_circular_errors = min(abs(hue_targets_vec - huePredictions_vec), ...
                         1 - abs(hue_targets_vec - huePredictions_vec));
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    nn_circular_SSE_hue = sum(nn_circular_errors.^2);
    nn_hue_r2_vals(i) = 1 - nn_circular_SSE_hue/circular_SST_hue;
    nn_hue_rmse_vals(i) = sqrt(mean(nn_circular_errors.^2));
    
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    nn_SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    nn_sat_r2_vals(i) = 1 - nn_SSE_sat/SST_sat;
    nn_sat_rmse_vals(i) = sqrt(mean((sat_targets_vec - satPredictions_vec).^2));
    
    % Calc metrics
    calc_hue_vec = all_hsv_calculated{i}(:,1);
    calc_sat_vec = all_hsv_calculated{i}(:,2);
    calc_circular_errors = min(abs(hue_targets_vec - calc_hue_vec), ...
                         1 - abs(hue_targets_vec - calc_hue_vec));
    calc_circular_SSE_hue = sum(calc_circular_errors.^2);
    calc_hue_r2_vals(i) = 1 - calc_circular_SSE_hue/circular_SST_hue;
    calc_hue_rmse_vals(i) = sqrt(mean(calc_circular_errors.^2));
    
    calc_SSE_sat = sum((sat_targets_vec - calc_sat_vec).^2);
    calc_sat_r2_vals(i) = 1 - calc_SSE_sat/SST_sat;
    calc_sat_rmse_vals(i) = sqrt(mean((sat_targets_vec - calc_sat_vec).^2));
end

fprintf('Neural Network       | %.4f | %.4f | %.4f   | %.4f\n', ...
        mean(nn_hue_r2_vals), mean(nn_sat_r2_vals), ...
        mean(nn_hue_rmse_vals), mean(nn_sat_rmse_vals));
fprintf('RGB->HSV Calculation | %.4f | %.4f | %.4f   | %.4f\n', ...
        mean(calc_hue_r2_vals), mean(calc_sat_r2_vals), ...
        mean(calc_hue_rmse_vals), mean(calc_sat_rmse_vals));

fprintf('\nImprovement (NN vs Calc):\n');
fprintf('Hue R² improvement: %.1f%%\n', 100 * (mean(nn_hue_r2_vals) - mean(calc_hue_r2_vals)) / (1 - mean(calc_hue_r2_vals)));
fprintf('Sat R² improvement: %.1f%%\n', 100 * (mean(nn_sat_r2_vals) - mean(calc_sat_r2_vals)) / (1 - mean(calc_sat_r2_vals)));
fprintf('Hue RMSE reduction: %.1f%%\n', 100 * (mean(calc_hue_rmse_vals) - mean(nn_hue_rmse_vals)) / mean(calc_hue_rmse_vals));
fprintf('Sat RMSE reduction: %.1f%%\n', 100 * (mean(calc_sat_rmse_vals) - mean(nn_sat_rmse_vals)) / mean(calc_sat_rmse_vals));

%% Display comparison of validation vs calibration performance
fprintf('\n\nValidation vs Calibration Performance:\n');
fprintf('%-20s | Avg Hue R² | Avg Sat R²\n', 'Dataset Type');
fprintf('%-20s-------------+-----------\n', '--------------------');

% Calculate averages for validation datasets (1-4)
val_hue_scores = zeros(4, 1);
val_sat_scores = zeros(4, 1);
for i = 1:4
    % Recalculate scores for consistency
    hue_targets_vec = all_hue_targets{i}(:);
    huePredictions_vec = all_hue_predictions{i}(:);
    circular_errors = min(abs(hue_targets_vec - huePredictions_vec), ...
                         1 - abs(hue_targets_vec - huePredictions_vec));
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    val_hue_scores(i) = 1 - circular_SSE_hue/circular_SST_hue;
    
    sat_targets_vec = all_sat_targets{i}(:);
    satPredictions_vec = all_sat_predictions{i}(:);
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    val_sat_scores(i) = 1 - SSE_sat/SST_sat;
end

% Calculate averages for calibration datasets (5-8)
cal_hue_scores = zeros(4, 1);
cal_sat_scores = zeros(4, 1);
for i = 5:8
    % Recalculate scores for consistency
    hue_targets_vec = all_hue_targets{i}(:);
    huePredictions_vec = all_hue_predictions{i}(:);
    circular_errors = min(abs(hue_targets_vec - huePredictions_vec), ...
                         1 - abs(hue_targets_vec - huePredictions_vec));
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    cal_hue_scores(i-4) = 1 - circular_SSE_hue/circular_SST_hue;
    
    sat_targets_vec = all_sat_targets{i}(:);
    satPredictions_vec = all_sat_predictions{i}(:);
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    cal_sat_scores(i-4) = 1 - SSE_sat/SST_sat;
end

fprintf('Validation (_ver)    | %.4f     | %.4f\n', mean(val_hue_scores), mean(val_sat_scores));
fprintf('Calibration (_cal)   | %.4f     | %.4f\n', mean(cal_hue_scores), mean(cal_sat_scores));

%% Create Simulink blocks for the neural networks
% % Uncomment the following lines to generate Simulink blocks for the neural networks
% 
% % Generate Simulink block for Hue network
% gensim(everything_hue_net, 'Name', 'HueNeuralNetwork');
% 
% % Generate Simulink block for Saturation network  
% gensim(everything_sat_net, 'Name', 'SaturationNeuralNetwork');
% 
% fprintf('\nSimulink blocks created:\n');
% fprintf('  - HueNeuralNetwork.slx\n');
% fprintf('  - SaturationNeuralNetwork.slx\n');

%% New code

%% Figure 5: 2D plots of Target Hue vs Calculated Hue
figure('Name', 'Target Hue vs Calculated Hue', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter(all_hue_targets{i}, all_hsv_calculated{i}(:,1), 20, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    % Add diagonal reference line
    plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Target Hue');
    ylabel('Calculated Hue');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    axis equal;
    xlim([0 1]);
    ylim([0 1]);
    % Add R² value to plot
    hue_targets_vec = all_hue_targets{i}(:);
    calc_hue_vec = all_hsv_calculated{i}(:,1);
    circular_errors = min(abs(hue_targets_vec - calc_hue_vec), ...
                         1 - abs(hue_targets_vec - calc_hue_vec));
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
    text(0.05, 0.95, sprintf('R² = %.4f', R2_hue), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end
sgtitle('Figure 5: Target Hue vs Calculated Hue (from RGB)');

%% Figure 6: 2D plots of Target Hue vs NN Predicted Hue
figure('Name', 'Target Hue vs NN Predicted Hue', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter(all_hue_targets{i}, all_hue_predictions{i}, 20, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    % Add diagonal reference line
    plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Target Hue');
    ylabel('NN Predicted Hue');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    axis equal;
    xlim([0 1]);
    ylim([0 1]);
    % Add R² value to plot
    hue_targets_vec = all_hue_targets{i}(:);
    huePredictions_vec = all_hue_predictions{i}(:);
    circular_errors = min(abs(hue_targets_vec - huePredictions_vec), ...
                         1 - abs(hue_targets_vec - huePredictions_vec));
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
    text(0.05, 0.95, sprintf('R² = %.4f', R2_hue), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end
sgtitle('Figure 6: Target Hue vs Neural Network Predicted Hue');

%% Figure 7: 2D plots of Target Saturation vs Calculated Saturation
figure('Name', 'Target Saturation vs Calculated Saturation', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter(all_sat_targets{i}, all_hsv_calculated{i}(:,2), 20, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    % Add diagonal reference line
    plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Target Saturation');
    ylabel('Calculated Saturation');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    axis equal;
    xlim([0 1]);
    ylim([0 1]);
    % Add R² value to plot
    sat_targets_vec = all_sat_targets{i}(:);
    calc_sat_vec = all_hsv_calculated{i}(:,2);
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - calc_sat_vec).^2);
    R2_sat = 1 - SSE_sat/SST_sat;
    text(0.05, 0.95, sprintf('R² = %.4f', R2_sat), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end
sgtitle('Figure 7: Target Saturation vs Calculated Saturation (from RGB)');

%% Figure 8: 2D plots of Target Saturation vs NN Predicted Saturation
figure('Name', 'Target Saturation vs NN Predicted Saturation', 'Position', [100, 100, 1600, 900]);
for i = 1:8
    subplot(2, 4, i);
    scatter(all_sat_targets{i}, all_sat_predictions{i}, 20, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    % Add diagonal reference line
    plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
    hold off;
    xlabel('Target Saturation');
    ylabel('NN Predicted Saturation');
    title(strrep(dataset_names{i}, '_', '\_'));
    grid on;
    axis equal;
    xlim([0 1]);
    ylim([0 1]);
    % Add R² value to plot
    sat_targets_vec = all_sat_targets{i}(:);
    satPredictions_vec = all_sat_predictions{i}(:);
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    R2_sat = 1 - SSE_sat/SST_sat;
    text(0.05, 0.95, sprintf('R² = %.4f', R2_sat), 'Units', 'normalized', ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end
sgtitle('Figure 8: Target Saturation vs Neural Network Predicted Saturation');