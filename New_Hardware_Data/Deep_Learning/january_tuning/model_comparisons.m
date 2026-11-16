%% HSV Analysis and Neural Network Prediction Script
% This script reads 7 datasets, analyzes HSV values, and uses trained neural networks

clear all; close all; clc;

%% Define dataset names and parameters
dataset_names = {'celeste_ver', 'tidal_ver', 'pacific_blue_ver', 'redwood_cal', 'celeste_cal', 'tidal_cal', 'pacific_blue_cal'};
dataset_files = {'celeste_ver.csv', 'tidal_ver.csv', 'pacific_blue_ver.csv', 'redwood_cal.csv', 'celeste_cal.csv', 'tidal_cal.csv', 'pacific_blue_cal.csv'};

%% Load the trained neural networks
fprintf('Loading trained neural networks...\n');
load('best_hue_net.mat', 'everything_hue_net');
load('best_sat_net.mat', 'everything_sat_net');

%% Initialize storage for all data
all_inputs = cell(7, 1);
all_hue_targets = cell(7, 1);
all_sat_targets = cell(7, 1);
all_hsv_calculated = cell(7, 1);

%% Read and process all datasets
fprintf('\nReading and processing datasets...\n');
for i = 1:7
    fprintf('Processing %s...\n', dataset_names{i});
    
    % Read data
    data = readmatrix(dataset_files{i});
    
    % Determine number of rows to use
    if i <= 3  % _ver files
        rows_to_use = 70;
    elseif i == 4  % redwood_cal
        rows_to_use = 480;
    else  % other _cal files
        rows_to_use = 480;  % 20 * 24
    end
    
    % Extract data
    all_inputs{i} = data(1:rows_to_use, 3:6);  % RGBK values
    all_hue_targets{i} = data(1:rows_to_use, 1);  % Hue targets
    all_sat_targets{i} = data(1:rows_to_use, 2);  % Saturation targets
    
    % Normalize RGB values (same as in training script)
    rgb_normalized = zeros(rows_to_use, 3);
    rgb_normalized(:,1) = (all_inputs{i}(:,1)-100)/1500;
    rgb_normalized(:,2) = (all_inputs{i}(:,2)-100)/1500;
    rgb_normalized(:,3) = (all_inputs{i}(:,3)-100)/1500;
    rgb_normalized = max(min(rgb_normalized, 1), 0);
    
    % Calculate HSV from normalized RGB
    all_hsv_calculated{i} = rgb2hsv(rgb_normalized);
end

%% Neural Network Predictions (compute before plotting)
fprintf('\n\nMaking predictions using trained neural networks...\n');

% Initialize storage for predictions
all_hue_predictions = cell(7, 1);
all_sat_predictions = cell(7, 1);

for i = 1:7
    fprintf('Predicting for %s...\n', dataset_names{i});
    
    % Prepare inputs (same normalization as training)
    inputs = all_inputs{i};
    inputs(:,1) = (inputs(:,1)-100)/1500;
    inputs(:,2) = (inputs(:,2)-100)/1500;
    inputs(:,3) = (inputs(:,3)-100)/1500;
    inputs(:,4) = (inputs(:,4)-500)/4500;
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
figure('Name', 'Calculated Hue vs Target Hue and Saturation', 'Position', [100, 100, 1400, 800]);
for i = 1:7
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
figure('Name', 'NN Predicted Hue vs Target Hue and Saturation', 'Position', [100, 100, 1400, 800]);
for i = 1:7
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
figure('Name', 'Calculated Saturation vs Target Hue and Saturation', 'Position', [100, 100, 1400, 800]);
for i = 1:7
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
figure('Name', 'NN Predicted Saturation vs Target Hue and Saturation', 'Position', [100, 100, 1400, 800]);
for i = 1:7
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
fprintf('\n\nPerformance Metrics:\n');
fprintf('%-20s | Hue R² | Sat R² | Calc Hue RMSE | Calc Sat RMSE\n', 'Dataset');
fprintf('%-20s---------+--------+---------------+--------------\n', '--------------------');

for i = 1:7
    % Calculate R-squared for NN predictions
    % Hue (circular R-squared)
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
    
    % Saturation (standard R-squared)
    sat_targets_vec = all_sat_targets{i}(:);
    satPredictions_vec = all_sat_predictions{i}(:);
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    R2_sat = 1 - SSE_sat/SST_sat;
    
    % Calculate RMSE for calculated HSV vs targets
    hue_rmse = sqrt(mean((all_hsv_calculated{i}(:,1) - all_hue_targets{i}).^2));
    sat_rmse = sqrt(mean((all_hsv_calculated{i}(:,2) - all_sat_targets{i}).^2));
    
    fprintf('%-20s | %.4f | %.4f | %.4f        | %.4f\n', ...
            dataset_names{i}, R2_hue, R2_sat, hue_rmse, sat_rmse);
end

%% Create Simulink blocks for the neural networks
% Uncomment the following lines to generate Simulink blocks for the neural networks

gensim(everything_hue_net, 'Name', 'HueNeuralNetwork');

gensim(everything_sat_net, 'Name', 'SaturationNeuralNetwork');

fprintf('\nSimulink blocks created:\n');
fprintf('  - HueNeuralNetwork.slx\n');
fprintf('  - SaturationNeuralNetwork.slx\n');