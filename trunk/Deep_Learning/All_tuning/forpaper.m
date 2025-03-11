%% Create and Train Feedforward Neural Networks for Hue and Saturation
% Copy this entire directory, then update the files individually as needed.
% Written by Christopher Waight
% Last update on Jan 29, 2025

%% Load Relevant models and data for Verification
clc; clear all;
load("everything_nets.mat");  % Load the pink neural networks

data1 = readmatrix("redwood_cal.csv");

num_lines = 480;
inputs = [data1(1:num_lines,3:6)];

hue_targets = [data1(1:num_lines,1)];
hue_targets_sin = sin(2 * pi * hue_targets);
hue_targets_cos = cos(2 * pi * hue_targets);

sat_targets = [data1(1:num_lines,2)];

%% Normalize data
inputs(:,1) = (inputs(:,1)-100)/1500;
inputs(:,2) = (inputs(:,2)-100)/1500;
inputs(:,3) = (inputs(:,3)-100)/1500;
inputs(:,4) = (inputs(:,4)-500)/4500;
inputs = max(min(inputs, 1), 0);

% Some feature Engineering
hsv = rgb2hsv(inputs(:,1:3));
inputs(:,5:7) = hsv; 
inputs = max(min(inputs, 1), 0); 

%% Transpose the inputs to make NN compatible
inputs = inputs';
hue_targets_sin = hue_targets_sin';
hue_targets_cos = hue_targets_cos';
sat_targets = sat_targets';

%% Normalizing the Target Data between [-1 and 1]
% Normalize the input data
inputs = (inputs*2) -1;
hue_targets_sin = (hue_targets_sin*2) -1;
hue_targets_cos = (hue_targets_cos*2) -1;
sat_targets = (sat_targets*2) -1;

%% Predict Hue values
allHuePredictionsNormalized = everything_hue_net(inputs);
allHuePredictions_sin = (allHuePredictionsNormalized(1,:)+1)/2;
allHuePredictions_cos = (allHuePredictionsNormalized(2,:)+1)/2;

% Bringing back into a single value from 0 to 1
allHuePredictions = atan2(allHuePredictions_sin, allHuePredictions_cos);
allHuePredictions(allHuePredictions < 0) = allHuePredictions(allHuePredictions < 0) + 2 * pi;
allHuePredictions = allHuePredictions / (2 * pi);

%% Predict Saturation values
allSatPredictionsNormalized = everything_sat_net(inputs);
allSatPredictions = (allSatPredictionsNormalized+1)/2;
allSatPredictions(allSatPredictions>1)=1;
sat_targets = (sat_targets+1)/2;

%% Plotting Results Side by Side
figure('Position', [100, 100, 1200, 500]);

% Hue Plot - Left Side
subplot(1, 2, 1);
scatter(hue_targets, allHuePredictions, 'o'); % Scatter plot of actual vs. predicted
hold on;
plot([0, 1], [0, 1], 'k--'); % Line for perfect prediction (black dashed)
hold off;
xlabel('Actual Hue');
ylabel('Predicted Hue');
title('Predicted vs. Actual Hue');
legend('Predictions', 'Perfect Prediction Line', 'Location', 'northwest');
grid on;

% Saturation Plot - Right Side
subplot(1, 2, 2);
scatter(sat_targets, allSatPredictions, 'o'); % Scatter plot of actual vs. predicted
hold on;
plot([0.3, 1], [0.3, 1], 'k--'); % Line for perfect prediction (black dashed)
hold off;
xlabel('Actual Sat');
ylabel('Predicted Sat');
title('Predicted vs. Actual Saturation');
legend('Predictions', 'Perfect Prediction', 'Location', 'northwest');
grid on;

%% Some Statistics
fprintf('\n--- Hue Model Metrics ---\n');

% Regular metrics for Hue
% Ensure all data is in the same format (vectors)
hue_targets_vec = hue_targets(:);
allHuePredictions_vec = allHuePredictions(:);

% Standard R-squared calculation
SST_hue = sum((hue_targets_vec - mean(hue_targets_vec)).^2);
SSE_hue = sum((hue_targets_vec - allHuePredictions_vec).^2);
R2_hue = 1 - SSE_hue/SST_hue;
fprintf('R-squared: %.4f\n', R2_hue);

% Standard RMSE calculation
RMSE_hue = sqrt(mean((hue_targets_vec - allHuePredictions_vec).^2));
fprintf('RMSE: %.4f\n', RMSE_hue);

% Circular metrics for Hue
circular_errors = min(abs(hue_targets_vec - allHuePredictions_vec), ...
                      1 - abs(hue_targets_vec - allHuePredictions_vec));

% Circular RMSE
circular_RMSE_hue = sqrt(mean(circular_errors.^2));
fprintf('Circular RMSE: %.4f\n', circular_RMSE_hue);

% Circular R-squared calculation
hue_mean = mean(hue_targets_vec);
circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                               1 - abs(hue_targets_vec - hue_mean));
                           
circular_SST_hue = sum(circular_errors_from_mean.^2);
circular_SSE_hue = sum(circular_errors.^2);

circular_R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
fprintf('Circular R-squared: %.4f\n', circular_R2_hue);

fprintf('\n--- Saturation Model Metrics ---\n');

% Metrics for Saturation (non-circular)
sat_targets_vec = sat_targets(:);
allSatPredictions_vec = allSatPredictions(:);

% Standard R-squared calculation
SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
SSE_sat = sum((sat_targets_vec - allSatPredictions_vec).^2);
R2_sat = 1 - SSE_sat/SST_sat;
fprintf('R-squared: %.4f\n', R2_sat);

% Standard RMSE calculation
RMSE_sat = sqrt(mean((sat_targets_vec - allSatPredictions_vec).^2));
fprintf('RMSE: %.4f\n', RMSE_sat);