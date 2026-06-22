%% Combined HSV Neural Network Prediction Analysis
% Publication-quality combined figure for IEEE Transactions
% This script creates a side-by-side comparison of Hue and Saturation predictions
% in a single figure with 1x2 subplot layout
% This file creates figure 6 for IEEE T-Mech Paper

fprintf('========================================\n');
fprintf('Combined HSV Neural Network Predictions\n');
fprintf('Publication Output - Validation Data\n');
fprintf('========================================\n\n');

%% Define dataset names and parameters (VALIDATION ONLY)
dataset_names = {'celeste_ver', 'tidal_ver', 'pacific_blue_ver', 'redwood_ver'};
dataset_files = {'data/celeste_ver.csv', 'data/tidal_ver.csv', 'data/pacific_blue_ver.csv', 'data/redwood_ver.csv'};
num_datasets = 4;

%% Load the trained neural networks
fprintf('Loading trained neural networks...\n');
% Get the directory where this script is located
script_dir = fileparts(mfilename('fullpath'));
nn_path = fullfile(script_dir, 'nn_training', 'everything_nets_final.mat');

if exist(nn_path, 'file') == 2
    load(nn_path, 'everything_hue_net', 'everything_sat_net');
    fprintf('  Loaded from: %s\n', nn_path);
else
    error('Cannot find everything_nets_final.mat at: %s', nn_path);
end

%% Initialize storage for validation data
all_inputs = cell(num_datasets, 1);
all_hue_targets = cell(num_datasets, 1);
all_sat_targets = cell(num_datasets, 1);
all_hue_predictions = cell(num_datasets, 1);
all_sat_predictions = cell(num_datasets, 1);

%% Read and process all validation datasets
fprintf('\nProcessing validation datasets...\n');
for i = 1:num_datasets
    fprintf('  [%d/%d] %s... ', i, num_datasets, dataset_names{i});

    % Read data (use full path relative to script location)
    data_path = fullfile(script_dir, dataset_files{i});
    data = readmatrix(data_path);
    rows_to_use = 70;  % All validation files have 70 rows

    % Extract data
    all_inputs{i} = data(1:rows_to_use, 3:6);  % RGBK values
    all_hue_targets{i} = data(1:rows_to_use, 1);  % Hue targets
    all_sat_targets{i} = data(1:rows_to_use, 2);  % Saturation targets

    fprintf('%d samples\n', rows_to_use);
end

%% Make Neural Network Predictions
fprintf('\nGenerating neural network predictions...\n');

for i = 1:num_datasets
    % Prepare inputs (same normalization as training)
    inputs = all_inputs{i};
    inputs(:,1) = (inputs(:,1)-120)/1300;  % R normalization
    inputs(:,2) = (inputs(:,2)-120)/1300;  % G normalization
    inputs(:,3) = (inputs(:,3)-120)/1300;  % B normalization
    inputs(:,4) = (inputs(:,4)-800)/3500;  % K normalization
    inputs = max(min(inputs, 1), 0);  % Clamp to [0,1]

    % Transpose and normalize to [-1, 1] for network input
    inputs = inputs';
    inputs_normalized = (inputs*2) - 1;

    % Predict Hue (using sin/cos encoding)
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

fprintf('Predictions completed.\n');

%% Combine all validation data for aggregated analysis
fprintf('\nCombining validation datasets...\n');
combined_hue_targets = vertcat(all_hue_targets{:});
combined_sat_targets = vertcat(all_sat_targets{:});
combined_hue_predictions = vertcat(all_hue_predictions{:});
combined_sat_predictions = vertcat(all_sat_predictions{:});

fprintf('  Total validation samples: %d\n', length(combined_hue_targets));

%% Calculate Performance Metrics
fprintf('\n========================================\n');
fprintf('PERFORMANCE METRICS - VALIDATION DATA\n');
fprintf('========================================\n\n');

% Hue metrics (circular statistics)
hue_circular_errors = min(abs(combined_hue_targets - combined_hue_predictions), ...
                          1 - abs(combined_hue_targets - combined_hue_predictions));
hue_mean = mean(combined_hue_targets);
hue_circular_errors_from_mean = min(abs(combined_hue_targets - hue_mean), ...
                                    1 - abs(combined_hue_targets - hue_mean));
hue_circular_SST = sum(hue_circular_errors_from_mean.^2);
hue_circular_SSE = sum(hue_circular_errors.^2);
R2_hue = 1 - hue_circular_SSE/hue_circular_SST;
RMSE_hue = sqrt(mean(hue_circular_errors.^2));

% Saturation metrics (standard statistics)
sat_SST = sum((combined_sat_targets - mean(combined_sat_targets)).^2);
sat_SSE = sum((combined_sat_targets - combined_sat_predictions).^2);
R2_sat = 1 - sat_SSE/sat_SST;
RMSE_sat = sqrt(mean((combined_sat_targets - combined_sat_predictions).^2));

fprintf('HUE PREDICTION:\n');
fprintf('  R² (circular):  %.4f\n', R2_hue);
fprintf('  RMSE (circular): %.4f\n', RMSE_hue);
fprintf('  Mean absolute error: %.4f\n', mean(hue_circular_errors));
fprintf('\n');

fprintf('SATURATION PREDICTION:\n');
fprintf('  R²:              %.4f\n', R2_sat);
fprintf('  RMSE:            %.4f\n', RMSE_sat);
fprintf('  Mean absolute error: %.4f\n', mean(abs(combined_sat_targets - combined_sat_predictions)));
fprintf('\n');

%% Create Combined Publication-Quality Figure
fprintf('Creating combined HSV prediction figure...\n');

% Create figure with 1x2 subplot layout (wider for side-by-side)
%fig = figure('Name', 'Combined HSV Neural Network Predictions', ...
%             'Units', 'inches', 'Position', [1, 1, 7, 3.5]);
fig = figure()
%% Subplot 1: Hue Prediction
subplot(1, 2, 1);

% Scatter plot with transparency
scatter(combined_hue_targets, combined_hue_predictions, 20, [0.2 0.4 0.8], ...
        'filled', 'MarkerFaceAlpha', 0.5);
hold on;

% Perfect prediction reference line
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);

% Formatting
xlabel('target', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('predicted', 'FontSize', 16, 'FontWeight', 'bold');
title('Hue', 'FontSize', 18, 'FontWeight', 'bold');
grid on;
axis equal;
xlim([0 1]);
ylim([0 1]);
set(gca, 'XTick', 0:0.1:1, 'YTick', 0:0.1:1);

% Add metrics text box
textStr = sprintf('R^2 = %.3f\nRMSE = %.3f\nn = %d', R2_hue, RMSE_hue, length(combined_hue_targets));
text(0.05, 0.95, textStr, 'Units', 'normalized', ...
     'FontSize', 13, 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'VerticalAlignment', 'top', 'FontWeight', 'bold');

% Legend
legend('Validation Data', 'Perfect Prediction', 'Location', 'southeast', 'FontSize', 13, 'FontWeight', 'bold');

% Fine-tune appearance
set(gca, 'FontSize', 14, 'LineWidth', 1, 'FontWeight', 'bold');
box on;

hold off;

%% Subplot 2: Saturation Prediction
subplot(1, 2, 2);

% Scatter plot with transparency
scatter(combined_sat_targets, combined_sat_predictions, 20, [0.8 0.2 0.4], ...
        'filled', 'MarkerFaceAlpha', 0.5);
hold on;

% Perfect prediction reference line
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);

% Formatting
xlabel('target', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('predicted', 'FontSize', 16, 'FontWeight', 'bold');
title('Saturation', 'FontSize', 18, 'FontWeight', 'bold');
grid on;
axis equal;
xlim([0 1]);
ylim([0 1]);
set(gca, 'XTick', 0:0.1:1, 'YTick', 0:0.1:1);

% Add metrics text box
textStr = sprintf('R^2 = %.3f\nRMSE = %.3f\nn = %d', R2_sat, RMSE_sat, length(combined_sat_targets));
text(0.05, 0.95, textStr, 'Units', 'normalized', ...
     'FontSize', 13, 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'VerticalAlignment', 'top', 'FontWeight', 'bold');

% Legend
legend('Validation Data', 'Perfect Prediction', 'Location', 'southeast', 'FontSize', 13, 'FontWeight', 'bold');

% Fine-tune appearance
set(gca, 'FontSize', 14, 'LineWidth', 1, 'FontWeight', 'bold');
box on;

hold off;

%% Add overall title (optional)
%sgtitle('Neural Network Color Sensor HSV Predictions', 'FontSize', 18, 'FontWeight', 'bold');

%% Adjust subplot spacing for better appearance
% Reduce white space between subplots
set(fig, 'PaperPositionMode', 'auto');
fig.Position = [100 100 900 400];  % Adjust figure size for better aspect ratio

%% Save high-resolution figures
print(fig, 'color_sensor_hsv_predictions', '-dpng', '-r300');
savefig(fig, 'color_sensor_hsv_predictions.fig');
fprintf('  Saved: color_sensor_hsv_predictions.png (300 DPI)\n');
fprintf('  Saved: color_sensor_hsv_predictions.fig\n');

%% Summary
fprintf('\n========================================\n');
fprintf('COMBINED FIGURE GENERATION COMPLETE\n');
fprintf('========================================\n\n');
fprintf('Output files generated:\n');
fprintf('  1. color_sensor_hsv_predictions.png - Combined plot (300 DPI)\n');
fprintf('  2. color_sensor_hsv_predictions.fig - Combined plot (MATLAB figure)\n');
fprintf('\n');
fprintf('Figure specifications:\n');
fprintf('  - Layout: 1x2 subplot (side-by-side)\n');
fprintf('  - Size: 7" x 3.5" (IEEE two-column format)\n');
fprintf('  - Resolution: 300 DPI (publication quality)\n');
fprintf('  - Data: Validation datasets only (n = %d samples)\n', length(combined_hue_targets));
fprintf('  - Left panel: Hue prediction (blue markers)\n');
fprintf('  - Right panel: Saturation prediction (red markers)\n');
fprintf('\n');