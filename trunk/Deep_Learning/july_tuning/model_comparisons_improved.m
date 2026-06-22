%% Improved HSV Neural Network Prediction Analysis
% Publication-quality plots for IEEE Transactions in Mechatronics
% This script analyzes NN prediction performance on validation datasets only
% and generates clean, publication-ready figures (3.5" single column format)

% Start diary to capture all command window output
diary('model_comparison_results.txt');
diary on;

fprintf('========================================\n');
fprintf('Neural Network HSV Prediction Analysis\n');
fprintf('Publication Output - Validation Data Only\n');
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
fprintf('  Max absolute error:  %.4f\n', max(hue_circular_errors));
fprintf('\n');

fprintf('SATURATION PREDICTION:\n');
fprintf('  R²:              %.4f\n', R2_sat);
fprintf('  RMSE:            %.4f\n', RMSE_sat);
fprintf('  Mean absolute error: %.4f\n', mean(abs(combined_sat_targets - combined_sat_predictions)));
fprintf('  Max absolute error:  %.4f\n', max(abs(combined_sat_targets - combined_sat_predictions)));
fprintf('\n');

%% Individual Dataset Performance
fprintf('========================================\n');
fprintf('INDIVIDUAL DATASET PERFORMANCE\n');
fprintf('========================================\n\n');
fprintf('%-20s | Hue R²  | Hue RMSE | Sat R²  | Sat RMSE | Samples\n', 'Dataset');
fprintf('%-20s---------+----------+---------+----------+--------\n', '--------------------');

for i = 1:num_datasets
    % Hue metrics
    hue_t = all_hue_targets{i};
    hue_p = all_hue_predictions{i};
    circ_err = min(abs(hue_t - hue_p), 1 - abs(hue_t - hue_p));
    hue_m = mean(hue_t);
    circ_err_mean = min(abs(hue_t - hue_m), 1 - abs(hue_t - hue_m));
    hue_SST = sum(circ_err_mean.^2);
    hue_SSE = sum(circ_err.^2);
    r2_h = 1 - hue_SSE/hue_SST;
    rmse_h = sqrt(mean(circ_err.^2));

    % Saturation metrics
    sat_t = all_sat_targets{i};
    sat_p = all_sat_predictions{i};
    sat_SST_local = sum((sat_t - mean(sat_t)).^2);
    sat_SSE_local = sum((sat_t - sat_p).^2);
    r2_s = 1 - sat_SSE_local/sat_SST_local;
    rmse_s = sqrt(mean((sat_t - sat_p).^2));

    fprintf('%-20s | %.4f  | %.4f   | %.4f  | %.4f   | %d\n', ...
            dataset_names{i}, r2_h, rmse_h, r2_s, rmse_s, length(hue_t));
end

fprintf('\n');

%% Publication-Quality Figure 1: Hue Prediction
fprintf('Generating publication-quality figures...\n');
fprintf('  Creating Figure 1: Hue Prediction...\n');

fig1 = figure('Name', 'NN Hue Prediction - Validation Data', ...
              'Units', 'inches', 'Position', [1, 1, 3.5, 3.5]);

% Scatter plot with transparency
scatter(combined_hue_targets, combined_hue_predictions, 25, [0.2 0.4 0.8], ...
        'filled', 'MarkerFaceAlpha', 0.5);
hold on;

% Perfect prediction reference line
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);

% Formatting
xlabel('Target Hue', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Predicted Hue', 'FontSize', 11, 'FontWeight', 'bold');
title('Neural Network Hue Prediction', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
axis equal;
xlim([0 1]);
ylim([0 1]);

% Add metrics text box
textStr = sprintf('R^2 = %.4f\nRMSE = %.4f\nn = %d', R2_hue, RMSE_hue, length(combined_hue_targets));
text(0.05, 0.88, textStr, 'Units', 'normalized', ...
     'FontSize', 9, 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'VerticalAlignment', 'top');

% Legend
legend('Validation Data', 'Perfect Prediction', 'Location', 'southeast', 'FontSize', 9);

% Fine-tune appearance
set(gca, 'FontSize', 10, 'LineWidth', 1);
box on;

hold off;

% Save high-resolution figure
print(fig1, 'hue_prediction_validation', '-dpng', '-r300');
savefig(fig1, 'hue_prediction_validation.fig');
fprintf('    Saved: hue_prediction_validation.png (300 DPI)\n');
fprintf('    Saved: hue_prediction_validation.fig\n');

%% Publication-Quality Figure 2: Saturation Prediction
fprintf('  Creating Figure 2: Saturation Prediction...\n');

fig2 = figure('Name', 'NN Saturation Prediction - Validation Data', ...
              'Units', 'inches', 'Position', [1, 1, 3.5, 3.5]);

% Scatter plot with transparency
scatter(combined_sat_targets, combined_sat_predictions, 25, [0.8 0.2 0.4], ...
        'filled', 'MarkerFaceAlpha', 0.5);
hold on;

% Perfect prediction reference line
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);

% Formatting
xlabel('Target Saturation', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Predicted Saturation', 'FontSize', 11, 'FontWeight', 'bold');
title('Neural Network Saturation Prediction', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
axis equal;
xlim([0 1]);
ylim([0 1]);

% Add metrics text box
textStr = sprintf('R^2 = %.4f\nRMSE = %.4f\nn = %d', R2_sat, RMSE_sat, length(combined_sat_targets));
text(0.05, 0.88, textStr, 'Units', 'normalized', ...
     'FontSize', 9, 'BackgroundColor', 'white', ...
     'EdgeColor', 'black', 'VerticalAlignment', 'top');

% Legend
legend('Validation Data', 'Perfect Prediction', 'Location', 'southeast', 'FontSize', 9);

% Fine-tune appearance
set(gca, 'FontSize', 10, 'LineWidth', 1);
box on;

hold off;

% Save high-resolution figure
print(fig2, 'saturation_prediction_validation', '-dpng', '-r300');
savefig(fig2, 'saturation_prediction_validation.fig');
fprintf('    Saved: saturation_prediction_validation.png (300 DPI)\n');
fprintf('    Saved: saturation_prediction_validation.fig\n');

%% Summary
fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE\n');
fprintf('========================================\n\n');
fprintf('Output files generated:\n');
fprintf('  1. model_comparison_results.txt - All console output\n');
fprintf('  2. hue_prediction_validation.png - Hue plot (300 DPI)\n');
fprintf('  3. hue_prediction_validation.fig - Hue plot (MATLAB figure)\n');
fprintf('  4. saturation_prediction_validation.png - Saturation plot (300 DPI)\n');
fprintf('  5. saturation_prediction_validation.fig - Saturation plot (MATLAB figure)\n');
fprintf('\n');
fprintf('Figure specifications:\n');
fprintf('  - Size: 3.5" x 3.5" (IEEE single column format)\n');
fprintf('  - Resolution: 300 DPI (publication quality)\n');
fprintf('  - Data: Validation datasets only (n = %d samples)\n', length(combined_hue_targets));
fprintf('\n');

% Stop diary
diary off;
fprintf('All results saved to model_comparison_results.txt\n');
