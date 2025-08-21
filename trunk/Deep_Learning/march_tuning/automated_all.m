clc; clear all; close all;
%% Load data
data1 = readmatrix("celeste_cal.csv");
data2 = readmatrix("tidal_cal.csv");
data3 = readmatrix("pacific_blue_cal.csv");
data4 = readmatrix("pacific_blue_cal.csv");
data5 = readmatrix("pacific_blue_cal.csv");
data6 = readmatrix("pacific_blue_cal.csv");
data7 = readmatrix("pacific_blue_cal.csv");

% Store datasets in a cell array for easier validation
validation_datasets = {
    "celeste_cal.csv",
    "tidal_cal.csv", 
    "pacific_blue_cal.csv", 
    "redwood_cal.csv", 
    "pacific_blue_cal.csv", 
    "pacific_blue_cal.csv", 
    "pacific_blue_cal.csv"
};

%% Random Search to find best architecture and model
best_score_hue = 1e6;
best_score_sat = 1e6;
for j = 1:10  
    
    fprintf('\n=== Training Iteration %d ===\n', j);
    
    %% Data Pre-Processing
    
    % Assign Input Variables and Target Values
    rows_to_include = 20;
    inputs = [data1(1:24*rows_to_include,3:6);data2(1:24*rows_to_include,3:6);data3(1:24*rows_to_include,3:6)];
    hue_targets = [data1(1:24*rows_to_include,1);data2(1:24*rows_to_include,1);data3(1:24*rows_to_include,1)];
    sat_targets = [data1(1:24*rows_to_include,2);data2(1:24*rows_to_include,2);data3(1:24*rows_to_include,2)];
    
    % Normalize data (only once)
    inputs(:,1) = (inputs(:,1)-100)/1500;
    inputs(:,2) = (inputs(:,2)-100)/1500;
    inputs(:,3) = (inputs(:,3)-100)/1500;
    inputs(:,4) = (inputs(:,4)-500)/4500;
    inputs = max(min(inputs, 1), 0); 
    
    % Some feature Engineering
    hsv = rgb2hsv(inputs(:,1:3));
    inputs(:,5:7) = hsv; 
    inputs = max(min(inputs, 1), 0); 
    
    % Data Augmentation
    noiseLevelRGB = 0.01; % Baseline 
    numAugmentations = 1;  
    
    augmentedInputs = [];
    augmentedHueTargets = [];
    augmentedSatTargets = [];
    
    for i = 1:size(inputs, 1)
        for k = 1:numAugmentations
            % 1. Add Gaussian noise to RGBK (first 4 features)
            noisyInput = inputs(i, 1:4) + noiseLevelRGB * randn(1, 4);
    
            % 2. Clip noisy RGBK to [0, 1]
            noisyInput = max(0, min(1, noisyInput));
    
            % 3. Add to augmented data
            augmentedInputs = [augmentedInputs; noisyInput];
            augmentedHueTargets = [augmentedHueTargets; hue_targets(i)];
            augmentedSatTargets = [augmentedSatTargets; sat_targets(i)];
        end
    end
    
    % Add HSV to augmented data
    hsv_augmented = rgb2hsv(augmentedInputs(:, 1:3));
    augmentedInputs = [augmentedInputs, hsv_augmented];
    
    % Combine original and augmented data
    inputs = [inputs; augmentedInputs];
    hue_targets = [hue_targets; augmentedHueTargets];
    sat_targets = [sat_targets; augmentedSatTargets];
    
    % Decomposing Hue into a 2 neuron output
    hue_targets_sin = sin(2 * pi * hue_targets);
    hue_targets_cos = cos(2 * pi * hue_targets);
    
    % Transpose the inputs to make NN compatible
    inputs = inputs';
    hue_targets_sin = hue_targets_sin';
    hue_targets_cos = hue_targets_cos';
    sat_targets = sat_targets';
    % Normalize the input data to range [-1, 1]
    inputs_normalized = (inputs*2) - 1;
    hue_targets_sin_normalized = (hue_targets_sin*2) - 1;
    hue_targets_cos_normalized = (hue_targets_cos*2) - 1;
    sat_targets_normalized = (sat_targets*2) - 1;
    
    %% Create neural network architecture and Train it
    
    hiddenLayerSizes1 = [randi([3, 8]) randi([3, 8])]; % 2 Nueron Output
    hiddenLayerSizes2 = [randi([3, 8]) randi([3, 8])];
    
    
    everything_hue_net = feedforwardnet(hiddenLayerSizes1);
    everything_sat_net = feedforwardnet(hiddenLayerSizes2);
    
    % Configure hue network
    everything_hue_net.trainFcn = 'trainlm';
    everything_hue_net.trainParam.epochs = 48000;
    everything_hue_net.trainParam.showWindow = true;
    everything_hue_net.divideFcn = 'dividerand';
    everything_hue_net.divideParam.trainRatio = 0.7;
    everything_hue_net.divideParam.valRatio = 0.15;
    everything_hue_net.divideParam.testRatio = 0.15;
    
    % Configure saturation network
    everything_sat_net.trainFcn = 'trainlm';
    everything_sat_net.trainParam.epochs = 4800;
    everything_sat_net.trainParam.showWindow = true;
    everything_sat_net.divideFcn = 'dividerand';
    everything_sat_net.divideParam.trainRatio = 0.7;
    everything_sat_net.divideParam.valRatio = 0.15;
    everything_sat_net.divideParam.testRatio = 0.15;
    
    
    % Train the networks
    [everything_hue_net, tr1] = train(everything_hue_net, inputs_normalized, [hue_targets_sin_normalized; hue_targets_cos_normalized]);
    [everything_sat_net, tr2] = train(everything_sat_net, inputs_normalized, sat_targets_normalized);
    
    %% Evaluate training performance - Hue
    allHuePredictionsNormalized = everything_hue_net(inputs_normalized);
    allHuePredictions_sin = (allHuePredictionsNormalized(1,:)+1)/2;
    allHuePredictions_cos = (allHuePredictionsNormalized(2,:)+1)/2;
    
    % Convert back to single hue value
    allHuePredictions = atan2(allHuePredictions_sin, allHuePredictions_cos);
    allHuePredictions(allHuePredictions < 0) = allHuePredictions(allHuePredictions < 0) + 2 * pi;
    allHuePredictions = allHuePredictions / (2 * pi);
    
    % Calculate circular errors
    circularErrors = min(abs(allHuePredictions - hue_targets'), 1 - abs(allHuePredictions - hue_targets'));
    
    %% Evaluate training performance - Saturation
    allSatPredictionsNormalized = everything_sat_net(inputs_normalized);
    allSatPredictions = (allSatPredictionsNormalized+1)/2;
    
    % Use original saturation targets for comparison
    errors_sat = allSatPredictions - sat_targets;
    
    %% Print training statistics
    fprintf('\n--- Hue Model Training Metrics ---\n');
    
    % Regular metrics for Hue
    hue_targets_vec = hue_targets';
    allHuePredictions_vec = allHuePredictions(:);
    
    % Standard R-squared calculation
    SST_hue = sum((hue_targets_vec - mean(hue_targets_vec)).^2);
    SSE_hue = sum((hue_targets_vec - allHuePredictions_vec).^2);
    R2_hue_train = 1 - SSE_hue/SST_hue;
    fprintf('R-squared: %.4f\n', R2_hue_train);
    
    % Circular metrics
    circular_errors = min(abs(hue_targets_vec - allHuePredictions_vec), ...
                         1 - abs(hue_targets_vec - allHuePredictions_vec));
    circular_RMSE_hue = sqrt(mean(circular_errors.^2));
    fprintf('Circular RMSE: %.4f\n', circular_RMSE_hue);
    
    % Circular R-squared
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    circular_R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
    fprintf('Circular R-squared: %.4f\n', circular_R2_hue);
    
    fprintf('\n--- Saturation Model Training Metrics ---\n');
    
    % Metrics for Saturation
    sat_targets_vec = sat_targets(:);
    allSatPredictions_vec = allSatPredictions(:);
    
    % Standard R-squared calculation
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - allSatPredictions_vec).^2);
    R2_sat_train = 1 - SSE_sat/SST_sat;
    fprintf('R-squared: %.4f\n', R2_sat_train);
    
    % Standard RMSE calculation
    RMSE_sat = sqrt(mean((sat_targets_vec - allSatPredictions_vec).^2));
    fprintf('RMSE: %.4f\n', RMSE_sat);
    
    % Save current networks temporarily
    save('everything_nets.mat', 'everything_hue_net', 'everything_sat_net');
    
    %% Validation on all datasets
    circular_RMSE_hue_all = zeros(1, length(validation_datasets));
    RMSE_sat_all = zeros(1, length(validation_datasets));
    
    for dataset_idx = 1:length(validation_datasets)
        % Load validation data
        val_data = readmatrix(validation_datasets{dataset_idx});
        
        fprintf('\n=== Validation on %s ===\n', validation_datasets{dataset_idx});
        
        num_lines = min(480, size(val_data, 1));
        val_inputs = [val_data(1:num_lines,3:6)];
        val_hue_targets = [val_data(1:num_lines,1)];
        val_sat_targets = [val_data(1:num_lines,2)];
        
        %% Normalize validation data
        val_inputs(:,1) = (val_inputs(:,1)-100)/1500;
        val_inputs(:,2) = (val_inputs(:,2)-100)/1500;
        val_inputs(:,3) = (val_inputs(:,3)-100)/1500;
        val_inputs(:,4) = (val_inputs(:,4)-500)/4500;
        val_inputs = max(min(val_inputs, 1), 0);
        
        % Feature engineering on validation data
        val_hsv = rgb2hsv(val_inputs(:,1:3));
        val_inputs(:,5:7) = val_hsv; 
        val_inputs = max(min(val_inputs, 1), 0);
        
        %% Prepare validation data
        val_inputs = val_inputs';
        val_inputs_normalized = (val_inputs*2) - 1;
        
        %% Test on validation data - Hue
        val_huePredictionsNormalized = everything_hue_net(val_inputs_normalized);
        val_huePredictions_sin = (val_huePredictionsNormalized(1,:)+1)/2;
        val_huePredictions_cos = (val_huePredictionsNormalized(2,:)+1)/2;
        
        % Convert to single hue value
        val_huePredictions = atan2(val_huePredictions_sin, val_huePredictions_cos);
        val_huePredictions(val_huePredictions < 0) = val_huePredictions(val_huePredictions < 0) + 2 * pi;
        val_huePredictions = val_huePredictions / (2 * pi);
        
        %% Test on validation data - Saturation
        val_satPredictionsNormalized = everything_sat_net(val_inputs_normalized);
        val_satPredictions = (val_satPredictionsNormalized+1)/2;
        val_satPredictions = max(min(val_satPredictions, 1), 0);  % Ensure values are in range
        
        %% Print validation statistics
        fprintf('\n--- Hue Model Validation Metrics ---\n');
        
        % Regular metrics for Hue
        val_hue_targets_vec = val_hue_targets(:);
        val_huePredictions_vec = val_huePredictions(:);
        
        % Standard R-squared calculation
        val_SST_hue = sum((val_hue_targets_vec - mean(val_hue_targets_vec)).^2);
        val_SSE_hue = sum((val_hue_targets_vec - val_huePredictions_vec).^2);
        R2_hue_val = 1 - val_SSE_hue/val_SST_hue;
        fprintf('R-squared: %.4f\n', R2_hue_val);
        
        % Circular metrics
        val_circular_errors = min(abs(val_hue_targets_vec - val_huePredictions_vec), ...
                                1 - abs(val_hue_targets_vec - val_huePredictions_vec));
        val_circular_RMSE_hue = sqrt(mean(val_circular_errors.^2));
        fprintf('Circular RMSE: %.4f\n', val_circular_RMSE_hue);
        
        % Store circular RMSE for this dataset
        circular_RMSE_hue_all(dataset_idx) = val_circular_RMSE_hue;
        
        % Circular R-squared
        val_hue_mean = mean(val_hue_targets_vec);
        val_circular_errors_from_mean = min(abs(val_hue_targets_vec - val_hue_mean), ...
                                          1 - abs(val_hue_targets_vec - val_hue_mean));
        val_circular_SST_hue = sum(val_circular_errors_from_mean.^2);
        val_circular_SSE_hue = sum(val_circular_errors.^2);
        val_circular_R2_hue = 1 - val_circular_SSE_hue/val_circular_SST_hue;
        fprintf('Circular R-squared: %.4f\n', val_circular_R2_hue);
        
        fprintf('\n--- Saturation Model Validation Metrics ---\n');
        
        % Metrics for Saturation
        val_sat_targets_vec = val_sat_targets(:);
        val_satPredictions_vec = val_satPredictions(:);
        
        % Standard R-squared calculation
        val_SST_sat = sum((val_sat_targets_vec - mean(val_sat_targets_vec)).^2);
        val_SSE_sat = sum((val_sat_targets_vec - val_satPredictions_vec).^2);
        R2_sat_val = 1 - val_SSE_sat/val_SST_sat;
        fprintf('R-squared: %.4f\n', R2_sat_val);
        
        % Standard RMSE calculation
        val_RMSE_sat = sqrt(mean((val_sat_targets_vec - val_satPredictions_vec).^2));
        fprintf('RMSE: %.4f\n', val_RMSE_sat);
        
        % Store RMSE for this dataset
        RMSE_sat_all(dataset_idx) = val_RMSE_sat;
    end
    
    % Calculate max RMSE across all datasets for both hue and saturation
    max_circular_RMSE_hue = max(circular_RMSE_hue_all);
    max_RMSE_sat = max(RMSE_sat_all);
    
    % Print the scores for all datasets
    fprintf('\n--- Overall Performance Metrics ---\n');
    fprintf('Hue Circular RMSE for all datasets: ');
    for i = 1:length(circular_RMSE_hue_all)
        fprintf('%.4f ', circular_RMSE_hue_all(i));
    end
    fprintf('\nMax Hue Circular RMSE: %.4f\n', max_circular_RMSE_hue);
    
    fprintf('Saturation RMSE for all datasets: ');
    for i = 1:length(RMSE_sat_all)
        fprintf('%.4f ', RMSE_sat_all(i));
    end
    fprintf('\nMax Saturation RMSE: %.4f\n', max_RMSE_sat);
    
    % Check if this model's hue performance is better than previous best
    if max_circular_RMSE_hue < best_score_hue
        best_score_hue = max_circular_RMSE_hue;
        fprintf('\n==== SUCCESS: Hue model performance meets criteria! ====\n');
        save('everything_hue_nets_best.mat', 'everything_hue_net');
    end
    
    % Check if this model's saturation performance is better than previous best
    if max_RMSE_sat < best_score_sat
        best_score_sat = max_RMSE_sat;
        fprintf('\n==== SUCCESS: Saturation model performance meets criteria! ====\n');
        save('everything_sat_nets_best.mat', 'everything_sat_net');
    end
    
    % Save the combined model only if both are best (optional)
    if max_circular_RMSE_hue < best_score_hue && max_RMSE_sat < best_score_sat
        save('everything_nets_10.mat', 'everything_hue_net', 'everything_sat_net');
    end
end