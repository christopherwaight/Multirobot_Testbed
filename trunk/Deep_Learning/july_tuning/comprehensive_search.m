% Initialize best scores
best_hue_score = 0;
best_sat_score = 0;
best_iteration_hue = 0;
best_iteration_sat = 0;
iteration_count = 100;

% Store scores for analysis
all_hue_scores = zeros(iteration_count, 8);
all_sat_scores = zeros(iteration_count, 8);
all_min_hue_scores = zeros(iteration_count, 1);
all_min_sat_scores = zeros(iteration_count, 1);

for j = 1:iteration_count  % Fixed loop count
    
    fprintf('\n=== Training Iteration %d ===\n', j);
    
    %% Load Training data
    data1 = readmatrix("celeste_cal.csv");
    data2 = readmatrix("tidal_cal.csv");
    data3 = readmatrix("pacific_blue_cal.csv");
    data4 = readmatrix("redwood_cal.csv");
    
    %% Assign Input Variables and Target Values
    rows_to_include = 20;
    inputs = [data1(1:24*rows_to_include,3:6);data2(1:24*rows_to_include,3:6);data3(1:24*rows_to_include,3:6);data4(1:24*rows_to_include,3:6)];
    hue_targets = [data1(1:24*rows_to_include,1);data2(1:24*rows_to_include,1);data3(1:24*rows_to_include,1),;data4(1:24*rows_to_include,1)];
    sat_targets = [data1(1:24*rows_to_include,2);data2(1:24*rows_to_include,2);data3(1:24*rows_to_include,2);data4(1:24*rows_to_include,2)];
    
    %% Normalize data (only once)
    inputs(:,1) = (inputs(:,1)-120)/1300;
    inputs(:,2) = (inputs(:,2)-120)/1300;
    inputs(:,3) = (inputs(:,3)-120)/1300;
    inputs(:,4) = (inputs(:,4)-800)/3500;
    inputs = max(min(inputs, 1), 0); 
    
    % Some feature Engineering

    inputs = max(min(inputs, 1), 0); 
    
    %% Data Augmentation
    noiseLevelRGB = 0.005;  % Adjust as needed
    numAugmentations = randi([0, 2]);  % Number of augmented samples to generate per original sample
    
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
    %hsv_augmented = rgb2hsv(augmentedInputs(:, 1:3));
    %augmentedInputs = [augmentedInputs, hsv_augmented];
    augmentedInputs = [augmentedInputs];

    % Combine original and augmented data
    inputs = [inputs; augmentedInputs];
    hue_targets = [hue_targets; augmentedHueTargets];
    sat_targets = [sat_targets; augmentedSatTargets];
    
    %% Decomposing Hue into a 2 neuron output
    hue_targets_sin = sin(2 * pi * hue_targets);
    hue_targets_cos = cos(2 * pi * hue_targets);
    
    %% Transpose the inputs to make NN compatible
    inputs = inputs';
    hue_targets_sin = hue_targets_sin';
    hue_targets_cos = hue_targets_cos';
    sat_targets = sat_targets';
    
    %% Create neural network architecture (varying with iteration)
    % Modify network architecture based on iteration number
    %hiddenLayerSizes1 =  [6 6]; % 2 Neuron Output
    hiddenLayerSizes1 = [randi([5, 10]) randi([4, 7])];

    if rand() < 0.8  % 50% chance for each
        % 2 layers
        hiddenLayerSizes2 = [randi([10, 20]) randi([5, 15])];
    else
        % 3 layers
        hiddenLayerSizes2 = [randi([12, 48])];
    end
    
    
    everything_hue_net = feedforwardnet(hiddenLayerSizes1);
    everything_sat_net = feedforwardnet(hiddenLayerSizes2);
    
    % Configure hue network
    everything_hue_net.trainFcn = 'trainlm';
    everything_hue_net.trainParam.epochs = 4000;
    everything_hue_net.trainParam.showWindow = false;
    everything_hue_net.divideFcn = 'dividerand';
    everything_hue_net.divideParam.trainRatio = 0.7;
    everything_hue_net.divideParam.valRatio = 0.15;
    everything_hue_net.divideParam.testRatio = 0.15;
    
    % Configure saturation network
    everything_sat_net.trainFcn = 'trainlm';
    everything_sat_net.trainParam.epochs = 4800;
    everything_sat_net.trainParam.showWindow = false;
    everything_sat_net.divideFcn = 'dividerand';
    everything_sat_net.divideParam.trainRatio = 0.8;
    everything_sat_net.divideParam.valRatio = 0.1;
    everything_sat_net.divideParam.testRatio = 0.1;
    everything_sat_net.performParam.regularization = 0.0001;
    % Reduce learning rate as training progresses
    everything_sat_net.trainParam.lr = 0.01 * (0.995^floor(j/10));
    
    %% Normalize the input data to range [-1, 1] (only do this ONCE)
    inputs_normalized = (inputs*2) - 1;
    hue_targets_sin_normalized = (hue_targets_sin*2) - 1;
    hue_targets_cos_normalized = (hue_targets_cos*2) - 1;
    sat_targets_normalized = (sat_targets*2) - 1;
    
    %% Train the networks
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
    hue_targets_unnormalized = (hue_targets+1)/2;  % Ensure proper comparison
    circularErrors = min(abs(allHuePredictions - hue_targets'), 1 - abs(allHuePredictions - hue_targets'));
    
    %% Evaluate training performance - Saturation
    allSatPredictionsNormalized = everything_sat_net(inputs_normalized);
    allSatPredictions = (allSatPredictionsNormalized+1)/2;
    
    % Unnormalize saturation targets for comparison
    sat_targets_unnormalized = (sat_targets+1)/2;
    errors_sat = allSatPredictions - sat_targets_unnormalized;
    
    %% Print training statistics
    %fprintf('\n--- Training Performance ---\n');
    
    % Regular metrics for Hue
    hue_targets_vec = hue_targets';
    allHuePredictions_vec = allHuePredictions(:);
    
    % Circular metrics
    circular_errors = min(abs(hue_targets_vec - allHuePredictions_vec), ...
                         1 - abs(hue_targets_vec - allHuePredictions_vec));
    circular_RMSE_hue = sqrt(mean(circular_errors.^2));
    
    % Circular R-squared
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    circular_R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
    %fprintf('Hue Circular R-squared: %.4f\n', circular_R2_hue);
    
    % Metrics for Saturation
    sat_targets_vec = sat_targets_unnormalized(:);
    allSatPredictions_vec = allSatPredictions(:);
    
    % Standard R-squared calculation
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - allSatPredictions_vec).^2);
    R2_sat_train = 1 - SSE_sat/SST_sat;
    %fprintf('Saturation R-squared: %.4f\n', R2_sat_train);
    
    %% Validation on all 8 datasets
    fprintf('\n=== Multi-Dataset Validation ===\n');
    
    % Initialize arrays to store scores
    hue_scores = zeros(1, 8);
    sat_scores = zeros(1, 8);
    dataset_names = {'celeste_ver', 'tidal_ver', 'pacific_blue_ver', 'redwood_ver','redwood_cal', 'celeste_cal', 'tidal_cal', 'pacific_blue_cal'};
    
    %% Validation Set 1-3: _ver.csv files
    ver_data{1} = readmatrix("celeste_ver.csv");
    ver_data{2} = readmatrix("tidal_ver.csv");
    ver_data{3} = readmatrix("pacific_blue_ver.csv");
    ver_data{4} = readmatrix("redwood_ver.csv");

    for v = 1:4
        val_inputs = ver_data{v}(1:70, 3:6);
        val_hue_targets = ver_data{v}(1:70, 1);
        val_sat_targets = ver_data{v}(1:70, 2);
        
        % Normalize with same scaling as training (1500)
        val_inputs(:,1) = (val_inputs(:,1)-120)/1300;
        val_inputs(:,2) = (val_inputs(:,2)-120)/1300;
        val_inputs(:,3) = (val_inputs(:,3)-120)/1300;
        val_inputs(:,4) = (val_inputs(:,4)-800)/3500;
        val_inputs = max(min(val_inputs, 1), 0);
        
        % Feature engineering
        %val_hsv = rgb2hsv(val_inputs(:,1:3));
        %val_inputs(:,5:7) = val_hsv;
        val_inputs = max(min(val_inputs, 1), 0);
        
        % Prepare data
        val_inputs = val_inputs';
        val_inputs_normalized = (val_inputs*2) - 1;
        
        % Test Hue
        val_huePredictionsNormalized = everything_hue_net(val_inputs_normalized);
        val_huePredictions_sin = (val_huePredictionsNormalized(1,:)+1)/2;
        val_huePredictions_cos = (val_huePredictionsNormalized(2,:)+1)/2;
        val_huePredictions = atan2(val_huePredictions_sin, val_huePredictions_cos);
        val_huePredictions(val_huePredictions < 0) = val_huePredictions(val_huePredictions < 0) + 2 * pi;
        val_huePredictions = val_huePredictions / (2 * pi);
        
        % Calculate Hue R-squared (using circular R-squared)
        val_hue_targets_vec = val_hue_targets(:);
        val_huePredictions_vec = val_huePredictions(:);
        val_hue_mean = mean(val_hue_targets_vec);
        val_circular_errors = min(abs(val_hue_targets_vec - val_huePredictions_vec), ...
                                1 - abs(val_hue_targets_vec - val_huePredictions_vec));
        val_circular_errors_from_mean = min(abs(val_hue_targets_vec - val_hue_mean), ...
                                          1 - abs(val_hue_targets_vec - val_hue_mean));
        val_circular_SST_hue = sum(val_circular_errors_from_mean.^2);
        val_circular_SSE_hue = sum(val_circular_errors.^2);
        hue_scores(v) = 1 - val_circular_SSE_hue/val_circular_SST_hue;
        
        % Test Saturation
        val_satPredictionsNormalized = everything_sat_net(val_inputs_normalized);
        val_satPredictions = (val_satPredictionsNormalized+1)/2;
        val_satPredictions = max(min(val_satPredictions, 1), 0);
        
        % Calculate Saturation R-squared
        val_sat_targets_vec = val_sat_targets(:);
        val_satPredictions_vec = val_satPredictions(:);
        val_SST_sat = sum((val_sat_targets_vec - mean(val_sat_targets_vec)).^2);
        val_SSE_sat = sum((val_sat_targets_vec - val_satPredictions_vec).^2);
        sat_scores(v) = 1 - val_SSE_sat/val_SST_sat;
    end
    
    %% Validation Set 4-7: _cal.csv files
    cal_data{1} = readmatrix("redwood_cal.csv");
    cal_data{2} = readmatrix("celeste_cal.csv");
    cal_data{3} = readmatrix("tidal_cal.csv");
    cal_data{4} = readmatrix("pacific_blue_cal.csv");
    
    num_lines = 480;  % For redwood
    rows_to_use = 20;  % For others
    
    for v = 1:4
        if v == 1  % Redwood
            val_inputs = cal_data{v}(1:num_lines, 3:6);
            val_hue_targets = cal_data{v}(1:num_lines, 1);
            val_sat_targets = cal_data{v}(1:num_lines, 2);
        else  % Other cal files (v=2,3,4 correspond to celeste, tidal, pacific_blue)
            val_inputs = cal_data{v}(1:24*rows_to_use, 3:6);
            val_hue_targets = cal_data{v}(1:24*rows_to_use, 1);
            val_sat_targets = cal_data{v}(1:24*rows_to_use, 2);
        end
        
        % Normalize with standard scaling
        val_inputs(:,1) = (val_inputs(:,1)-120)/1300;
        val_inputs(:,2) = (val_inputs(:,2)-120)/1300;
        val_inputs(:,3) = (val_inputs(:,3)-120)/1300;
        val_inputs(:,4) = (val_inputs(:,4)-800)/3500;
        val_inputs = max(min(val_inputs, 1), 0);
        
        % Feature engineering
        %val_hsv = rgb2hsv(val_inputs(:,1:3));
        %val_inputs(:,5:7) = val_hsv;
        val_inputs = max(min(val_inputs, 1), 0);
        
        % Prepare data
        val_inputs = val_inputs';
        val_inputs_normalized = (val_inputs*2) - 1;
        
        % Test Hue
        val_huePredictionsNormalized = everything_hue_net(val_inputs_normalized);
        val_huePredictions_sin = (val_huePredictionsNormalized(1,:)+1)/2;
        val_huePredictions_cos = (val_huePredictionsNormalized(2,:)+1)/2;
        val_huePredictions = atan2(val_huePredictions_sin, val_huePredictions_cos);
        val_huePredictions(val_huePredictions < 0) = val_huePredictions(val_huePredictions < 0) + 2 * pi;
        val_huePredictions = val_huePredictions / (2 * pi);
        
        % Calculate Hue R-squared (using circular R-squared)
        val_hue_targets_vec = val_hue_targets(:);
        val_huePredictions_vec = val_huePredictions(:);
        val_hue_mean = mean(val_hue_targets_vec);
        val_circular_errors = min(abs(val_hue_targets_vec - val_huePredictions_vec), ...
                                1 - abs(val_hue_targets_vec - val_huePredictions_vec));
        val_circular_errors_from_mean = min(abs(val_hue_targets_vec - val_hue_mean), ...
                                          1 - abs(val_hue_targets_vec - val_hue_mean));
        val_circular_SST_hue = sum(val_circular_errors_from_mean.^2);
        val_circular_SSE_hue = sum(val_circular_errors.^2);
        hue_scores(4+v) = 1 - val_circular_SSE_hue/val_circular_SST_hue;
        
        % Test Saturation
        val_satPredictionsNormalized = everything_sat_net(val_inputs_normalized);
        val_satPredictions = (val_satPredictionsNormalized+1)/2;
        val_satPredictions = max(min(val_satPredictions, 1), 0);
        
        % Calculate Saturation R-squared
        val_sat_targets_vec = val_sat_targets(:);
        val_satPredictions_vec = val_satPredictions(:);
        val_SST_sat = sum((val_sat_targets_vec - mean(val_sat_targets_vec)).^2);
        val_SSE_sat = sum((val_sat_targets_vec - val_satPredictions_vec).^2);
        sat_scores(4+v) = 1 - val_SSE_sat/val_SST_sat;
    end
    
    %% Display all scores
    fprintf('\nValidation Scores:\n');
    fprintf('%-20s | Hue R² | Sat R²\n', 'Dataset');
    fprintf('%-20s---------+--------\n', '--------------------');
    for i = 1:8
        fprintf('%-20s | %.4f | %.4f\n', dataset_names{i}, hue_scores(i), sat_scores(i));
    end
    
    % Find minimum scores
    min_hue_score = min(hue_scores);
    min_sat_score = min(sat_scores);
    [~, min_hue_idx] = min(hue_scores);
    [~, min_sat_idx] = min(sat_scores);
    
    fprintf('\nMinimum Scores:\n');
    fprintf('Hue: %.4f (from %s)\n', min_hue_score, dataset_names{min_hue_idx});
    fprintf('Saturation: %.4f (from %s)\n', min_sat_score, dataset_names{min_sat_idx});
    fprintf('Best Scores: %.4f %.4f', best_hue_score,best_sat_score)
    % Store scores for analysis
    all_hue_scores(j, :) = hue_scores;
    all_sat_scores(j, :) = sat_scores;
    all_min_hue_scores(j) = min_hue_score;
    all_min_sat_scores(j) = min_sat_score;
    
    %% Save networks separately if better than previous best
    if min_hue_score > best_hue_score
        best_hue_score = min_hue_score;
        best_iteration_hue = j;
        fprintf('\n*** NEW BEST HUE MODEL! ***\n');
        save('best_hue_net.mat', 'everything_hue_net', 'hue_scores', 'min_hue_score', 'j');
        fprintf('Hue model saved to best_hue_net.mat\n');
        
        % Also save with iteration number for tracking
        filename = sprintf('hue_net_iter_%d.mat', j);
        save(filename, 'everything_hue_net', 'hue_scores', 'min_hue_score');
    end
    
    if min_sat_score > best_sat_score
        best_sat_score = min_sat_score;
        best_iteration_sat = j;
        fprintf('\n*** NEW BEST SATURATION MODEL! ***\n');
        save('best_sat_net.mat', 'everything_sat_net', 'sat_scores', 'min_sat_score', 'j');
        fprintf('Saturation model saved to best_sat_net.mat\n');
        
        % Also save with iteration number for tracking
        filename = sprintf('sat_net_iter_%d.mat', j);
        save(filename, 'everything_sat_net', 'sat_scores', 'min_sat_score');
    end
end

%% Final Summary
fprintf('\n\n========== TRAINING COMPLETE ==========\n');
fprintf('Best Hue Model:\n');
fprintf('  Iteration: %d\n', best_iteration_hue);
fprintf('  Minimum Score: %.4f\n', best_hue_score);
fprintf('  All scores from that iteration:\n');
for i = 1:7
    fprintf('    %-20s: %.4f\n', dataset_names{i}, all_hue_scores(best_iteration_hue, i));
end

fprintf('\nBest Saturation Model:\n');
fprintf('  Iteration: %d\n', best_iteration_sat);
fprintf('  Minimum Score: %.4f\n', best_sat_score);
fprintf('  All scores from that iteration:\n');
for i = 1:8
    fprintf('    %-20s: %.4f\n', dataset_names{i}, all_sat_scores(best_iteration_sat, i));
end

% Save training history
save('training_history.mat', 'all_hue_scores', 'all_sat_scores', 'all_min_hue_scores', 'all_min_sat_scores', ...
     'best_iteration_hue', 'best_iteration_sat', 'best_hue_score', 'best_sat_score');

%% Plotting Progress

% Plot progress
figure;
subplot(2,1,1);
plot(1:iteration_count, all_min_hue_scores, 'b-');
hold on;
plot(best_iteration_hue, best_hue_score, 'ro', 'MarkerSize', 10);
xlabel('Iteration');
ylabel('Minimum Hue R²');
title('Hue Model Performance Progress');
grid on;

subplot(2,1,2);
plot(1:iteration_count, all_min_sat_scores, 'r-');
hold on;
plot(best_iteration_sat, best_sat_score, 'go', 'MarkerSize', 10);
xlabel('Iteration');
ylabel('Minimum Saturation R²');
title('Saturation Model Performance Progress');
grid on;

%% Code to combine best models after training
% fprintf('\n\nTo combine the best models into a single file, run:\n');
% fprintf('load(''best_hue_net.mat'');\n');
% fprintf('load(''best_sat_net.mat'');\n');
% fprintf('save(''everything_nets_final.mat'', ''everything_hue_net'', ''everything_sat_net'');\n');