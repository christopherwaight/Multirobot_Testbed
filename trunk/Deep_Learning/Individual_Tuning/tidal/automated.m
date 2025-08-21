for j = 1:1000

    %% Create and Train Feedforward Neural Networks for Hue and Saturation
    % Copy this entire directory, then update the files individually as needed.
    % Written by Christopher Waight
    % Last update on Jan 29, 2025
    
    %% Clearing the workspace
    %clc; clear all; close all;
    
    %% Read Data from the Calibration CSV
    data = readmatrix("tidal_cal.csv");
    
    %% Assign Input Variables and Target Values
    rows_to_include = 20;
    inputs = data(1:24*rows_to_include,3:6);
    hue_targets = data(1:24*rows_to_include,1);
    sat_targets = data(1:24*rows_to_include,2);
    
    % Normalize data
    inputs(:,1) = (inputs(:,1)-100)/1200;
    inputs(:,2) = (inputs(:,2)-150)/1250;
    inputs(:,3) = (inputs(:,3)-150)/1250;
    inputs(:,4) = (inputs(:,4)-600)/3500;
    inputs = max(min(inputs, 1), 0); 
    
    % Some feature Engineering
    hsv = rgb2hsv(inputs(:,1:3));
    inputs(:,5:7) = hsv; 
    inputs = max(min(inputs, 1), 0); 
    
    
    %% Data Augmentation
    noiseLevelRGB = 0.01;  % Adjust as needed
    numAugmentations = 2; % Number of augmented samples to generate per original sample
    
    augmentedInputs = [];
    augmentedHueTargets = [];
    augmentedSatTargets = [];
    
    for i = 1:size(inputs, 1)
        for j = 1:numAugmentations
            % 1. Add Gaussian noise to RGBK (first 4 features)
            noisyInput = inputs(i, 1:4) + noiseLevelRGB * randn(1, 4);
    
            % 2. Clip noisy RGBK to [0, 1]
            noisyInput = max(0, min(1, noisyInput));
    
    
            % 5. Add to augmented data
            augmentedInputs = [augmentedInputs; noisyInput];
            augmentedHueTargets = [augmentedHueTargets; hue_targets(i)]; % Hue target remains the same
            augmentedSatTargets = [augmentedSatTargets; sat_targets(i)];
        end
    end
    
    
    % Add HSV to augmented data
    hsv_augmented = rgb2hsv(augmentedInputs(:, 1:3));
    augmentedInputs = [augmentedInputs, hsv_augmented];
    
    inputs = [inputs; augmentedInputs];
    hue_targets = [hue_targets; augmentedHueTargets];
    
    sat_targets = [sat_targets; augmentedSatTargets];
    
    
    
    %% Decomposing Hue into a 2 neuron output
    hue_targets_sin = sin(2 * pi * hue_targets);
    hue_targets_cos = cos(2 * pi * hue_targets);
    
    %% Debugging Steps
    % figure();
    % plot(hue_targets);
    % hold on;
    % hue_target_verify =  atan2(hue_targets_sin, hue_targets_cos);
    % hue_target_verify(hue_target_verify < 0) = hue_target_verify(hue_target_verify < 0) + 2 * pi;
    % hue_target_verify = hue_target_verify / (2 * pi);
    % plot(hue_target_verify, '--');
    
    
    
    %% Transpose the inputs to make NN compatible
    inputs = inputs';
    hue_targets_sin = hue_targets_sin';
    hue_targets_cos = hue_targets_cos';
    sat_targets = sat_targets';
    
    %% Normaling the Target Data to the range [-1, 1]
    inputs = (inputs*2) -1;
    hue_targets_sin = (hue_targets_sin*2) -1;
    hue_targets_cos = (hue_targets_cos*2) -1;
    sat_targets = (sat_targets*2) -1;
    
    
    %% Create and Train the Deeper Feedforward Network
    
    hiddenLayerSizes1 =  [5 4]; % 2 Nueron Output
    hiddenLayerSizes2 = [randi([3, 8]) randi([3, 8])];
    
    
    tidal_hue_net = feedforwardnet(hiddenLayerSizes1);
    tidal_sat_net = feedforwardnet(hiddenLayerSizes2);
    
    % Customize training parameters of hue net
    tidal_hue_net.trainFcn = 'trainlm';  % Use Levenberg-Marquardt algorithm (you can change this)
    tidal_hue_net.trainParam.epochs = 1000; % Set the maximum number of epochs
    tidal_hue_net.trainParam.showWindow = true; % Turn off display progress dialog.
    tidal_hue_net.divideFcn = 'dividerand'; % Randomly divide data into training, validation, and test sets (default)
    tidal_hue_net.divideParam.trainRatio = 0.7; % 70% of data for training
    tidal_hue_net.divideParam.valRatio = 0.15; % 15% of data for validation
    tidal_hue_net.divideParam.testRatio = 0.15; % 15% of data for testing
    %pink_hue_net.trainParam.mu_max = 1e20; %You can uncomment these if the model trains too slowly.
    %pink_hue_net.trainParam.mu = 10;
    
    % Customize training parameters of sat net
    tidal_sat_net.trainFcn = 'trainlm';  % Use Levenberg-Marquardt algorithm (you can change this)
    tidal_sat_net.trainParam.epochs = 4800; % Set the maximum number of epochs
    tidal_sat_net.trainParam.showWindow = true; % Turn off display progress dialog.
    tidal_sat_net.divideFcn = 'dividerand'; % Randomly divide data into training, validation, and test sets (default)
    tidal_sat_net.divideParam.trainRatio = 0.7; % 70% of data for training
    tidal_sat_net.divideParam.valRatio = 0.15; % 15% of data for validation
    tidal_sat_net.divideParam.testRatio = 0.15; % 15% of data for testing
    %pink_sat_net.trainParam.mu_max = 1e20; %You can uncomment these if the model trains too slowly.
    %pink_sat_net.trainParam.mu = 10;
    
    
    
    %% Now, train the networks with normalized data
    [tidal_hue_net, tr1] = train(tidal_hue_net, inputs, [hue_targets_sin; hue_targets_cos]); % Train on both sine and cosine
    [tidal_sat_net, tr2] = train(tidal_sat_net, inputs, sat_targets);
    
    %% Saving Trained network
    save('tidal_nets.mat', 'tidal_hue_net', 'tidal_sat_net');
    
    
    %% Identify Outliers in Predictions in Hue
    % Predict on the entire dataset
    allHuePredictionsNormalized = tidal_hue_net(inputs);
    allHuePredictions_sin =(allHuePredictionsNormalized(1,:)+1)/2;
    allHuePredictions_cos = (allHuePredictionsNormalized(2,:)+1)/2;
    
    % Bringing back into a single value from 0 to 1
    allHuePredictions = atan2(allHuePredictions_sin, allHuePredictions_cos);
    allHuePredictions(allHuePredictions < 0) = allHuePredictions(allHuePredictions < 0) + 2 * pi;
    allHuePredictions = allHuePredictions / (2 * pi);
    
    % Calculate the circular error element-wise
    circularErrors = min(abs(allHuePredictions - hue_targets'), 1 - abs(allHuePredictions - hue_targets'));
    
    % Define an outlier threshold (e.g., 3 standard deviations away from the mean error)
    errorMean = mean(circularErrors);
    errorStd = std(circularErrors);
    outlierThreshold =3 * errorStd;
    
    % Find outlier indices
    outlierIndices = find(circularErrors > outlierThreshold);
    
    % Display outlier information
    disp('Hue Outliers (based on circular error):');
    count = 0;
    for i = 1:length(outlierIndices)
        index = outlierIndices(i);
        disp(['  Sample Index: ', num2str(index), ', Circular Error: ', num2str(circularErrors(index)), ', Predicted Hue: ', num2str(allHuePredictions(index)), ', Actual Hue: ', num2str(hue_targets(index))]);
        count = count+1;
    end
    count
    %%  Identify Outliers in Predictions in Saturation
    
    allSatPredictionsNormalized = tidal_sat_net(inputs);
    allSatPredictions = (allSatPredictionsNormalized+1)/2;
    
    % Denormalize sat_targets using the stored parameters (ts2)
    sat_targets = (sat_targets+1)/2;
    
    % Calculate the error
    errors2 = allSatPredictions - sat_targets; %error
    
    % Define an outlier threshold (e.g., 3 standard deviations away from the mean error)
    errorMean2 = mean(errors2);
    errorStd2 = std(errors2);
    outlierThreshold2 = 4 * errorStd2;
    
    % Find outlier indices
    outlierIndices2 = find(abs(errors2) > outlierThreshold2);
    
    % Display outlier information
    disp('Saturation Outliers:');
    count2 = 0;
    for i = 1:length(outlierIndices2)
        index = outlierIndices2(i);
        disp(['  Sample Index: ', num2str(index), ', Error: ', num2str(errors2(index)), ', Predicted Sat: ', num2str(allSatPredictions(index)), ', Actual Sat: ', num2str(sat_targets(index))]);
        count2 = count2+1;
    end
    count2
    
    
    
    %% Plotting Results
    
    % Create the figure
    figure;
    scatter(hue_targets, allHuePredictions, 'o'); % Scatter plot of actual vs. predicted
    hold on;
    
    % --- Add Line of Best Fit ---
    p = polyfit(hue_targets, allHuePredictions, 1);  % Fit a 1st-degree polynomial (line)
    f = polyval(p, hue_targets);          % Evaluate the polynomial at hue_targets
    plot(hue_targets, f, 'r-');            % Plot the line of best fit in red
    
    % --- Rest of the Plotting Code ---
    plot([0, 1], [0, 1], 'k--'); % Line for perfect prediction (black dashed)
    hold off;
    xlabel('Actual Hue');
    ylabel('Predicted Hue');
    title('Predicted vs. Actual Hue');
    legend('Predictions', 'Line of Best Fit', 'Perfect Prediction Line', 'Location', 'northwest');
    grid on;
    
    
    % Now for Saturation
    figure;
    scatter(sat_targets, allSatPredictions, 'o'); % Scatter plot of actual vs. predicted
    hold on;
    
    % --- Add Line of Best Fit ---
    p = polyfit(sat_targets, allSatPredictions, 1);  % Fit a 1st-degree polynomial (line)
    f = polyval(p, sat_targets);          % Evaluate the polynomial at hue_targets
    plot(sat_targets, f, 'r-');            % Plot the line of best fit in red
    
    % --- Rest of the Plotting Code ---
    plot([0.3, 1], [0.3, 1], 'k--'); % Line for perfect prediction (black dashed)
    hold off;
    xlabel('Actual Sat');
    ylabel('Predicted Sat');
    title('Predicted vs. Actual Saturation');
    legend('Predictions', 'Line of Best Fit', 'Perfect Prediction Line', 'Location', 'northwest');
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
    % For circular data, we need to account for the circular nature
    % Calculate circular error (smallest angle between predictions and targets)
    circular_errors = min(abs(hue_targets_vec - allHuePredictions_vec), ...
                          1 - abs(hue_targets_vec - allHuePredictions_vec));
    
    % Circular RMSE
    circular_RMSE_hue = sqrt(mean(circular_errors.^2));
    fprintf('Circular RMSE: %.4f\n', circular_RMSE_hue);
    
    % Circular R-squared calculation
    % Calculate circular distance from mean
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                   1 - abs(hue_targets_vec - hue_mean));
    
    % Circular SST and SSE                           
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    
    % Circular R-squared
    circular_R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
    fprintf('Circular R-squared: %.4f\n', circular_R2_hue);
    
    fprintf('\n--- Saturation Model Metrics ---\n');
    
    % Metrics for Saturation (non-circular)
    % Ensure all data is in the same format (vectors)
    sat_targets_vec = sat_targets(:);
    allSatPredictions_vec = allSatPredictions(:);
    
    % Standard R-squared calculation
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - allSatPredictions_vec).^2);
    R2_sat_cas1 = 1 - SSE_sat/SST_sat;
    fprintf('R-squared: %.4f\n', R2_sat_cas1);
    
    % Standard RMSE calculation
    RMSE_sat = sqrt(mean((sat_targets_vec - allSatPredictions_vec).^2));
    fprintf('RMSE: %.4f\n', RMSE_sat);
    
    % For consistency in output, include "N/A" for circular metrics
    fprintf('Circular RMSE: N/A (saturation is not circular)\n');
    fprintf('Circular R-squared: N/A (saturation is not circular)\n');
    

    load("tidal_nets.mat"); 
    
    
    %% Read Data from the Calibration CSV
    data = readmatrix("tidal_ver.csv");
    
    %% Assign Input Variables and Target Values
    total_squares = 80;
    inputs = data(1:total_squares,3:6);
    
    hue_targets = data(1:total_squares,1);
    hue_targets_sin = sin(2 * pi * hue_targets);
    hue_targets_cos = cos(2 * pi * hue_targets);
    
    sat_targets = data(1:total_squares,2);
    
    % Normalize data
    inputs(:,1) = (inputs(:,1)-100)/1200;
    inputs(:,2) = (inputs(:,2)-150)/1250;
    inputs(:,3) = (inputs(:,3)-150)/1250;
    inputs(:,4) = (inputs(:,4)-600)/3500;
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
    
    %% Normaling the Target Data to the range [-1, 1]
    inputs = (inputs*2) -1;
    hue_targets_sin = (hue_targets_sin*2) -1;
    hue_targets_cos = (hue_targets_cos*2) -1;
    sat_targets = (sat_targets*2) -1;
    
    % Add this before calling the network
    disp(['Size of inputs: ', num2str(size(inputs))]);
    
    % Then check the expected size of the network's input
    disp(['Expected input size: ', num2str(tidal_sat_net.inputs{1}.size)]);
    
    %% Identify Outliers in Predictions in Hue
    % Predict on the entire dataset
    allHuePredictionsNormalized =tidal_hue_net(inputs);
    allHuePredictions_sin =(allHuePredictionsNormalized(1,:)+1)/2;
    allHuePredictions_cos = (allHuePredictionsNormalized(2,:)+1)/2;
    
    % Bringing back into a single value from 0 to 1
    allHuePredictions = atan2(allHuePredictions_sin, allHuePredictions_cos);
    allHuePredictions(allHuePredictions < 0) = allHuePredictions(allHuePredictions < 0) + 2 * pi;
    allHuePredictions = allHuePredictions / (2 * pi);
    
    % Calculate the circular error element-wise
    circularErrors = min(abs(allHuePredictions - hue_targets'), 1 - abs(allHuePredictions - hue_targets'));
    
    % Define an outlier threshold (e.g., 3 standard deviations away from the mean error)
    errorMean = mean(circularErrors);
    errorStd = std(circularErrors);
    outlierThreshold =3 * errorStd;
    
    % Find outlier indices
    outlierIndices = find(circularErrors > outlierThreshold);
    
    % % Display outlier information
    % disp('Hue Outliers (based on circular error):');
    % count = 0;
    % for i = 1:length(outlierIndices)
    %     index = outlierIndices(i);
    %     disp(['  Sample Index: ', num2str(index), ', Circular Error: ', num2str(circularErrors(index)), ', Predicted Hue: ', num2str(allHuePredictions(index)), ', Actual Hue: ', num2str(hue_targets(index))]);
    %     count = count+1;
    % end
    count
    %%  Identify Outliers in Predictions in Saturation
    
    % UnNormalizing
    allSatPredictionsNormalized = tidal_sat_net(inputs);
    allSatPredictions = (allSatPredictionsNormalized+1)/2;
    sat_targets = (sat_targets+1)/2;
    allSatPredictions(allSatPredictions>1)=1;
    % Calculate the error
    errors2 = allSatPredictions - sat_targets; %error
    
    % Define an outlier threshold (e.g., 3 standard deviations away from the mean error)
    errorMean2 = mean(errors2);
    errorStd2 = std(errors2);
    outlierThreshold2 = 3 * errorStd2;
    
    % Find outlier indices
    outlierIndices2 = find(abs(errors2) > outlierThreshold2);
    
    % % Display outlier information
    % disp('Saturation Outliers:');
    % count2 = 0;
    % for i = 1:length(outlierIndices2)
    %     index = outlierIndices2(i);
    %     disp(['  Sample Index: ', num2str(index), ', Error: ', num2str(errors2(index)), ', Predicted Sat: ', num2str(allSatPredictions(index)), ', Actual Sat: ', num2str(sat_targets(index))]);
    %     count2 = count2+1;
    % end
    % count2
    % %% Plotting Results
    % 
    % % Create the figure
    % figure;
    % scatter(hue_targets, allHuePredictions, 'o'); % Scatter plot of actual vs. predicted
    % hold on;
    % 
    % % --- Add Line of Best Fit ---
    % p = polyfit(hue_targets, allHuePredictions, 1);  % Fit a 1st-degree polynomial (line)
    % f = polyval(p, hue_targets);          % Evaluate the polynomial at hue_targets
    % plot(hue_targets, f, 'r-');            % Plot the line of best fit in red
    % 
    % % --- Rest of the Plotting Code ---
    % plot([0, 1], [0, 1], 'k--'); % Line for perfect prediction (black dashed)
    % hold off;
    % xlabel('Actual Hue');
    % ylabel('Predicted Hue');
    % title('Predicted vs. Actual Hue');
    % legend('Predictions', 'Line of Best Fit', 'Perfect Prediction Line', 'Location', 'northwest');
    % grid on;
    % 
    % 
    % % Now for Saturation
    % figure;
    % scatter(sat_targets, allSatPredictions, 'o'); % Scatter plot of actual vs. predicted
    % hold on;
    % 
    % % --- Add Line of Best Fit ---
    % p = polyfit(sat_targets, allSatPredictions, 1);  % Fit a 1st-degree polynomial (line)
    % f = polyval(p, sat_targets);          % Evaluate the polynomial at hue_targets
    % plot(sat_targets, f, 'r-');            % Plot the line of best fit in red
    % 
    % % --- Rest of the Plotting Code ---
    % plot([0.3, 1], [0.3, 1], 'k--'); % Line for perfect prediction (black dashed)
    % hold off;
    % xlabel('Actual Sat');
    % ylabel('Predicted Sat');
    % title('Predicted vs. Actual Saturation');
    % legend('Predictions', 'Line of Best Fit', 'Perfect Prediction Line', 'Location', 'northwest');
    % grid on;
    
    
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
    % For circular data, we need to account for the circular nature
    % Calculate circular error (smallest angle between predictions and targets)
    circular_errors = min(abs(hue_targets_vec - allHuePredictions_vec), ...
                          1 - abs(hue_targets_vec - allHuePredictions_vec));
    
    % Circular RMSE
    circular_RMSE_hue = sqrt(mean(circular_errors.^2));
    fprintf('Circular RMSE: %.4f\n', circular_RMSE_hue);
    
    % Circular R-squared calculation
    % Calculate circular distance from mean
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                   1 - abs(hue_targets_vec - hue_mean));
    
    % Circular SST and SSE                           
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    
    % Circular R-squared
    circular_R2_hue = 1 - circular_SSE_hue/circular_SST_hue;
    fprintf('Circular R-squared: %.4f\n', circular_R2_hue);
    
    fprintf('\n--- Saturation Model Metrics ---\n');
    
    % Metrics for Saturation (non-circular)
    % Ensure all data is in the same format (vectors)
    sat_targets_vec = sat_targets(:);
    allSatPredictions_vec = allSatPredictions(:);
    
    % Standard R-squared calculation
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - allSatPredictions_vec).^2);
    R2_sat_cas2 = 1 - SSE_sat/SST_sat;
    fprintf('R-squared: %.4f\n', R2_sat_cas2);
    
    % Standard RMSE calculation
    RMSE_sat = sqrt(mean((sat_targets_vec - allSatPredictions_vec).^2));
    fprintf('RMSE: %.4f\n', RMSE_sat);
    
    % For consistency in output, include "N/A" for circular metrics
    fprintf('Circular RMSE: N/A (saturation is not circular)\n');
    fprintf('Circular R-squared: N/A (saturation is not circular)\n');

    if R2_sat_cas2 >0.9 && R2_sat_cas1 > 0.9
        fprintf('Winner winner chicken dinner')
        break
    end

end