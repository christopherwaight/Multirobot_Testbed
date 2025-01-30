%% Create and Train Feedforward Neural Networks for Hue and Saturation
% Copy this entire directory, then update the files individually as needed.
% Written by Christopher Waight
% Last update on Jan 29, 2025

%% Clearing the workspace
clc; clear all; close all;

%% Read Data from the Calibration CSV
data = readmatrix("pink_cal.csv");

%% Assign Input Variables and Target Values
rows_to_include = 20;
inputs = data(1:24*rows_to_include,3:6);
hue_targets = data(1:24*rows_to_include,1);
sat_targets = data(1:24*rows_to_include,2);

% Normalize data
inputs(:,1) = (inputs(:,1)-171)/1718;
inputs(:,2) = (inputs(:,2)-262)/2023;
inputs(:,3) = (inputs(:,3)-253)/1713;
inputs(:,4) = (inputs(:,4)-991)/5292;
inputs = max(min(inputs, 1), 0); 

% Some feature Engineering
hsv = rgb2hsv(inputs(:,1:3));
inputs(:,5:7) = hsv; 
inputs = max(min(inputs, 1), 0); 


%% Data Augmentation
noiseLevelRGB = 0.01;  % Adjust as needed
numAugmentations = 3; % Number of augmented samples to generate per original sample

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
figure();
plot(hue_targets);
hold on;
hue_target_verify =  atan2(hue_targets_sin, hue_targets_cos);
hue_target_verify(hue_target_verify < 0) = hue_target_verify(hue_target_verify < 0) + 2 * pi;
hue_target_verify = hue_target_verify / (2 * pi);
plot(hue_target_verify, '--');



%% Transpose the inputs to make NN compatible
inputs = inputs';
hue_targets_sin = hue_targets_sin';
hue_targets_cos = hue_targets_cos';
sat_targets = sat_targets';

%% Create and Train the Deeper Feedforward Network
%hiddenLayerSizes1 = [8 8 2]; % 2 Nueron Output
%hiddenLayerSizes2 = [8 6 4 2 1]; % Define the number of neurons in each hidden layer for a deeper network
hiddenLayerSizes1 =  [6 4 3 2]; % 2 Nueron Output
hiddenLayerSizes2 = [6 4 2 1]; % Define the number of neurons in each hidden layer for a deeper network


pink_hue_net = feedforwardnet(hiddenLayerSizes1);
pink_sat_net = feedforwardnet(hiddenLayerSizes2);

% Customize training parameters of hue net
pink_hue_net.trainFcn = 'trainlm';  % Use Levenberg-Marquardt algorithm (you can change this)
pink_hue_net.trainParam.epochs = 1000; % Set the maximum number of epochs
pink_hue_net.trainParam.showWindow = true; % Turn off display progress dialog.
pink_hue_net.divideFcn = 'dividerand'; % Randomly divide data into training, validation, and test sets (default)
pink_hue_net.divideParam.trainRatio = 0.7; % 70% of data for training
pink_hue_net.divideParam.valRatio = 0.15; % 15% of data for validation
pink_hue_net.divideParam.testRatio = 0.15; % 15% of data for testing
%pink_hue_net.trainParam.mu_max = 1e20; %You can uncomment these if the model trains too slowly.
%pink_hue_net.trainParam.mu = 10;

% Customize training parameters of sat net
pink_sat_net.trainFcn = 'trainlm';  % Use Levenberg-Marquardt algorithm (you can change this)
pink_sat_net.trainParam.epochs = 4800; % Set the maximum number of epochs
pink_sat_net.trainParam.showWindow = true; % Turn off display progress dialog.
pink_sat_net.divideFcn = 'dividerand'; % Randomly divide data into training, validation, and test sets (default)
pink_sat_net.divideParam.trainRatio = 0.7; % 70% of data for training
pink_sat_net.divideParam.valRatio = 0.15; % 15% of data for validation
pink_sat_net.divideParam.testRatio = 0.15; % 15% of data for testing
%pink_sat_net.trainParam.mu_max = 1e20; %You can uncomment these if the model trains too slowly.
%pink_sat_net.trainParam.mu = 10;


%% Normaling the Target Data
% Normalize the input data
[inputs,ps1] = mapminmax(inputs); % Normalize inputs to the range [-1, 1]
[hue_targets_sin, ts1_sin] = mapminmax(hue_targets_sin);
[hue_targets_cos, ts1_cos] = mapminmax(hue_targets_cos);
[sat_targets,ts2] = mapminmax(sat_targets); % Normalize targets to the range [-1, 1]

%% Now, train the networks with normalized data
[pink_hue_net, tr1] = train(pink_hue_net, inputs, [hue_targets_sin; hue_targets_cos]); % Train on both sine and cosine
[pink_sat_net, tr2] = train(pink_sat_net, inputs, sat_targets);



%% Verification - Use the Trained Model to Predict Data and Denormalize
% % Predict on a sample
% sample_index = 4517; 
% predicted_hue_sin_cos_normalized = pink_hue_net(inputs(:, sample_index));  % Between -1 and +1
% predicted_hue_sin_normalized = predicted_hue_sin_cos_normalized(1,:);
% predicted_hue_cos_normalized = predicted_hue_sin_cos_normalized(2,:);
% predicted_hue_sin = mapminmax('reverse', predicted_hue_sin_normalized, ts1_sin); % Brings it between -1 and 1
% predicted_hue_cos = mapminmax('reverse', predicted_hue_cos_normalized, ts1_cos); % Brings it between -1 and 1
% 
% % Recover the hue angle using atan2
% predicted_hue = atan2(predicted_hue_sin, predicted_hue_cos); % Result is between -pi and pi
% predicted_hue(predicted_hue < 0) = predicted_hue(predicted_hue < 0) + 2 * pi;
% predicted_hue = predicted_hue / (2 * pi);
% 
% predicted_sat_Output_Normalized = pink_sat_net(inputs(:, sample_index));
% predicted_sat_Output = mapminmax('reverse', predicted_sat_Output_Normalized, ts2); % Denormalize the sat prediction
% 
% % Denormalize the actual target for comparison using original range
% actual_hue_target = hue_targets(sample_index); 
% actual_sat_target = sat_targets(sample_index);
% 
% % Display the denormalized predicted and actual values
% disp(['Predicted Hue: ', num2str(predicted_hue)]);
% disp(['Actual Hue: ', num2str(actual_hue_target)]);
% disp(['Predicted Saturation: ', num2str(predicted_sat_Output)]);
% disp(['Actual Saturation: ', num2str(actual_sat_target)]);

%% Identify Outliers in Predictions in Hue
% Predict on the entire dataset
allHuePredictionsNormalized = pink_hue_net(inputs);
allHuePredictions_sin = mapminmax('reverse', allHuePredictionsNormalized(1,:), ts1_sin);
allHuePredictions_cos = mapminmax('reverse', allHuePredictionsNormalized(2,:), ts1_cos);

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

allSatPredictionsNormalized = pink_sat_net(inputs);
allSatPredictions = mapminmax('reverse', allSatPredictionsNormalized, ts2); % Denormalize

% Denormalize sat_targets using the stored parameters (ts2)
sat_targets = mapminmax('reverse', sat_targets, ts2);

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
%% Saving Trained network
%% Saving the Networks
save('pink_nets.mat', 'pink_hue_net', 'pink_sat_net');



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