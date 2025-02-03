%% Create and Train Feedforward Neural Networks for Hue and Saturation
% Copy this entire directory, then update the files individually as needed.
% Written by Christopher Waight
% Last update on Jan 29, 2025

%% Clearing the workspace
clc; clear all; close all;

%% Read Data from the Calibration CSV
data = readmatrix("pacific_blue_cal.csv");

%% Assign Input Variables and Target Values
rows_to_include = 20;
inputs = data(1:24*rows_to_include,3:6);
hue_targets = data(1:24*rows_to_include,1);
sat_targets = data(1:24*rows_to_include,2);

% Normalize data
inputs(:,1) = (inputs(:,1)-127)/1012;
inputs(:,2) = (inputs(:,2)-218)/1121;
inputs(:,3) = (inputs(:,3)-219)/971;
inputs(:,4) = (inputs(:,4)-842)/2824;
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

%% Transpose the inputs to make NN compatible
inputs = inputs';
hue_targets_sin = hue_targets_sin';
hue_targets_cos = hue_targets_cos';
sat_targets = sat_targets';

%% Create Deep Learning Networks for Hue and Saturation

% Define the number of neurons in each hidden layer
hiddenLayerSizes1 = [5 4];
hiddenLayerSizes2 = [6 4 2];

% Create the hue network
hueInputSize = size(inputs, 1);  % Number of input features
hueOutputSize = 2;  % Number of output neurons (sin and cos)

hueLayers = [
    featureInputLayer(hueInputSize, 'Name', 'Input')
    fullyConnectedLayer(hiddenLayerSizes1(1), 'Name', 'Hidden1')
    reluLayer('Name', 'ReLU1')
    fullyConnectedLayer(hiddenLayerSizes1(2), 'Name', 'Hidden2')
    reluLayer('Name', 'ReLU2')
    fullyConnectedLayer(hueOutputSize, 'Name', 'Output')
    tanhLayer('Name', 'Tanh', 'OutputLayer', true)  % Explicitly mark as output layer
];

hueNet = dlnetwork(hueLayers);

% Create the saturation network
satInputSize = size(inputs, 1);  % Number of input features
satOutputSize = 1;  % Number of output neurons

satLayers = [
    featureInputLayer(satInputSize, 'Name', 'Input')
    fullyConnectedLayer(hiddenLayerSizes2(1), 'Name', 'Hidden1')
    reluLayer('Name', 'ReLU1')
    fullyConnectedLayer(hiddenLayerSizes2(2), 'Name', 'Hidden2')
    reluLayer('Name', 'ReLU2')
    fullyConnectedLayer(hiddenLayerSizes2(3), 'Name', 'Hidden3')
    reluLayer('Name', 'ReLU3')
    fullyConnectedLayer(satOutputSize, 'Name', 'Output')
    tanhLayer('Name', 'Tanh', 'OutputLayer', true)  % Explicitly mark as output layer
];

satNet = dlnetwork(satLayers);



%% Normaling the Target Data to the range [-1, 1]
inputs = (inputs*2) -1;
hueTargets = [hue_targets_sin; hue_targets_cos];

% Normalize the target data to the range [-1, 1]
hueTargets = (hueTargets * 2) - 1;
satTargets = (sat_targets * 2) - 1;

% Specify training options
opts = trainingOptions('sgdm', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 250, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the hue network
hueNet = trainNetwork(inputs, hueTargets, hueLayers, opts);

% Train the saturation network
satNet = trainNetwork(inputs, satTargets, satLayers, opts);

%% Make Predictions and Evaluate

% Predict hue
allHuePredictionsNormalized = predict(hueNet, inputs);
allHuePredictions_sin = (allHuePredictionsNormalized(1,:) + 1) / 2;
allHuePredictions_cos = (allHuePredictionsNormalized(2,:) + 1) / 2;

% Convert hue predictions back to [0, 1] range
allHuePredictions = atan2(allHuePredictions_sin, allHuePredictions_cos);
allHuePredictions(allHuePredictions < 0) = allHuePredictions(allHuePredictions < 0) + 2 * pi;
allHuePredictions = allHuePredictions / (2 * pi);

% Predict saturation
allSatPredictionsNormalized = predict(satNet, inputs);
allSatPredictions = (allSatPredictionsNormalized + 1) / 2;

% Denormalize sat_targets
sat_targets = (sat_targets + 1) / 2;

%% Identify Outliers in Predictions in Hue
% Predict on the entire dataset
allHuePredictionsNormalized = pacific_blue_hue_net(inputs);
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

allSatPredictionsNormalized = pacific_blue_sat_net(inputs);
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
%% Saving Trained network
save(['pacific_blue_nets.mat'], 'pacific_blue_hue_net', 'pacific_blue_sat_net');

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