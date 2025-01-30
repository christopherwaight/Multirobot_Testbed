%% Create and Train a Feedforward Neural Network
% This example shows how to train a feedforward neural network to predict temperature.
% Copyright 2018-2024 The MathWorks, Inc.
%% Read Data from the Weather Station ThingSpeak Channel
% ThingSpeak&trade; channel 12397 contains data from the MathWorks&reg; weather station, located in Natick, Massachusetts.
% The data is collected once every minute. Fields 2, 3, 4, and 6 contain wind speed (mph), relative humidity, temperature (F), and atmospheric pressure (inHg) data, respectively.
% Read the data from channel 12397 using the |thingSpeakRead| function.
data = thingSpeakRead(12397,'Fields',[2 3 4 6],'DateRange',[datetime('January 7, 2018'),datetime('January 9, 2018')],...
    'outputFormat','table');

%% Assign Input Variables and Target Values
inputs = [data.Humidity'; data.TemperatureF'; data.PressureHg'; data.WindSpeedmph'];
tempC = (5/9)*(data.TemperatureF-32);
b = 17.62;
c = 243.5;
gamma = log(data.Humidity/100) + b*tempC ./ (c+tempC);
dewPointC = c*gamma ./ (b-gamma);
dewPointF = (dewPointC*1.8) + 32;
targets = dewPointF';

%% Create and Train the Deeper Feedforward Network
hiddenLayerSizes = [10 8 5]; % Define the number of neurons in each hidden layer for a deeper network
net = feedforwardnet(hiddenLayerSizes);

% Customize training parameters
net.trainFcn = 'trainlm';  % Use Levenberg-Marquardt algorithm (you can change this)
net.trainParam.epochs = 1000; % Set the maximum number of epochs
net.trainParam.showWindow = true; % Turn off display progress dialog.
net.divideFcn = 'dividerand'; % Randomly divide data into training, validation, and test sets (default)
net.divideParam.trainRatio = 0.7; % 70% of data for training
net.divideParam.valRatio = 0.15; % 15% of data for validation
net.divideParam.testRatio = 0.15; % 15% of data for testing
%net.trainParam.mu_max = 1e20; %You can uncomment these if the model trains too slowly.
%net.trainParam.mu = 10;
% Normalize the input data (good practice!)
[inputs,ps] = mapminmax(inputs); % Normalize inputs to the range [-1, 1]
[targets,ts] = mapminmax(targets); % Normalize targets to the range [-1, 1]

% Now, train the network with normalized data
[net,tr] = train(net,inputs,targets);

%% Use the Trained Model to Predict Data and Denormalize
% Predict on a specific sample (e.g., the 5th sample)
predictedOutputNormalized = net(inputs(:,5));
predictedOutput = mapminmax('reverse', predictedOutputNormalized, ts); % Denormalize the prediction

%% Verification
% Denormalize the actual target for comparison
actualTarget = mapminmax('reverse', targets(5), ts); % Denormalize the actual target

% Display the denormalized predicted and actual values
disp(['Predicted Dew Point (F): ', num2str(predictedOutput)]);
disp(['Actual Dew Point (F): ', num2str(actualTarget)]);
%% Identify Outliers in Predictions
% Predict on the entire dataset
allPredictionsNormalized = net(inputs);
allPredictions = mapminmax('reverse', allPredictionsNormalized, ts); % Denormalize

% Calculate the error
errors = targets - allPredictionsNormalized; % Use normalized targets here for accurate error calculation

% Define an outlier threshold (e.g., 2 standard deviations away from the mean error)
errorMean = mean(errors);
errorStd = std(errors);
outlierThreshold = 3 * errorStd;

% Find outlier indices
outlierIndices = find(abs(errors - errorMean) > outlierThreshold);

% Display outlier information
disp('Outliers:');
for i = 1:length(outlierIndices)
    index = outlierIndices(i);
    disp(['  Sample Index: ', num2str(index), ', Error: ', num2str(errors(index)), ', Predicted: ', num2str(allPredictions(index)), ', Actual: ', num2str(mapminmax('reverse', targets(index), ts))]);
end

%% Saving
% Save the trained network
save('trainedNetwork.mat', 'net');