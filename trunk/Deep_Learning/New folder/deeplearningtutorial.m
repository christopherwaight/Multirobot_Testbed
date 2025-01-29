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
%% Create and Train the Two-Layer Feedforward Network

net = feedforwardnet(10); % Use the |feedforwardnet| function to create a two-layer feedforward network. The network has one hidden layer with 10 neurons and an output layer.
net.trainParam.showWindow = true; % Turn off display progress dialog.
[net,tr] = train(net,inputs,targets); % Use the |train| function to train the feedforward network using the inputs.

%% Use the Trained Model to Predict Data
output = net(inputs(:,5))

%% Verification
targets(5)