function [everything_hue_net, everything_sat_net] = createNetworks()


    % Generate random network architectures
    hiddenLayerSizes1 = [randi([4, 12]) randi([4, 12]) ]; % 2 Neuron Output for Hue
    hiddenLayerSizes2 = [randi([4, 12]) randi([4, 12]) ]; % 1 Neuron Output for Saturation

    % Occasionally try a 3-layer network (20% of the time)
    if rand < 0.1
        hiddenLayerSizes1 = [randi([4, 12]) randi([4, 10]) randi([2, 10])];
        hiddenLayerSizes2 = [randi([4, 12]) randi([4, 10]) randi([2, 10])];
    end


    
    % Create hue network
    everything_hue_net = feedforwardnet(hiddenLayerSizes1);
    
    % Create saturation network
    everything_sat_net = feedforwardnet(hiddenLayerSizes2);
    
    % Configure hue network
    everything_hue_net.trainFcn = 'trainlm';
    everything_hue_net.trainParam.epochs = 48000;
    everything_hue_net.trainParam.showWindow = false;
    everything_hue_net.divideFcn = 'dividerand';
    everything_hue_net.divideParam.trainRatio = 0.7;
    everything_hue_net.divideParam.valRatio = 0.15;
    everything_hue_net.divideParam.testRatio = 0.15;
    everything_hue_net.trainParam.max_fail = randi([15, 30]);     % Increase patience
    everything_hue_net.trainParam.mu = randi([5, 20])*0.001;  % Random initial mu

    
    % Configure saturation network
    everything_sat_net.trainFcn = 'trainlm';
    everything_sat_net.trainParam.epochs = 4800;
    everything_sat_net.trainParam.showWindow = false;
    everything_sat_net.divideFcn = 'dividerand';
    everything_sat_net.divideParam.trainRatio = 0.7;
    everything_sat_net.divideParam.valRatio = 0.15;
    everything_sat_net.divideParam.testRatio = 0.15;
    everything_sat_net.trainParam.max_fail = 25;
    everything_sat_net.trainParam.mu = randi([5, 20])*0.001;
end