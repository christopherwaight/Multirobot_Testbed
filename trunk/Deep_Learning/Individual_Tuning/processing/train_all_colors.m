%% Main Script for Training All Color-Specific Networks
clc; clear all; close all;

% Define color variants to train
color_variants = {'celeste', 'tidal', 'pacific_blue'};

% Number of iterations per color
iterations = 333;

% Train model for each color variant
for i = 1:length(color_variants)
    color_name = color_variants{i};
    fprintf('\n\n========= TRAINING %s MODEL =========\n\n', upper(color_name));
    
    % Train the model
    [hue_net, sat_net] = train_single_model(color_name, iterations);
    
    % Save the final models with specific color names
    save(['outputs/' color_name '/' color_name '_nets.mat'], 'hue_net', 'sat_net');
    save(['outputs/' color_name '/' color_name '_hue_net.mat'], 'hue_net');
    save(['outputs/' color_name '/' color_name '_sat_net.mat'], 'sat_net');
    
    fprintf('\n%s models trained and saved successfully.\n', color_name);
end

fprintf('\n\nAll color models have been trained and saved!\n');