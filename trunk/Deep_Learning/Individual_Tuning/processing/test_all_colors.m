%% Script to test all color-specific models
clc; clear all; close all;

% Define color variants to test
color_variants = {'celeste', 'tidal', 'pacific_blue'};

% Test each color model
for i = 1:length(color_variants)
    color_name = color_variants{i};
    fprintf('\n\n========= TESTING %s MODEL =========\n\n', upper(color_name));
    
    % Test the model
    test_color_model(color_name);
end

fprintf('\n\nAll color models have been tested!\n');