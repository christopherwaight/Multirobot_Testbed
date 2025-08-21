function [inputs_normalized, hue_targets_sin_normalized, hue_targets_cos_normalized, sat_targets_normalized] = preprocessData(inputs, hue_targets, sat_targets)
% PREPROCESSDATA Preprocess data for neural network training
%   [inputs_normalized, hue_targets_sin_normalized, hue_targets_cos_normalized, sat_targets_normalized] = preprocessData(inputs, hue_targets, sat_targets)
%
%   Inputs:
%       inputs - Input features with RGB+K and potentially HSV
%       hue_targets - hue target values
%       sat_targets - saturation target values
%
%   Outputs:
%       inputs_normalized - normalized inputs ready for NN (transposed and in [-1,1] range)
%       hue_targets_sin_normalized - normalized sin component of hue
%       hue_targets_cos_normalized - normalized cos component of hue
%       sat_targets_normalized - normalized saturation targets

    % Ensure inputs are normalized and have HSV features
    inputs = normalizeData(inputs);
    inputs = addHSVFeatures(inputs);
    
    % Decompose Hue into sine and cosine components for circular representation
    hue_targets_sin = sin(2 * pi * hue_targets);
    hue_targets_cos = cos(2 * pi * hue_targets);
    
    % Transpose inputs and targets for NN compatibility
    inputs = inputs';
    hue_targets_sin = hue_targets_sin';
    hue_targets_cos = hue_targets_cos';
    sat_targets = sat_targets';
    
    % Normalize to range [-1, 1] for better neural network performance
    inputs_normalized = (inputs*2) - 1;
    hue_targets_sin_normalized = (hue_targets_sin*2) - 1;
    hue_targets_cos_normalized = (hue_targets_cos*2) - 1;
    sat_targets_normalized = (sat_targets*2) - 1;
end