function [augmentedInputs, augmentedHueTargets, augmentedSatTargets] = augmentData(inputs, hue_targets, sat_targets)
% AUGMENTDATA Apply data augmentation to the training data
%   [augmentedInputs, augmentedHueTargets, augmentedSatTargets] = augmentData(inputs, hue_targets, sat_targets)
%
%   Inputs:
%       inputs - RGB+K input features (can be raw or normalized)
%       hue_targets - hue target values
%       sat_targets - saturation target values
%
%   Outputs:
%       augmentedInputs - combined original and augmented inputs with HSV features
%       augmentedHueTargets - combined original and augmented hue targets
%       augmentedSatTargets - combined original and augmented saturation targets

    % Data Augmentation parameters
    noiseLevelRGB = randi([5, 20])*0.001;; % Baseline noise level
    numAugmentations = randi([1, 2]); % Number of augmented copies per original sample
    
    % First normalize the inputs if they aren't already
    inputsNorm = normalizeData(inputs);
    
    % Storage for augmented data
    augmentedInputs_temp = [];
    augmentedHueTargets_temp = [];
    augmentedSatTargets_temp = [];
    
    % Generate augmented data by adding noise
    for i = 1:size(inputsNorm, 1)
        for k = 1:numAugmentations
            % Add Gaussian noise to RGBK (first 4 features)
            noisyInput = inputsNorm(i, 1:4) + noiseLevelRGB * randn(1, 4);
    
            % Clip to [0, 1] range
            noisyInput = max(0, min(1, noisyInput));
    
            % Add to augmented data collections
            augmentedInputs_temp = [augmentedInputs_temp; noisyInput];
            augmentedHueTargets_temp = [augmentedHueTargets_temp; hue_targets(i)];
            augmentedSatTargets_temp = [augmentedSatTargets_temp; sat_targets(i)];
        end
    end
    
    % Add HSV features to both original and augmented inputs
    inputsWithHSV = addHSVFeatures(inputsNorm);
    augmentedInputs_temp = addHSVFeatures(augmentedInputs_temp);
    
    % Combine original and augmented data
    augmentedInputs = [inputsWithHSV; augmentedInputs_temp];
    augmentedHueTargets = [hue_targets; augmentedHueTargets_temp];
    augmentedSatTargets = [sat_targets; augmentedSatTargets_temp];
end