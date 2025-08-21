function [inputs_with_hsv] = addHSVFeatures(inputs)
% ADDHSVFEATURES Add HSV features to normalized RGB+K inputs
%   [inputs_with_hsv] = addHSVFeatures(inputs)
%
%   Inputs:
%       inputs - normalized RGB+K input features (N x 4 matrix)
%
%   Outputs:
%       inputs_with_hsv - inputs with HSV features added (N x 7 matrix)

    % Check if HSV features need to be added
    if size(inputs, 2) < 5
        % Calculate HSV from RGB
        hsv = rgb2hsv(inputs(:, 1:3));
        
        % Concatenate HSV features to the input
        inputs_with_hsv = [inputs, hsv];
        
        % Ensure all values are in [0, 1] range
        inputs_with_hsv = max(min(inputs_with_hsv, 1), 0);
    else
        % HSV features already exist
        inputs_with_hsv = inputs;
    end
end