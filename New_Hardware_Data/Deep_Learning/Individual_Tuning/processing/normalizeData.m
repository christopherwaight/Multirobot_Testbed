function [normalized_inputs] = normalizeData(inputs)


    % Check if normalization is needed
    if max(inputs(:,1)) > 1
        % Normalize RGB+K values to [0, 1] range
        normalized_inputs = inputs;
        normalized_inputs(:,1) = (normalized_inputs(:,1)-100)/1450;  % R
        normalized_inputs(:,2) = (normalized_inputs(:,2)-175)/1700;  % G
        normalized_inputs(:,3) = (normalized_inputs(:,3)-150)/1500;  % B
        normalized_inputs(:,4) = (normalized_inputs(:,4)-675)/4350;  % K
        normalized_inputs = max(min(normalized_inputs, 1), 0);  % Clip to [0, 1]
    else
        % Data is already normalized
        normalized_inputs = inputs;
    end
end