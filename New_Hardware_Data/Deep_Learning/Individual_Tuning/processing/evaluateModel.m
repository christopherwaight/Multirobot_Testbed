function [circular_RMSE_hue, circular_R2_hue, R2_sat, RMSE_sat] = evaluateModel(hue_net, sat_net, inputs_normalized, hue_targets, sat_targets, label,dataset_name)


    %% Make Predictions
    
    % Prediction Hue (2 neurons)
    huePredictionsNormalized = hue_net(inputs_normalized);
    huePredictions_sin = (huePredictionsNormalized(1,:)+1)/2;
    huePredictions_cos = (huePredictionsNormalized(2,:)+1)/2;
    
    % Convert back to single hue value
    huePredictions = atan2(huePredictions_sin, huePredictions_cos);
    huePredictions(huePredictions < 0) = huePredictions(huePredictions < 0) + 2 * pi;
    huePredictions = huePredictions / (2 * pi);
    
    % Predict Saturation
    satPredictionsNormalized = sat_net(inputs_normalized);
    satPredictions = (satPredictionsNormalized+1)/2;
    satPredictions = max(min(satPredictions, 1), 0);  % Ensure values are in range
    
    %% Calculate metrics
       
    % Specify Predictions and Targets

    hue_targets_vec = hue_targets(:);
    huePredictions_vec = huePredictions(:);
    sat_targets_vec = sat_targets(:);
    satPredictions_vec = satPredictions(:);
    
    
    % Circular RMSE for Hue
    circular_errors = min(abs(hue_targets_vec - huePredictions_vec), ...
                         1 - abs(hue_targets_vec - huePredictions_vec));
    circular_RMSE_hue = sqrt(mean(circular_errors.^2));
    
    
    % Circular R-squared for HUE
    hue_mean = mean(hue_targets_vec);
    circular_errors_from_mean = min(abs(hue_targets_vec - hue_mean), ...
                                  1 - abs(hue_targets_vec - hue_mean));
    circular_SST_hue = sum(circular_errors_from_mean.^2);
    circular_SSE_hue = sum(circular_errors.^2);
    circular_R2_hue = 1 - circular_SSE_hue/circular_SST_hue;

    
    % Standard R-squared calculation for Saturation
    SST_sat = sum((sat_targets_vec - mean(sat_targets_vec)).^2);
    SSE_sat = sum((sat_targets_vec - satPredictions_vec).^2);
    R2_sat = 1 - SSE_sat/SST_sat;
    
    
    % Standard RMSE calculation for Saturation
    RMSE_sat = sqrt(mean((sat_targets_vec - satPredictions_vec).^2));


    % Print Statistics
    fprintf('\n=== %s on %s ===\n', label, dataset_name);
    fprintf('Hue: Circular RMSE = %.4f, Circular R² = %.4f\n', circular_RMSE_hue, circular_R2_hue);
    fprintf('Sat: RMSE = %.4f, R² = %.4f\n', RMSE_sat, R2_sat);
end