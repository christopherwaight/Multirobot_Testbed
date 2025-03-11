function test_color_model(color_name)
% TEST_COLOR_MODEL Test a trained color-specific model on its calibration and verification data
%   test_color_model(color_name)
%
%   Inputs:
%       color_name - String name of the color variant (e.g., 'celeste', 'tidal', 'pacific_blue')

    % Define validation datasets for this color
    validation_datasets = {
        ['data/' color_name '_cal.csv'],
        ['data/' color_name '_ver.csv']
    };
    
    % Load the best models
    model_path = ['outputs/' color_name '/' color_name '_nets.mat'];
    fprintf('Loading models from %s\n', model_path);
    
    try
        models = load(model_path);
        hue_net = models.hue_net;
        sat_net = models.sat_net;
    catch e
        fprintf('Error loading models: %s\n', e.message);
        return;
    end
    
    % Initialize arrays to hold evaluation metrics across all validation datasets
    hue_circular_RMSE_all = zeros(1, length(validation_datasets));
    hue_circular_R2_all = zeros(1, length(validation_datasets));
    sat_RMSE_all = zeros(1, length(validation_datasets));
    sat_R2_all = zeros(1, length(validation_datasets));
    
    % Evaluate on all validation datasets
    for dataset_idx = 1:length(validation_datasets)
        dataset_name = validation_datasets{dataset_idx};
        
        % Load validation data
        [val_inputs_raw, val_hue_targets, val_sat_targets] = loadTestingData(dataset_name);
        
        % Preprocess validation data
        [val_inputs_processed, ~, ~, ~] = preprocessData(val_inputs_raw, val_hue_targets, val_sat_targets);
        
        % Evaluate model on validation data
        [hue_circular_RMSE, hue_circular_R2, sat_R2, sat_RMSE] = evaluateModel(hue_net, sat_net, val_inputs_processed, val_hue_targets, val_sat_targets, ['Test: ' color_name], dataset_name);
        
        % Store metrics for this dataset
        hue_circular_RMSE_all(dataset_idx) = hue_circular_RMSE;
        hue_circular_R2_all(dataset_idx) = hue_circular_R2;
        sat_RMSE_all(dataset_idx) = sat_RMSE;
        sat_R2_all(dataset_idx) = sat_R2;
    end
    
    % Print summary of results
    fprintf('\n%s Model Evaluation Results:\n', color_name);
    fprintf('Hue Circular RMSE for all datasets: ');
    fprintf('%.4f ', hue_circular_RMSE_all);
    fprintf('\n');
    fprintf('Hue Circular R² for all datasets: ');
    fprintf('%.4f ', hue_circular_R2_all);
    fprintf('\n');
    fprintf('Saturation RMSE for all datasets: ');
    fprintf('%.4f ', sat_RMSE_all);
    fprintf('\n');
    fprintf('Saturation R² for all datasets: ');
    fprintf('%.4f ', sat_R2_all);
    fprintf('\n');
end