function [hue_net, sat_net] = train_single_model(color_name, iterations)

    % Set default iterations if not provided
    if nargin < 2
        iterations = 50;
    end
    
    % Define validation datasets for this color
    validation_datasets = {
        ['data/' color_name '_cal.csv'],
        ['data/' color_name '_ver.csv']
    };
    
    % Initialize best scores
    best_score_hue = 1e6;
    best_score_sat = 1e6;
    
    % Initialize arrays to track scores over time
    score_history_hue = zeros(1, iterations);
    score_history_sat = zeros(1, iterations);
    best_score_history_hue = ones(1, iterations) * 1e6;
    best_score_history_sat = ones(1, iterations) * 1e6;
    
    % Create figure for real-time plotting
    figure('Position', [100, 100, 1000, 600], 'Name', ['Training Progress - ' color_name]);
    
    % Create subplots for Hue and Saturation
    subplot(2, 1, 1);
    h_plot_hue = plot(1, 1e6, 'b-', 'LineWidth', 2);
    hold on;
    h_current_hue = plot(1, 1e6, 'r.', 'MarkerSize', 10);
    title(['Hue Model - Max Circular RMSE Progress (' color_name ')']);
    xlabel('Iteration');
    ylabel('Max Circular RMSE');
    ylim([0, 0.1]);  % Adjust based on your expected range
    xlim([1, min(50, iterations)]);  % Start with first 50 iterations
    grid on;
    
    subplot(2, 1, 2);
    h_plot_sat = plot(1, 1e6, 'b-', 'LineWidth', 2);
    hold on;
    h_current_sat = plot(1, 1e6, 'r.', 'MarkerSize', 10);
    title(['Saturation Model - Max RMSE Progress (' color_name ')']);
    xlabel('Iteration');
    ylabel('Max RMSE');
    ylim([0, 0.1]);  % Adjust based on your expected range
    xlim([1, min(50, iterations)]);  % Start with first 50 iterations
    grid on;
    
    drawnow;
    
    % Random Search to find best architecture and model
    for j = 1:iterations
        fprintf('\n=== Training Iteration %d for %s ===\n', j, color_name);
        
        %% Load Data and Train Network
        % Load raw training data for this specific color
        [inputs_raw, hue_targets_raw, sat_targets_raw] = loadSingleColorData(color_name);
        
        % Data Augmentation
        [inputs_augmented, hue_targets, sat_targets] = augmentData(inputs_raw, hue_targets_raw, sat_targets_raw);
        
        % Data Preprocessing for neural network training
        [inputs_processed, hue_targets_sin_norm, hue_targets_cos_norm, sat_targets_norm] = preprocessData(inputs_augmented, hue_targets, sat_targets);
        
        % Create neural networks with random architecture
        [hue_net, sat_net] = createNetworks();
        
        % Train neural networks
        [hue_net, sat_net, tr1, tr2] = trainNetworks(hue_net, sat_net, inputs_processed, hue_targets_sin_norm, hue_targets_cos_norm, sat_targets_norm);
        
        % Save current networks temporarily
        output_dir = 'outputs/' + string(color_name);
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        save(output_dir + '/nets_current.mat', 'hue_net', 'sat_net');
        
        % Initialize arrays to hold evaluation metrics across all validation datasets
        circular_RMSE_hue_all = zeros(1, length(validation_datasets));
        RMSE_sat_all = zeros(1, length(validation_datasets));
        
        %% Evaluate on all validation datasets
        for dataset_idx = 1:length(validation_datasets)
            dataset_name = validation_datasets{dataset_idx};
                    
            % Load validation data
            [val_inputs_raw, val_hue_targets, val_sat_targets] = loadTestingData(dataset_name);
            
            % Preprocess validation data (no augmentation for validation)
            [val_inputs_processed, ~, ~, ~] = preprocessData(val_inputs_raw, val_hue_targets, val_sat_targets);
            
            % Evaluate model on validation data
            [val_circular_RMSE_hue, val_circular_R2_hue, R2_sat_val, val_RMSE_sat] = evaluateModel(hue_net, sat_net, val_inputs_processed, val_hue_targets, val_sat_targets, 'Validation', dataset_name);
            
            % Store metrics for this dataset
            circular_RMSE_hue_all(dataset_idx) = val_circular_RMSE_hue;
            RMSE_sat_all(dataset_idx) = val_RMSE_sat;
        end
        
        % Calculate max RMSE across all datasets for both hue and saturation
        max_circular_RMSE_hue = max(circular_RMSE_hue_all);
        max_RMSE_sat = max(RMSE_sat_all);
        
        % Store current scores
        score_history_hue(j) = max_circular_RMSE_hue;
        score_history_sat(j) = max_RMSE_sat;
        
        % Update best scores history
        if j == 1
            best_score_history_hue(j) = max_circular_RMSE_hue;
            best_score_history_sat(j) = max_RMSE_sat;
        else
            best_score_history_hue(j) = min(best_score_history_hue(j-1), max_circular_RMSE_hue);
            best_score_history_sat(j) = min(best_score_history_sat(j-1), max_RMSE_sat);
        end
        
        % Update plots
        set(h_plot_hue, 'XData', 1:j, 'YData', best_score_history_hue(1:j));
        set(h_current_hue, 'XData', j, 'YData', max_circular_RMSE_hue);
        
        set(h_plot_sat, 'XData', 1:j, 'YData', best_score_history_sat(1:j));
        set(h_current_sat, 'XData', j, 'YData', max_RMSE_sat);
        
        % Adjust x-axis as needed
        if j > 50
            subplot(2, 1, 1);
            xlim([1, j]);
            subplot(2, 1, 2);
            xlim([1, j]);
        end
        
        % Add iteration number and best scores to figure title
        sgtitle(sprintf('%s Training Progress - Iteration %d / %d\nBest Hue: %.4f, Best Sat: %.4f', ...
            color_name, j, iterations, best_score_history_hue(j), best_score_history_sat(j)));
        
        drawnow;
        
        % Print overall performance metrics
        fprintf('\n--- Overall Performance Metrics for %s ---\n', color_name);
        fprintf('Hue Circular RMSE for all datasets: ');
        fprintf('%.4f ', circular_RMSE_hue_all);
        fprintf('\nMax Hue Circular RMSE: %.4f\n', max_circular_RMSE_hue);
        
        fprintf('Saturation RMSE for all datasets: ');
        fprintf('%.4f ', RMSE_sat_all);
        fprintf('\nMax Saturation RMSE: %.4f\n', max_RMSE_sat);
        
        %% Saving the best model
        hue_updated = false;
        sat_updated = false;
        
        % Checking if Hue Improved
        if max_circular_RMSE_hue < best_score_hue
            best_score_hue = max_circular_RMSE_hue;
            fprintf('\n==== UPDATE: HUE MODEL for %s! ====\n', color_name);
            save(output_dir + '/hue_net_best.mat', 'hue_net');
            hue_updated = true;
        end
        
        % Checking if Sat Improved
        if max_RMSE_sat < best_score_sat
            best_score_sat = max_RMSE_sat;
            fprintf('\n==== UPDATE: SAT MODEL for %s! ====\n', color_name);
            save(output_dir + '/sat_net_best.mat', 'sat_net');
            sat_updated = true;
        end
        
        % Update nets_best if there is any improvement
        if hue_updated || sat_updated
            % If both models were updated in this iteration, we can save directly
            if hue_updated && sat_updated
                save(output_dir + '/nets_best.mat', 'hue_net', 'sat_net');
            else
                % At least one model needs to be loaded from file
                % First, create temporary variables for the best models
                best_hue_net = hue_net;  % Start with current models
                best_sat_net = sat_net;
                
                % If hue wasn't updated but there's a saved best version, load it
                if ~hue_updated
                    hue_data = load(output_dir + '/hue_net_best.mat');
                    best_hue_net = hue_data.hue_net;
                end
                
                % If saturation wasn't updated, load the saved best version
                if ~sat_updated
                    sat_data = load(output_dir + '/sat_net_best.mat');
                    best_sat_net = sat_data.sat_net;
                end
                
                % Now save the combined file with the best of both models
                hue_net = best_hue_net;
                sat_net = best_sat_net;
                save(output_dir + '/nets_best.mat', 'hue_net', 'sat_net');
            end
            
            fprintf('\n==== Updated combined model file with best versions for %s ====\n', color_name);
        end
    end
    
    % Save the score history
    save(output_dir + '/training_progress.mat', 'score_history_hue', 'score_history_sat', ...
        'best_score_history_hue', 'best_score_history_sat');
    
    % Create final summary figure
    figure('Position', [200, 200, 1200, 800], 'Name', ['Training Summary - ' color_name]);
    subplot(2, 1, 1);
    plot(1:iterations, score_history_hue, 'r.', 'MarkerSize', 6);
    hold on;
    plot(1:iterations, best_score_history_hue, 'b-', 'LineWidth', 2);
    title(['Hue Model Training Progress (' color_name ')']);
    xlabel('Iteration');
    ylabel('Max Circular RMSE');
    grid on;
    legend('Current Iteration Score', 'Best Score So Far', 'Location', 'northeast');
    
    subplot(2, 1, 2);
    plot(1:iterations, score_history_sat, 'r.', 'MarkerSize', 6);
    hold on;
    plot(1:iterations, best_score_history_sat, 'b-', 'LineWidth', 2);
    title(['Saturation Model Training Progress (' color_name ')']);
    xlabel('Iteration');
    ylabel('Max RMSE');
    grid on;
    legend('Current Iteration Score', 'Best Score So Far', 'Location', 'northeast');
    
    sgtitle(sprintf('%s Training Summary - %d Iterations\nFinal Best Scores - Hue: %.4f, Saturation: %.4f', ...
        color_name, iterations, best_score_history_hue(end), best_score_history_sat(end)));
    
    saveas(gcf, output_dir + '/training_summary.png');
    
    %% Training Complete
    fprintf('\nTraining complete for %s. Best Hue RMSE: %.4f, Best Saturation RMSE: %.4f\n', color_name, best_score_hue, best_score_sat);
    
    %% Final Evaluation of Best Models
    fprintf('\n==== Final Evaluation of Best Models for %s ====\n', color_name);
    
    % Load the best models
    fprintf('Loading best models from %s/nets_best.mat\n', output_dir);
    best_models = load(output_dir + '/nets_best.mat');
    best_hue_net = best_models.hue_net;
    best_sat_net = best_models.sat_net;
    
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
        [hue_circular_RMSE, hue_circular_R2, sat_R2, sat_RMSE] = evaluateModel(best_hue_net, best_sat_net, val_inputs_processed, val_hue_targets, val_sat_targets, 'Final Evaluation', dataset_name);
        
        % Store metrics for this dataset
        hue_circular_RMSE_all(dataset_idx) = hue_circular_RMSE;
        hue_circular_R2_all(dataset_idx) = hue_circular_R2;
        sat_RMSE_all(dataset_idx) = sat_RMSE;
        sat_R2_all(dataset_idx) = sat_R2;
    end
    
    % Print summary of results
    fprintf('\n%s Training complete.\n', color_name);
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
    
    % Return the best models
    hue_net = best_hue_net;
    sat_net = best_sat_net;
end