% Aggregate Statistics Calculator
% This script loads multiple run files and calculates aggregate statistics
% (mean and standard deviation) for specified variables across all runs

% Define the run files to process - modify these filenames as needed
run_files = {'run1.mat', 'run2.mat', 'run3.mat', 'run4.mat', 'run5.mat', ...
             'run6.mat', 'run7.mat', 'run8.mat', 'run9.mat', 'run10.mat'};

% Define the variables to analyze
variables_to_analyze = {'p', 'q', 'beta', 'r'};

% Display info message at start
fprintf('Calculating aggregate statistics for variables: ');
fprintf('%s ', variables_to_analyze{:});
fprintf('\n');

% Initialize data containers
all_data = struct();
for i = 1:length(variables_to_analyze)
    var_name = variables_to_analyze{i};
    all_data.(var_name) = [];
end

% Count valid files
valid_files = 0;

% Process each run file
for run_idx = 1:length(run_files)
    current_file = run_files{run_idx};
    
    % Skip if file doesn't exist
    if ~exist(current_file, 'file')
        fprintf('File not found: %s - skipping\n', current_file);
        continue;
    end
    
    fprintf('Processing file: %s\n', current_file);
    valid_files = valid_files + 1;
    
    % Load data
    temp_data = load(current_file);
    
    % Extract and concatenate each variable
    for i = 1:length(variables_to_analyze)
        var_name = variables_to_analyze{i};
        
        if isfield(temp_data, var_name)
            var_data = temp_data.(var_name);
            
            % Handle different data types
            if isstruct(var_data)
                % If it's a struct with a data field (common for timeseries)
                if isfield(var_data, 'data')
                    var_data = var_data.data;
                else
                    fprintf('Warning: Variable %s is a struct but has no data field - skipping\n', var_name);
                    continue;
                end
            end
            
            % Ensure data is a column vector
            if size(var_data, 2) > 1 && size(var_data, 1) > 1
                fprintf('Warning: Variable %s has multiple columns, using only the first column\n', var_name);
                var_data = var_data(:, 1);
            end
            
            % If it's a row vector, transpose it
            if size(var_data, 1) == 1
                var_data = var_data';
            end
            
            % Concatenate with existing data
            all_data.(var_name) = [all_data.(var_name); var_data];
        else
            fprintf('Warning: Variable %s not found in file %s\n', var_name, current_file);
        end
    end
end

% Check if we found any valid data
if valid_files == 0
    error('No valid run files found. Please check file paths and names.');
end

fprintf('\nProcessed %d valid run files\n', valid_files);

% Calculate statistics for each variable
fprintf('\n%-10s %-15s %-15s %-15s %-15s %-15s\n', 'Variable', 'Mean', 'Std Dev', 'Min', 'Median', 'Max');
fprintf('-------------------------------------------------------------------\n');

for i = 1:length(variables_to_analyze)
    var_name = variables_to_analyze{i};
    
    if ~isempty(all_data.(var_name))
        % Calculate statistics
        mean_val = mean(all_data.(var_name));
        std_val = std(all_data.(var_name));
        min_val = min(all_data.(var_name));
        median_val = median(all_data.(var_name));
        max_val = max(all_data.(var_name));
        
        % Print results
        fprintf('%-10s %-15.6f %-15.6f %-15.6f %-15.6f %-15.6f\n', ...
                var_name, mean_val, std_val, min_val, median_val, max_val);
    else
        fprintf('%-10s %-15s %-15s %-15s %-15s %-15s\n', ...
                var_name, 'No data', '-', '-', '-', '-');
    end
end

% Create a more detailed results struct
results = struct();
for i = 1:length(variables_to_analyze)
    var_name = variables_to_analyze{i};
    
    if ~isempty(all_data.(var_name))
        results.(var_name).mean = mean(all_data.(var_name));
        results.(var_name).std = std(all_data.(var_name));
        results.(var_name).min = min(all_data.(var_name));
        results.(var_name).max = max(all_data.(var_name));
        results.(var_name).median = median(all_data.(var_name));
        results.(var_name).data = all_data.(var_name);
        results.(var_name).count = length(all_data.(var_name));
    else
        results.(var_name).mean = NaN;
        results.(var_name).std = NaN;
        results.(var_name).min = NaN;
        results.(var_name).max = NaN;
        results.(var_name).median = NaN;
        results.(var_name).data = [];
        results.(var_name).count = 0;
    end
end

% Save results
save('aggregate_statistics_results.mat', 'results');
fprintf('\nFull results saved to aggregate_statistics_results.mat\n');

% Create histograms for visual analysis
figure('Position', [100, 100, 1000, 800]);

for i = 1:length(variables_to_analyze)
    var_name = variables_to_analyze{i};
    
    if ~isempty(all_data.(var_name))
        subplot(2, 2, i);
        histogram(all_data.(var_name), min(50, ceil(sqrt(length(all_data.(var_name))))));
        title([var_name, ' Distribution']);
        xlabel('Value');
        ylabel('Frequency');
        
        % Add mean and std lines
        hold on;
        mean_val = results.(var_name).mean;
        std_val = results.(var_name).std;
        
        % Get y-limits to scale the lines
        y_lim = ylim;
        
        % Plot mean line
        plot([mean_val, mean_val], y_lim, 'r-', 'LineWidth', 2);
        text(mean_val, 0.95*y_lim(2), [' \mu = ', num2str(mean_val, '%.4f')], ...
             'Color', 'r', 'VerticalAlignment', 'top');
        
        % Plot mean ± std lines
        plot([mean_val-std_val, mean_val-std_val], y_lim, 'r--', 'LineWidth', 1.5);
        plot([mean_val+std_val, mean_val+std_val], y_lim, 'r--', 'LineWidth', 1.5);
        text(mean_val-std_val, 0.85*y_lim(2), [' \mu-\sigma'], ...
             'Color', 'r', 'VerticalAlignment', 'top');
        text(mean_val+std_val, 0.85*y_lim(2), [' \mu+\sigma'], ...
             'Color', 'r', 'VerticalAlignment', 'top');
        
        hold off;
    end
end

% Add overall title
sgtitle('Aggregate Statistics Across All Runs');

% Save figure
saveas(gcf, 'aggregate_statistics_histograms.png');
fprintf('Histogram visualization saved as aggregate_statistics_histograms.png\n');

% Optional: Create a box plot for comparison
figure('Position', [150, 150, 800, 600]);

% Prepare data for boxplot
boxplot_data = [];
boxplot_labels = {};
for i = 1:length(variables_to_analyze)
    var_name = variables_to_analyze{i};
    if ~isempty(all_data.(var_name))
        boxplot_data = [boxplot_data; all_data.(var_name)];
        boxplot_labels = [boxplot_labels; repmat({var_name}, length(all_data.(var_name)), 1)];
    end
end

% Create boxplot if data is available
if ~isempty(boxplot_data)
    boxplot(boxplot_data, boxplot_labels);
    title('Variable Distributions Across All Runs');
    ylabel('Value');
    grid on;
    
    % Save boxplot
    saveas(gcf, 'aggregate_statistics_boxplot.png');
    fprintf('Boxplot visualization saved as aggregate_statistics_boxplot.png\n');
end