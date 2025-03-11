%% Find Min and Max Values Across Multiple Files
% This script stacks all specified CSV files and finds the min/max of each column

clc; clear all; close all;

% Define the list of files to process
validation_datasets = {
    'data/celeste_cal.csv',
    'data/tidal_cal.csv',
    'data/pacific_blue_cal.csv',
    'data/redwood_cal.csv',
    'data/celeste_ver.csv',
    'data/tidal_ver.csv',
    'data/pacific_blue_ver.csv'
};

% Initialize empty array for stacked data
all_data = [];

% Load and stack all files
fprintf('Loading and stacking all files...\n');
for i = 1:length(validation_datasets)
    file_name = validation_datasets{i};
    fprintf('Loading %s...\n', file_name);
    
    % Read the file
    try
        data = readmatrix(file_name);
        fprintf('  Loaded %d rows and %d columns\n', size(data, 1), size(data, 2));
        
        % Stack the data
        all_data = [all_data; data];
    catch e
        fprintf('  ERROR: Could not load file: %s\n', e.message);
    end
end

fprintf('\nTotal stacked data: %d rows\n\n', size(all_data, 1));

% Calculate min and max for each column
min_values = min(all_data);
max_values = max(all_data);

% Determine column names/numbers for display
num_columns = size(all_data, 2);
column_labels = cell(1, num_columns);
for i = 1:num_columns
    column_labels{i} = sprintf('Column %d', i);
end

% Try to identify columns if we have typical color data
if num_columns >= 6
    column_labels{1} = 'Hue';
    column_labels{2} = 'Saturation';
    column_labels{3} = 'R';
    column_labels{4} = 'G';
    column_labels{5} = 'B';
    column_labels{6} = 'K';
end

% Display results
fprintf('============================================\n');
fprintf('Min and Max Values for Each Column\n');
fprintf('============================================\n');
fprintf('%-15s %-15s %-15s\n', 'Column', 'Min', 'Max');
fprintf('============================================\n');

for i = 1:num_columns
    fprintf('%-15s %-15.4f %-15.4f\n', column_labels{i}, min_values(i), max_values(i));
end

% Save results to a file
results_table = table(column_labels', min_values', max_values', ...
                     'VariableNames', {'Column', 'Min', 'Max'});
writetable(results_table, 'column_min_max_values.csv');
fprintf('\nResults saved to column_min_max_values.csv\n');