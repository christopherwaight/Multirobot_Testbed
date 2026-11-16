function [inputs, hue_targets, sat_targets] = loadSingleColorData(color_name)

    % Construct the filename
    filename = ['data/' color_name '_cal.csv'];
    
    % Load data
    data = readmatrix(filename);
    
    % Assign Input Variables and Target Values
    rows_to_include = 20;
    inputs = data(1:24*rows_to_include, 3:6);
    hue_targets = data(1:24*rows_to_include, 1);
    sat_targets = data(1:24*rows_to_include, 2);
end