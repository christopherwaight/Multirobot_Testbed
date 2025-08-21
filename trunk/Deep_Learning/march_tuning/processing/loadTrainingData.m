function [inputs, hue_targets, sat_targets] = loadTrainingData()



    % Load training data from multiple files
    data1 = readmatrix("data/celeste_cal.csv");
    data2 = readmatrix("data/tidal_cal.csv");
    data3 = readmatrix("data/pacific_blue_cal.csv");
    
    % Assign Input Variables and Target Values
    rows_to_include = 20;
    inputs = [data1(1:24*rows_to_include,3:6);
             data2(1:24*rows_to_include,3:6);
             data3(1:24*rows_to_include,3:6)];
    
    hue_targets = [data1(1:24*rows_to_include,1);
                  data2(1:24*rows_to_include,1);
                  data3(1:24*rows_to_include,1)];
    
    sat_targets = [data1(1:24*rows_to_include,2);
                  data2(1:24*rows_to_include,2);
                  data3(1:24*rows_to_include,2)];
                
end