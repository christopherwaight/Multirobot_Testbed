function [inputs, hue_targets, sat_targets] = loadTestingData(dataset_file)

        
        val_data = readmatrix(dataset_file);        
        num_lines = min(480, size(val_data, 1));
        inputs = val_data(1:num_lines, 3:6);
        hue_targets = val_data(1:num_lines, 1);
        sat_targets = val_data(1:num_lines, 2);

end