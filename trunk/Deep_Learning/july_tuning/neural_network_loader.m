%% Loading all Nueral Networks

% Just try to load and let MATLAB handle the path
filename = 'everything_nets_final.mat';

try
    % Load first to see if it works
    load(filename, 'everything_hue_net', 'everything_sat_net');
    
    % If successful, get info about where it was found
    file_path = which(filename);
    file_info = dir(file_path);
    
    fprintf('=== File Information ===\n');
    fprintf('Loaded from: %s\n', file_path);
    fprintf('File size: %.2f MB\n', file_info.bytes / (1024^2));
    fprintf('Last modified: %s\n', datestr(file_info.datenum));
    fprintf('========================\n');
    
    disp('Networks loaded into base workspace!');
    
catch ME
    fprintf('Error loading file: %s\n', ME.message);
    fprintf('Make sure the file is in your current directory or MATLAB path.\n');
end