% Aggregate Orbital Statistics Across All Robot Tests
% This script reads orbital control data from:
%   - 3-robot vortex tests (trunk/robots_3/vortex_tests/orbit*.mat)
%   - 3-robot saddle tests (trunk/robots_3/saddle_tests/radius*.mat)
%   - 4-robot tests (trunk/robots_4/radius*.mat)
%
% Groups trials by commanded radius and calculates mean ± std
% Outputs formatted table for paper


% This file generates TABLE V of my paper.
clear; clc;

% ============================================================================
% CONFIGURATION
% ============================================================================

% Number of decimal places for output (2, 3, or 4)
DECIMAL_PLACES = 3;

% Directory paths (relative to this script)
script_dir = fileparts(mfilename('fullpath'));
vortex_dir = fullfile(script_dir, 'robots_3', 'vortex_tests');
saddle_dir = fullfile(script_dir, 'robots_3', 'saddle_tests');
robot4_dir = fullfile(script_dir, 'robots_4');

% ============================================================================
% MAIN EXECUTION
% ============================================================================

fprintf('======================================================================\n');
fprintf('AGGREGATE ORBITAL CONTROL STATISTICS\n');
fprintf('======================================================================\n');
fprintf('Decimal places: %d\n', DECIMAL_PLACES);
fprintf('Vortex directory: %s\n', vortex_dir);
fprintf('Saddle directory: %s\n', saddle_dir);
fprintf('4-robot directory: %s\n', robot4_dir);
fprintf('======================================================================\n\n');

% ============================================================================
% PROCESS 3-ROBOT VORTEX TESTS (Series 0: orbit001-070)
% ============================================================================
fprintf('Processing 3-robot VORTEX tests (series 0)...\n');
fprintf('----------------------------------------------------------------------\n');

vortex_3robot_results = process_orbit_files(vortex_dir, 'orbit', 'cluster_position', 0);

% ============================================================================
% PROCESS 4-ROBOT VORTEX TESTS (Series 1: orbit101-160)
% ============================================================================
fprintf('\nProcessing 4-robot VORTEX tests (series 1)...\n');
fprintf('----------------------------------------------------------------------\n');

vortex_4robot_results = process_orbit_files(vortex_dir, 'orbit', 'cluster_position', 1);

% ============================================================================
% PROCESS 3-ROBOT SADDLE TESTS
% ============================================================================
fprintf('\nProcessing 3-robot SADDLE tests...\n');
fprintf('----------------------------------------------------------------------\n');

saddle_results = process_radius_files(saddle_dir, 'radius', 'cluster_pose');

% ============================================================================
% PROCESS 4-ROBOT TESTS
% ============================================================================
fprintf('\nProcessing 4-robot tests...\n');
fprintf('----------------------------------------------------------------------\n');

robot4_results = process_radius_files(robot4_dir, 'radius', 'cluster_pose');

% ============================================================================
% DISPLAY RESULTS
% ============================================================================
fprintf('\n\n');
fprintf('======================================================================\n');
fprintf('RESULTS SUMMARY\n');
fprintf('======================================================================\n\n');

% Create format string based on decimal places
fmt_val = sprintf('%%.%df', DECIMAL_PLACES);
fmt_std = sprintf('%%.%df', DECIMAL_PLACES);

% Check if we have any results
if isempty(vortex_3robot_results) && isempty(vortex_4robot_results) && isempty(saddle_results) && isempty(robot4_results)
    fprintf('ERROR: No data loaded from any source!\n');
    fprintf('Please check file paths and field names.\n');
    return;
end

% Display 3-Robot Vortex Results
if ~isempty(vortex_3robot_results)
    fprintf('3-ROBOT VORTEX TESTS (Series 0):\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('%-15s | %-15s | %-15s | %-10s\n', 'Commanded (m)', 'Actual (m)', 'Std (m)', 'N trials');
    fprintf('----------------------------------------------------------------------\n');

    radii = sort([vortex_3robot_results.commanded_radius]);
    for r = radii
        idx = find([vortex_3robot_results.commanded_radius] == r);
        fprintf(['%-15s | %-15s | %-15s | %-10d\n'], ...
            num2str(r, fmt_val), ...
            num2str(vortex_3robot_results(idx).mean_radius, fmt_val), ...
            num2str(vortex_3robot_results(idx).std_radius, fmt_std), ...
            vortex_3robot_results(idx).num_trials);
    end
    fprintf('\n');
end

% Display 4-Robot Vortex Results
if ~isempty(vortex_4robot_results)
    fprintf('4-ROBOT VORTEX TESTS (Series 1):\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('%-15s | %-15s | %-15s | %-10s\n', 'Commanded (m)', 'Actual (m)', 'Std (m)', 'N trials');
    fprintf('----------------------------------------------------------------------\n');

    radii = sort([vortex_4robot_results.commanded_radius]);
    for r = radii
        idx = find([vortex_4robot_results.commanded_radius] == r);
        fprintf(['%-15s | %-15s | %-15s | %-10d\n'], ...
            num2str(r, fmt_val), ...
            num2str(vortex_4robot_results(idx).mean_radius, fmt_val), ...
            num2str(vortex_4robot_results(idx).std_radius, fmt_std), ...
            vortex_4robot_results(idx).num_trials);
    end
    fprintf('\n');
end

% Display 3-Robot Saddle Results
if ~isempty(saddle_results)
    fprintf('3-ROBOT SADDLE TESTS:\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('%-15s | %-15s | %-15s | %-10s\n', 'Commanded (m)', 'Actual (m)', 'Std (m)', 'N trials');
    fprintf('----------------------------------------------------------------------\n');

    radii = sort([saddle_results.commanded_radius]);
    for r = radii
        idx = find([saddle_results.commanded_radius] == r);
        fprintf(['%-15s | %-15s | %-15s | %-10d\n'], ...
            num2str(r, fmt_val), ...
            num2str(saddle_results(idx).mean_radius, fmt_val), ...
            num2str(saddle_results(idx).std_radius, fmt_std), ...
            saddle_results(idx).num_trials);
    end
    fprintf('\n');
end

% Display 4-Robot Results
if ~isempty(robot4_results)
    fprintf('4-ROBOT TESTS:\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('%-15s | %-15s | %-15s | %-10s\n', 'Commanded (m)', 'Actual (m)', 'Std (m)', 'N trials');
    fprintf('----------------------------------------------------------------------\n');

    radii = sort([robot4_results.commanded_radius]);
    for r = radii
        idx = find([robot4_results.commanded_radius] == r);
        fprintf(['%-15s | %-15s | %-15s | %-10d\n'], ...
            num2str(r, fmt_val), ...
            num2str(robot4_results(idx).mean_radius, fmt_val), ...
            num2str(robot4_results(idx).std_radius, fmt_std), ...
            robot4_results(idx).num_trials);
    end
    fprintf('\n');
end

% ============================================================================
% TABLE V FORMAT (for paper)
% ============================================================================
fprintf('\n');
fprintf('======================================================================\n');
fprintf('TABLE V FORMAT (Hardware Orbital Control Results)\n');
fprintf('======================================================================\n\n');

% Combine all commanded radii
all_commanded = [];
if ~isempty(vortex_3robot_results)
    all_commanded = [all_commanded, [vortex_3robot_results.commanded_radius]];
end
if ~isempty(vortex_4robot_results)
    all_commanded = [all_commanded, [vortex_4robot_results.commanded_radius]];
end
if ~isempty(saddle_results)
    all_commanded = [all_commanded, [saddle_results.commanded_radius]];
end
if ~isempty(robot4_results)
    all_commanded = [all_commanded, [robot4_results.commanded_radius]];
end

all_commanded = sort(unique(all_commanded));

if isempty(all_commanded)
    fprintf('No data to display in table format.\n');
    return;
end

fprintf('Commanded | Vortex      | Vortex      | Saddle\n');
fprintf('Radius    | 3-Robot     | 4-Robot     | 3-Robot\n');
fprintf('(m)       | (± std)     | (± std)     | (± std)\n');
fprintf('----------+-------------+-------------+-------------\n');

for cmd_r = all_commanded
    % Find vortex 3-robot data (series 0)
    idx_v3 = find([vortex_3robot_results.commanded_radius] == cmd_r);
    if ~isempty(idx_v3)
        vortex_3_str = sprintf([fmt_val ' ± ' fmt_std], ...
            vortex_3robot_results(idx_v3).mean_radius, ...
            vortex_3robot_results(idx_v3).std_radius);
    else
        vortex_3_str = '--';
    end

    % Find vortex 4-robot data (series 1)
    idx_v4 = find([vortex_4robot_results.commanded_radius] == cmd_r);
    if ~isempty(idx_v4)
        vortex_4_str = sprintf([fmt_val ' ± ' fmt_std], ...
            vortex_4robot_results(idx_v4).mean_radius, ...
            vortex_4robot_results(idx_v4).std_radius);
    else
        vortex_4_str = '--';
    end

    % Find saddle 3-robot data
    idx_s3 = find([saddle_results.commanded_radius] == cmd_r);
    if ~isempty(idx_s3)
        saddle_3_str = sprintf([fmt_val ' ± ' fmt_std], ...
            saddle_results(idx_s3).mean_radius, ...
            saddle_results(idx_s3).std_radius);
    else
        saddle_3_str = '--';
    end

    fprintf([fmt_val ' | %-11s | %-11s | %-11s\n'], ...
        cmd_r, vortex_3_str, vortex_4_str, saddle_3_str);
end

fprintf('\n');
fprintf('======================================================================\n');
fprintf('COMPLETE\n');
fprintf('======================================================================\n');

% ============================================================================
% HELPER FUNCTIONS
% ============================================================================

function results = process_orbit_files(dir_path, prefix, position_field, series_filter)
    % Process orbit*.mat files (vortex tests)
    % Files: orbit001, orbit010, ..., orbit070 (series 0 = 3-robot)
    %        orbit101, orbit110, ..., orbit160 (series 1 = 4-robot)
    %        orbit201, orbit210, ..., orbit250 (series 2 = experimental/excluded)
    % series_filter: 0, 1, or 2 to select which series to process

    % Define all possible orbit files
    orbit_files = {};

    % Series 0: orbit001, orbit010, orbit020, ..., orbit070
    orbit_files{end+1} = 'orbit001.mat';
    for i = 10:10:70
        orbit_files{end+1} = sprintf('orbit%03d.mat', i);
    end

    % Series 1: orbit101, orbit110, orbit120, ..., orbit160
    orbit_files{end+1} = 'orbit101.mat';
    for i = 110:10:160
        orbit_files{end+1} = sprintf('orbit%03d.mat', i);
    end

    % Series 2: orbit201, orbit210, orbit220, ..., orbit250
    orbit_files{end+1} = 'orbit201.mat';
    for i = 210:10:250
        orbit_files{end+1} = sprintf('orbit%03d.mat', i);
    end

    % Also check for orbit233 (appears in file listing)
    orbit_files{end+1} = 'orbit233.mat';

    % Group files by commanded radius
    radius_groups = struct();

    for i = 1:length(orbit_files)
        filepath = fullfile(dir_path, orbit_files{i});

        if ~exist(filepath, 'file')
            continue;
        end

        % Extract commanded radius from filename
        % orbitXYZ where X = series (0,1,2), YZ = radius index
        % orbit001 -> series 0, radius 01 -> 0.01m
        % orbit101 -> series 1, radius 01 -> 0.01m
        % orbit210 -> series 2, radius 10 -> 0.10m
        filename = orbit_files{i};
        num_str = filename(6:8); % Extract 3 digits (e.g., "001", "101", "210")

        % Extract series number (first digit)
        series_num = str2double(num_str(1));

        % Skip if not in the requested series
        if series_num ~= series_filter
            continue;
        end

        % Last 2 digits are the radius index
        radius_index = str2double(num_str(2:3));
        commanded_r = radius_index / 100.0;

        % Load file and extract radius data
        try
            data = load(filepath);

            if ~isfield(data, position_field)
                fprintf('  Warning: %s missing field %s\n', orbit_files{i}, position_field);
                continue;
            end

            pos = data.(position_field);
            x = pos(:, 1);
            y = pos(:, 2);
            r = sqrt(x.^2 + y.^2);

            % Store in groups (use valid field name without periods)
            key = sprintf('r%04d', round(commanded_r * 1000)); % e.g., r0010 for 0.01m
            if ~isfield(radius_groups, key)
                radius_groups.(key).commanded_radius = commanded_r;
                radius_groups.(key).all_radii = [];
                radius_groups.(key).files = {};
            end

            radius_groups.(key).all_radii = [radius_groups.(key).all_radii; r];
            radius_groups.(key).files{end+1} = orbit_files{i};

            fprintf('  ✓ Loaded %s: commanded=%.3f, mean=%.4f, n=%d\n', ...
                orbit_files{i}, commanded_r, mean(r), length(r));

        catch ME
            fprintf('  ✗ Error loading %s: %s\n', orbit_files{i}, ME.message);
        end
    end

    % Calculate statistics for each radius group
    results = [];
    field_names = fieldnames(radius_groups);

    for i = 1:length(field_names)
        group = radius_groups.(field_names{i});

        result.commanded_radius = group.commanded_radius;
        result.mean_radius = mean(group.all_radii);
        result.std_radius = std(group.all_radii);
        result.num_trials = length(group.files);
        result.files = group.files;

        results = [results; result];
    end
end

function results = process_radius_files(dir_path, prefix, position_field)
    % Process radius*.mat files (saddle and 4-robot tests)
    % Files: radius001, radius01, radius02, radius03, radius04, radius05, radius06

    % Define all possible radius files
    radius_files = {'radius001.mat', 'radius01.mat', 'radius02.mat', ...
                    'radius03.mat', 'radius04.mat', 'radius05.mat', 'radius06.mat'};

    % Map filename to commanded radius
    radius_map = containers.Map(...
        {'radius001.mat', 'radius01.mat', 'radius02.mat', 'radius03.mat', ...
         'radius04.mat', 'radius05.mat', 'radius06.mat'}, ...
        {0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6});

    % Group files by commanded radius (should be 1-to-1 for these tests)
    radius_groups = struct();

    for i = 1:length(radius_files)
        filepath = fullfile(dir_path, radius_files{i});

        if ~exist(filepath, 'file')
            continue;
        end

        % Get commanded radius from map
        commanded_r = radius_map(radius_files{i});

        % Load file and extract radius data
        try
            data = load(filepath);

            if ~isfield(data, position_field)
                % Try alternative field names
                fields = fieldnames(data);
                fprintf('  Warning: %s missing field %s. Available fields: %s\n', ...
                    radius_files{i}, position_field, strjoin(fields, ', '));

                % Try common alternatives
                if isfield(data, 'cluster_position')
                    position_field_actual = 'cluster_position';
                elseif isfield(data, 'cluster_pose')
                    position_field_actual = 'cluster_pose';
                else
                    continue;
                end
            else
                position_field_actual = position_field;
            end

            pos = data.(position_field_actual);
            x = pos(:, 1);
            y = pos(:, 2);
            r = sqrt(x.^2 + y.^2);

            % Store in groups (use valid field name without periods)
            key = sprintf('r%04d', round(commanded_r * 1000)); % e.g., r0010 for 0.01m
            if ~isfield(radius_groups, key)
                radius_groups.(key).commanded_radius = commanded_r;
                radius_groups.(key).all_radii = [];
                radius_groups.(key).files = {};
            end

            radius_groups.(key).all_radii = [radius_groups.(key).all_radii; r];
            radius_groups.(key).files{end+1} = radius_files{i};

            fprintf('  ✓ Loaded %s: commanded=%.3f, mean=%.4f, n=%d\n', ...
                radius_files{i}, commanded_r, mean(r), length(r));

        catch ME
            fprintf('  ✗ Error loading %s: %s\n', radius_files{i}, ME.message);
        end
    end

    % Calculate statistics for each radius group
    results = [];
    field_names = fieldnames(radius_groups);

    for i = 1:length(field_names)
        group = radius_groups.(field_names{i});

        result.commanded_radius = group.commanded_radius;
        result.mean_radius = mean(group.all_radii);
        result.std_radius = std(group.all_radii);
        result.num_trials = length(group.files);
        result.files = group.files;

        results = [results; result];
    end
end
