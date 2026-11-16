function S = load_run(f)
% Load a 3-robot hardware run (.mat), old or new logging names, into a struct with BOTH name sets.
%   S = load_run('data_hardware_campaign_11_11_2026/vortex_3r_0_10_run1.mat');
% - Timeseries variables are converted to plain N x k arrays (rows = time steps).
% - Old files (before 2026-11-12) get the new names (pose_r1, sensor_hue, field_cp, ...).
% - New files get the old names too (robot1_pose, hue, detected_critical_point, ...), so older
%   plotting scripts work after replacing load(f) with S = load_run(f).
% Column meanings: see run_config.log_legend (new files) or trunk/robots_3/README_hardware.md.
S = load(f);
fn = fieldnames(S);
for k = 1:numel(fn)
    v = S.(fn{k});
    if isa(v, 'timeseries')
        d = squeeze(v.Data);
        if size(d, 1) ~= numel(v.Time), d = d.'; end
        S.(fn{k}) = d;
    end
end

% new name <-> old name
map = { ...
    'pose_r1',             'robot1_pose'; ...
    'pose_r2',             'robot2_pose'; ...
    'pose_r3',             'robot3_pose'; ...
    'sensor_hue',          'hue'; ...
    'sensor_sat',          'saturation'; ...
    'sensor_rgbk_norm_r1', 'rgbk_robot_1'; ...
    'sensor_rgbk_norm_r2', 'rgbk_robot_2'; ...
    'sensor_rgbk_norm_r3', 'rgbk_robot_3'; ...
    'field_cp',            'detected_critical_point'; ...
    'cluster_state',       'cluster_pose'; ...
    'simfield_hue',        'target_hue_adjusted'; ...
    'simfield_sat',        'target_sat_adjusted'};
% 4-robot model (four_robot_cluster_all_primitives / chris_4R_all_primitives): its critical point
% logs map differently (two Jacobian primitives)
% Robot count: new logs record it in what the run itself wrote (cmd_robots_selected is 3 columns per
% robot). Don't go by the presence of pose_r4: 3-robot files saved with "save" after a 4-robot session
% carry STALE pose_r4 / field_cp_jacobian etc. from the 4R workspace (seen in every 2026-11-15 saddle run
% r_d 0.01-0.3). Old files (no run_config) fall back to robot4_pose.
if isfield(S, 'run_config') && isfield(S, 'cmd_robots_selected')
    is4R = size(S.cmd_robots_selected, 2) == 12;
else
    is4R = isfield(S, 'robot4_pose') || isfield(S, 'pose_r4');
end
if is4R
    map(strcmp(map(:, 1), 'field_cp'), :) = [];
    map = [map; {'pose_r4', 'robot4_pose'; 'field_cp_jacobian', 'detected_critical_point'; ...
                 'field_cp_jacobians', 'detected_critical_point_avg'; 'diag_angle_error_jacobians', 'angle_error_diagnostics'}];
end
% New logs WIN: files saved with "save" of the whole workspace can also carry STALE old-name
% variables left over from earlier runs (seen 2026-11-13: cluster_pose, robot1_pose, p, q, ... with
% 601 rows from the 11/11 runs inside 30 s / 301-row files). So when the new name exists, the old
% name is overwritten with it; the stale copy is kept as <old>_STALE for inspection.
newLogging = isfield(S, 'run_config');
for k = 1:size(map, 1)
    nw = map{k, 1}; od = map{k, 2};
    if isfield(S, nw)
        if isfield(S, od) && ~isequal(S.(od), S.(nw)), S.([od '_STALE']) = S.(od); end
        S.(od) = S.(nw);
    elseif isfield(S, od)
        S.(nw) = S.(od);
    end
end

% p, q, beta were dropped from the new 3-robot logging: they are columns 4-6 of cluster_state
if ~is4R && isfield(S, 'cluster_state') && size(S.cluster_state, 2) >= 6
    pqb = {'p', 4; 'q', 5; 'beta', 6};
    for k = 1:3
        nm = pqb{k, 1};
        if newLogging && isfield(S, nm), S.([nm '_STALE']) = S.(nm); end
        if newLogging || ~isfield(S, nm), S.(nm) = S.cluster_state(:, pqb{k, 2}); end
    end
end

% Older files logged the simulator per robot (hue_anticipated1..3 / sat_anticipated1..3)
if ~isfield(S, 'simfield_hue') && all(isfield(S, {'hue_anticipated1', 'hue_anticipated2', 'hue_anticipated3'}))
    S.simfield_hue = [S.hue_anticipated1, S.hue_anticipated2, S.hue_anticipated3];
end
if ~isfield(S, 'simfield_sat') && all(isfield(S, {'sat_anticipated1', 'sat_anticipated2', 'sat_anticipated3'}))
    S.simfield_sat = [S.sat_anticipated1, S.sat_anticipated2, S.sat_anticipated3];
end
end
