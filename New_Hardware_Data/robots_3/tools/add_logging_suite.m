function add_logging_suite(mdlFile)
% Add the full logging suite to three_robot_cluster_all_primitives.slx (written 2026-11-12).
%   add_logging_suite('trunk/robots_3/three_robot_cluster_all_primitives.slx')
% - Renames the existing To Workspace logs to the new naming scheme (see README_hardware.md, sec. 6).
% - Removes duplicate logs (p, q, beta, desired_velocities, desired_velocities_gradient_descent).
% - Fixes the swapped robot-3 columns of the analytic-simulator logs.
% - Adds new logs: raw sensors, every velocity command, errors, decoded field, field Jacobian,
%   orbit radius, wall-clock timing. All new blocks are named 'LOG <variable>' and coloured cyan.
% - Sets the model StopFcn to call log_run_config (writes run_config to the base workspace).
% Logging blocks only TAP existing signals; no control signal is changed. Safe to re-run: blocks
% that already exist are skipped. Saves the model.
here = fileparts(mfilename('fullpath'));
[~, mdl] = fileparts(mdlFile);
load_system(mdlFile);
cleanup = onCleanup(@() close_system(mdl, 0));
M = @(p) [mdl '/' p];
place('reset');

%% 1. Rename / reformat existing To Workspace logs
ren = { ...
    'robot1_pose', 'pose_r1'; 'robot2_pose', 'pose_r2'; 'robot3_pose', 'pose_r3'; ...
    'hue', 'sensor_hue'; 'saturation', 'sensor_sat'; ...
    'rgbk_robot_1', 'sensor_rgbk_norm_r1'; 'rgbk_robot_2', 'sensor_rgbk_norm_r2'; 'rgbk_robot_3', 'sensor_rgbk_norm_r3'; ...
    'detected_critical_point', 'field_cp'; 'cluster_pose', 'cluster_state'; ...
    'target_hue_adjusted', 'simfield_hue'; 'target_sat_adjusted', 'simfield_sat'};
for k = 1:size(ren, 1)
    tw = find_tw(mdl, ren{k, 1});
    if isempty(tw)
        if isempty(find_tw(mdl, ren{k, 2})), warning('No To Workspace logging %s (or %s)', ren{k, 1}, ren{k, 2}); end
        continue
    end
    set_param(tw, 'VariableName', ren{k, 2});
    style_tw(tw);
    set_param(tw, 'Name', ['LOG ' ren{k, 2}]);
    fprintf('renamed  %-38s -> %s\n', ren{k, 1}, ren{k, 2});
end

%% 2. Remove duplicate logs
for v = {'p', 'q', 'beta', 'desired_velocities', 'desired_velocities_gradient_descent'}
    tw = find_tw(mdl, v{1});
    if isempty(tw), continue; end
    lh = get_param(tw, 'LineHandles');
    if lh.Inport ~= -1, delete_line(lh.Inport); end
    delete_block(tw);
    fprintf('removed  %s (duplicate)\n', v{1});
end

%% 3. Fix the simulator logs: robot 3's hue and sat were wired to the wrong muxes
fv = M('Fixed_Vortex_Simulated_Environment');
sim3 = [fv '/Fixed_Vortex_Simulator2'];
op = find_system(sim3, 'SearchDepth', 1, 'LookUnderMasks', 'all', 'BlockType', 'Outport');
hueP = str2double(get_param(op{strcmp(get_param(op, 'Name'), 'hue_anticipated')}, 'Port'));
satP = str2double(get_param(op{strcmp(get_param(op, 'Name'), 'sat_anticipated')}, 'Port'));
hueMux = src_block(find_tw(mdl, 'simfield_hue'), 1);
satMux = src_block(find_tw(mdl, 'simfield_sat'), 1);
pc = get_param(hueMux, 'PortConnectivity');
if pc(3).SrcPort + 1 ~= hueP
    for mx = {hueMux, satMux}
        lh = get_param(mx{1}, 'LineHandles'); if lh.Inport(3) ~= -1, delete_line(lh.Inport(3)); end
    end
    s3 = get_param(sim3, 'PortHandles');
    add_line(fv, s3.Outport(hueP), ph_in(hueMux, 3), 'autorouting', 'smart');
    add_line(fv, s3.Outport(satP), ph_in(satMux, 3), 'autorouting', 'smart');
    fprintf('fixed    simulator robot-3 hue/sat wiring\n');
end

%% 4. New logs
% Root: raw sensors (Decabot outputs, before any normalisation)
robots = {'celeste', 'tidal', 'pacific-blue'};
for i = 1:3
    ph = get_param(M(robots{i}), 'PortHandles');
    add_log(mdl, sprintf('sensor_rgbk_raw_r%d', i), ph.Outport(1:4));
end
% Root: set point controller's desired values [xc zc thc] + [p q beta]
add_log(mdl, 'setpoint', [out_ph(M('Setpoint or trajectory'), 1), out_ph(M('Mux4'), 1)]);
% Root: each primitive's per-robot velocities, and what the switch chain passes on
prims = {'Cluster_Controller_setpoint', 'setpoint'; 'Jacobian_primitive', 'jacobian'; 'vector_sum', 'vectorsum'; ...
         'vector2scalar_mag_steepest_ascent', 'ascent'; 'vector2scalar_mag_steepest_descent', 'descent'};
for i = 1:size(prims, 1)
    mx = root_mux_of(mdl, prims{i, 1});
    add_log(mdl, ['cmd_robots_' prims{i, 2}], out_ph(mx, 1));
end
add_log(mdl, 'cmd_robots_selected', out_ph(M('Manual Switch'), 1));
% Root: wall-clock timing
if isempty(find_system(mdl, 'SearchDepth', 1, 'Name', 'LOG timing'))
    clk = [mdl '/LOG timing clock'];
    if isempty(find_system(mdl, 'SearchDepth', 1, 'Name', 'LOG timing clock'))
        add_block('simulink/Sources/Digital Clock', clk, 'SampleTime', get_param(mdl, 'FixedStep'), 'Position', row(mdl, 40, 0, 40));
        set_param(clk, 'BackgroundColor', 'cyan');
    end
    cp = get_param(clk, 'PortHandles');
    add_fn_log(mdl, 'timing', fileread(fullfile(here, 'wall_clock_log.m')), cp.Outport(1));
end

% Inside each primitive: what enters its inverse Jacobian
for i = 1:size(prims, 1)
    sys = M(prims{i, 1});
    add_log(sys, ['cmd_cluster_' prims{i, 2}], src_ports([sys '/Cluster Velocities to Cartesian Velocities'], 2:7));
end
% Errors
ce = M('Cluster_Controller_setpoint/Compute_error');
add_log(M('Cluster_Controller_setpoint'), 'err_setpoint', out_ph(ce, 1:6));
jp = M('Jacobian_primitive');
add_log(jp, 'err_jacobian', [out_ph([jp '/position_error'], [2 3 1]), out_ph([jp '/formation_error'], 1:3)]);
% Orbit radius (same inputs as the Orbiter: xc, zc, x_cp, z_cp, desired_radius)
add_fn_log(jp, 'orbit_radius', fileread(fullfile(here, 'orbit_radius_log.m')), src_ports([jp '/MATLAB Function'], [1 2 3 4 6]));
% Decoded field and field Jacobian (same inputs as Criticalpointfinder)
cpp = [jp '/Critical_point_predicter'];
add_log(cpp, 'field_uv', [out_ph([cpp '/Subsystem'], 1:2), out_ph([cpp '/Subsystem1'], 1:2), out_ph([cpp '/Subsystem2'], 1:2)]);
add_fn_log(cpp, 'field_jacobian', fileread(fullfile(here, 'field_jacobian_log.m')), src_ports([cpp '/MATLAB Function'], 1:12));
% Body-frame command to each robot
for i = 1:3
    sys = M(sprintf('Robot_signal_processing_%d', i));
    add_log(sys, sprintf('cmd_body_r%d', i), out_ph([sys '/Global to Body Velocities'], 1), true);
end

%% 5. StopFcn -> run_config
cb = get_param(mdl, 'StopFcn');
if ~contains(cb, 'log_run_config')
    add = sprintf(['%% Logging suite: save the run configuration as run_config (trunk/robots_3/log_run_config.m)\n' ...
        'try\n    addpath(fileparts(get_param(''%s'', ''FileName'')));\n    log_run_config(''%s'');\n' ...
        'catch logErr\n    warning(''log_run_config failed: %%s'', logErr.message);\nend'], mdl, mdl);
    if isempty(strtrim(cb)), cb = add; else, cb = [cb newline add]; end
    set_param(mdl, 'StopFcn', cb);
    fprintf('set      StopFcn -> log_run_config\n');
end

save_system(mdl);
fprintf('SAVED %s\n', mdlFile);
end

%% ---------------- helpers ----------------
function tw = find_tw(mdl, var)
tw = find_system(mdl, 'LookUnderMasks', 'all', 'BlockType', 'ToWorkspace', 'VariableName', var);
if ~isempty(tw), tw = tw{1}; else, tw = ''; end
end

function style_tw(tw)
set_param(tw, 'SaveFormat', 'Array', 'SampleTime', '-1', 'MaxDataPoints', 'inf', 'Decimation', '1', 'BackgroundColor', 'cyan');
try
    set_param(tw, 'Save2DSignal', '2-D array (concatenate along first dimension)');
catch
end
end

function b = src_block(blk, port)
pc = get_param(blk, 'PortConnectivity');
b = getfullname(pc(port).SrcBlock);
end

function h = ph_in(blk, k)
ph = get_param(blk, 'PortHandles'); h = ph.Inport(k);
end

function h = out_ph(blk, ks)
ph = get_param(blk, 'PortHandles'); h = ph.Outport(ks);
end

function h = src_ports(blk, ins)
% Output-port handles of whatever drives the given input ports of blk
pc = get_param(blk, 'PortConnectivity');
h = zeros(1, numel(ins));
for k = 1:numel(ins)
    s = pc(ins(k));
    ph = get_param(s.SrcBlock, 'PortHandles');
    h(k) = ph.Outport(s.SrcPort + 1);
end
end

function mx = root_mux_of(mdl, prim)
% The root Mux whose first input comes from primitive `prim` (its per-robot velocity bundle)
mxs = find_system(mdl, 'SearchDepth', 1, 'BlockType', 'Mux');
for k = 1:numel(mxs)
    pc = get_param(mxs{k}, 'PortConnectivity');
    if ~isempty(pc(1).SrcBlock) && pc(1).SrcBlock ~= -1 && strcmp(get_param(pc(1).SrcBlock, 'Name'), prim)
        mx = mxs{k}; return
    end
end
error('No root Mux fed by %s', prim);
end

function add_log(sys, var, src, doReshape)
% src: output-port handles to log. Several -> a Mux first. doReshape: force a 1-D vector first.
if nargin < 4, doReshape = false; end
if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', ['LOG ' var])), fprintf('exists   %s\n', var); return; end
n = numel(src);
p = row(sys, max(30, 14 * n), 0, 0);
x = p(1); y = p(2); hgt = p(4) - p(2);
feed = src;
if n > 1
    mxp = [sys '/LOG ' var ' mux'];
    add_block('simulink/Signal Routing/Mux', mxp, 'Inputs', num2str(n), 'DisplayOption', 'bar', ...
        'Position', [x, y, x + 5, y + hgt], 'BackgroundColor', 'cyan');
    for k = 1:n, add_line(sys, src(k), ph_in(mxp, k), 'autorouting', 'smart'); end
    feed = out_ph(mxp, 1); x = x + 60;
end
if doReshape
    rp = [sys '/LOG ' var ' reshape'];
    add_block('simulink/Math Operations/Reshape', rp, 'OutputDimensionality', '1-D array', ...
        'Position', [x, y + hgt/2 - 15, x + 40, y + hgt/2 + 15], 'BackgroundColor', 'cyan');
    add_line(sys, feed, ph_in(rp, 1), 'autorouting', 'smart');
    feed = out_ph(rp, 1); x = x + 80;
end
tw = [sys '/LOG ' var];
add_block('simulink/Sinks/To Workspace', tw, 'VariableName', var, ...
    'Position', [x, y + hgt/2 - 15, x + 140, y + hgt/2 + 15]);
style_tw(tw);
add_line(sys, feed, ph_in(tw, 1), 'autorouting', 'smart');
fprintf('added    %-30s in %s\n', var, strrep(sys, newline, ' '));
end

function add_fn_log(sys, var, code, src)
% A logging-only MATLAB Function block (code from tools/*_log.m) feeding a To Workspace.
if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', ['LOG ' var])), fprintf('exists   %s\n', var); return; end
fn = [sys '/LOG ' var ' fn'];
n = numel(src);
p = row(sys, max(40, 14 * n), 0, 0);
add_block('simulink/User-Defined Functions/MATLAB Function', fn, 'Position', [p(1), p(2), p(1) + 120, p(4)], 'BackgroundColor', 'cyan');
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', fn);
ch.Script = code;
for k = 1:n, add_line(sys, src(k), ph_in(fn, k), 'autorouting', 'smart'); end
% place the To Workspace to the right of the function on the same row
tw = [sys '/LOG ' var];
yc = (p(2) + p(4)) / 2;
add_block('simulink/Sinks/To Workspace', tw, 'VariableName', var, 'Position', [p(1) + 180, yc - 15, p(1) + 320, yc + 15]);
style_tw(tw);
add_line(sys, out_ph(fn, 1), ph_in(tw, 1), 'autorouting', 'smart');
fprintf('added    %-30s in %s (computed)\n', var, strrep(sys, newline, ' '));
end

function pos = row(sys, hgt, ~, ~)
% Next free row in a logging column to the right of everything else in `sys`.
persistent nxt
if ischar(sys) && strcmp(sys, 'reset'), nxt = containers.Map; pos = []; return; end
if isempty(nxt), nxt = containers.Map; end
key = getfullname(sys);
if ~isKey(nxt, key)
    b = find_system(sys, 'SearchDepth', 1, 'Type', 'Block');
    b = b(~strcmp(b, key));
    P = get_param(b, 'Position'); if ~iscell(P), P = {P}; end
    P = cell2mat(P(:));
    nxt(key) = [max(P(:, 3)) + 150, min(P(:, 2))];
end
xy = nxt(key);
pos = [xy(1), xy(2), xy(1) + 40, xy(2) + hgt];
nxt(key) = [xy(1), xy(2) + hgt + 30];
end

function place(cmd)
row(cmd, 0, 0, 0);
end
