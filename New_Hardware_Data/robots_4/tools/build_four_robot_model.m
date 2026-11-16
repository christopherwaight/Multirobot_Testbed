function build_four_robot_model(srcFile, dstFile)
% Build four_robot_cluster_all_primitives.slx (written by Claude, 2026-11-12).
%   build_four_robot_model('trunk/robots_4/chris_4R_all_primitives.slx', ...
%                          'trunk/robots_4/four_robot_cluster_all_primitives.slx')
% Copies the existing 4-robot model (srcFile is NOT modified) and ports the changes made to the
% 3-robot model three_robot_cluster_all_primitives.slx on 2026-11-11/12, extended to 4 robots:
%   1. hue -> vector decode fix: z component negated in all 8 critical-point decoders
%   2. Orbiter = Python law structure, k_t 0.15, k_r 1 (user gains; same code as the 3-robot Orbiter), radial speed clip 0.3, direction flag +1
%   3. no theta_c spin: Gain thC_gain = 0 before the inverse Jacobian in all 3 primitives
%   4. full logging suite (LOG blocks, cyan) for 4 robots; StopFcn -> log_run_config
%   5. a tag (annotation + model Description) saying how the model was made
% The kinematics blocks are NOT changed here (see README_four_robot.md, phase 2).
% Safe to re-run on the built file (pass srcFile = dstFile): finished steps are skipped.
here = fileparts(mfilename('fullpath'));
r3tools = fullfile(here, '..', '..', 'robots_3', 'tools');
if ~strcmp(srcFile, dstFile), copyfile(srcFile, dstFile, 'f'); end
[~, mdl] = fileparts(dstFile);
load_system(dstFile);
cleanup = onCleanup(@() close_system(mdl, 0));
M = @(p) [mdl '/' p];
row('reset');
prims = {'Cluster_Controller_setpoint', 'setpoint'; 'Jacobian', 'jacobian'; 'Jacobians', 'jacobians'};

%% 1. Decode fix (same as the 3-robot model): Y (z) component * -1
for P = {'Jacobian', 'Jacobians'}
    for S = {'Subsystem', 'Subsystem1', 'Subsystem2', 'Subsystem3'}
        sys = M([P{1} '/Critical_point_predicter/' S{1}]);
        if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', 'z_sign')), continue; end
        yo = [sys '/Y_Component'];
        lh = get_param(yo, 'LineHandles'); src = get_param(lh.Inport, 'SrcPortHandle'); delete_line(lh.Inport);
        p = get_param(yo, 'Position');
        g = [sys '/z_sign'];
        add_block('simulink/Math Operations/Gain', g, 'Gain', '-1', 'Position', [p(1) - 70, p(2) - 2, p(1) - 30, p(4) + 2], ...
            'BackgroundColor', 'orange', 'Description', ...
            'Hue decode fix (2026-11-12, same as 3-robot model): v = -m*cos(2*pi*h). See trunk/robots_3/README_hardware.md section 4.');
        gp = get_param(g, 'PortHandles'); yp = get_param(yo, 'PortHandles');
        add_line(sys, src, gp.Inport(1), 'autorouting', 'smart');
        add_line(sys, gp.Outport(1), yp.Inport(1), 'autorouting', 'smart');
        fprintf('decode   z_sign = -1 in %s\n', strrep(sys, newline, ' '));
    end
end

%% 2. Orbiters -> Python law, direction +1
orbCode = fileread(fullfile(r3tools, 'orbiter_python_match.m'));
rt = sfroot;
for P = {'Jacobian', 'Jacobians'}
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', M([P{1} '/MATLAB Function']));
    before = port_sig(ch);
    if ~strcmp(ch.Script, orbCode)      % same code as the 3-robot Orbiter (single source file)
        ch.Script = orbCode;
        if ~isequal(port_sig(ch), before), error('Orbiter port signature changed in %s', P{1}); end
        fprintf('orbiter  Python law in %s (ports unchanged)\n', P{1});
    end
    set_param(M([P{1} '/Constant']), 'Value', '1');
end

%% 3. No theta_c spin
for i = 1:size(prims, 1)
    insert_thc_gain(M(prims{i, 1}));
end

%% 4. Logging suite
ren = {'robot1_pose', 'pose_r1'; 'robot2_pose', 'pose_r2'; 'robot3_pose', 'pose_r3'; 'robot4_pose', 'pose_r4'; ...
       'hue', 'sensor_hue'; 'saturation', 'sensor_sat'; ...
       'detected_critical_point', 'field_cp_jacobian'; 'detected_critical_point_avg', 'field_cp_jacobians'; ...
       'angle_error_diagnostics', 'diag_angle_error_jacobians'};
for k = 1:size(ren, 1)
    tw = find_tw(mdl, ren{k, 1});
    if isempty(tw), continue; end
    set_param(tw, 'VariableName', ren{k, 2}); style_tw(tw); set_param(tw, 'Name', ['LOG ' ren{k, 2}]);
    fprintf('renamed  %-30s -> %s\n', ren{k, 1}, ren{k, 2});
end
tw = find_tw(mdl, 'cluster_position');                % = cluster_state columns 1-2
if ~isempty(tw)
    lh = get_param(tw, 'LineHandles'); if lh.Inport ~= -1, delete_line(lh.Inport); end
    delete_block(tw); fprintf('removed  cluster_position (duplicate of cluster_state(:,1:2))\n');
end
robots = {'celeste', 'tidal', 'pacific-blue', 'redwood'};
for i = 1:4
    add_log(mdl, sprintf('sensor_rgbk_raw_r%d', i), out_ph(M(robots{i}), 1:4));
    rg = M(sprintf('RGB to HSV to vel - all robots%d', i));
    add_log(rg, sprintf('sensor_rgbk_norm_r%d', i), out_ph([rg '/Mux'], 1));
    rs = M(sprintf('Robot_signal_processing_%d', i));
    add_log(rs, sprintf('cmd_body_r%d', i), out_ph([rs '/Global to Body Velocities'], 1), true);
end
add_log(mdl, 'setpoint', [out_ph(M('Center and Rotation'), 1), out_ph(M('Mux4'), 1)]);
for i = 1:size(prims, 1)
    add_log(mdl, ['cmd_robots_' prims{i, 2}], out_ph(root_mux_of(mdl, prims{i, 1}), 1));
end
add_log(mdl, 'cmd_robots_selected', out_ph(M('Manual Switch'), 1));
if isempty(find_system(mdl, 'SearchDepth', 1, 'Name', 'LOG timing'))
    clk = [mdl '/LOG timing clock'];
    add_block('simulink/Sources/Digital Clock', clk, 'SampleTime', get_param(mdl, 'FixedStep'), 'Position', row(mdl, 40), 'BackgroundColor', 'cyan');
    add_fn_log(mdl, 'timing', fileread(fullfile(r3tools, 'wall_clock_log.m')), out_ph(clk, 1));
end
for i = 1:size(prims, 1)
    sys = M(prims{i, 1});
    add_log(sys, ['cmd_cluster_' prims{i, 2}], src_ports([sys '/Cluster Velocities to Cartesian Velocities'], 2:9));
end
cs = M('Cluster_Controller_setpoint');
add_log(cs, 'cluster_state', out_ph([cs '/Cartesian to Cluster Variables'], 2:9));
add_log(cs, 'err_setpoint', out_ph([cs '/Compute_error'], 1:8));
for P = {'Jacobian', 'jacobian'; 'Jacobians', 'jacobians'}'
    sys = M(P{1});
    add_log(sys, ['err_' P{2}], [out_ph([sys '/position_error'], 1:3), out_ph([sys '/Formation_error'], 1:5)]);
    add_fn_log(sys, ['orbit_radius_' P{2}], fileread(fullfile(r3tools, 'orbit_radius_log.m')), src_ports([sys '/MATLAB Function'], [1 2 3 4 6]));
end
cpp = M('Jacobian/Critical_point_predicter');
add_log(cpp, 'field_uv', [out_ph([cpp '/Subsystem'], 1:2), out_ph([cpp '/Subsystem1'], 1:2), out_ph([cpp '/Subsystem2'], 1:2), out_ph([cpp '/Subsystem3'], 1:2)]);
add_fn_log(cpp, 'field_jacobian', fileread(fullfile(here, 'field_jacobian_log4.m')), src_ports([cpp '/MATLAB Function'], 1:16));

%% 5. StopFcn -> run_config (log_run_config.m lives in trunk/robots_3)
cb = get_param(mdl, 'StopFcn');
if ~contains(cb, 'log_run_config')
    add = sprintf(['%% Logging suite: save the run configuration as run_config (trunk/robots_3/log_run_config.m)\n' ...
        'try\n    addpath(fullfile(fileparts(get_param(''%s'', ''FileName'')), ''..'', ''robots_3''));\n    log_run_config(''%s'');\n' ...
        'catch logErr\n    warning(''log_run_config failed: %%s'', logErr.message);\nend'], mdl, mdl);
    if isempty(strtrim(cb)), cb = add; else, cb = [cb newline add]; end
    set_param(mdl, 'StopFcn', cb);
end

%% 6. Tag
tag = sprintf(['Built by Claude (AI assistant) on 2026-11-12 at the user''s request.\n' ...
    'Made from chris_4R_all_primitives.slx (the existing 4-robot model; left unchanged), by porting the\n' ...
    'changes made to the 3-robot model three_robot_cluster_all_primitives.slx and extending them to 4 robots:\n' ...
    '  - hue decode fix (z component * -1, block z_sign) in all 8 critical-point decoders\n' ...
    '  - Orbiter = Python law structure, k_t 0.15, k_r 1 (user gains; same code as the 3-robot Orbiter), radial speed clip 0.3, direction flag +1\n' ...
    '  - no theta_c spin: thC_gain = 0 in all three primitives (set 1 to restore)\n' ...
    '  - logging suite extended to 4 robots (cyan LOG blocks) + run_config via StopFcn\n' ...
    'Kinematics: unchanged; checked against Python quad_kinematics.py (they match), but the angle\n' ...
    'between the diagonals (phi) is not wrapped - see README_four_robot.md before trusting phi control.\n' ...
    'Rebuild: trunk/robots_4/tools/build_four_robot_model.m']);
set_param(mdl, 'Description', tag);
an = find_system(mdl, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'annotation');
hasTag = false;
for k = 1:numel(an), hasTag = hasTag || startsWith(get_param(an(k), 'Text'), 'Built by Claude'); end
if ~hasTag
    b = find_system(mdl, 'SearchDepth', 1, 'Type', 'Block'); b = b(~strcmp(b, mdl));
    P = cell2mat(get_param(b, 'Position'));
    x0 = min(P(:, 1)); y0 = min(P(:, 2));
    try
        h = Simulink.Annotation([mdl '/Built by Claude']);
        h.Text = tag;
        h.Position = [x0, y0 - 260, x0 + 760, y0 - 40];
        h.BackgroundColor = 'yellow';
        fprintf('tag      annotation + model Description\n');
    catch e
        warning('annotation not placed (%s); the tag is still in the model Description', e.message);
    end
end

save_system(mdl);
fprintf('SAVED %s\n', dstFile);
end

%% ---------------- helpers (same as robots_3/tools/add_logging_suite.m) ----------------
function insert_thc_gain(sys)
g = [sys '/thC_gain'];
if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', 'thC_gain')), return; end
ik = [sys '/Cluster Velocities to Cartesian Velocities'];
pc = get_param(ik, 'PortConnectivity');
src = pc(4);
sph = get_param(src.SrcBlock, 'PortHandles'); srcPort = sph.Outport(src.SrcPort + 1);
lh = get_param(ik, 'LineHandles'); delete_line(lh.Inport(4));
p = get_param(ik, 'Position');
add_block('simulink/Math Operations/Gain', g, 'Gain', '0', 'Position', [p(1) - 110, p(2) - 70, p(1) - 70, p(2) - 40], ...
    'BackgroundColor', 'orange', 'Description', ...
    'theta_c control gain. 0 = no spin (matches Python omega_c = 0, since 2026-11-12). 1 = old behaviour.');
set_param(g, 'AttributesFormatString', 'theta_c: 0 = no spin');
gph = get_param(g, 'PortHandles'); iph = get_param(ik, 'PortHandles');
add_line(sys, srcPort, gph.Inport(1), 'autorouting', 'smart');
add_line(sys, gph.Outport(1), iph.Inport(4), 'autorouting', 'smart');
fprintf('no-spin  thC_gain = 0 in %s\n', strrep(sys, newline, ' '));
end

function tw = find_tw(mdl, var)
tw = find_system(mdl, 'LookUnderMasks', 'all', 'BlockType', 'ToWorkspace', 'VariableName', var);
if ~isempty(tw), tw = tw{1}; else, tw = ''; end
end

function style_tw(tw)
set_param(tw, 'SaveFormat', 'Array', 'SampleTime', '-1', 'MaxDataPoints', 'inf', 'Decimation', '1', 'BackgroundColor', 'cyan');
try, set_param(tw, 'Save2DSignal', '2-D array (concatenate along first dimension)'); catch, end
end

function h = ph_in(blk, k)
ph = get_param(blk, 'PortHandles'); h = ph.Inport(k);
end

function h = out_ph(blk, ks)
ph = get_param(blk, 'PortHandles'); h = ph.Outport(ks);
end

function h = src_ports(blk, ins)
pc = get_param(blk, 'PortConnectivity');
h = zeros(1, numel(ins));
for k = 1:numel(ins)
    s = pc(ins(k)); ph = get_param(s.SrcBlock, 'PortHandles'); h(k) = ph.Outport(s.SrcPort + 1);
end
end

function mx = root_mux_of(mdl, prim)
mxs = find_system(mdl, 'SearchDepth', 1, 'BlockType', 'Mux');
for k = 1:numel(mxs)
    pc = get_param(mxs{k}, 'PortConnectivity');
    if ~isempty(pc(1).SrcBlock) && pc(1).SrcBlock ~= -1 && strcmp(get_param(pc(1).SrcBlock, 'Name'), prim), mx = mxs{k}; return; end
end
error('No root Mux fed by %s', prim);
end

function add_log(sys, var, src, doReshape)
if nargin < 4, doReshape = false; end
if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', ['LOG ' var])), return; end
n = numel(src); p = row(sys, max(30, 14 * n)); x = p(1); y = p(2); hgt = p(4) - p(2); feed = src;
if n > 1
    mxp = [sys '/LOG ' var ' mux'];
    add_block('simulink/Signal Routing/Mux', mxp, 'Inputs', num2str(n), 'DisplayOption', 'bar', 'Position', [x, y, x + 5, y + hgt], 'BackgroundColor', 'cyan');
    for k = 1:n, add_line(sys, src(k), ph_in(mxp, k), 'autorouting', 'smart'); end
    feed = out_ph(mxp, 1); x = x + 60;
end
if doReshape
    rp = [sys '/LOG ' var ' reshape'];
    add_block('simulink/Math Operations/Reshape', rp, 'OutputDimensionality', '1-D array', 'Position', [x, y + hgt/2 - 15, x + 40, y + hgt/2 + 15], 'BackgroundColor', 'cyan');
    add_line(sys, feed, ph_in(rp, 1), 'autorouting', 'smart'); feed = out_ph(rp, 1); x = x + 80;
end
tw = [sys '/LOG ' var];
add_block('simulink/Sinks/To Workspace', tw, 'VariableName', var, 'Position', [x, y + hgt/2 - 15, x + 140, y + hgt/2 + 15]);
style_tw(tw);
add_line(sys, feed, ph_in(tw, 1), 'autorouting', 'smart');
fprintf('added    %-28s in %s\n', var, strrep(sys, newline, ' '));
end

function add_fn_log(sys, var, code, src)
if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', ['LOG ' var])), return; end
fn = [sys '/LOG ' var ' fn']; n = numel(src); p = row(sys, max(40, 14 * n));
add_block('simulink/User-Defined Functions/MATLAB Function', fn, 'Position', [p(1), p(2), p(1) + 120, p(4)], 'BackgroundColor', 'cyan');
rt = sfroot; ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', fn); ch.Script = code;
for k = 1:n, add_line(sys, src(k), ph_in(fn, k), 'autorouting', 'smart'); end
tw = [sys '/LOG ' var]; yc = (p(2) + p(4)) / 2;
add_block('simulink/Sinks/To Workspace', tw, 'VariableName', var, 'Position', [p(1) + 180, yc - 15, p(1) + 320, yc + 15]);
style_tw(tw);
add_line(sys, out_ph(fn, 1), ph_in(tw, 1), 'autorouting', 'smart');
fprintf('added    %-28s in %s (computed)\n', var, strrep(sys, newline, ' '));
end

function pos = row(sys, hgt)
persistent nxt
if ischar(sys) && strcmp(sys, 'reset'), nxt = containers.Map; pos = []; return; end
if isempty(nxt), nxt = containers.Map; end
key = getfullname(sys);
if ~isKey(nxt, key)
    b = find_system(sys, 'SearchDepth', 1, 'Type', 'Block'); b = b(~strcmp(b, key));
    P = get_param(b, 'Position'); if ~iscell(P), P = {P}; end; P = cell2mat(P(:));
    nxt(key) = [max(P(:, 3)) + 150, min(P(:, 2))];
end
xy = nxt(key); pos = [xy(1), xy(2), xy(1) + 40, xy(2) + hgt]; nxt(key) = [xy(1), xy(2) + hgt + 30];
end

function s = port_sig(ch)
d = ch.find('-isa', 'Stateflow.Data');
in = d(strcmp({d.Scope}, 'Input')); out = d(strcmp({d.Scope}, 'Output'));
[~, i] = sort([in.Port]); [~, o] = sort([out.Port]);
s = sprintf('in(%s) out(%s)', strjoin({in(i).Name}, ','), strjoin({out(o).Name}, ','));
end
