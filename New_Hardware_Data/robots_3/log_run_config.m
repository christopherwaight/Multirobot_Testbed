function cfg = log_run_config(mdl)
% Snapshot of everything that defines a run of the 3-robot hardware model, written to the base
% workspace as `run_config` so it is saved together with the logged data.
% Called automatically from the model's StopFcn; it can also be called by hand: log_run_config(bdroot)
%
% Values are read from the model IN MEMORY, so they are correct even when the model has unsaved
% edits (model_dirty = true). model_file_sha256 identifies the file on disk.
% Optional: set a string `run_notes` in the base workspace before the run (map placement,
% batteries, purpose, ...); it is copied into run_config.notes.
if nargin < 1, mdl = bdroot; end
cfg = struct();
cfg.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
cfg.model = mdl;
cfg.model_file = get_param(mdl, 'FileName');
cfg.model_dirty = strcmp(get_param(mdl, 'Dirty'), 'on');
cfg.model_file_sha256 = sha256_file(cfg.model_file);
d = dir(cfg.model_file); if ~isempty(d), cfg.model_file_modified = d.date; end
cfg.stop_time = get_param(mdl, 'StopTime');
cfg.fixed_step = get_param(mdl, 'FixedStep');

% Robot index -> robot name, following the wiring Robot_signal_processing_i -> Demux_velocity* -> robot
nR = numel(find_system(mdl, 'SearchDepth', 1, 'Regexp', 'on', 'Name', '^Robot_signal_processing_\d+$'));
cfg.robots = cell(nR, 2);
for i = 1:nR
    cfg.robots{i, 1} = sprintf('r%d', i);
    try
        pc = get_param(sprintf('%s/Robot_signal_processing_%d', mdl, i), 'PortConnectivity');
        dm = pc(3).DstBlock(1);                                  % out1 (body velocity) -> Demux_velocity*
        pc2 = get_param(dm, 'PortConnectivity');
        cfg.robots{i, 2} = get_param(pc2(2).DstBlock(1), 'Name');
    catch
        cfg.robots{i, 2} = '?';
    end
end

% Hardware blocks
try
    ot = [mdl '/Optitrack'];
    cfg.optitrack.rigidBodyNames = get_param(ot, 'rigidBodyNames');
    cfg.optitrack.coordinateTransformationMatrix = get_param(ot, 'coordinateTransformationMatrix');
    cfg.optitrack.orientationFormat = get_param(ot, 'orientationFormat');
catch
end
db = find_system(mdl, 'LookUnderMasks', 'all', 'BlockType', 'MATLABSystem', 'System', 'Decabot');
cfg.decabots = cellfun(@(b) get_param(b, 'robotName'), db, 'UniformOutput', false);

% Switches: position and what they pass on
sw = find_system(mdl, 'LookUnderMasks', 'all', 'BlockType', 'ManualSwitch');
cfg.switches = struct('block', {}, 'sw', {}, 'passes', {});
for k = 1:numel(sw)
    pc = get_param(sw{k}, 'PortConnectivity');
    sel = 1 + strcmp(get_param(sw{k}, 'sw'), '0');
    src = '(unconnected)';
    if ~isempty(pc(sel).SrcBlock) && pc(sel).SrcBlock ~= -1, src = clean(getfullname(pc(sel).SrcBlock)); end
    cfg.switches(end + 1) = struct('block', clean(sw{k}), 'sw', get_param(sw{k}, 'sw'), 'passes', src);
end

% Every Constant and Gain value (setpoints, orbit radius/direction, decode signs, gains, ...)
cfg.constants = block_values(mdl, 'Constant', 'Value');
cfg.gains = block_values(mdl, 'Gain', 'Gain');

% Full code of every MATLAB Function block (captures K_p, orbit_speed, kinematics, decoders, ...)
rt = sfroot;
m = rt.find('-isa', 'Simulink.BlockDiagram', 'Name', mdl);
ch = m.find('-isa', 'Stateflow.EMChart');
cfg.matlab_functions = struct('block', {}, 'code', {});
for k = 1:numel(ch)
    cfg.matlab_functions(end + 1) = struct('block', clean(ch(k).Path), 'code', ch(k).Script);
end

% Notes from the user
cfg.notes = '';
try
    if evalin('base', 'exist(''run_notes'', ''var'')'), cfg.notes = evalin('base', 'run_notes'); end
catch
end

if nR == 4, cfg.log_legend = log_legend4(); else, cfg.log_legend = log_legend(); end
assignin('base', 'run_config', cfg);
end

function s = block_values(mdl, type, param)
b = find_system(mdl, 'LookUnderMasks', 'all', 'BlockType', type);
s = struct('block', {}, 'value', {});
for k = 1:numel(b)
    s(end + 1) = struct('block', clean(b{k}), 'value', get_param(b{k}, param)); %#ok<AGROW>
end
end

function t = clean(p)
t = strrep(p, newline, ' ');
end

function h = sha256_file(f)
h = '';
try
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(f, 'r'); bytes = fread(fid, inf, '*uint8'); fclose(fid);
    md.update(bytes);
    h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2)', 1, []));
catch
end
end

function L = log_legend()
% Column meanings of every logged variable of the 3-robot model (r1 = celeste, r2 = tidal,
% r3 = pacific-blue). The 4-robot model has its own legend (see robots_4/README_four_robot.md);
% run_config.robots always gives the actual robot order.
L = struct( ...
    'pose_r1_r2_r3',          'OptiTrack [x z heading] per robot (m, m, rad)', ...
    'sensor_rgbk_raw_rN',     'Decabot colour sensor raw counts in Decabot output order [K(clear) R G B]', ...
    'sensor_rgbk_norm_rN',    '[R G B K] after (x-120)/1300, K (x-800)/3500, clipped to [0,1] (neural-net input)', ...
    'sensor_hue',             'hue 0..1 per robot [r1 r2 r3] (after the nets and post-processing)', ...
    'sensor_sat',             'saturation 0..1 per robot [r1 r2 r3]', ...
    'field_uv',               'decoded field [u1 v1 u2 v2 u3 v3] fed to the critical-point finder (u=m sin(2pi h), v=-m cos(2pi h))', ...
    'field_jacobian',         '[J11 J12 J21 J22 trace det curl type], type 1 vortex 2 spiral 3 node -1 saddle 0 degenerate', ...
    'field_cp',               'critical-point estimate [x z]', ...
    'orbit_radius',           '[current radius, desired - current] of the Orbiter', ...
    'cluster_state',          '[xc zc thc p q beta phi1 phi2 phi3]', ...
    'setpoint',               'set point controller desired [xc zc thc p q beta]', ...
    'err_setpoint',           'Compute_error outputs [xc zc thc p q beta] (xc, zc clipped to +-0.3)', ...
    'err_jacobian',           'Jacobian_primitive errors [xc zc thc p q beta]', ...
    'cmd_cluster_PRIM',       'input to PRIM''s inverse Jacobian [v_xc v_zc w_c v_p v_q v_beta]', ...
    'cmd_robots_PRIM',        'PRIM''s per-robot global velocities [vx1 vz1 w1 vx2 vz2 w2 vx3 vz3 w3]', ...
    'cmd_robots_selected',    'per-robot global velocities after the root switch chain (what is sent on)', ...
    'cmd_body_rN',            'body-frame command sent to robot N [vx vy w]', ...
    'simfield_hue_sat',       'analytic simulator (NOT sensor data, not used for control)', ...
    'timing',                 '[sim time, wall-clock posix seconds]; diff(:,2) = real duration of each step');
end

function L = log_legend4()
% Column meanings for the 4-robot model four_robot_cluster_all_primitives.slx
% (r1 = celeste, r2 = tidal, r3 = pacific-blue, r4 = redwood; check run_config.robots).
% Quad shape variables: d1 = |r1->r3|, d2 = |r2->r4| (diagonals), r1/r2 = where the diagonals cross
% (fraction along each), phi = angle(diag 1-3) - angle(diag 2-4), thc = angle of diagonal 1-3.
L = struct( ...
    'pose_r1_to_r4',          'OptiTrack [x z heading] per robot (m, m, rad)', ...
    'sensor_rgbk_raw_rN',     'Decabot colour sensor raw counts in Decabot output order [K(clear) R G B]', ...
    'sensor_rgbk_norm_rN',    '[R G B K] after normalisation and clipping to [0,1] (neural-net input)', ...
    'sensor_hue',             'hue 0..1 per robot [r1 r2 r3 r4]', ...
    'sensor_sat',             'saturation 0..1 per robot [r1 r2 r3 r4]', ...
    'field_uv',               'decoded field [u1 v1 ... u4 v4] (u=m sin(2pi h), v=-m cos(2pi h)) as fed to the critical-point finders', ...
    'field_jacobian',         'least-squares 4-point fit [J11 J12 J21 J22 trace det curl type]', ...
    'field_cp_jacobian',      'critical point of the Jacobian primitive (4-point least squares) [x z]', ...
    'field_cp_jacobians',     'critical point of the Jacobians primitive (average of the 1-2-3 and 1-3-4 fits) [x z]', ...
    'orbit_radius_PRIM',      '[current radius, desired - current] of PRIM''s Orbiter', ...
    'cluster_state',          '[xc zc thc d1 d2 r1 r2 phi]', ...
    'setpoint',               'set point controller desired [xc zc thc d1 d2 r1 r2 phi]', ...
    'err_PRIM',               'errors [xc zc thc d1 d2 r1 r2 phi] of PRIM (setpoint: Compute_error outputs)', ...
    'cmd_cluster_PRIM',       'input to PRIM''s inverse Jacobian [v_xc v_zc w_c v_d1 v_d2 v_r1 v_r2 v_phi]', ...
    'cmd_robots_PRIM',        'PRIM''s per-robot global velocities [vx1 vz1 w1 ... vx4 vz4 w4]', ...
    'cmd_robots_selected',    'per-robot global velocities after the root switch chain (what is sent on)', ...
    'cmd_body_rN',            'body-frame command sent to robot N [vx vy w]', ...
    'diag_angle_error_jacobians', 'Jacobians primitive diagnostics [thc, angle to target, Theta_controller thc error]', ...
    'timing',                 '[sim time, wall-clock posix seconds]; diff(:,2) = real duration of each step');
end
