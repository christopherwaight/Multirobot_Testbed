function apply_phi_wrap_only(mdlFile)
% PHASE 2, minimal version (user's choice 2026-11-12: "just angle wrapping").
% Angle-wrapping only, to match Python's quad_kinematics.py / quad_cluster.py. Nothing else changes.
%  1. Forward kinematics in Cluster_Controller_setpoint and Jacobian: after the existing line
%        phi = -atan2(y4 - y2, x4 - x2) + atan2(y3 - y1, x3 - x1);
%     add the same wrap line the Jacobians FK already has:
%        phi = mod(phi + pi, 2*pi) - pi;
%  2. Wrap the phi ERROR to [-pi, pi] (MATLAB Function 'wrap_phi_err' before the th1_err outport) in
%     Cluster_Controller_setpoint/Compute_error, Jacobian/Formation_error, Jacobians/Formation_error,
%     like Python's error_psi = mod(error_psi + pi, 2*pi) - pi.
% NOT changed (left as the user had them): the rest of the FK code, the FK_results debug assignin in
% the Jacobians FK, the critical-point finders' per-step fprintf, the inverse Jacobians.
% The fuller alternative is apply_quad_python_match.m (not applied).
% Safe to re-run. Saves the model.
[~, mdl] = fileparts(mdlFile);
load_system(mdlFile); cleanup = onCleanup(@() close_system(mdl, 0));
rt = sfroot;
phiLine = 'phi = -atan2(y4 - y2, x4 - x2) + atan2(y3 - y1, x3 - x1);';
wrapLine = 'phi = mod(phi + pi, 2*pi) - pi;   % wrap to [-pi, pi] like Python quad_kinematics.py (added 2026-11-12)';
for P = {'Cluster_Controller_setpoint', 'Jacobian', 'Jacobians'}
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/' P{1} '/Cartesian to Cluster Variables']);
    s = ch.Script;
    if contains(s, 'phi = mod(phi + pi, 2*pi) - pi'), fprintf('FK       %s already wraps phi\n', P{1}); continue; end
    if numel(strfind(s, phiLine)) ~= 1, error('phi line not found exactly once in %s FK', P{1}); end
    ch.Script = strrep(s, phiLine, [phiLine newline wrapLine]);
    fprintf('FK       %s: phi wrap line added\n', P{1});
end
for S = {'Cluster_Controller_setpoint/Compute_error', 'Jacobian/Formation_error', 'Jacobians/Formation_error'}
    sys = [mdl '/' S{1}];
    if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', 'wrap_phi_err')), fprintf('wrap     %s already has wrap_phi_err\n', S{1}); continue; end
    op = [sys '/th1_err'];
    lh = get_param(op, 'LineHandles'); src = get_param(lh.Inport, 'SrcPortHandle'); delete_line(lh.Inport);
    p = get_param(op, 'Position');
    w = [sys '/wrap_phi_err'];
    add_block('simulink/User-Defined Functions/MATLAB Function', w, 'Position', [p(1) - 150, p(2) - 10, p(1) - 50, p(4) + 10], 'BackgroundColor', 'orange');
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', w);
    ch.Script = sprintf(['function e = wrap_phi_err(u)\n%% Wrap the phi (angle between diagonals) error to [-pi, pi], like Python''s\n' ...
        '%% error_psi = mod(error_psi + pi, 2*pi) - pi in quad_cluster.py (added 2026-11-12).\ne = mod(u + pi, 2*pi) - pi;\nend\n']);
    wp = get_param(w, 'PortHandles'); oph = get_param(op, 'PortHandles');
    add_line(sys, src, wp.Inport(1), 'autorouting', 'smart');
    add_line(sys, wp.Outport(1), oph.Inport(1), 'autorouting', 'smart');
    fprintf('wrap     phi error wrapped in %s\n', S{1});
end
save_system(mdl);
fprintf('SAVED %s\n', mdlFile);
end
