function apply_quad_python_match(mdlFile)
% PHASE 2, FULL version (prepared 2026-11-12, NOT applied: the user chose the minimal
% apply_phi_wrap_only.m instead, which is installed. Kept as an option.)
% Make four_robot_cluster_all_primitives.slx match the Python quad kinematics behaviour:
%  1. The three forward-kinematics blocks -> tools/fk_quad_python_match.m: phi wrapped to [-pi, pi],
%     parallel-diagonal guard, debug code removed (FK_results / dead unwrap code in Jacobians).
%  2. Wrap the phi ERROR to [-pi, pi] in Compute_error (setpoint) and Formation_error (Jacobian,
%     Jacobians), like Python's error_psi wrap. Before: an unwrapped error could jump by 2*pi.
%  3. Silence the per-step fprintf in both critical-point finders (console spam every 0.1 s).
% Inverse Jacobian blocks are unchanged: they already match the Python inverse kinematics.
% Safe to re-run. Saves the model.
here = fileparts(mfilename('fullpath'));
[~, mdl] = fileparts(mdlFile);
load_system(mdlFile); cleanup = onCleanup(@() close_system(mdl, 0));
rt = sfroot;
fk = fileread(fullfile(here, 'fk_quad_python_match.m'));
for P = {'Cluster_Controller_setpoint', 'Jacobian', 'Jacobians'}
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/' P{1} '/Cartesian to Cluster Variables']);
    if contains(ch.Script, 'fk_quad_python_match'), continue; end
    before = port_sig(ch);
    ch.Script = strrep(fk, 'quad_kinematics.py (prepared 2026-11-12)', 'quad_kinematics.py (prepared 2026-11-12; source: tools/fk_quad_python_match.m)');
    if ~isequal(port_sig(ch), before), error('FK port signature changed in %s', P{1}); end
    fprintf('FK       %s -> Python-matching FK (ports unchanged)\n', P{1});
end
for S = {'Cluster_Controller_setpoint/Compute_error', 'Jacobian/Formation_error', 'Jacobians/Formation_error'}
    sys = [mdl '/' S{1}];
    if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', 'wrap_phi_err')), continue; end
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
for P = {'Jacobian', 'Jacobians'}
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [mdl '/' P{1} '/Critical_point_predicter/MATLAB Function']);
    s = ch.Script;
    s2 = regexprep(s, '(?m)^(\s*)(fprintf\()', '$1% $2');
    s2 = strrep(s2, 'if nargout == 0 || true', 'if false % per-step debug print disabled 2026-11-12 (was: nargout == 0 || true)');
    if ~strcmp(s, s2), ch.Script = s2; fprintf('quiet    per-step prints disabled in %s critical-point finder\n', P{1}); end
end
save_system(mdl);
fprintf('SAVED %s\n', mdlFile);
end

function s = port_sig(ch)
d = ch.find('-isa', 'Stateflow.Data');
in = d(strcmp({d.Scope}, 'Input')); out = d(strcmp({d.Scope}, 'Output'));
[~, i] = sort([in.Port]); [~, o] = sort([out.Port]);
s = sprintf('in(%s) out(%s)', strjoin({in(i).Name}, ','), strjoin({out(o).Name}, ','));
end
