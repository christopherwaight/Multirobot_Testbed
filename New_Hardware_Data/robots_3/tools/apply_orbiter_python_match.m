function apply_orbiter_python_match(mdlFile)
% Make the 3-robot model's orbit primitive match the Python code (2026-11-12):
%  1. Orbiter (Jacobian_primitive/MATLAB Function) -> Python law structure, radial speed clipped to
%     +-0.3 m/s (code AND gains in tools/orbiter_python_match.m: k_t 0.15, k_r 1 since 2026-11-12).
%  2. Orbit direction flag Jacobian_primitive/Constant -> +1 (Python's direction).
%  3. "No spin" like Python (omega_c = 0): a Gain 'thC_gain' = 0 between the theta_c error and the
%     inverse Jacobian in Jacobian_primitive AND Cluster_Controller_setpoint. Set it to 1 to hold
%     theta_c at its setpoint again. The cmd_cluster_* logs are re-tapped after the gain (what is
%     commanded); err_* logs still show the theta_c error.
% Safe to re-run (skips what is already done). Saves the model.
here = fileparts(mfilename('fullpath'));
[~, mdl] = fileparts(mdlFile);
load_system(mdlFile);
cleanup = onCleanup(@() close_system(mdl, 0));
jp = [mdl '/Jacobian_primitive'];

% 1. Orbiter code (port names/order must not change)
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', [jp '/MATLAB Function']);
before = port_sig(ch);
ch.Script = fileread(fullfile(here, 'orbiter_python_match.m'));
if ~isequal(port_sig(ch), before), error('Orbiter port signature changed: %s -> %s', before, port_sig(ch)); end
fprintf('Orbiter code -> Python law (ports unchanged: %s)\n', before);

% 2. Direction
fprintf('Orbit direction flag %s -> 1\n', get_param([jp '/Constant'], 'Value'));
set_param([jp '/Constant'], 'Value', '1');

% 3. No spin
insert_thc_gain(jp, 'LOG cmd_cluster_jacobian mux');
insert_thc_gain([mdl '/Cluster_Controller_setpoint'], 'LOG cmd_cluster_setpoint mux');

save_system(mdl);
fprintf('SAVED %s\n', mdlFile);
end

function insert_thc_gain(sys, logMux)
g = [sys '/thC_gain'];
if ~isempty(find_system(sys, 'SearchDepth', 1, 'Name', 'thC_gain')), fprintf('exists   %s\n', g); return; end
ik = [sys '/Cluster Velocities to Cartesian Velocities'];
pc = get_param(ik, 'PortConnectivity');
src = pc(4);                                            % IK input 4 = thC_err
sph = get_param(src.SrcBlock, 'PortHandles'); srcPort = sph.Outport(src.SrcPort + 1);
lh = get_param(ik, 'LineHandles'); delete_line(lh.Inport(4));
p = get_param(ik, 'Position');
add_block('simulink/Math Operations/Gain', g, 'Gain', '0', 'Position', [p(1) - 110, p(2) - 70, p(1) - 70, p(2) - 40], ...
    'BackgroundColor', 'orange', 'Description', ...
    'theta_c control gain. 0 = no spin (matches Python omega_c = 0, since 2026-11-12). 1 = hold theta_c at its setpoint (old behaviour).');
set_param(g, 'AttributesFormatString', 'theta_c: 0 = no spin');
gph = get_param(g, 'PortHandles'); iph = get_param(ik, 'PortHandles');
add_line(sys, srcPort, gph.Inport(1), 'autorouting', 'smart');
add_line(sys, gph.Outport(1), iph.Inport(4), 'autorouting', 'smart');
% the cmd_cluster log should show what is commanded (after the gain): its input 3 is theta_c
mx = [sys '/' logMux];
ml = get_param(mx, 'LineHandles'); delete_line(ml.Inport(3));
mph = get_param(mx, 'PortHandles');
add_line(sys, gph.Outport(1), mph.Inport(3), 'autorouting', 'smart');
fprintf('added    %s = 0 (no spin); %s input 3 re-tapped after it\n', strrep(g, newline, ' '), logMux);
end

function s = port_sig(ch)
d = ch.find('-isa', 'Stateflow.Data');
in = d(strcmp({d.Scope}, 'Input')); out = d(strcmp({d.Scope}, 'Output'));
[~, i] = sort([in.Port]); [~, o] = sort([out.Port]);
s = sprintf('in(%s) out(%s)', strjoin({in(i).Name}, ','), strjoin({out(o).Name}, ','));
end
