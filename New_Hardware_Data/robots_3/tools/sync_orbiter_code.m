function sync_orbiter_code(mdlFile)
% Copy tools/orbiter_python_match.m (the ONE source of the Orbiter code and its gains k_t, k_r) into
% the Orbiter block(s) of a 3-robot or 4-robot model, so both models run identical orbit code.
%   3-robot: Jacobian_primitive/MATLAB Function
%   4-robot: Jacobian/MATLAB Function and Jacobians/MATLAB Function
% Port names/order are checked before and after. Safe to re-run. Saves the model only if it changed.
% (Added 2026-11-12, when the user set k_t 0.15, k_r 1 for both models.)
here = fileparts(mfilename('fullpath'));
code = fileread(fullfile(here, 'orbiter_python_match.m'));
[~, mdl] = fileparts(mdlFile);
load_system(mdlFile); cleanup = onCleanup(@() close_system(mdl, 0));
cand = {'Jacobian_primitive/MATLAB Function', 'Jacobian/MATLAB Function', 'Jacobians/MATLAB Function'};
rt = sfroot; changed = false; found = 0;
for k = 1:numel(cand)
    p = [mdl '/' cand{k}];
    ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', p);
    if isempty(ch), continue; end
    if ~contains(ch.Script, 'radial_speed'), error('%s does not look like the Orbiter', cand{k}); end
    found = found + 1;
    if strcmp(ch.Script, code), fprintf('same     %s\n', cand{k}); continue; end
    before = port_sig(ch);
    ch.Script = code;
    if ~isequal(port_sig(ch), before), error('Orbiter port signature changed in %s', cand{k}); end
    fprintf('updated  %s (ports unchanged: %s)\n', cand{k}, before);
    changed = true;
end
if found == 0, error('no Orbiter block found in %s', mdl); end
if changed, save_system(mdl); fprintf('SAVED %s\n', mdlFile); else, fprintf('no change %s\n', mdlFile); end
end

function s = port_sig(ch)
d = ch.find('-isa', 'Stateflow.Data');
in = d(strcmp({d.Scope}, 'Input')); out = d(strcmp({d.Scope}, 'Output'));
[~, i] = sort([in.Port]); [~, o] = sort([out.Port]);
s = sprintf('in(%s) out(%s)', strjoin({in(i).Name}, ','), strjoin({out(o).Name}, ','));
end
