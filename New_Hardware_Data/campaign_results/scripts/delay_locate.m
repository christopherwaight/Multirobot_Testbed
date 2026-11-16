function delay_locate(root)
% Where is the one-step delay?
% (a) inside the model: is the logged Orbiter command cmd(k) computed from cluster_state(k) or (k-1)?
% (b) command -> motion: v(k) = (p(k+1)-p(k))/dt correlates with u(k) or u(k-1)?
fs = dir(fullfile(root, 'Vortex', 'three_robot', '**', '*point_1_run*.mat'));
eSame = []; ePrev = []; cA = [];
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name));
    cs = S.cluster_state; cp = S.field_cp; cmd = S.cmd_cluster_jacobian(:, 1:2);
    P0 = orb(cs(:, 1:2), cp, 0.1);
    P1 = [nan(1, 2); orb(cs(1:end-1, 1:2), cp(1:end-1, :), 0.1)];
    eSame = [eSame; vecnorm(P0 - cmd, 2, 2)]; ePrev = [ePrev; vecnorm(P1 - cmd, 2, 2)]; %#ok<AGROW>
    for r = 1:3
        p = S.(sprintf('pose_r%d', r))(:, 1:2); u = S.cmd_robots_selected(:, 3*r-2:3*r-1);
        v = diff(p) ./ diff(S.timing(:, 2)); n = size(v, 1);
        cA = [cA; v(2:n, :) u(2:n, :) u(1:n-1, :)]; %#ok<AGROW>
    end
end
fprintf('(a) Orbiter law on cluster_state/field_cp of the SAME step: median |pred - logged| = %.2e m/s\n', median(eSame, 'omitnan'));
fprintf('    ... of the PREVIOUS step:                               median |pred - logged| = %.2e m/s\n', median(ePrev, 'omitnan'));
ok = all(isfinite(cA), 2); cA = cA(ok, :);
for c = 1:2
    fprintf('(b) axis %d: corr(v(k), u(k)) %.3f | corr(v(k), u(k-1)) %.3f\n', c, corr(cA(:, c), cA(:, 2 + c)), corr(cA(:, c), cA(:, 4 + c)));
end
end
function v = orb(c, q, rd)
kt = 0.15; kr = 1; d = q - c; r = vecnorm(d, 2, 2); rh = d ./ r;
tang = [rh(:, 2) -rh(:, 1)];                 % clockwise = +1
rad = min(max(kr * (r - rd), -0.3), 0.3);
v = kt * tang + rad .* rh;
end
