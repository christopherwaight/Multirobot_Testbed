function calib_fix_test(d, ctrue)
% Recompute the 4R (and 3R) CP estimate with per-robot reading corrections (bias rotation + magnitude scale,
% from robot_accuracy.m) and see whether the toward-cluster bias at large distance goes away.
biasDeg = struct('celeste', -4.8, 'tidal', -5.3, 'pacific_blue', -2.3, 'redwood', -8.0);
slope = struct('celeste', 0.88, 'tidal', 0.83, 'pacific_blue', 0.87, 'redwood', 0.70); ref = 0.86;
variants = {'as logged', 'fix redwood only', 'fix all robots'};
fs = dir(fullfile(d, 'vortex_*robots_*.mat'));
R = struct();
for g = [3 4], for v = 1:3, R(g, v).D = []; R(g, v).B = []; R(g, v).E = []; end, end
for i = 1:numel(fs)
    S = load(fullfile(d, fs(i).name)); N = 3 + isfield(S, 'pose_r4');
    if N == 4 && contains(fs(i).name, '0_point_01_run') && any(strcmp(regexp(fs(i).name, 'run\d+', 'match', 'once'), {'run3', 'run4', 'run5', 'run6', 'run7'})), continue; end
    rn = matlab.lang.makeValidName(S.run_config.robots(:, 2)); h = S.tout > 5; T = find(h)';
    P = zeros(numel(S.tout), N, 2); U = P;
    for r = 1:N, P(:, r, :) = S.(sprintf('pose_r%d', r))(:, 1:2); U(:, r, :) = S.field_uv(:, 2 * r - 1:2 * r); end
    C = S.cluster_state(:, 1:2);
    for v = 1:3
        rot = zeros(1, N); sc = ones(1, N);
        for r = 1:N
            if v == 3 || (v == 2 && strcmp(rn{r}, 'redwood')), rot(r) = -deg2rad(biasDeg.(rn{r})); sc(r) = ref / slope.(rn{r}); end
        end
        for t = T
            Y = squeeze(U(t, :, :)); Yc = Y;
            for r = 1:N, Yc(r, :) = sc(r) * [cos(rot(r)) -sin(rot(r)); sin(rot(r)) cos(rot(r))] * Y(r, :)'; end
            A = [squeeze(P(t, :, :)) ones(N, 1)]; c = A \ Yc; J = c(1:2, :)'; if abs(det(J)) < 1e-10, continue; end
            cp = (-J \ c(3, :)')'; dist = norm(C(t, :) - ctrue); u = (ctrue - C(t, :)) / dist;
            R(N, v).D(end+1) = dist; R(N, v).E(end+1) = norm(cp - ctrue); R(N, v).B(end+1) = dot(cp - ctrue, u);
        end
    end
end
edges = [0.15 0.25 0.35 0.45 0.55];
for N = [3 4]
    fprintf('\n== %d robots: radial bias (- = toward cluster) / median error, by cluster distance from true centre\n', N);
    fprintf('   %-18s', 'variant'); for b = 1:numel(edges) - 1, fprintf('| %.2f-%.2f m          ', edges(b), edges(b + 1)); end; fprintf('\n');
    for v = 1:3
        if N == 3 && v == 2, continue; end
        fprintf('   %-18s', variants{v});
        for b = 1:numel(edges) - 1
            m = R(N, v).D >= edges(b) & R(N, v).D < edges(b + 1);
            fprintf('| %+.3f / %.3f  (%4d) ', median(R(N, v).B(m)), median(R(N, v).E(m)), sum(m));
        end
        fprintf('\n');
    end
end
end
