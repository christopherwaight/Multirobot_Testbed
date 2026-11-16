function cp_vs_dist_3R4R(d, ctrue)
% Live CP estimate quality vs cluster distance from the true centre, 3R vs 4R, all runs pooled.
fs = dir(fullfile(d, 'vortex_*robots_*.mat'));
D = {[], []}; E = {[], []}; B = {[], []};
for i = 1:numel(fs)
    S = load(fullfile(d, fs(i).name)); g = 1 + isfield(S, 'pose_r4');
    if g == 1, cp = S.field_cp; else, cp = S.field_cp_jacobian; end
    if g == 2 && contains(fs(i).name, '0_point_01_run') && any(strcmp(regexp(fs(i).name, 'run\d+', 'match', 'once'), {'run3', 'run4', 'run5', 'run6', 'run7'})), continue; end  % weak redwood battery
    C = S.cluster_state(:, 1:2); ok = all(isfinite(cp), 2) & S.tout > 5;
    dist = vecnorm(C(ok, :) - ctrue, 2, 2); u = (ctrue - C(ok, :)) ./ dist;
    D{g} = [D{g}; dist]; E{g} = [E{g}; vecnorm(cp(ok, :) - ctrue, 2, 2)]; B{g} = [B{g}; sum((cp(ok, :) - ctrue) .* u, 2)];
end
edges = [0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.55];
fprintf('cluster distance | 3R: steps  median err  >10cm  radial bias | 4R: steps  median err  >10cm  radial bias   (bias - = toward cluster)\n');
for b = 1:numel(edges) - 1
    s = cell(1, 2);
    for g = 1:2
        m = D{g} >= edges(b) & D{g} < edges(b + 1);
        if sum(m) < 50, s{g} = sprintf('%5d   %-36s', sum(m), '(too few)'); continue; end
        s{g} = sprintf('%5d   %.3f       %4.1f%%  %+.3f      ', sum(m), median(E{g}(m)), 100 * mean(E{g}(m) > 0.1), median(B{g}(m)));
    end
    fprintf('%.2f-%.2f m      | %s| %s\n', edges(b), edges(b + 1), s{1}, s{2});
end
end
