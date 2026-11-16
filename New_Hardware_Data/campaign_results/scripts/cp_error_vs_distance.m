function cp_error_vs_distance(root)
% Why does the CP estimate get worse far from the centre?
% (1) data: CP error vs cluster distance from the (run-median) centre, pooled over all runs.
% (2) noise model fitted from the data (direction + magnitude noise), then the same 3-point
%     linear fit on an ideal vortex with that noise -> predicted error vs distance.
fs = dir(fullfile(root, '**', '*.mat'));
D = []; E = []; dirErr = []; magRes = []; MAG = []; RR = [];
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name));
    cp = S.field_cp; ok = all(isfinite(cp), 2); cpm = median(cp(ok, :));
    C = S.cluster_state(:, 1:2);
    D = [D; vecnorm(C(ok, :) - cpm, 2, 2)]; E = [E; vecnorm(cp(ok, :) - cpm, 2, 2)]; %#ok<AGROW>
    % per-robot reading errors vs the ideal vortex about cpm (direction + magnitude)
    for rb = 1:3
        pose = S.(sprintf('pose_r%d', rb)); uv = S.field_uv(:, 2 * rb - 1:2 * rb);
        d = pose(:, 1:2) - cpm; r = vecnorm(d, 2, 2); m = vecnorm(uv, 2, 2);
        okr = all(isfinite(uv), 2) & m > 1e-6;
        ideal = [-d(:, 2), d(:, 1)];
        dirErr = [dirErr; wrapToPi(atan2(uv(okr, 2), uv(okr, 1)) - atan2(ideal(okr, 2), ideal(okr, 1)))]; %#ok<AGROW>
        MAG = [MAG; m(okr)]; RR = [RR; r(okr)]; %#ok<AGROW>
    end
end
% magnitude model m = g*r + m0
bm = [RR ones(size(RR))] \ MAG; magRes = MAG - [RR ones(size(RR))] * bm;
fprintf('reading model from data: |uv| = %.2f*r %+.3f, magnitude noise std %.3f; direction error median |.| %.1f deg, bias %+.1f deg, p90 %.1f deg\n', ...
    bm, std(magRes), rad2deg(median(abs(dirErr))), rad2deg(median(dirErr)), rad2deg(prctile(abs(dirErr), 90)));

edges = [0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.45];
fprintf('\n%-13s %7s | %-34s | %-34s\n', 'cluster dist', 'steps', 'DATA: CP error median / p90 / >10cm', 'MODEL (same noise): median / p90 / >10cm');
% simulated cluster: equilateral triangle side 0.3 (p = q = 0.3, beta 60 deg), random orientation
rng(2); side = 0.3; tri0 = side / sqrt(3) * [cos([0 2 4] * pi / 3)' sin([0 2 4] * pi / 3)'];
dirPool = dirErr; magPool = magRes;
for b = 1:numel(edges) - 1
    m = D >= edges(b) & D < edges(b + 1);
    if sum(m) < 20, continue; end
    % model
    N = 4000; es = zeros(N, 1);
    for k = 1:N
        dd = edges(b) + rand * (edges(b + 1) - edges(b)); ang = 2 * pi * rand; cen = dd * [cos(ang) sin(ang)];
        th = 2 * pi * rand; Rm = [cos(th) -sin(th); sin(th) cos(th)];
        P = cen + tri0 * Rm';
        F = zeros(3, 2);
        for j = 1:3
            r = norm(P(j, :)); dir0 = atan2(P(j, 1), -P(j, 2));          % ideal vortex direction [-z x]
            dirj = dir0 + dirPool(randi(numel(dirPool)));
            mj = max(bm(1) * r + bm(2) + magPool(randi(numel(magPool))), 0);
            F(j, :) = mj * [cos(dirj) sin(dirj)];
        end
        A = [P ones(3, 1)] \ F;  J = A(1:2, :)'; c = (-J \ A(3, :)')';
        es(k) = norm(c);
    end
    fprintf('%.2f-%.2f m   %7d | %.3f / %.3f / %4.1f%%                | %.3f / %.3f / %4.1f%%\n', edges(b), edges(b + 1), sum(m), ...
        median(E(m)), prctile(E(m), 90), 100 * mean(E(m) > 0.1), median(es), prctile(es, 90), 100 * mean(es > 0.1));
end
% power law on medians
ok = D > 0.03;
bb = polyfit(log(D(ok)), log(E(ok)), 1);
fprintf('\nlog-log slope of CP error vs distance (data): %.2f  (2 = error grows with distance squared)\n', bb(1));
% same model WITHOUT magnitude noise and WITHOUT direction noise -> which one matters
end
