function why_4R_worse(d, ctrue)
% (1) Leave-one-out on the logged 4R data: CP from all 4 vs each 3-robot subset (same positions, same readings).
% (2) Noise-model comparison of formation geometry: 3R triangle (sides 0.3) vs 4R square (diagonals 0.3).
names = {'celeste', 'tidal', 'pacific-blue', 'redwood'};
for rdv = {'0_point_2', '0_point_3'}
    fs = dir(fullfile(d, ['vortex_4robots_radius_of_' rdv{1} '_run*.mat']));
    subsets = {1:4, [2 3 4], [1 3 4], [1 2 4], [1 2 3]};
    lab = {'all 4', 'without celeste', 'without tidal', 'without pacific-blue', 'without redwood'};
    E = cell(1, 5); B = cell(1, 5);
    for i = 1:numel(fs)
        S = load(fullfile(d, fs(i).name)); h = S.tout > 10;
        P = zeros(numel(S.tout), 4, 2); U = P;
        for r = 1:4, P(:, r, :) = S.(sprintf('pose_r%d', r))(:, 1:2); U(:, r, :) = S.field_uv(:, 2 * r - 1:2 * r); end
        C = S.cluster_state(:, 1:2);
        for s = 1:5
            idx = subsets{s};
            for t = find(h)'
                A = [squeeze(P(t, idx, :)) ones(numel(idx), 1)]; Y = squeeze(U(t, idx, :));
                c = A \ Y; J = c(1:2, :)'; if abs(det(J)) < 1e-10, continue; end
                cp = (-J \ c(3, :)')';
                u = (ctrue - C(t, :)) / norm(ctrue - C(t, :));
                E{s}(end+1) = norm(cp - ctrue); B{s}(end+1) = dot(cp - ctrue, u);   % >0: beyond the centre; <0: toward the cluster
            end
        end
    end
    fprintf('\n== 4R r_d %s (%d runs, t > 10 s), CP error vs true centre, recomputed from the logged data\n', strrep(rdv{1}, '_point_', '.'), numel(fs));
    fprintf('   %-22s | median error | >10 cm | mean radial bias (- = toward cluster)\n', 'robots used');
    for s = 1:5, fprintf('   %-22s | %.3f m      | %4.1f%%  | %+.3f m\n', lab{s}, median(E{s}), 100 * mean(E{s} > 0.1), median(B{s})); end
end

% (2) geometry with the measured noise (direction 6.9 deg median (heavy tail), magnitude |uv| = 0.86 r - 0.027, noise 0.054)
rng(3); M = 20000;
tri = 0.3 / sqrt(3) * [cos([0 2 4] * pi / 3)' sin([0 2 4] * pi / 3)'];      % equilateral, side 0.3
sq = 0.15 * [cos([0 1 2 3] * pi / 2)' sin([0 1 2 3] * pi / 2)'];            % square, diagonals 0.3
fprintf('\n== noise model: same reading noise, formation geometry only\n   %-6s | %-34s | %-34s\n', 'dist', '3R triangle (sides 0.30, spread r 0.17)', '4R square (diag 0.30, sides 0.21, spread r 0.15)');
for dist = [0.2 0.3 0.4]
    out = zeros(2, 3);
    for g = 1:2
        F = tri; if g == 2, F = sq; end
        e = zeros(M, 1); b = e;
        for k = 1:M
            a = 2 * pi * rand; cen = dist * [cos(a) sin(a)]; th = 2 * pi * rand; R = [cos(th) -sin(th); sin(th) cos(th)];
            Pp = cen + F * R'; Y = zeros(size(Pp));
            for j = 1:size(Pp, 1)
                r = norm(Pp(j, :)); dir0 = atan2(Pp(j, 1), -Pp(j, 2)) + deg2rad(9) * trnd(3) / sqrt(3);
                m = max(0.86 * r - 0.027 + 0.054 * randn, 0); Y(j, :) = m * [cos(dir0) sin(dir0)];
            end
            c = [Pp ones(size(Pp, 1), 1)] \ Y; J = c(1:2, :)'; cp = (-J \ c(3, :)')';
            e(k) = norm(cp); b(k) = dot(cp, -cen / norm(cen));     % centre is the origin; -cen/|cen| points from cluster toward centre... radial bias: + beyond
        end
        out(g, :) = [median(e), mean(e > 0.1), median(-b)];
    end
    fprintf('   %.2f m | median %.3f, >10 cm %4.1f%%, bias %+.3f | median %.3f, >10 cm %4.1f%%, bias %+.3f\n', dist, out(1, 1), 100 * out(1, 2), out(1, 3), out(2, 1), 100 * out(2, 2), out(2, 3));
end
fprintf('   (bias: median of radial error, - = estimate pulled toward the cluster)\n');
end
