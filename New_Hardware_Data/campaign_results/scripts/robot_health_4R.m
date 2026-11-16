function robot_health_4R(files)
% Per robot, per run: speed reached vs command, direction error, fitted response gain; + run outcome.
names = {'celeste', 'tidal', 'pacific-blue', 'redwood'};
fprintf('%-6s %-9s | %-44s | %-44s | %-44s | %s\n', 'run', 'time', 'speed ratio actual/cmd  c / t / p / R', 'angle vs command (deg)  c / t / p / R', 'fitted gain b/(1-a)  c / t / p / R', 'radius, laps, shape');
for i = 1:numel(files)
    S = load(files{i}); rc = S.run_config; t = S.tout; h = t > 5; n = numel(t);
    sr = zeros(1, 4); an = sr; gn = sr;
    for r = 1:4
        pose = S.(sprintf('pose_r%d', r)); u = S.cmd_robots_selected(:, 3 * r - 2:3 * r - 1);
        v = [diff(pose(:, 1:2)) / 0.1; nan(1, 2)];
        ok = h & vecnorm(u, 2, 2) > 0.02 & vecnorm(v, 2, 2) > 0.02;
        sr(r) = median(vecnorm(v(ok, :), 2, 2) ./ vecnorm(u(ok, :), 2, 2));
        an(r) = rad2deg(median(angle(exp(1i * (atan2(v(ok, 2), v(ok, 1)) - atan2(u(ok, 2), u(ok, 1)))))));
        k = (3:n - 1)'; k = k(h(k));
        X = [reshape(v(k - 1, :), [], 1), reshape(u(k, :), [], 1)]; y = reshape(v(k, :), [], 1); okk = all(isfinite([X y]), 2);
        th = X(okk, :) \ y(okk); gn(r) = th(2) / (1 - th(1));
    end
    cs = S.cluster_state; C = cs(:, 1:2); rr = vecnorm(C - [0.840 0.856], 2, 2); ang = unwrap(atan2(C(:, 2) - 0.856, C(:, 1) - 0.840));
    [~, nm] = fileparts(files{i}); run = regexp(nm, 'run\d+', 'match', 'once');
    fprintf('%-6s %-9s | %.2f / %.2f / %.2f / %.2f                      | %+4.0f / %+4.0f / %+4.0f / %+4.0f                   | %.2f / %.2f / %.2f / %.2f                      | r %.3f, %.1f laps, d1 %.3f d2 %.3f phi %.2f\n', ...
        run, rc.created(12:end), sr, an, gn, mean(rr(h)), abs(ang(end) - ang(find(h, 1))) / 2 / pi, mean(cs(h, 4)), mean(cs(h, 5)), mean(cs(h, 8)));
end
fprintf('(c = %s, t = %s, p = %s, R = %s)\n', names{:});
end
