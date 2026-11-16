function failed_run(f, ctr, okDir)
S = load(f); rc = S.run_config; t = S.tout; n = numel(t);
k = find(endsWith(strrep({rc.constants.block}, newline, ' '), '/Jacobian/Constant1'), 1);
fprintf('created %s | %.1f s | r_d %s | sha %s\n', rc.created, t(end), rc.constants(k).value, rc.model_file_sha256(1:12));
% reference: sensor values seen on the map in good 4R runs
fs = dir(fullfile(okDir, '*.mat')); satok = []; rawok = [];
for i = 1:numel(fs), G = load(fullfile(okDir, fs(i).name), 'sensor_sat', 'sensor_rgbk_raw_r1'); satok = [satok; G.sensor_sat(:)]; rawok = [rawok; G.sensor_rgbk_raw_r1]; end %#ok<AGROW>
fprintf('on-map reference (4R r_d 0.4 runs): sat %.2f-%.2f (p1-p99), raw r1 clear K %.0f-%.0f\n', prctile(satok, 1), prctile(satok, 99), prctile(rawok(:, 1), 1), prctile(rawok(:, 1), 99));
C = S.cluster_state(:, 1:2); rr = vecnorm(C - ctr, 2, 2); cp = S.field_cp_jacobian;
fprintf('\n t(s) | centroid dist | robot dist from centre (c t p R)  | sat (c t p R)          | raw K (c t p R)        | est dist from centre | |cmd| max robot | type\n');
for tt = [1:20:n n]
    d = zeros(1, 4); K = d;
    for r = 1:4, p = S.(sprintf('pose_r%d', r)); d(r) = norm(p(tt, 1:2) - ctr); K(r) = S.(sprintf('sensor_rgbk_raw_r%d', r))(tt, 1); end
    cm = max(vecnorm(reshape(S.cmd_robots_selected(tt, :), 3, 4)', 2, 2));
    fprintf('%5.1f | %.3f         | %.2f %.2f %.2f %.2f            | %.2f %.2f %.2f %.2f    | %4.0f %4.0f %4.0f %4.0f    | %.3f               | %.3f           | %d\n', ...
        t(tt), rr(tt), d, S.sensor_sat(tt, :), K, norm(cp(tt, :) - ctr), cm, S.field_jacobian(tt, 8));
end
fprintf('\nshape: d1 %.3f-%.3f d2 %.3f-%.3f phi %.2f-%.2f | NaN in commands: %d rows\n', min(S.cluster_state(:, 4)), max(S.cluster_state(:, 4)), min(S.cluster_state(:, 5)), max(S.cluster_state(:, 5)), min(S.cluster_state(:, 8)), max(S.cluster_state(:, 8)), sum(any(~isfinite(S.cmd_robots_selected), 2)));
fprintf('position extent: x %.2f..%.2f  z %.2f..%.2f (all robots)\n', min(arrayfun(@(r) min(S.(sprintf('pose_r%d', r))(:, 1)), 1:4)), max(arrayfun(@(r) max(S.(sprintf('pose_r%d', r))(:, 1)), 1:4)), ...
    min(arrayfun(@(r) min(S.(sprintf('pose_r%d', r))(:, 2)), 1:4)), max(arrayfun(@(r) max(S.(sprintf('pose_r%d', r))(:, 2)), 1:4)));
end
