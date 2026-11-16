function cp_drift_vs_radius(root, c0)
% Separate "map moved over time" from "centre depends on which radius is sampled".
% (1) inner vs outer readings of the SAME runs (same time, different radius)
% (2) early vs late runs of the SAME condition (same radius, different time)
fs = dir(fullfile(root, '**', '*.mat')); [~, o] = sort([fs.datenum]); fs = fs(o);
opt = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-7, 'TolFun', 1e-9);
D = struct('P', {}, 'ang', {}, 'RB', {}, 'cond', {}, 't', {});
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name)); [~, cn] = fileparts(fs(i).folder);
    P = []; A = []; R = [];
    for rb = 1:3
        pose = S.(sprintf('pose_r%d', rb)); uv = S.field_uv(:, 2 * rb - 1:2 * rb);
        ok = all(isfinite([pose uv]), 2) & vecnorm(uv, 2, 2) > 1e-6;
        P = [P; pose(ok, 1:2)]; A = [A; atan2(uv(ok, 2), uv(ok, 1))]; R = [R; rb * ones(sum(ok), 1)]; %#ok<AGROW>
    end
    D(i).P = P; D(i).ang = A; D(i).RB = R; D(i).cond = cn; D(i).t = fs(i).datenum;
end
q0 = [c0 0 0 0];
fprintf('(1) SAME runs, inner vs outer readings (distance from %.3f, %.3f):\n', c0);
conds = unique({D.cond}, 'stable');
for k = 1:numel(conds)
    idx = find(strcmp({D.cond}, conds{k}));
    P = vertcat(D(idx).P); A = vertcat(D(idx).ang); R = vertcat(D(idx).RB);
    r = vecnorm(P - c0, 2, 2); med = median(r);
    qi = fminsearch(@(q) dc(q, P(r < med, :), A(r < med), R(r < med)), q0, opt);
    qo = fminsearch(@(q) dc(q, P(r >= med, :), A(r >= med), R(r >= med)), q0, opt);
    fprintf('   %-11s inner (r < %.2f m): (%.4f, %.4f) | outer: (%.4f, %.4f) | outer - inner (%+.4f, %+.4f)\n', conds{k}, med, qi(1:2), qo(1:2), qo(1:2) - qi(1:2));
end
fprintf('(2) SAME condition, first half vs second half of its runs (time order):\n');
for k = 1:numel(conds)
    idx = find(strcmp({D.cond}, conds{k})); h = floor(numel(idx) / 2); if h < 1, continue; end
    a = idx(1:h); b = idx(h + 1:end);
    qa = fminsearch(@(q) dc(q, vertcat(D(a).P), vertcat(D(a).ang), vertcat(D(a).RB)), q0, opt);
    qb = fminsearch(@(q) dc(q, vertcat(D(b).P), vertcat(D(b).ang), vertcat(D(b).RB)), q0, opt);
    fprintf('   %-11s early (%.4f, %.4f) | late (%.4f, %.4f) | late - early (%+.4f, %+.4f)  [%s - %s]\n', conds{k}, qa(1:2), qb(1:2), qb(1:2) - qa(1:2), datestr(D(a(1)).t, 'HH:MM'), datestr(D(b(end)).t, 'HH:MM'));
end
% (3) one common radius band across all conditions (0.15-0.30 m) -> per-condition centre; isolates time
fprintf('(3) only readings 0.15-0.30 m from the centre, per condition (same radius band, different time):\n');
for k = 1:numel(conds)
    idx = find(strcmp({D.cond}, conds{k}));
    P = vertcat(D(idx).P); A = vertcat(D(idx).ang); R = vertcat(D(idx).RB); r = vecnorm(P - c0, 2, 2); m = r >= 0.15 & r < 0.30;
    if sum(m) < 300, fprintf('   %-11s too few (%d)\n', conds{k}, sum(m)); continue; end
    q = fminsearch(@(q) dc(q, P(m, :), A(m), R(m)), q0, opt);
    fprintf('   %-11s (%5d readings, %s): (%.4f, %.4f)\n', conds{k}, sum(m), datestr(D(idx(1)).t, 'HH:MM'), q(1:2));
end
end
function f = dc(q, P, ang, RB)
d = P - q(1:2); pred = atan2(d(:, 1), -d(:, 2)) + q(2 + RB)'; f = sum(1 - cos(ang - pred));
end
