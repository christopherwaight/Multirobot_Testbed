function fit_true_cp(root)
% Where is the true vortex centre (in Motive x,z)? Fit models to ALL readings at once.
fs = dir(fullfile(root, '**', '*.mat'));
P = []; TH = []; UV = []; RB = []; RUN = []; COND = {}; rdRun = []; cpMed = []; condRun = {};
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name)); rc = S.run_config;
    k = find(endsWith({rc.constants.block}, 'Jacobian_primitive/Constant1'), 1); rdRun(i) = str2double(rc.constants(k).value); %#ok<AGROW>
    [~, cn] = fileparts(fs(i).folder); condRun{i} = cn; %#ok<AGROW>
    cp = S.field_cp; ok = all(isfinite(cp), 2); cpMed(i, :) = median(cp(ok, :)); %#ok<AGROW>
    for rb = 1:3
        pose = S.(sprintf('pose_r%d', rb)); uv = S.field_uv(:, 2 * rb - 1:2 * rb);
        okr = all(isfinite([pose uv]), 2) & vecnorm(uv, 2, 2) > 1e-6;
        P = [P; pose(okr, 1:2)]; TH = [TH; pose(okr, 3)]; UV = [UV; uv(okr, :)]; %#ok<AGROW>
        RB = [RB; rb * ones(sum(okr), 1)]; RUN = [RUN; i * ones(sum(okr), 1)]; %#ok<AGROW>
    end
end
fprintf('%d runs, %d readings. r_d values: %s\n', numel(fs), size(P, 1), mat2str(unique(rdRun)));
ang = atan2(UV(:, 2), UV(:, 1)); mag = vecnorm(UV, 2, 2);
opt = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-7, 'TolFun', 1e-9);
c0 = median(cpMed);

% A: affine fit of all readings
A = [P ones(size(P, 1), 1)] \ UV; J = A(1:2, :)'; cA = (-J \ A(3, :)')';
fprintf('\nA  linear field, all readings:            centre (%.4f, %.4f)   J = %s\n', cA, mat2str(J, 2));

% B: direction-only vortex, per-robot angle bias
fB = @(q) dircost(q, P, ang, RB);
qB = fminsearch(fB, [c0 0 0 0], opt);
fprintf('B  vortex directions + robot angle bias:  centre (%.4f, %.4f)   bias celeste %+.1f, tidal %+.1f, pacific-blue %+.1f deg\n', qB(1:2), rad2deg(qB(3:5)));

% D: magnitude-only: |uv| = g*|p - c| + m0
qD = fminsearch(@(c) magcost(c, P, mag), c0, opt);
[~, gm] = magcost(qD, P, mag);
fprintf('D  magnitude only (|uv| = %.2f r %+.3f):    centre (%.4f, %.4f)\n', gm, qD);

% C: directions + per-robot sensor offset in body frame (q = p + R(th) s)
fprintf('   heading spread per robot (std, deg): %s\n', mat2str(round(rad2deg(arrayfun(@(r) std(unwrap(TH(RB == r))), 1:3)))));
qC = fminsearch(@(q) dircostOff(q, P, TH, ang, RB), [qB zeros(1, 6)], opt);
qC = fminsearch(@(q) dircostOff(q, P, TH, ang, RB), qC, opt);
fprintf('C  B + sensor offset per robot:           centre (%.4f, %.4f)   offsets (body x,z, cm): c %s t %s p %s\n', qC(1:2), ...
    mat2str(round(100 * qC(6:7), 1)), mat2str(round(100 * qC(8:9), 1)), mat2str(round(100 * qC(10:11), 1)));

% residual quality and direction-dependent error (model B)
e = resid(qB, P, ang, RB);
fprintf('\n   model B residual: median |e| %.1f deg, p90 %.1f deg\n', rad2deg(median(abs(e))), rad2deg(prctile(abs(e), 90)));
d = P - qB(1:2); pa = atan2(d(:, 1), -d(:, 2));
fprintf('   residual vs field direction (mean, deg, 8 bins of 45 deg) - tests direction-dependent NN error:\n');
for rb = 1:3
    m = zeros(1, 8);
    for b = 1:8
        sel = RB == rb & mod(pa + pi, 2 * pi) >= (b - 1) * pi / 4 & mod(pa + pi, 2 * pi) < b * pi / 4;
        m(b) = rad2deg(atan2(mean(sin(e(sel))), mean(cos(e(sel)))));
    end
    fprintf('     robot %d: %s\n', rb, mat2str(round(m, 1)));
end
r = vecnorm(d, 2, 2);
fprintf('   readings by distance from centre: <5cm %d, 5-15 %d, 15-30 %d, >30cm %d\n', sum(r < 0.05), sum(r >= 0.05 & r < 0.15), sum(r >= 0.15 & r < 0.3), sum(r >= 0.3));

% consistency: per condition and per run (model B)
fprintf('\n   model B per condition (did the map move?):\n');
conds = unique(condRun);
for k = 1:numel(conds)
    runs = find(strcmp(condRun, conds{k})); sel = ismember(RUN, runs);
    q = fminsearch(@(q) dircost(q, P(sel, :), ang(sel), RB(sel)), qB, opt);
    fprintf('     %-12s (r_d %s, %2d runs): centre (%.4f, %.4f) | live estimator run-median mean (%.4f, %.4f)\n', conds{k}, mat2str(unique(rdRun(runs))), numel(runs), q(1:2), mean(cpMed(runs, :), 1));
end
cr = zeros(numel(fs), 2);
for i = 1:numel(fs)
    sel = RUN == i; q = fminsearch(@(q) dircost(q, P(sel, :), ang(sel), RB(sel)), qB, opt); cr(i, :) = q(1:2);
end
fprintf('   per-run centres (model B): std x %.4f z %.4f m -> standard error of the all-data centre ~ %.4f m\n', std(cr), max(std(cr)) / sqrt(numel(fs)));
fprintf('   chronological drift check (run order by file time): first 10 runs (%.4f, %.4f) vs last 10 (%.4f, %.4f)\n', mean(cr(order(fs, 1:10), :)), mean(cr(order(fs, numel(fs) - 9:numel(fs)), :)));
fprintf('\n   live estimator (mean of run medians) minus model B centre: (%+.4f, %+.4f) m\n', mean(cpMed) - qB(1:2));
end

function idx = order(fs, k)
[~, o] = sort([fs.datenum]); idx = o(k);
end
function e = resid(q, P, ang, RB)
d = P - q(1:2); pred = atan2(d(:, 1), -d(:, 2)) + q(2 + RB)';  % vortex direction [-dz, dx] + robot bias
e = angle(exp(1i * (ang - pred)));
end
function f = dircost(q, P, ang, RB)
e = resid(q, P, ang, RB); f = sum(1 - cos(e));
end
function f = dircostOff(q, P, TH, ang, RB)
s = reshape(q(6:11), 2, 3)'; sr = s(RB, :);
Q = P + [cos(TH) .* sr(:, 1) - sin(TH) .* sr(:, 2), sin(TH) .* sr(:, 1) + cos(TH) .* sr(:, 2)];
f = dircost(q(1:5), Q, ang, RB);
end
function [f, gm] = magcost(c, P, mag)
r = vecnorm(P - c, 2, 2); X = [r ones(size(r))]; gm = X \ mag; f = sum((mag - X * gm).^2);
end
