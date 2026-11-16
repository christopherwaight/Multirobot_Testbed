function T = campaign_report(dataDir)
% CAMPAIGN_REPORT  All numbers for the 2026-11-13 vortex-orbit hardware campaign (3 and 4 robots).
%   T = campaign_report('data_hardware_campaign_11_11_2026')
% Reads every vortex_<N>robots_radius_of_0_point_<d>_run<k>.mat under dataDir (recursively; duplicate
% file names are read once, newest copy). Prints every table used in
% trunk/campaign_results/2026-11-13_vortex_orbit_campaign.md and returns them in struct T.
% Written by Claude (AI assistant), 2026-11-13. Read-only: it never modifies data files.
%
% Conventions
%  - Motive floor plane (x, z). "True centre" = vortex centre fitted from ALL sensor readings
%    (direction-only vortex model with a constant angle bias per robot), see section B.
%  - Steady-state window: t > 10 s. Battery screen: a run is "weak" if any robot reaches < 0.7 of its
%    commanded speed (median |actual|/|commanded| over t > 5 s); weak runs are excluded from the
%    condition statistics, the plant fit and the estimator tables (they are listed in section A).
%  - Estimator = the critical-point estimate the controller actually used: 3 robots field_cp (exact
%    3-point linear fit), 4 robots field_cp_jacobian (4-point least squares).

if nargin < 1, dataDir = 'data_hardware_campaign_11_11_2026'; end
fs = dir(fullfile(dataDir, '**', 'vortex_*robots_radius_of_0_point_*_run*.mat'));
[~, o] = sort([fs.datenum], 'descend'); fs = fs(o);
[~, keep] = unique({fs.name}, 'stable'); fs = fs(keep);
[~, o] = sort([fs.datenum]); fs = fs(o);
opt = optimset('MaxFunEvals', 200000, 'MaxIter', 200000, 'TolX', 1e-8, 'TolFun', 1e-10);
names4 = {'celeste', 'tidal', 'pacific-blue', 'redwood'};

%% load
R = struct([]);
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name)); rc = S.run_config;
    tok = regexp(fs(i).name, 'vortex_(\d)robots_radius_of_0_point_(\d+)_run(\d+)', 'tokens', 'once');
    N = str2double(tok{1}); rdName = str2double(['0.' tok{2}]); run = str2double(tok{3});
    if N == 3, cpBlk = 'Jacobian_primitive/Constant1'; cp = S.field_cp; lr = S.orbit_radius(:, 1); cmdc = S.cmd_cluster_jacobian;
    else, cpBlk = '/Jacobian/Constant1'; cp = S.field_cp_jacobian; lr = S.orbit_radius_jacobian(:, 1); cmdc = S.cmd_cluster_jacobian; end
    k = find(endsWith(strrep({rc.constants.block}, newline, ' '), cpBlk), 1); rd = str2double(rc.constants(k).value);
    orb = rc.matlab_functions(contains(strrep({rc.matlab_functions.block}, newline, ' '), ternary(N == 3, 'Jacobian_primitive/MATLAB Function', '/Jacobian/MATLAB Function'))).code;
    t = S.tout; P = zeros(numel(t), N, 2); U = P; TH = zeros(numel(t), N);
    for r = 1:N, p = S.(sprintf('pose_r%d', r)); P(:, r, :) = p(:, 1:2); TH(:, r) = p(:, 3); U(:, r, :) = S.field_uv(:, 2 * r - 1:2 * r); end
    ratio = zeros(1, N); angE = ratio;
    for r = 1:N
        u = S.cmd_robots_selected(:, 3 * r - 2:3 * r - 1); v = [diff(squeeze(P(:, r, :))) / 0.1; nan(1, 2)];
        m = t > 5 & vecnorm(u, 2, 2) > 0.02 & vecnorm(v, 2, 2) > 0.02;
        ratio(r) = median(vecnorm(v(m, :), 2, 2) ./ vecnorm(u(m, :), 2, 2));
        angE(r) = rad2deg(median(angle(exp(1i * (atan2(v(m, 2), v(m, 1)) - atan2(u(m, 2), u(m, 1)))))));
    end
    sel = {'setpoint', 'jacobian', 'jacobians', 'vectorsum', 'ascent', 'descent'}; act = '';
    for q = 1:numel(sel), f = ['cmd_robots_' sel{q}]; if isfield(S, f) && isequal(size(S.(f)), size(S.cmd_robots_selected)) && max(abs(S.(f)(:) - S.cmd_robots_selected(:))) < 1e-12, act = sel{q}; break; end, end
    R(end+1).file = fs(i).name; R(end).N = N; R(end).rd = rd; R(end).rdName = rdName; R(end).run = run; %#ok<AGROW>
    R(end).time = rc.created; R(end).dur = t(end); R(end).sha = rc.model_file_sha256(1:12); R(end).dirty = rc.model_dirty;
    R(end).kt = regexp(orb, 'k_t = [\d.]+', 'match', 'once'); R(end).kr = regexp(orb, 'k_r = [\d.]+', 'match', 'once'); R(end).active = act;
    R(end).robots = rc.robots(:, 2)'; R(end).t = t; R(end).P = P; R(end).U = U; R(end).C = S.cluster_state(:, 1:2);
    R(end).cs = S.cluster_state; R(end).sp = S.setpoint(1, :); R(end).cp = cp; R(end).lr = lr; R(end).cmdc = cmdc(:, 1:2);
    R(end).ratio = ratio; R(end).angE = angE; R(end).weak = any(ratio < 0.7); R(end).fj = S.field_jacobian(:, 8);
end
T.runs = R;

%% A. inventory
fprintf('\n=== A. INVENTORY (%d runs)\n', numel(R));
key = arrayfun(@(x) sprintf('%dR r_d %.2f', x.N, x.rd), R, 'uni', 0); [uk, ~, ic] = unique(key, 'stable');
fprintf('%-13s | runs | weak | dur (s) | time span   | model sha (dirty) | Orbiter        | active  | name=config\n', 'condition');
for k = 1:numel(uk)
    g = R(ic == k);
    fprintf('%-13s | %4d | %4d | %-7s | %s-%s | %-17s | %s %s | %-7s | %d\n', uk{k}, numel(g), sum([g.weak]), strjoin(unique(arrayfun(@(x) sprintf('%g', x.dur), g, 'uni', 0)), '/'), ...
        g(1).time(12:16), g(end).time(12:16), strjoin(unique(arrayfun(@(x) sprintf('%s(%d)', x.sha, x.dirty), g, 'uni', 0)), ' '), g(1).kt, g(1).kr, ...
        strjoin(unique({g.active}), '/'), all(abs([g.rd] - [g.rdName]) < 1e-9));
end
w = R([R.weak]); fprintf('weak-battery runs (any robot < 0.7): %s\n', strjoin(arrayfun(@(x) sprintf('%s [%s]', x.file, num2str(x.ratio, '%.2f ')), w, 'uni', 0), '; '));

%% B. true centre
fprintf('\n=== B. TRUE CENTRE (direction-only vortex fit, constant angle bias per robot)\n');
Pa = []; Aa = []; Ra = []; Ga = []; Ka = [];
for i = 1:numel(R)
    for r = 1:R(i).N
        p = squeeze(R(i).P(:, r, :)); u = squeeze(R(i).U(:, r, :)); ok = all(isfinite([p u]), 2) & vecnorm(u, 2, 2) > 1e-6;
        Pa = [Pa; p(ok, :)]; Aa = [Aa; atan2(u(ok, 2), u(ok, 1))]; Ra = [Ra; find(strcmp(names4, R(i).robots{r})) * ones(sum(ok), 1)]; %#ok<AGROW>
        Ga = [Ga; R(i).N * ones(sum(ok), 1)]; Ka = [Ka; ic(i) * ones(sum(ok), 1)]; %#ok<AGROW>
    end
end
q0 = [0.84 0.86 0 0 0 0];
qa = fminsearch(@(q) dcost(q, Pa, Aa, Ra), q0, opt);
fprintf('all runs:        centre (%.4f, %.4f) | robot angle bias c %+.1f t %+.1f p %+.1f r %+.1f deg\n', qa(1:2), rad2deg(qa(3:6)));
T.centre = qa(1:2);
for N = [3 4]
    m = Ga == N; q = fminsearch(@(q) dcost(q, Pa(m, :), Aa(m), Ra(m)), qa, opt); T.(sprintf('centre%dR', N)) = q(1:2);
    fprintf('%d-robot runs:    centre (%.4f, %.4f)  (shift from all-runs %+.4f, %+.4f)\n', N, q(1:2), q(1:2) - qa(1:2));
end
for k = 1:numel(uk)
    m = Ka == k; q = fminsearch(@(q) dcost(q, Pa(m, :), Aa(m), Ra(m)), qa, opt);
    fprintf('   %-13s centre (%.4f, %.4f)\n', uk{k}, q(1:2));
end
pr = zeros(numel(R), 2);
for i = 1:numel(R)
    m = []; P1 = []; A1 = []; R1 = [];
    for r = 1:R(i).N
        p = squeeze(R(i).P(:, r, :)); u = squeeze(R(i).U(:, r, :)); ok = all(isfinite([p u]), 2) & vecnorm(u, 2, 2) > 1e-6;
        P1 = [P1; p(ok, :)]; A1 = [A1; atan2(u(ok, 2), u(ok, 1))]; R1 = [R1; find(strcmp(names4, R(i).robots{r})) * ones(sum(ok), 1)]; %#ok<AGROW>
    end
    q = fminsearch(@(q) dcost(q, P1, A1, R1), qa, opt); pr(i, :) = q(1:2);
end
fprintf('per-run centres: std x %.4f z %.4f m (n %d) -> standard error of the all-runs centre ~%.4f m\n', std(pr), numel(R), max(std(pr)) / sqrt(numel(R)));
ctr = qa(1:2);

%% C. per condition
fprintf('\n=== C. PER CONDITION (healthy runs, t > 10 s, radius about the all-runs centre)\n');
fprintf('%-13s | n  | radius mean±sd | last 5 s | wobble | r-r_d  | logged r | est. jitter | est >10cm | est offset | speed | laps/10s | shape err | vortex%%\n', 'condition');
C = struct([]);
for k = 1:numel(uk)
    g = R(ic == k & ~[R.weak]'); if isempty(g), continue; end
    v = zeros(numel(g), 12);
    for j = 1:numel(g)
        x = g(j); h = x.t > 10; r = vecnorm(x.C - ctr, 2, 2); ok = all(isfinite(x.cp), 2) & h;
        cpm = median(x.cp(ok, :)); dv = vecnorm(x.cp(ok, :) - cpm, 2, 2);
        a = unwrap(atan2(x.C(:, 2) - ctr(2), x.C(:, 1) - ctr(1))); i0 = find(h, 1);
        v(j, :) = [mean(r(h)), mean(r(x.t > x.t(end) - 5)), std(r(h)), mean(r(h)) - x.rd, mean(x.lr(h)), 1.4826 * median(dv), mean(dv > 0.1), norm(cpm - ctr), ...
            median(vecnorm(diff(x.C(h, :)), 2, 2)) / 0.1, abs(a(end) - a(i0)) / 2 / pi / (x.t(end) - x.t(i0)) * 10, ...
            sqrt(mean(mean((x.cs(h, 4:5) - x.sp(4:5)).^2))), mean(x.fj(h) == 1 | x.fj(h) == 2)];
    end
    mv = mean(v, 1);
    fprintf('%-13s | %-2d | %.3f ± %.3f  | %.3f    | %.3f  | %+.3f | %.3f    | %.3f       | %4.1f%%     | %.3f      | %.3f | %.2f     | %.3f     | %3.0f%%\n', ...
        uk{k}, numel(g), mv(1), std(v(:, 1)), mv(2), mv(3), mv(4), mv(5), mv(6), 100 * mv(7), mv(8), mv(9), mv(10), mv(11), 100 * mv(12));
    C(end+1).cond = uk{k}; C(end).N = g(1).N; C(end).rd = g(1).rd; C(end).n = numel(g); C(end).v = v; %#ok<AGROW>
end
T.conditions = C;
fprintf('(wobble = std of radius within a run; est. = the live critical-point estimate; est offset = run-median estimate vs centre; shape err = RMS of p,q (3R) / d1,d2 (4R) vs setpoint)\n');

%% D. estimator vs distance
fprintf('\n=== D. ESTIMATOR vs CLUSTER DISTANCE FROM CENTRE (healthy runs, t > 5 s)\n');
edges = [0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.55 0.7];
fprintf('distance     | 3R: steps  median err  >10cm  radial bias | 4R: steps  median err  >10cm  radial bias   (bias - = toward cluster)\n');
Dd = {[], []}; Ee = Dd; Bb = Dd;
for x = R(~[R.weak])
    g = x.N - 2; ok = all(isfinite(x.cp), 2) & x.t > 5; d = vecnorm(x.C(ok, :) - ctr, 2, 2); u = (ctr - x.C(ok, :)) ./ d;
    Dd{g} = [Dd{g}; d]; Ee{g} = [Ee{g}; vecnorm(x.cp(ok, :) - ctr, 2, 2)]; Bb{g} = [Bb{g}; sum((x.cp(ok, :) - ctr) .* u, 2)];
end
for b = 1:numel(edges) - 1
    s = cell(1, 2);
    for g = 1:2
        m = Dd{g} >= edges(b) & Dd{g} < edges(b + 1);
        if sum(m) < 50, s{g} = sprintf('%5d   %-34s', sum(m), '(too few)'); else, s{g} = sprintf('%5d   %.3f       %4.1f%%  %+.3f     ', sum(m), median(Ee{g}(m)), 100 * mean(Ee{g}(m) > 0.1), median(Bb{g}(m))); end
    end
    fprintf('%.2f-%.2f m  | %s | %s\n', edges(b), edges(b + 1), s{1}, s{2});
end

%% E. plant fit
fprintf('\n=== E. ROBOT RESPONSE (centroid v(k) = a v(k-1) + b u(k), u = Orbiter command; healthy runs)\n');
for N = [3 4]
    X = []; y = [];
    for x = R(~[R.weak] & [R.N] == N)
        v = [diff(x.C) / 0.1; nan(1, 2)]; kk = (3:numel(x.t) - 1)';
        X = [X; reshape(v(kk - 1, :), [], 1), reshape(x.cmdc(kk, :), [], 1)]; y = [y; reshape(v(kk, :), [], 1)]; %#ok<AGROW>
    end
    ok = all(isfinite([X y]), 2); th = X(ok, :) \ y(ok); R2 = 1 - sum((y(ok) - X(ok, :) * th).^2) / sum((y(ok) - mean(y(ok))).^2);
    fprintf('%dR: a %.3f  b %.3f  speed reached b/(1-a) %.2f  R^2 %.3f  (%d samples)\n', N, th, th(2) / (1 - th(1)), R2, sum(ok));
    T.(sprintf('plant%dR', N)) = [th' th(2) / (1 - th(1)) R2];
end
fprintf('per robot speed reached |actual|/|cmd| (healthy runs, median of run medians) and direction error vs command:\n');
for nm = names4
    rr = []; aa = [];
    for x = R(~[R.weak]), k = find(strcmp(x.robots, nm{1})); if ~isempty(k), rr(end+1) = x.ratio(k); aa(end+1) = x.angE(k); end, end %#ok<AGROW>
    fprintf('   %-13s ratio %.2f [%.2f-%.2f]  direction %+.1f deg  (%d runs)\n', nm{1}, median(rr), min(rr), max(rr), median(aa), numel(rr));
end

%% F. sensor accuracy per robot
fprintf('\n=== F. SENSOR READING ACCURACY per robot (all runs; readings > 5 cm from centre; vs ideal vortex about the centre)\n');
allr = []; allm = [];
for x = R, for r = 1:x.N, p = squeeze(x.P(:, r, :)); u = squeeze(x.U(:, r, :)); ok = all(isfinite(u), 2) & vecnorm(u, 2, 2) > 1e-6; allr = [allr; vecnorm(p(ok, :) - ctr, 2, 2)]; allm = [allm; vecnorm(u(ok, :), 2, 2)]; end, end %#ok<AGROW>
gm = [allr ones(size(allr))] \ allm; fprintf('shared strength model |uv| = %.3f r %+.3f\n', gm);
fprintf('robot         | trained | median | p90    | bias   | dir-dependent p2p | strength slope | strength RMS | readings\n');
for nm = names4
    e = []; pa = []; rr = []; mm = [];
    for x = R
        k = find(strcmp(x.robots, nm{1})); if isempty(k), continue; end
        p = squeeze(x.P(:, k, :)); u = squeeze(x.U(:, k, :)); d = p - ctr; r = vecnorm(d, 2, 2); ok = all(isfinite(u), 2) & vecnorm(u, 2, 2) > 1e-6 & r > 0.05;
        e = [e; angle(exp(1i * (atan2(u(ok, 2), u(ok, 1)) - atan2(d(ok, 1), -d(ok, 2)))))]; pa = [pa; atan2(d(ok, 1), -d(ok, 2))]; %#ok<AGROW>
        rr = [rr; r(ok)]; mm = [mm; vecnorm(u(ok, :), 2, 2)]; %#ok<AGROW>
    end
    b8 = zeros(1, 8); for b = 1:8, s = mod(pa + pi, 2 * pi) >= (b - 1) * pi / 4 & mod(pa + pi, 2 * pi) < b * pi / 4; b8(b) = atan2(mean(sin(e(s))), mean(cos(e(s)))); end
    gr = [rr ones(size(rr))] \ mm; res = mm - [rr ones(size(rr))] * gm;
    fprintf('%-13s | %-7s | %5.1f° | %5.1f° | %+5.1f° | %5.1f°            | %.2f           | %.3f        | %d\n', nm{1}, ternary(strcmp(nm{1}, 'redwood'), 'NO', 'yes'), ...
        rad2deg(median(abs(e))), rad2deg(prctile(abs(e), 90)), rad2deg(atan2(mean(sin(e)), mean(cos(e)))), rad2deg(max(b8) - min(b8)), gr(1), sqrt(mean(res.^2)), numel(e));
end
fprintf('\nDone. Centre used throughout: (%.4f, %.4f).\n', ctr);
end

function f = dcost(q, P, ang, RB)
d = P - q(1:2); pred = atan2(d(:, 1), -d(:, 2)) + q(2 + RB)'; f = sum(1 - cos(ang - pred));
end
function o = ternary(c, a, b), if c, o = a; else, o = b; end, end
