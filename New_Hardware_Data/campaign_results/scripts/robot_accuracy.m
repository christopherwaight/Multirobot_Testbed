function robot_accuracy(d, ctrue)
% Rank each robot's field reading (colour sensor + NN -> hue/sat -> decoded u,v) against the ideal vortex.
fs = dir(fullfile(d, '*.mat'));
names = {}; A = struct();
for i = 1:numel(fs)
    S = load(fullfile(d, fs(i).name)); rc = S.run_config; N = size(rc.robots, 1);
    for rb = 1:N
        nm = matlab.lang.makeValidName(rc.robots{rb, 2});
        pose = S.(sprintf('pose_r%d', rb)); uv = S.field_uv(:, 2 * rb - 1:2 * rb);
        dd = pose(:, 1:2) - ctrue; r = vecnorm(dd, 2, 2); m = vecnorm(uv, 2, 2);
        ok = all(isfinite(uv), 2) & m > 1e-6 & r > 0.05;
        e = angle(exp(1i * (atan2(uv(ok, 2), uv(ok, 1)) - atan2(dd(ok, 1), -dd(ok, 2)))));   % ideal vortex dir [-dz, dx]
        pa = atan2(dd(ok, 1), -dd(ok, 2));
        if ~isfield(A, nm), A.(nm) = struct('e', [], 'pa', [], 'r', [], 'm', [], 'runs', 0, 'N3', 0, 'N4', 0); names{end+1} = rc.robots{rb, 2}; end %#ok<AGROW>
        A.(nm).e = [A.(nm).e; e]; A.(nm).pa = [A.(nm).pa; pa]; A.(nm).r = [A.(nm).r; r(ok)]; A.(nm).m = [A.(nm).m; m(ok)];
        A.(nm).runs = A.(nm).runs + 1; if N == 3, A.(nm).N3 = A.(nm).N3 + 1; else, A.(nm).N4 = A.(nm).N4 + 1; end
    end
end
% shared magnitude model from all robots
allr = []; allm = []; for k = 1:numel(names), x = A.(matlab.lang.makeValidName(names{k})); allr = [allr; x.r]; allm = [allm; x.m]; end %#ok<AGROW>
g = [allr ones(size(allr))] \ allm;
rows = zeros(numel(names), 7);
for k = 1:numel(names)
    x = A.(matlab.lang.makeValidName(names{k}));
    bias = atan2(mean(sin(x.e)), mean(cos(x.e)));
    b8 = zeros(1, 8);
    for b = 1:8
        s = mod(x.pa + pi, 2 * pi) >= (b - 1) * pi / 4 & mod(x.pa + pi, 2 * pi) < b * pi / 4;
        b8(b) = atan2(mean(sin(x.e(s))), mean(cos(x.e(s))));
    end
    gr = [x.r ones(size(x.r))] \ x.m;                         % this robot's own magnitude curve
    mres = x.m - [x.r ones(size(x.r))] * g;                  % vs the shared curve
    rows(k, :) = [rad2deg(median(abs(x.e))), rad2deg(prctile(abs(x.e), 90)), rad2deg(bias), rad2deg(max(b8) - min(b8)), sqrt(mean(mres.^2)), gr(1), numel(x.e)];
end
[~, o] = sort(rows(:, 1));
fprintf('shared magnitude model |uv| = %.2f r %+.3f\n', g);
fprintf('%-4s %-13s | %-8s | %-8s | %-8s | %-20s | %-14s | %-10s | %s\n', 'rank', 'robot', 'median', 'p90', 'bias', 'direction-dependent', 'magnitude RMS', 'mag slope', 'readings (runs 3R/4R)');
for j = 1:numel(o)
    k = o(j); x = A.(matlab.lang.makeValidName(names{k}));
    fprintf('%-4d %-13s | %5.1f°   | %5.1f°   | %+5.1f°   | %5.1f° peak-to-peak     | %.3f          | %.2f       | %d (%d/%d)\n', j, names{k}, rows(k, 1:5), rows(k, 6), rows(k, 7), x.N3, x.N4);
end
end
