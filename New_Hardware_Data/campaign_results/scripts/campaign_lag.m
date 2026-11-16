function campaign_lag(root)
% (1) CP outliers: robust jitter + outlier fraction. (2) Where is the live CP estimate relative to the
% cluster's motion (latency signature)? (3) Sensor-to-pose lag: shift readings by L steps vs poses and
% see which L fits an ideal linear vortex best. (4) Radius prediction with robust jitter and with a
% CP that lags the cluster.
subs = dir(root); subs = subs([subs.isdir] & ~startsWith({subs.name}, '.'));
Ls = 0:4; dt = 0.1;
for s = 1:numel(subs)
    fs = dir(fullfile(root, subs(s).name, '*.mat'));
    off = []; outl = []; err = zeros(numel(fs), numel(Ls)); rds = 0; robust = [];
    for i = 1:numel(fs)
        S = load(fullfile(fs(i).folder, fs(i).name)); rc = S.run_config;
        k = find(endsWith({rc.constants.block}, 'Jacobian_primitive/Constant1'), 1); rds = str2double(rc.constants(k).value);
        h = S.tout > 10; cp = S.field_cp; ok = all(isfinite(cp), 2) & h; cpm = median(cp(ok, :));
        dev = cp - cpm; nd = vecnorm(dev, 2, 2);
        outl(end+1) = mean(nd(ok) > 0.1); %#ok<AGROW>
        robust(end+1, :) = 1.4826 * median(abs(dev(ok & nd < 0.1, :))); %#ok<AGROW>
        % live CP offset in the cluster frame: radial (from median CP toward centroid), tangential (along motion)
        C = S.cluster_state(:, 1:2); v = [diff(C); nan(1, 2)];
        er = (C - cpm) ./ vecnorm(C - cpm, 2, 2); et = v ./ vecnorm(v, 2, 2);
        m = ok & nd < 0.1 & all(isfinite(et), 2);
        off = [off; sum(dev(m, :) .* er(m, :), 2), sum(dev(m, :) .* et(m, :), 2)]; %#ok<AGROW>
        % lag test: readings at step k vs poses at step k-L
        for j = 1:numel(Ls)
            L = Ls(j); X = []; Y = [];
            for rb = 1:3
                pose = S.(sprintf('pose_r%d', rb)); uv = S.field_uv(:, 2 * rb - 1:2 * rb);
                X = [X; pose(1:end - L, 1:2)]; Y = [Y; uv(1 + L:end, :)]; %#ok<AGROW>
            end
            okr = all(isfinite([X Y]), 2) & vecnorm(Y, 2, 2) > 1e-6;
            A = [X(okr, :) ones(sum(okr), 1)] \ Y(okr, :); Yh = [X(okr, :) ones(sum(okr), 1)] * A;
            err(i, j) = rad2deg(median(abs(wrapToPi(atan2(Y(okr, 2), Y(okr, 1)) - atan2(Yh(:, 2), Yh(:, 1))))));
        end
    end
    fprintf('\n== %s (r_d %.2f, %d runs)\n', subs(s).name, rds, numel(fs));
    fprintf('   CP outliers (> 10 cm from run median): %.1f%% of steps (worst run %.1f%%) | robust jitter std x %.3f z %.3f m\n', 100 * mean(outl), 100 * max(outl), mean(robust));
    fprintf('   live CP offset vs cluster: radial %+.3f m (+ = toward cluster), along-motion %+.3f m (- = behind)\n', mean(off(:, 1)), mean(off(:, 2)));
    fprintf('   direction error vs best linear field when readings are shifted L steps later than poses:\n     ');
    fprintf('L=%d: %.1f deg   ', [Ls; mean(err, 1)]); fprintf('\n');
    S2(s).rd = rds; S2(s).jit = mean(robust); S2(s).off = mean(off); %#ok<AGROW>
end
fprintf('\n== radius prediction, re-fitted plant a 0.354 b 0.552:\n');
for s = 1:numel(S2)
    fprintf('   r_d %.2f: robust jitter %.3f | + live CP displaced as measured (rotating with cluster) %.3f\n', S2(s).rd, ...
        simcl(0.354, 0.552, S2(s).jit, S2(s).rd, [0 0]), simcl(0.354, 0.552, S2(s).jit, S2(s).rd, S2(s).off));
end
end

function rbar = simcl(a, b, jit, rd, off)
% off = [radial tangential] displacement of the CP estimate in the cluster frame
dt = 0.1; N = 600; rng(1); c = [0.3 0]; v = [0 0]; R = zeros(N, 1); kt = 0.15; kr = 1;
for k = 1:N
    er = c / norm(c); et = v / max(norm(v), 1e-9); if norm(v) < 1e-9, et = [-er(2) er(1)]; end
    cp = jit .* randn(1, 2) + off(1) * er + off(2) * et;
    d = cp - c; r = norm(d); un = d / r;
    rs = max(min(kr * (r - rd), 0.3), -0.3);
    u = kt * [un(2), -un(1)] + rs * un;
    v = a * v + b * u; c = c + dt * v; R(k) = norm(c);
end
rbar = mean(R(200:end));
end
