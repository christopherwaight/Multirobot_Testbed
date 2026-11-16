function campaign_sanity(root)
% Does the campaign data make sense? Per condition (sub-folder): sensor accuracy vs an ideal vortex,
% CP repeatability, measured radius/speed, plant re-fit, and closed-loop radius prediction.
subs = dir(root); subs = subs([subs.isdir] & ~startsWith({subs.name}, '.'));
dt = 0.1; k_t = 0.15; k_r = 1; clipv = 0.3;
fprintf('%-11s %-5s %-4s | %-15s | %-24s | %-27s | %-18s | %s\n', 'condition', 'r_d', 'runs', 'radius mean±sd', 'logged r (to live CP)', 'CP median (x,z) / jitter', 'speed m/s', 'name matches config');
P = struct();
for s = 1:numel(subs)
    fs = dir(fullfile(root, subs(s).name, '*.mat'));
    R = []; RL = []; SP = []; CPm = []; CPj = []; names_ok = true; rds = [];
    U = []; V = []; Vp = []; Jit = [];
    E = struct('ang', {[], [], []}); SAT = []; DIST = []; R2u = []; R2v = []; ANGfit = [];
    for i = 1:numel(fs)
        S = load(fullfile(fs(i).folder, fs(i).name)); rc = S.run_config;
        k = find(endsWith({rc.constants.block}, 'Jacobian_primitive/Constant1'), 1); rd = str2double(rc.constants(k).value); rds(end+1) = rd; %#ok<AGROW>
        tok = regexp(fs(i).name, 'radius_of_0_point_(\d+)', 'tokens', 'once'); rdn = str2double(['0.' tok{1}]);
        if abs(rdn - rd) > 1e-9, names_ok = false; fprintf('   NAME MISMATCH %s: config r_d %g\n', fs(i).name, rd); end
        h = S.tout > 10;
        cp = S.field_cp; ok = all(isfinite(cp), 2); cpm = median(cp(ok, :));
        C = S.cluster_state(:, 1:2);
        r = vecnorm(C - cpm, 2, 2); R(end+1) = mean(r(h)); RL(end+1) = mean(S.orbit_radius(h, 1), 'omitnan'); %#ok<AGROW>
        v = [diff(C) / dt; nan(1, 2)]; SP(end+1) = median(vecnorm(v(h, :), 2, 2), 'omitnan'); %#ok<AGROW>
        CPm(end+1, :) = cpm; CPj(end+1, :) = std(cp(ok & h, :)); %#ok<AGROW>
        Jit = [Jit; cp(ok, :) - cpm]; %#ok<AGROW>
        % plant samples: centroid velocity vs commanded centroid velocity
        u = S.cmd_cluster_jacobian(:, 1:2); kk = (3:numel(S.tout) - 1)';
        U = [U; u(kk, :)]; V = [V; v(kk, :)]; Vp = [Vp; v(kk - 1, :)]; %#ok<AGROW>
        % sensors: linear field fit over all robots, all samples
        X = []; Y = []; SATi = []; P3 = [];
        for rb = 1:3
            pose = S.(sprintf('pose_r%d', rb)); uv = S.field_uv(:, 2 * rb - 1:2 * rb);
            X = [X; pose(:, 1:2)]; Y = [Y; uv]; SATi = [SATi; S.sensor_sat(:, rb)]; P3 = [P3; rb * ones(size(pose, 1), 1)]; %#ok<AGROW>
        end
        okr = all(isfinite([X Y]), 2) & vecnorm(Y, 2, 2) > 1e-6;
        A = [X(okr, :) ones(sum(okr), 1)] \ Y(okr, :);
        Yh = [X(okr, :) ones(sum(okr), 1)] * A;
        R2u(end+1) = 1 - sum((Y(okr, 1) - Yh(:, 1)).^2) / sum((Y(okr, 1) - mean(Y(okr, 1))).^2); %#ok<AGROW>
        R2v(end+1) = 1 - sum((Y(okr, 2) - Yh(:, 2)).^2) / sum((Y(okr, 2) - mean(Y(okr, 2))).^2); %#ok<AGROW>
        J = A(1:2, :)'; c = (-J \ A(3, :)')';            % fitted centre
        curl = J(2, 1) - J(1, 2);
        d = X - c; ideal = sign(curl) * [-d(:, 2), d(:, 1)];
        ang = abs(wrapToPi(atan2(Y(:, 2), Y(:, 1)) - atan2(ideal(:, 2), ideal(:, 1))));
        ANGfit(end+1) = rad2deg(median(abs(wrapToPi(atan2(Y(okr, 2), Y(okr, 1)) - atan2(Yh(:, 2), Yh(:, 1)))))); %#ok<AGROW>
        for rb = 1:3, E(rb).ang = [E(rb).ang; ang(P3 == rb & okr)]; end
        SAT = [SAT; SATi(okr)]; DIST = [DIST; vecnorm(d(okr, :), 2, 2)]; %#ok<AGROW>
    end
    cond = subs(s).name;
    fprintf('%-11s %-5.2f %-4d | %.3f ± %.3f   | %.3f                    | (%.3f, %.3f) / %.3f       | %.3f              | %d\n', ...
        cond, mean(rds), numel(fs), mean(R), std(R), mean(RL), mean(CPm), mean(vecnorm(CPj, 2, 2)), mean(SP), names_ok);
    P(s).cond = cond; P(s).rd = mean(rds); P(s).R = R; P(s).U = U; P(s).V = V; P(s).Vp = Vp; P(s).jit = std(Jit); P(s).CPm = CPm;
    P(s).R2u = R2u; P(s).R2v = R2v; P(s).ANGfit = ANGfit; P(s).E = E; P(s).SAT = SAT; P(s).DIST = DIST; P(s).SP = SP;
end

fprintf('\n== CRITICAL POINT repeatability (per-run median, across runs)\n');
allc = vertcat(P.CPm);
for s = 1:numel(P), fprintf('   %-11s spread across runs: std x %.3f z %.3f m | within-run jitter std x %.3f z %.3f\n', P(s).cond, std(P(s).CPm), P(s).jit); end
fprintf('   all %d runs: mean (%.3f, %.3f), std across runs x %.3f z %.3f m\n', size(allc, 1), mean(allc), std(allc));

fprintf('\n== SENSORS vs an ideal vortex (how well the decoded field reads)\n');
for s = 1:numel(P)
    fprintf('   %-11s linear-field fit R^2: u %.2f, v %.2f (per-run median) | direction error vs best linear fit: median %.0f deg\n', P(s).cond, median(P(s).R2u), median(P(s).R2v), median(P(s).ANGfit));
    fprintf('              direction error vs ideal vortex, median (p90): celeste %.0f (%.0f), tidal %.0f (%.0f), pacific-blue %.0f (%.0f) deg\n', ...
        rad2deg(median(P(s).E(1).ang)), rad2deg(prctile(P(s).E(1).ang, 90)), rad2deg(median(P(s).E(2).ang)), rad2deg(prctile(P(s).E(2).ang, 90)), rad2deg(median(P(s).E(3).ang)), rad2deg(prctile(P(s).E(3).ang, 90)));
    b = [ones(numel(P(s).DIST), 1) P(s).DIST] \ P(s).SAT; res = P(s).SAT - [ones(numel(P(s).DIST), 1) P(s).DIST] * b;
    fprintf('              saturation vs distance from centre: sat = %.2f + %.2f*r  (R^2 %.2f; robots at r = %.2f-%.2f m)\n', b, 1 - var(res) / var(P(s).SAT), prctile(P(s).DIST, 5), prctile(P(s).DIST, 95));
end

fprintf('\n== PLANT re-fit: centroid v(k) = a v(k-1) + b u(k)   (11/11 fit: a 0.398, b 0.471, gain 0.78)\n');
allU = vertcat(P.U); allV = vertcat(P.V); allVp = vertcat(P.Vp);
for s = 0:numel(P)
    if s == 0, Uu = allU; Vv = allV; Vvp = allVp; nm = 'ALL'; else, Uu = P(s).U; Vv = P(s).V; Vvp = P(s).Vp; nm = P(s).cond; end
    ok = all(isfinite([Uu Vv Vvp]), 2);
    Xf = [reshape(Vvp(ok, :), [], 1), reshape(Uu(ok, :), [], 1)]; y = reshape(Vv(ok, :), [], 1);
    th = Xf \ y; R2 = 1 - sum((y - Xf * th).^2) / sum((y - mean(y)).^2);
    fprintf('   %-11s a %.3f  b %.3f  gain %.2f  R^2 %.3f  (%d samples)\n', nm, th, th(2) / (1 - th(1)), R2, numel(y));
    if s == 0, aa = th(1); bb = th(2); end
end

fprintf('\n== RADIUS: measured vs closed-loop prediction (re-fitted plant, measured CP jitter, k_t %.2f k_r %g)\n', k_t, k_r);
for s = 1:numel(P)
    rp0 = simcl(aa, bb, [0 0], P(s).rd, k_t, k_r, clipv);
    rp1 = simcl(aa, bb, P(s).jit, P(s).rd, k_t, k_r, clipv);
    fprintf('   r_d %.2f: measured %.3f ± %.3f | predicted %.3f (no jitter), %.3f (with jitter) | old 11/11-plant prediction %.3f\n', ...
        P(s).rd, mean(P(s).R), std(P(s).R), rp0, rp1, simcl(0.398, 0.471, P(s).jit, P(s).rd, k_t, k_r, clipv));
end
end

function rbar = simcl(a, b, jit, rd, kt, kr, clipv)
dt = 0.1; N = 600; rng(1); c = [0.3 0]; v = [0 0]; R = zeros(N, 1);
for k = 1:N
    cp = jit .* randn(1, 2);
    d = cp - c; r = norm(d); un = d / r;
    rs = max(min(kr * (r - rd), clipv), -clipv);
    u = kt * [un(2), -un(1)] + rs * un;
    v = a * v + b * u; c = c + dt * v;
    R(k) = norm(c);       % radius about the TRUE centre (0,0)
end
rbar = mean(R(200:end));
end
