function orbit_predict(files, params, labels)
% 1) Reconstruct the Simulink Orbiter command each step from logged data; fit how the real cluster
%    centroid follows it: v(k) = a v(k-1) + b u(k-d) (lag pole a, gain b, delay d samples).
% 2) Closed-loop simulation with the fitted plant (+ measured critical-point jitter) for each run's
%    settings -> compare predicted vs measured radius (validation).
% 3) Predict the Python law (k_t 0.2, k_r 0.5, radial speed clamp 0.3) vs the current Simulink law.
% params(i,:) = [v_t, K_p, r_d, cw]
dt = 0.1;
U = []; V = []; Vp = []; Up1 = []; Up2 = []; cpn = []; meas = zeros(numel(files), 1);
for f = 1:numel(files)
    S = load(files{f});
    C = S.cluster_pose(:, 1:2); cp = S.detected_critical_point; n = size(C, 1);
    vt = params(f, 1); kp = params(f, 2); rd = params(f, 3); cw = params(f, 4);
    u = zeros(n, 2);
    for k = 1:n, u(k, :) = orbiter_sim(C(k, :), cp(k, :), vt, kp, rd, cw); end
    v = [diff(C) / dt; nan(1, 2)];
    % rows k = 3..n-1: v(k), v(k-1), u(k), u(k-1), u(k-2)
    k = (3:n-1)';
    V = [V; v(k, :)]; Vp = [Vp; v(k-1, :)]; U = [U; u(k, :)]; Up1 = [Up1; u(k-1, :)]; Up2 = [Up2; u(k-2, :)]; %#ok<AGROW>
    cpm = median(cp); cpn = [cpn; cp - cpm]; %#ok<AGROW>
    r = vecnorm(C - cpm, 2, 2); h = S.tout > 10; meas(f) = mean(r(h));
end
fprintf('== plant fit over %d samples (both axes stacked): v(k) = a v(k-1) + b u(k-d)\n', size(V, 1));
best = [];
cands = {U, Up1, Up2};
for d = 0:2
    X = [Vp(:), cands{d + 1}(:)]; y = V(:);
    th = X \ y; res = y - X * th; R2 = 1 - sum(res.^2) / sum((y - mean(y)).^2);
    fprintf('   d=%d: a=%.3f b=%.3f (DC gain b/(1-a)=%.2f)  R^2=%.3f\n', d, th(1), th(2), th(2) / (1 - th(1)), R2);
    if isempty(best) || R2 > best(4), best = [d, th(1), th(2), R2]; end
end
d = best(1); a = best(2); b = best(3);
fprintf('   -> using d=%d a=%.3f b=%.3f\n', d, a, b);
cpstd = std(cpn); fprintf('   critical-point jitter std: x %.3f z %.3f m\n', cpstd);

fprintf('\n== validation: closed loop with the fitted plant (mean radius, 60 s, last 40 s)\n');
for f = 1:numel(files)
    p = params(f, :);
    r0 = sim_cl(@(c, q) orbiter_sim(c, q, p(1), p(2), p(3), p(4)), a, b, d, [0 0], 60, 0);
    r1 = sim_cl(@(c, q) orbiter_sim(c, q, p(1), p(2), p(3), p(4)), a, b, d, cpstd, 60, 1);
    fprintf('   %-42s v_t %.2f K_p %.1f r_d %.2f : measured %.3f | predicted %.3f (no jitter) %.3f (with jitter)\n', labels{f}, p(1), p(2), p(3), meas(f), r0, r1);
end

fprintf('\n== prediction: Python law vs current Simulink law (fitted hardware plant, with jitter)\n');
fprintf('   %-6s  %-26s  %-26s  %-26s\n', 'r_d', 'Simulink now (0.15, K_p 1)', 'Python (0.2, 0.5, clamp)', 'Python gains, no clamp');
for rd = [0.01 0.05 0.1 0.2 0.4]
    rs = sim_cl(@(c, q) orbiter_sim(c, q, 0.15, 1, rd, -1), a, b, d, cpstd, 60, 1);
    rp = sim_cl(@(c, q) orbiter_py(c, q, 0.2, 0.5, rd, 0.3), a, b, d, cpstd, 60, 1);
    rn = sim_cl(@(c, q) orbiter_py(c, q, 0.2, 0.5, rd, inf), a, b, d, cpstd, 60, 1);
    fprintf('   %-6.2f  %-26.3f  %-26.3f  %-26.3f\n', rd, rs, rp, rn);
end
% speeds: is anything near stiction?
fprintf('\n== per-step commanded centroid speed in steady state (Python law, r_d 0.1): ');
[~, sp] = sim_cl(@(c, q) orbiter_py(c, q, 0.2, 0.5, 0.1, 0.3), a, b, d, cpstd, 60, 1);
fprintf('min %.3f median %.3f max %.3f m/s\n', min(sp), median(sp), max(sp));
% Python's own simulator model (momentum 0.7, cap 0.3, stiction 0.025), for reference
fprintf('\n== reference: Python simulator robot model (momentum 0.7) with the Python law\n');
for rd = [0.01 0.1 0.4]
    fprintf('   r_d %.2f -> %.3f\n', rd, sim_cl(@(c, q) orbiter_py(c, q, 0.2, 0.5, rd, 0.3), 0.7, 0.3, 0, [0 0], 60, 0, true));
end
end

function u = orbiter_sim(c, cp, vt, kp, rd, cw)
d = cp - c; r = norm(d); un = d / r;
e = max(min(rd - r, 5), -5);
u = vt * [cw * un(2), -cw * un(1)] - kp * e * un;
end

function u = orbiter_py(c, cp, kt, kr, rd, clampv)
d = cp - c; r = norm(d); un = d / r;
rs = max(min(kr * (r - rd), clampv), -clampv);
u = kt * [un(2), -un(1)] + rs * un;
end

function [rbar, sp] = sim_cl(law, a, b, d, cpstd, T, jitter, pyModel)
if nargin < 8, pyModel = false; end
dt = 0.1; N = round(T / dt); rng(1);
c = [0.3, 0]; v = [0 0]; q = zeros(d + 1, 2); R = zeros(N, 1); sp = zeros(N, 1);
for k = 1:N
    cp = [0 0] + jitter * cpstd .* randn(1, 2);
    u = law(c, cp);
    q = [q(2:end, :); u]; ud = q(1, :);
    if pyModel                                  % omnibot.py: momentum, cap, stiction
        v = a * v + (1 - a) * ud; m = norm(v);
        if m > 0.3, v = v / m * 0.3; end
        if m < 0.025, v = [0 0]; end
    else
        v = a * v + b * ud;
    end
    c = c + dt * v; R(k) = norm(c); sp(k) = norm(u);
end
h = round(N / 3):N; rbar = mean(R(h)); sp = sp(h);
end
