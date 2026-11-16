function fit_coulomb_delay(root)
% Same Omnibot "coulomb" model, now with an input delay d (steps):
%   free = a*v(k-1) + (1-a)*g*u(k-d);  v(k) = free*max(0, 1-(1-a)*vk/|free|);  |v| <= vmax
names = {'celeste', 'tidal', 'pacific-blue', 'redwood'};
fs = dir(fullfile(root, '**', '*.mat'));
D = struct('rob', {}, 'u', {}, 'v', {});
for i = 1:numel(fs)
    if contains(fs(i).name, 'off_the_mat'), continue; end
    S = load(fullfile(fs(i).folder, fs(i).name));
    nr = size(S.cmd_robots_selected, 2) / 3;
    dtw = diff(S.timing(:, 2)); dtw(dtw <= 0.02 | dtw > 0.5) = NaN;
    for r = 1:nr
        p = S.(sprintf('pose_r%d', r)); u = S.cmd_robots_selected(1:end-1, 3*r-2:3*r-1);
        v = diff(p(:, 1:2)) ./ dtw;
        k = vecnorm(u, 2, 2) > 0.05 & all(isfinite(v), 2);
        if median(vecnorm(v(k, :), 2, 2) ./ vecnorm(u(k, :), 2, 2)) < 0.75, continue; end
        D(end+1) = struct('rob', r, 'u', u, 'v', v); %#ok<AGROW>
    end
end
fprintf('%d robot-runs\n', numel(D));
o = optimset('Display', 'off', 'MaxFunEvals', 4000);
for d = 0:2
    f = @(x) cost(D, x(1), x(2), max(x(3), 0), d, inf);
    x = fminsearch(f, [0.3 0.85 0.01], o); x(3) = max(x(3), 0);
    fprintf('delay %d step(s): a %.3f (tau %.3f s) g %.3f vk %.4f | 1-step %.4f | free-run %.4f\n', d, x(1), -0.1/log(max(x(1), 1e-6)), x(2), x(3), f(x), freerun(D, x(1), x(2), x(3), d, inf));
end
fprintf('\nper robot, delay 1:\n');
for r = 1:4
    J = find([D.rob] == r);
    f = @(x) cost(D(J), x(1), x(2), max(x(3), 0), 1, inf);
    x = fminsearch(f, [0.1 0.85 0.01], o); x(3) = max(x(3), 0);
    fprintf('%-13s n=%3d  a %.3f  g %.3f  vk %.4f | 1-step %.4f free-run %.4f\n', names{r}, numel(J), x(1), x(2), x(3), f(x), freerun(D(J), x(1), x(2), x(3), 1, inf));
end
fprintf('\nreference models, free-run RMSE on all data:\n');
fprintf('  python OmniCluster now (a 0.70, g 1, vk 0.025, d 0, vmax 0.4): %.4f\n', freerun(D, 0.70, 1, 0.025, 0, 0.4));
fprintf('  old plant fit (a 0.44, g 0.85, vk 0, d 0):                    %.4f\n', freerun(D, 0.44, 0.85, 0, 0, inf));
% speed cap: what happens at large commands?
U = vecnorm(vertcat(D.u), 2, 2); V = vecnorm(vertcat(D.v), 2, 2); U1 = [NaN; U(1:end-1)];
for e = [0.30 0.35 0.40 0.45]
    m = U1 >= e & U1 < e + 0.05; if sum(m) > 3, fprintf('|u(k-1)| %.2f-%.2f: n %d, |v| median %.3f max %.3f (ratio %.2f)\n', e, e + 0.05, sum(m), median(V(m), 'omitnan'), max(V(m)), median(V(m), 'omitnan') / median(U1(m))); end
end
end
function c = cost(D, a, g, vk, d, vmax)
if a < 0 || a >= 1 || g <= 0, c = 1e3; return; end
se = 0; n = 0;
for j = 1:numel(D)
    u = D(j).u; v = D(j).v;
    for k = 2 + d:size(u, 1)
        if any(~isfinite(v(k-1:k, :)), 'all'), continue; end
        vp = step(v(k-1, :), u(k-d, :), a, g, vk, vmax);
        se = se + sum((vp - v(k, :)).^2); n = n + 1;
    end
end
c = sqrt(se / max(n, 1));
end
function c = freerun(D, a, g, vk, d, vmax)
se = 0; n = 0;
for j = 1:numel(D)
    u = D(j).u; v = D(j).v; vv = [0 0];
    for k = 1:size(u, 1)
        if k - d >= 1, uu = u(k-d, :); else, uu = [0 0]; end
        vv = step(vv, uu, a, g, vk, vmax);
        if all(isfinite(v(k, :))), se = se + sum((vv - v(k, :)).^2); n = n + 1; end
    end
end
c = sqrt(se / max(n, 1));
end
function v = step(v, u, a, g, vk, vmax)
free = a * v + (1 - a) * g * u; fm = norm(free); drop = (1 - a) * vk;
if fm <= drop, v = [0 0]; else, v = free * (1 - drop / fm); end
if norm(v) > vmax, v = v / norm(v) * vmax; end
end
