function fit_coulomb(root)
% Fit the Python Omnibot "coulomb" model to every robot in the 2026-11 campaign.
%   free = a*v_prev + (1-a)*g*u ;  v = free*max(0, 1 - (1-a)*vk/|free|)   (+ stuck if |u|<=vs)
% v(k) = (p(k+1)-p(k))/dt(k) using wall-clock timing; u(k) = cmd_robots_selected at step k.
names = {'celeste', 'tidal', 'pacific-blue', 'redwood'};
fs = dir(fullfile(root, '**', '*.mat'));
D = struct('rob', {}, 'u', {}, 'v', {}, 'dt', {}, 'run', {});
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name));
    if contains(fs(i).name, 'off_the_mat'), continue; end
    nr = size(S.cmd_robots_selected, 2) / 3;
    tw = S.timing(:, 2); dtw = diff(tw); dtw(dtw <= 0.02 | dtw > 0.5) = NaN;
    for r = 1:nr
        p = S.(sprintf('pose_r%d', r)); u = S.cmd_robots_selected(:, 3*r-2:3*r-1);
        v = diff(p(:, 1:2)) ./ dtw;  u = u(1:end-1, :);
        % battery screen: median speed ratio for this robot in this run
        k = vecnorm(u, 2, 2) > 0.05 & all(isfinite(v), 2);
        ratio = median(vecnorm(v(k, :), 2, 2) ./ vecnorm(u(k, :), 2, 2));
        if ratio < 0.75, continue; end
        D(end+1) = struct('rob', r, 'u', u, 'v', v, 'dt', dtw, 'run', i); %#ok<AGROW>
    end
end
fprintf('%d robot-runs kept after battery screen (speed ratio >= 0.75)\n', numel(D));
allDt = vertcat(D.dt); fprintf('wall-clock step: median %.3f s, p5 %.3f, p95 %.3f\n', median(allDt, 'omitnan'), prctile(allDt, 5), prctile(allDt, 95));

% ---- 1. steady-state speed vs command (command steady for the last 5 steps)
U = []; V = []; R = [];
for j = 1:numel(D)
    u = D(j).u; v = D(j).v; n = size(u, 1);
    for k = 6:n
        w = u(k-5:k, :); if any(~isfinite(v(k, :))), continue; end
        if max(vecnorm(w - u(k, :), 2, 2)) < 0.02
            U(end+1) = norm(u(k, :)); V(end+1) = dot(v(k, :), u(k, :)) / max(norm(u(k, :)), 1e-9); R(end+1) = D(j).rob; %#ok<AGROW>
        end
    end
end
fprintf('\n-- steady state (command constant within 2 cm/s for 0.5 s): %d samples\n', numel(U));
ed = [0 0.02 0.04 0.06 0.08 0.10 0.13 0.16 0.20 0.25 0.30 0.40 0.6];
fprintf('|u| bin      n     speed along u: median  [p25 p75]   ratio\n');
for b = 1:numel(ed) - 1
    m = U >= ed(b) & U < ed(b+1); if sum(m) < 5, continue; end
    fprintf('%.2f-%.2f  %5d   %.3f  [%.3f %.3f]   %.2f\n', ed(b), ed(b+1), sum(m), median(V(m)), prctile(V(m), 25), prctile(V(m), 75), median(V(m)) / median(U(m)));
end
m = U > 0.05; c = [U(m)' ones(sum(m), 1)] \ V(m)';
fprintf('line fit (|u| > 0.05): speed = %.3f*|u| %+.3f  -> gain %.2f, offset (kinetic friction) %.3f m/s\n', c(1), c(2), c(1), -c(2));

% ---- 2. dynamic fits per robot and pooled
models = {'python default (a 0.70, g 1, vk=vs 0.025)', 'linear lag (a, g)', 'coulomb (a, g, vk)'};
fprintf('\n-- one-step-ahead and free-run fits (vector velocity, m/s RMSE)\n');
for r = [0 1 2 3 4]
    if r == 0, J = 1:numel(D); nm = 'ALL'; else, J = find([D.rob] == r); nm = names{r}; end
    if isempty(J), continue; end
    f2 = @(x) cost(D(J), x(1), x(2), 0, 0);
    x2 = fminsearch(f2, [0.5 0.85], optimset('Display', 'off'));
    f3 = @(x) cost(D(J), x(1), x(2), max(x(3), 0), max(x(3), 0));
    x3 = fminsearch(f3, [x2 0.02], optimset('Display', 'off')); x3(3) = max(x3(3), 0);
    c0 = cost(D(J), 0.70, 1, 0.025, 0.025); c2 = f2(x2); c3 = f3(x3);
    fr0 = freerun(D(J), 0.70, 1, 0.025, 0.025); fr2 = freerun(D(J), x2(1), x2(2), 0, 0); fr3 = freerun(D(J), x3(1), x3(2), x3(3), x3(3));
    fprintf('%-13s n=%3d | python: 1-step %.4f free %.4f | linear a %.2f g %.2f: 1-step %.4f free %.4f | coulomb a %.2f g %.2f vk %.3f: 1-step %.4f free %.4f | tau = %.2f s\n', ...
        nm, numel(J), c0, fr0, x2(1), x2(2), c2, fr2, x3(1), x3(2), x3(3), c3, fr3, -0.1 / log(x3(1)));
end

% ---- 3. delay check (pooled): does u(k-1) explain v(k) better than u(k)?
for dly = 0:2
    e = []; for j = 1:numel(D), u = D(j).u; v = D(j).v; n = size(u, 1);
        X = [v(1+dly:n-1, :) u(1:n-1-dly, :)]; Y = v(2+dly:n, :); % v(k) ~ v(k-1), u(k-1-dly) (index aligned)
        ok = all(isfinite([X Y]), 2); e = [e; X(ok, :) Y(ok, :)]; end %#ok<AGROW>
    Aa = [e(:, [1 3]); e(:, [2 4])]; y = [e(:, 5); e(:, 6)]; b = Aa \ y; res = sqrt(mean((y - Aa*b).^2));
    fprintf('delay test: v(k) = %.2f v(k-1) + %.2f u(k-%d)   rmse %.4f\n', b(1), b(2), dly + 1, res);
end
fprintf('max measured speed %.3f m/s, max command %.3f m/s\n', max(vecnorm(vertcat(D.v), 2, 2)), max(vecnorm(vertcat(D.u), 2, 2)));
end

function c = cost(D, a, g, vs, vk)
% one-step-ahead: start from the MEASURED previous velocity
if a <= 0 || a >= 1 || g <= 0, c = 1e3; return; end
se = 0; n = 0;
for j = 1:numel(D)
    u = D(j).u; v = D(j).v;
    for k = 2:size(u, 1)
        if any(~isfinite(v(k-1:k, :)), 'all'), continue; end
        vp = step(v(k-1, :), u(k, :), a, g, vs, vk, norm(v(k-1, :)) < 1e-3);
        se = se + sum((vp - v(k, :)).^2); n = n + 1;
    end
end
c = sqrt(se / max(n, 1));
end
function c = freerun(D, a, g, vs, vk)
% simulate each robot from rest using only its commands; compare velocities
se = 0; n = 0;
for j = 1:numel(D)
    u = D(j).u; v = D(j).v; vv = [0 0]; stuck = true;
    for k = 1:size(u, 1)
        [vv, stuck] = step(vv, u(k, :), a, g, vs, vk, stuck);
        if all(isfinite(v(k, :))), se = se + sum((vv - v(k, :)).^2); n = n + 1; end
    end
end
c = sqrt(se / max(n, 1));
end
function [v, stuck] = step(v, u, a, g, vs, vk, stuck)
u = g * u;
if stuck && norm(u) <= vs, v = [0 0]; return; end
stuck = false;
free = a * v + (1 - a) * u; fm = norm(free); drop = (1 - a) * vk;
if fm <= drop, v = [0 0]; stuck = norm(u) <= vs; else, v = free * (1 - drop / fm); end
end
