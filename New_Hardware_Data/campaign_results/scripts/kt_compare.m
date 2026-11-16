function kt_compare
% Steady orbit with the fitted hardware plant v = 0.398 v_prev + 0.471 u, CP jitter std [0.022 0.017].
% Breaks the command into tangential / radial parts and reports actual centroid speed.
a = 0.398; b = 0.471; cpstd = [0.022 0.017];
laws = {'Python k_t 0.20 k_r 0.5', @(c, q, rd) py(c, q, 0.20, 0.5, rd); ...
        'k_t 0.15 k_r 0.5       ', @(c, q, rd) py(c, q, 0.15, 0.5, rd); ...
        'old Simulink 0.15 K_p 1', @(c, q, rd) old(c, q, 0.15, 1, rd)};
fprintf('%-24s %5s | %6s | %8s %8s %8s | %8s\n', 'law', 'r_d', 'radius', 'tang cmd', 'rad cmd', '|cmd|', 'actual');
for rd = [0.01 0.1 0.4]
    for L = 1:size(laws, 1)
        for jit = [0 1]
            [r, ut, ur, um, va] = sim_cl(laws{L, 2}, rd, a, b, cpstd * jit);
            tag = ''; if jit, tag = ' (jitter)'; end
            fprintf('%-24s %5.2f | %6.3f | %8.3f %8.3f %8.3f | %8.3f%s\n', laws{L, 1}, rd, r, ut, ur, um, va, tag);
        end
    end
    fprintf('\n');
end
end

function u = py(c, cp, kt, kr, rd)
d = cp - c; r = norm(d); un = d / r;
rs = max(min(kr * (r - rd), 0.3), -0.3);
u = kt * [un(2), -un(1)] + rs * un;
end

function u = old(c, cp, vt, kp, rd)
d = cp - c; r = norm(d); un = d / r; cw = -1;
e = max(min(rd - r, 5), -5);
u = vt * [cw * un(2), -cw * un(1)] - kp * e * un;
end

function [rbar, ut, ur, um, va] = sim_cl(law, rd, a, b, cpstd)
dt = 0.1; N = 600; rng(1);
c = [0.3, 0]; v = [0 0];
R = zeros(N, 1); UT = R; UR = R; UM = R; VA = R;
for k = 1:N
    cp = cpstd .* randn(1, 2);
    u = law(c, cp, rd);
    d = cp - c; un = d / norm(d);
    UR(k) = dot(u, un); UT(k) = abs(u(1) * un(2) - u(2) * un(1)); UM(k) = norm(u);
    v = a * v + b * u; VA(k) = norm(v);
    c = c + dt * v; R(k) = norm(c);
end
h = 200:N;
rbar = mean(R(h)); ut = mean(UT(h)); ur = mean(UR(h)); um = mean(UM(h)); va = mean(VA(h));
end
