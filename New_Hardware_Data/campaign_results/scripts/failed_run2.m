function failed_run2(f, ctr)
S = load(f); t = S.tout; cs = S.cluster_state; C = cs(:, 1:2); rr = vecnorm(C - ctr, 2, 2);
lr = S.orbit_radius_jacobian(:, 1);
fprintf(' t   | centroid r | ctrl r | d1    d2    r1    r2    phi  | |cmd| per robot c t p R      | actual speed c t p R\n');
for tt = 1:10:numel(t)
    cm = vecnorm(reshape(S.cmd_robots_selected(tt, :), 3, 4)', 2, 2)';
    sp = zeros(1, 4); for r = 1:4, p = S.(sprintf('pose_r%d', r)); k = min(tt + 1, numel(t)); sp(r) = norm(p(k, 1:2) - p(max(k - 1, 1), 1:2)) / 0.1; end
    fprintf('%4.1f | %.3f      | %.3f  | %.2f  %.2f  %.2f  %.2f  %.2f | %.2f %.2f %.2f %.2f   | %.2f %.2f %.2f %.2f\n', t(tt), rr(tt), lr(tt), cs(tt, 4:8), cm, sp);
end
end
