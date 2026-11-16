function r04_detail(dirName)
fs = dir(fullfile(dirName, '*.mat'));
C1 = [0.840 0.856];   % true-centre estimate (all data)
C2 = [0.810 0.874];   % centre fitted from the r_d 0.4 runs only
for i = 1:numel(fs)
    S = load(fullfile(fs(i).folder, fs(i).name)); t = S.tout; C = S.cluster_state(:, 1:2);
    cp = S.field_cp; ok = all(isfinite(cp), 2); C3 = median(cp(ok, :));
    r1 = vecnorm(C - C1, 2, 2); r2 = vecnorm(C - C2, 2, 2); r3 = vecnorm(C - C3, 2, 2);
    ang = unwrap(atan2(C(:, 2) - C1(2), C(:, 1) - C1(1)));
    fprintf('%s\n', fs(i).name);
    fprintf('   start r %.3f | t 10-20 s %.3f | t 20-30 s %.3f | last 5 s %.3f  (vs true-centre estimate)\n', r1(1), mean(r1(t > 10 & t <= 20)), mean(r1(t > 20)), mean(r1(t > 25)));
    fprintf('   t > 10 s: vs centre from 0.4 runs %.3f | vs this run''s median live estimate %.3f | logged orbit_radius %.3f\n', mean(r2(t > 10)), mean(r3(t > 10)), mean(S.orbit_radius(t > 10, 1)));
    fprintf('   laps in 30 s: %.2f | centroid speed %.3f m/s\n', abs(ang(end) - ang(1)) / 2 / pi, median(vecnorm(diff(C), 2, 2)) / 0.1);
end
end
