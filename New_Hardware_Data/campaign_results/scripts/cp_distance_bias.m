function cp_distance_bias(root, ctrue)
% Why is the cluster-to-estimate distance usually LONGER than cluster-to-true-centre?
fs = dir(fullfile(root, '**', '*.mat'));
conds = {}; for i = 1:numel(fs), [~, conds{i}] = fileparts(fs(i).folder); end %#ok<AGROW>
uc = unique(conds, 'stable');
fprintf('%-11s | %-9s | %-21s | %-10s | %-26s | %-28s | %s\n', 'condition', 'true r', 'r to live est mean/med', 'longer %', 'error radial/tang mean (cm)', 'scatter sigma (robust, cm)', 'sqrt(r^2+2s^2)');
for k = 1:numel(uc)
    rt = []; re = []; er = []; et = [];
    for i = find(strcmp(conds, uc{k}))
        S = load(fullfile(fs(i).folder, fs(i).name));
        h = S.tout > 10; C = S.cluster_state(h, 1:2); cp = S.field_cp(h, :); ok = all(isfinite(cp), 2);
        C = C(ok, :); cp = cp(ok, :);
        u = (ctrue - C) ./ vecnorm(ctrue - C, 2, 2);            % unit vector cluster -> true centre
        e = cp - ctrue;                                         % estimate error
        rt = [rt; vecnorm(ctrue - C, 2, 2)]; re = [re; vecnorm(cp - C, 2, 2)]; %#ok<AGROW>
        er = [er; sum(e .* u, 2)]; et = [et; e(:, 1) .* u(:, 2) - e(:, 2) .* u(:, 1)]; %#ok<AGROW>
    end
    s = 1.4826 * median(abs([er - median(er); et - median(et)]));   % robust per-axis scatter
    fprintf('%-11s | %.3f m   | %.3f / %.3f m         | %4.0f%%     | %+5.1f / %+5.1f               | %.1f                          | %.3f\n', ...
        uc{k}, mean(rt), mean(re), median(re), 100 * mean(re > rt), 100 * mean(er), 100 * mean(et), 100 * s, sqrt(mean(rt)^2 + 2 * s^2));
end
fprintf('(radial error > 0 = estimate lies BEYOND the true centre, as seen from the cluster)\n');
% pure geometry demo: unbiased isotropic scatter sigma around the true centre, cluster at distance r
fprintf('\npure geometry, zero-mean scatter (sigma 2.5 cm): mean distance / true distance\n');
rng(1); n = randn(1e5, 2) * 0.025;
for r = [0.02 0.05 0.1 0.2 0.3]
    d = vecnorm([r 0] + n, 2, 2); fprintf('   r %.2f m: mean %.3f (+%.1f cm), longer in %.0f%% of steps\n', r, mean(d), 100 * (mean(d) - r), 100 * mean(d > r));
end
end
