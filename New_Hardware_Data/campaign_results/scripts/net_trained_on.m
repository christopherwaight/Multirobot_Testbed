function net_trained_on(dl)
% Was redwood in the training set of july_tuning/everything_nets_final.mat?
% Same preprocessing/decoding as july_tuning/comprehensive_search.m.
L = load(fullfile(dl, 'everything_nets_final.mat'));
fprintf('variables in the net file: %s\n', strjoin(fieldnames(L)', ', '));
hn = L.everything_hue_net; sn = L.everything_sat_net;
robots = {'celeste', 'tidal', 'pacific_blue', 'redwood'};
fprintf('%-13s | %-34s | %-34s\n', 'robot', 'CAL (first 480 rows; training rows)', 'VER (held-out file)');
fprintf('%-13s | %-10s %-10s %-10s | %-10s %-10s %-10s\n', '', 'hue med', 'hue p90', 'sat RMSE', 'hue med', 'hue p90', 'sat RMSE');
for k = 1:4
    out = cell(1, 2);
    for w = 1:2
        if w == 1, D = readmatrix(fullfile(dl, [robots{k} '_cal.csv'])); D = D(1:min(480, end), :);
        else, D = readmatrix(fullfile(dl, [robots{k} '_ver.csv'])); end
        D = D(all(isfinite(D(:, 1:6)), 2), :);
        x = D(:, 3:6); x(:, 1:3) = (x(:, 1:3) - 120) / 1300; x(:, 4) = (x(:, 4) - 800) / 3500; x = max(min(x, 1), 0);
        xn = x' * 2 - 1;
        ho = hn(xn); s = (ho(1, :) + 1) / 2; c = (ho(2, :) + 1) / 2;
        hp = atan2(s, c); hp(hp < 0) = hp(hp < 0) + 2 * pi; hp = hp / (2 * pi);
        he = min(abs(D(:, 1)' - hp), 1 - abs(D(:, 1)' - hp)) * 360;
        sp = max(min((sn(xn) + 1) / 2, 1), 0); se = sqrt(mean((D(:, 2)' - sp).^2));
        out{w} = [median(he), prctile(he, 90), se, size(D, 1)];
    end
    fprintf('%-13s | %6.1f°    %6.1f°    %.3f      | %6.1f°    %6.1f°    %.3f     (n %d / %d)\n', robots{k}, out{1}(1:3), out{2}(1:3), out{1}(4), out{2}(4));
end
end
