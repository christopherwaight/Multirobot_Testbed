%% Analyze all datasets to find optimal normalization constants

% Load all datasets
datasets = {
    'celeste_cal.csv', 'tidal_cal.csv', 'pacific_blue_cal.csv', 'redwood_cal.csv',
    'celeste_ver.csv', 'tidal_ver.csv', 'pacific_blue_ver.csv', 'redwood_ver.csv'
};

% Initialize arrays to store all values
all_R = [];
all_G = [];
all_B = [];
all_K = [];

% Collect all data
fprintf('Loading datasets...\n');
for i = 1:length(datasets)
    data = readmatrix(datasets{i});
    all_R = [all_R; data(:,3)];
    all_G = [all_G; data(:,4)];
    all_B = [all_B; data(:,5)];
    all_K = [all_K; data(:,6)];
    fprintf('Loaded %s\n', datasets{i});
end

%% Method 1: Min-Max Normalization (maps to [0,1])
fprintf('\n=== Method 1: Min-Max Normalization ===\n');
fprintf('Channel | Min    | Max    | Range  | Suggested Formula\n');
fprintf('--------|--------|--------|--------|------------------\n');

% R channel
R_min = min(all_R); R_max = max(all_R);
fprintf('R       | %6.1f | %6.1f | %6.1f | (R - %.1f) / %.1f\n', R_min, R_max, R_max-R_min, R_min, R_max-R_min);

% G channel
G_min = min(all_G); G_max = max(all_G);
fprintf('G       | %6.1f | %6.1f | %6.1f | (G - %.1f) / %.1f\n', G_min, G_max, G_max-G_min, G_min, G_max-G_min);

% B channel
B_min = min(all_B); B_max = max(all_B);
fprintf('B       | %6.1f | %6.1f | %6.1f | (B - %.1f) / %.1f\n', B_min, B_max, B_max-B_min, B_min, B_max-B_min);

% K channel
K_min = min(all_K); K_max = max(all_K);
fprintf('K       | %6.1f | %6.1f | %6.1f | (K - %.1f) / %.1f\n', K_min, K_max, K_max-K_min, K_min, K_max-K_min);

%% Method 2: Percentile-based (more robust to outliers)
fprintf('\n=== Method 2: Percentile-based Normalization ===\n');
fprintf('Using 1st and 99th percentiles to avoid outliers\n');
fprintf('Channel | P1     | P99    | Range  | Suggested Formula\n');
fprintf('--------|--------|--------|--------|------------------\n');

% R channel
R_p1 = prctile(all_R, 0.5); R_p99 = prctile(all_R, 99.5);
fprintf('R       | %6.1f | %6.1f | %6.1f | (R - %.1f) / %.1f\n', R_p1, R_p99, R_p99-R_p1, R_p1, R_p99-R_p1);

% G channel
G_p1 = prctile(all_G, 0.5); G_p99 = prctile(all_G, 99.5);
fprintf('G       | %6.1f | %6.1f | %6.1f | (G - %.1f) / %.1f\n', G_p1, G_p99, G_p99-G_p1, G_p1, G_p99-G_p1);

% B channel
B_p1 = prctile(all_B, 0.5); B_p99 = prctile(all_B, 99.5);
fprintf('B       | %6.1f | %6.1f | %6.1f | (B - %.1f) / %.1f\n', B_p1, B_p99, B_p99-B_p1, B_p1, B_p99-B_p1);

% K channel
K_p1 = prctile(all_K, 0.5); K_p99 = prctile(all_K, 99.5);
fprintf('K       | %6.1f | %6.1f | %6.1f | (K - %.1f) / %.1f\n', K_p1, K_p99, K_p99-K_p1, K_p1, K_p99-K_p1);

%% Method 3: Your current approach analysis
fprintf('\n=== Method 3: Analysis of Current Approach ===\n');
fprintf('Current formulas:\n');
fprintf('R: (R - 120) / 1300\n');
fprintf('G: (G - 120) / 1300\n');
fprintf('B: (B - 130) / 1300\n');
fprintf('K: (K - 800) / 3500\n');

fprintf('\nWhat range this covers:\n');
fprintf('Channel | Your Min | Your Max | Actual Min | Actual Max | Coverage\n');
fprintf('--------|----------|----------|------------|------------|----------\n');
fprintf('R       | 120      | 1300     | %6.1f     | %6.1f     | %.1f%%\n', R_min, R_max, 100*sum(all_R >= 100 & all_R <= 1600)/length(all_R));
fprintf('G       | 120      | 1300     | %6.1f     | %6.1f     | %.1f%%\n', G_min, G_max, 100*sum(all_G >= 100 & all_G <= 1600)/length(all_G));
fprintf('B       | 120      | 1300     | %6.1f     | %6.1f     | %.1f%%\n', B_min, B_max, 100*sum(all_B >= 100 & all_B <= 1600)/length(all_B));
fprintf('K       | 800      | 3500     | %6.1f     | %6.1f     | %.1f%%\n', K_min, K_max, 100*sum(all_K >= 500 & all_K <= 5000)/length(all_K));

%% Visualization
figure('Position', [100, 100, 1200, 800]);

% Histograms for each channel
subplot(2,2,1);
histogram(all_R, 50);
hold on;
xline(120, 'r--', 'LineWidth', 2);
xline(1520, 'r--', 'LineWidth', 2);
xline(R_min, 'g-', 'LineWidth', 2);
xline(R_max, 'g-', 'LineWidth', 2);
xlabel('R values');
ylabel('Count');
title('R Channel Distribution');
legend('Data', 'Your bounds', 'Actual min/max');

subplot(2,2,2);
histogram(all_G, 50);
hold on;
xline(120, 'r--', 'LineWidth', 2);
xline(1520, 'r--', 'LineWidth', 2);
xline(G_min, 'g-', 'LineWidth', 2);
xline(G_max, 'g-', 'LineWidth', 2);
xlabel('G values');
ylabel('Count');
title('G Channel Distribution');

subplot(2,2,3);
histogram(all_B, 50);
hold on;
xline(120, 'r--', 'LineWidth', 2);
xline(1520, 'r--', 'LineWidth', 2);
xline(B_min, 'g-', 'LineWidth', 2);
xline(B_max, 'g-', 'LineWidth', 2);
xlabel('B values');
ylabel('Count');
title('B Channel Distribution');

subplot(2,2,4);
histogram(all_K, 50);
hold on;
xline(800, 'r--', 'LineWidth', 2);
xline(4200, 'r--', 'LineWidth', 2);
xline(K_min, 'g-', 'LineWidth', 2);
xline(K_max, 'g-', 'LineWidth', 2);
xlabel('K values');
ylabel('Count');
title('K Channel Distribution');

%% Recommended approach
fprintf('\n=== RECOMMENDATION ===\n');
fprintf('Based on the analysis, here are the recommended normalization formulas:\n\n');

% Use nice round numbers that cover most of the data
R_rec_min = floor(R_p1/10)*10;  % Round down to nearest 10
R_rec_max = ceil(R_p99/10)*10;   % Round up to nearest 10
G_rec_min = floor(G_p1/10)*10;
G_rec_max = ceil(G_p99/10)*10;
B_rec_min = floor(B_p1/10)*10;
B_rec_max = ceil(B_p99/10)*10;
K_rec_min = floor(K_p1/50)*50;   % Round to nearest 50 for K
K_rec_max = ceil(K_p99/50)*50;

fprintf('inputs(:,1) = (inputs(:,1) - %d) / %d;\n', R_rec_min, R_rec_max - R_rec_min);
fprintf('inputs(:,2) = (inputs(:,2) - %d) / %d;\n', G_rec_min, G_rec_max - G_rec_min);
fprintf('inputs(:,3) = (inputs(:,3) - %d) / %d;\n', B_rec_min, B_rec_max - B_rec_min);
fprintf('inputs(:,4) = (inputs(:,4) - %d) / %d;\n', K_rec_min, K_rec_max - K_rec_min);

fprintf('\nDon''t forget to clip to [0,1] after normalization:\n');
fprintf('inputs = max(min(inputs, 1), 0);\n');

%% Check for correlation between channels
fprintf('\n=== Channel Correlations ===\n');
corr_matrix = corrcoef([all_R, all_G, all_B, all_K]);
fprintf('     R      G      B      K\n');
fprintf('R  %.3f  %.3f  %.3f  %.3f\n', corr_matrix(1,:));
fprintf('G  %.3f  %.3f  %.3f  %.3f\n', corr_matrix(2,:));
fprintf('B  %.3f  %.3f  %.3f  %.3f\n', corr_matrix(3,:));
fprintf('K  %.3f  %.3f  %.3f  %.3f\n', corr_matrix(4,:));