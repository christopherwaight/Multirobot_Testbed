% Combined Trajectory Plotter - Vortex and Saddle Tests
% Creates a 2-row, 1-column figure showing average cluster trajectories
% from both vortex and saddle experiments

% =========================================================================
% CONFIGURATION - Edit these parameters as needed
% =========================================================================

% Axis limits for vortex plot (top subplot)
VORTEX_XLIM = [-0.7, 0.7];
VORTEX_YLIM = [-0.7, 0.7];

% Axis limits for saddle plot (bottom subplot)
SADDLE_XLIM = [-0.7, 0.7];
SADDLE_YLIM = [-0.7, 0.7];

% Figure size [x, y, width, height]
FIGURE_SIZE = [100, 100, 900, 1200];

% Number of runs and grouping
NUM_RUNS = 80;
RUNS_PER_GROUP = 10;
NUM_GROUPS = NUM_RUNS / RUNS_PER_GROUP;

% Colors for different groups (8 groups total)
GROUP_COLORS = [
    0.8 0.2 0.2;  % Red
    0.2 0.6 0.2;  % Green
    0.2 0.3 0.8;  % Blue
    0.8 0.5 0.2;  % Orange
    0.6 0.2 0.8;  % Purple
    0.8 0.8 0.2;  % Yellow
    0.2 0.8 0.8;  % Cyan
    0.8 0.2 0.6;  % Magenta
];

% Font sizes for plot text
TITLE_FONT_SIZE = 16;
AXIS_LABEL_FONT_SIZE = 14;
TICK_FONT_SIZE = 12;
LEGEND_FONT_SIZE = 12;

% =========================================================================
% PROCESS VORTEX DATA
% =========================================================================

fprintf('=== PROCESSING VORTEX DATA ===\n');

% Change to vortex_tests directory
vortex_dir = 'vortex_tests';
original_dir = pwd;
cd(vortex_dir);

% Define run files
vortex_run_files = cell(1, NUM_RUNS);
for i = 1:NUM_RUNS
    vortex_run_files{i} = sprintf('run%d.mat', i);
end

% Storage for vortex average trajectories
vortex_group_avg = cell(1, NUM_GROUPS);
vortex_group_std = cell(1, NUM_GROUPS);

% Process each group
for group_idx = 1:NUM_GROUPS
    fprintf('\nProcessing Vortex Group %d (runs %d-%d)\n', group_idx, ...
        (group_idx-1)*RUNS_PER_GROUP + 1, group_idx*RUNS_PER_GROUP);

    % Create data storage for trajectories in this group
    all_cluster_traj = {};

    % Get the run indices for this group
    start_idx = (group_idx - 1) * RUNS_PER_GROUP + 1;
    end_idx = group_idx * RUNS_PER_GROUP;

    % Check which files exist in this group
    existing_files = {};
    for i = start_idx:end_idx
        if exist(vortex_run_files{i}, 'file')
            existing_files{end+1} = vortex_run_files{i};
        else
            fprintf('Warning: File not found - %s\n', vortex_run_files{i});
        end
    end

    fprintf('Found %d existing run files in vortex group %d\n', length(existing_files), group_idx);

    if isempty(existing_files)
        fprintf('Warning: No run files found for vortex group %d. Skipping...\n', group_idx);
        continue;
    end

    % Process each run file in this group
    for run_idx = 1:length(existing_files)
        current_file = existing_files{run_idx};

        % Load data
        temp_data = load(current_file);

        % Store trajectories if they exist (vortex uses 'cluster_position')
        if isfield(temp_data, 'cluster_position')
            all_cluster_traj{end+1} = temp_data.cluster_position(:,1:2); % Just keep X and Y
        end
    end

    % Process and average trajectories for this group
    if ~isempty(all_cluster_traj)
        [vortex_group_avg{group_idx}, vortex_group_std{group_idx}] = averageTrajectories(all_cluster_traj);
        fprintf('Processed %d cluster trajectories for vortex group %d\n', length(all_cluster_traj), group_idx);
    end
end

% Return to original directory
cd(original_dir);

% =========================================================================
% PROCESS SADDLE DATA
% =========================================================================

fprintf('\n=== PROCESSING SADDLE DATA ===\n');

% Change to saddle_tests directory
saddle_dir = 'saddle_tests';
cd(saddle_dir);

% Define run files
saddle_run_files = cell(1, NUM_RUNS);
for i = 1:NUM_RUNS
    saddle_run_files{i} = sprintf('run%d.mat', i);
end

% Storage for saddle average trajectories
saddle_group_avg = cell(1, NUM_GROUPS);
saddle_group_std = cell(1, NUM_GROUPS);

% Process each group
for group_idx = 1:NUM_GROUPS
    fprintf('\nProcessing Saddle Group %d (runs %d-%d)\n', group_idx, ...
        (group_idx-1)*RUNS_PER_GROUP + 1, group_idx*RUNS_PER_GROUP);

    % Create data storage for trajectories in this group
    all_cluster_traj = {};

    % Get the run indices for this group
    start_idx = (group_idx - 1) * RUNS_PER_GROUP + 1;
    end_idx = group_idx * RUNS_PER_GROUP;

    % Check which files exist in this group
    existing_files = {};
    for i = start_idx:end_idx
        if exist(saddle_run_files{i}, 'file')
            existing_files{end+1} = saddle_run_files{i};
        else
            fprintf('Warning: File not found - %s\n', saddle_run_files{i});
        end
    end

    fprintf('Found %d existing run files in saddle group %d\n', length(existing_files), group_idx);

    if isempty(existing_files)
        fprintf('Warning: No run files found for saddle group %d. Skipping...\n', group_idx);
        continue;
    end

    % Process each run file in this group
    for run_idx = 1:length(existing_files)
        current_file = existing_files{run_idx};

        % Load data
        temp_data = load(current_file);

        % Store trajectories if they exist (saddle uses 'cluster_pose')
        if isfield(temp_data, 'cluster_pose')
            all_cluster_traj{end+1} = temp_data.cluster_pose(:,1:2); % Just keep X and Y
        end
    end

    % Process and average trajectories for this group
    if ~isempty(all_cluster_traj)
        [saddle_group_avg{group_idx}, saddle_group_std{group_idx}] = averageTrajectories(all_cluster_traj);
        fprintf('Processed %d cluster trajectories for saddle group %d\n', length(all_cluster_traj), group_idx);
    end
end

% Return to original directory
cd(original_dir);

% =========================================================================
% CREATE COMBINED FIGURE
% =========================================================================

% fprintf('\n=== CREATING COMBINED FIGURE ===\n');
%
% fig = figure('Position', FIGURE_SIZE);

% % -------------------------------------------------------------------------
% % TOP SUBPLOT: VORTEX TRAJECTORIES
% % -------------------------------------------------------------------------
%
% subplot(2, 1, 1);
% hold on;
% title('(a) Vortex Field Convergence', 'FontSize', TITLE_FONT_SIZE);
% xlabel('x (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% ylabel('y (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% grid on;
% axis equal;
% xlim(VORTEX_XLIM);
% ylim(VORTEX_YLIM);
% set(gca, 'FontSize', TICK_FONT_SIZE);
%
% % Add vortex field
% [X_v, Y_v] = meshgrid(linspace(VORTEX_XLIM(1), VORTEX_XLIM(2), 15), ...
%                        linspace(VORTEX_YLIM(1), VORTEX_YLIM(2), 15));
% center_x = 0; center_y = 0;
% r_v = sqrt((X_v - center_x).^2 + (Y_v - center_y).^2);
% theta_v = atan2(Y_v - center_y, X_v - center_x);
% U_v = -r_v .* sin(theta_v);
% V_v = r_v .* cos(theta_v);
% scale_factor = 0.5;
% U_v_norm = scale_factor * U_v;
% V_v_norm = scale_factor * V_v;
% q_v = quiver(X_v, Y_v, U_v_norm, V_v_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
% q_v.HandleVisibility = 'off';
%
% % Plot vortex cluster trajectories for each group
% for group_idx = 1:NUM_GROUPS
%     if ~isempty(vortex_group_avg{group_idx})
%         plotTrajectoryWithConfidence(vortex_group_avg{group_idx}, vortex_group_std{group_idx}, ...
%             GROUP_COLORS(group_idx,:), 2, 0.15, sprintf('Runs %d-%d', ...
%             (group_idx-1)*RUNS_PER_GROUP + 1, group_idx*RUNS_PER_GROUP));
%     end
% end
% legend('Location', 'best', 'FontSize', LEGEND_FONT_SIZE);
%
% % -------------------------------------------------------------------------
% % BOTTOM SUBPLOT: SADDLE TRAJECTORIES
% % -------------------------------------------------------------------------
%
% subplot(2, 1, 2);
% hold on;
% title('(b) Saddle Field Convergence', 'FontSize', TITLE_FONT_SIZE);
% xlabel('x (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% ylabel('y (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% grid on;
% axis equal;
% xlim(SADDLE_XLIM);
% ylim(SADDLE_YLIM);
% set(gca, 'FontSize', TICK_FONT_SIZE);
%
% % Add saddle field
% [X_s, Y_s] = meshgrid(linspace(-1, 1, 15), linspace(-1, 1, 15));
% r_s = sqrt((X_s - center_x).^2 + (Y_s - center_y).^2);
% theta_s = atan2(Y_s - center_y, X_s - center_x);
% U_s = -r_s .* sin(theta_s);  % Negated for correct saddle orientation
% V_s = -r_s .* cos(theta_s);  % Negated for correct saddle orientation
% U_s_norm = scale_factor * U_s;
% V_s_norm = scale_factor * V_s;
% q_s = quiver(X_s, Y_s, U_s_norm, V_s_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
% q_s.HandleVisibility = 'off';
%
% % Plot saddle cluster trajectories for each group
% for group_idx = 1:NUM_GROUPS
%     if ~isempty(saddle_group_avg{group_idx})
%         plotTrajectoryWithConfidence(saddle_group_avg{group_idx}, saddle_group_std{group_idx}, ...
%             GROUP_COLORS(group_idx,:), 2, 0.15, sprintf('Runs %d-%d', ...
%             (group_idx-1)*RUNS_PER_GROUP + 1, group_idx*RUNS_PER_GROUP));
%     end
% end
% legend('Location', 'best', 'FontSize', LEGEND_FONT_SIZE);
%
% % Save combined figure
% saveas(fig, 'combined_trajectories.png');
% saveas(fig, 'combined_trajectories.fig');
%
% fprintf('\nCombined figure saved successfully!\n');
% fprintf('  - combined_trajectories.png\n');
% fprintf('  - combined_trajectories.fig\n');

% =========================================================================
% CREATE SECOND FIGURE WITH START/END MARKERS
% =========================================================================

% fprintf('\n=== CREATING SECOND FIGURE WITH MARKERS ===\n');
%
% fig2 = figure('Position', FIGURE_SIZE);

% % -------------------------------------------------------------------------
% % TOP SUBPLOT: VORTEX TRAJECTORIES WITH MARKERS
% % -------------------------------------------------------------------------
%
% subplot(2, 1, 1);
% hold on;
% title('(a) Vortex Field Convergence', 'FontSize', TITLE_FONT_SIZE);
% xlabel('x (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% ylabel('y (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% grid on;
% axis equal;
% xlim(VORTEX_XLIM);
% ylim(VORTEX_YLIM);
% set(gca, 'FontSize', TICK_FONT_SIZE);
%
% % Add vortex field
% [X_v, Y_v] = meshgrid(linspace(VORTEX_XLIM(1), VORTEX_XLIM(2), 15), ...
%                        linspace(VORTEX_YLIM(1), VORTEX_YLIM(2), 15));
% center_x = 0; center_y = 0;
% r_v = sqrt((X_v - center_x).^2 + (Y_v - center_y).^2);
% theta_v = atan2(Y_v - center_y, X_v - center_x);
% U_v = -r_v .* sin(theta_v);
% V_v = r_v .* cos(theta_v);
% scale_factor = 0.5;
% U_v_norm = scale_factor * U_v;
% V_v_norm = scale_factor * V_v;
% q_v = quiver(X_v, Y_v, U_v_norm, V_v_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
% q_v.HandleVisibility = 'off';
%
% % Plot vortex cluster trajectories without legend
% for group_idx = 1:NUM_GROUPS
%     if ~isempty(vortex_group_avg{group_idx})
%         avg_traj = vortex_group_avg{group_idx};
%         std_traj = vortex_group_std{group_idx};
%
%         % Plot trajectory line
%         plot(avg_traj(:,1), avg_traj(:,2), 'Color', GROUP_COLORS(group_idx,:), ...
%              'LineWidth', 2, 'HandleVisibility', 'off');
%
%         % Plot confidence band
%         x_band = [avg_traj(:,1); flipud(avg_traj(:,1))];
%         y_band = [avg_traj(:,2) + std_traj(:,2); flipud(avg_traj(:,2) - std_traj(:,2))];
%         fill(x_band, y_band, GROUP_COLORS(group_idx,:), 'FaceAlpha', 0.15, ...
%              'EdgeColor', 'none', 'HandleVisibility', 'off');
%
%         % Plot start point (circle)
%         plot(avg_traj(1,1), avg_traj(1,2), 'o', 'MarkerSize', 8, ...
%              'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
%              'LineWidth', 1, 'HandleVisibility', 'off');
%
%         % Plot end point (square)
%         plot(avg_traj(end,1), avg_traj(end,2), 's', 'MarkerSize', 8, ...
%              'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
%              'LineWidth', 1, 'HandleVisibility', 'off');
%     end
% end
%
% % Plot origin marker (red x)
% plot(0, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'HandleVisibility', 'off');
%
% % Create legend with dummy plots
% h_start = plot(NaN, NaN, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
% h_end = plot(NaN, NaN, 'sk', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
% h_origin = plot(NaN, NaN, 'rx', 'MarkerSize', 12, 'LineWidth', 3);
% legend([h_start, h_end, h_origin], {'Start', 'End', 'Origin'}, ...
%        'Location', 'best', 'FontSize', LEGEND_FONT_SIZE);
%
% % -------------------------------------------------------------------------
% % BOTTOM SUBPLOT: SADDLE TRAJECTORIES WITH MARKERS
% % -------------------------------------------------------------------------
%
% subplot(2, 1, 2);
% hold on;
% title('(b) Saddle Field Convergence', 'FontSize', TITLE_FONT_SIZE);
% xlabel('x (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% ylabel('y (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
% grid on;
% axis equal;
% xlim(SADDLE_XLIM);
% ylim(SADDLE_YLIM);
% set(gca, 'FontSize', TICK_FONT_SIZE);
%
% % Add saddle field
% [X_s, Y_s] = meshgrid(linspace(-1, 1, 15), linspace(-1, 1, 15));
% r_s = sqrt((X_s - center_x).^2 + (Y_s - center_y).^2);
% theta_s = atan2(Y_s - center_y, X_s - center_x);
% U_s = -r_s .* sin(theta_s);  % Negated for correct saddle orientation
% V_s = -r_s .* cos(theta_s);  % Negated for correct saddle orientation
% U_s_norm = scale_factor * U_s;
% V_s_norm = scale_factor * V_s;
% q_s = quiver(X_s, Y_s, U_s_norm, V_s_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
% q_s.HandleVisibility = 'off';
%
% % Plot saddle cluster trajectories without legend
% for group_idx = 1:NUM_GROUPS
%     if ~isempty(saddle_group_avg{group_idx})
%         avg_traj = saddle_group_avg{group_idx};
%         std_traj = saddle_group_std{group_idx};
%
%         % Plot trajectory line
%         plot(avg_traj(:,1), avg_traj(:,2), 'Color', GROUP_COLORS(group_idx,:), ...
%              'LineWidth', 2, 'HandleVisibility', 'off');
%
%         % Plot confidence band
%         x_band = [avg_traj(:,1); flipud(avg_traj(:,1))];
%         y_band = [avg_traj(:,2) + std_traj(:,2); flipud(avg_traj(:,2) - std_traj(:,2))];
%         fill(x_band, y_band, GROUP_COLORS(group_idx,:), 'FaceAlpha', 0.15, ...
%              'EdgeColor', 'none', 'HandleVisibility', 'off');
%
%         % Plot start point (circle)
%         plot(avg_traj(1,1), avg_traj(1,2), 'o', 'MarkerSize', 8, ...
%              'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
%              'LineWidth', 1, 'HandleVisibility', 'off');
%
%         % Plot end point (square)
%         plot(avg_traj(end,1), avg_traj(end,2), 's', 'MarkerSize', 8, ...
%              'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
%              'LineWidth', 1, 'HandleVisibility', 'off');
%     end
% end
%
% % Plot origin marker (red x)
% plot(0, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'HandleVisibility', 'off');
%
% % Create legend with dummy plots
% h_start = plot(NaN, NaN, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
% h_end = plot(NaN, NaN, 'sk', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
% h_origin = plot(NaN, NaN, 'rx', 'MarkerSize', 12, 'LineWidth', 3);
% legend([h_start, h_end, h_origin], {'Start', 'End', 'Origin'}, ...
%        'Location', 'best', 'FontSize', LEGEND_FONT_SIZE);
%
% % Save second combined figure
% saveas(fig2, 'combined_trajectories_with_markers.png');
% saveas(fig2, 'combined_trajectories_with_markers.fig');
%
% fprintf('\nSecond figure with markers saved successfully!\n');
% fprintf('  - combined_trajectories_with_markers.png\n');
% fprintf('  - combined_trajectories_with_markers.fig\n');

% =========================================================================
% CREATE THIRD FIGURE WITH 180-DEGREE ROTATED TRAJECTORIES
% =========================================================================

fprintf('\n=== CREATING THIRD FIGURE WITH ROTATED TRAJECTORIES ===\n');

fig3 = figure('Position', FIGURE_SIZE);

% -------------------------------------------------------------------------
% TOP SUBPLOT: VORTEX TRAJECTORIES ROTATED 180 DEGREES
% -------------------------------------------------------------------------

subplot(2, 1, 1);
hold on;
title('(a) Vortex Field Convergence', 'FontSize', 12, 'FontWeight', 'normal');
xlabel('x (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
ylabel('y (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
grid on;
axis equal;
xlim(VORTEX_XLIM);
ylim(VORTEX_YLIM);
set(gca, 'FontSize', TICK_FONT_SIZE);

% Add vortex field (NOT rotated)
[X_v, Y_v] = meshgrid(linspace(VORTEX_XLIM(1), VORTEX_XLIM(2), 15), ...
                       linspace(VORTEX_YLIM(1), VORTEX_YLIM(2), 15));
center_x = 0; center_y = 0;
r_v = sqrt((X_v - center_x).^2 + (Y_v - center_y).^2);
theta_v = atan2(Y_v - center_y, X_v - center_x);
U_v = -r_v .* sin(theta_v);
V_v = r_v .* cos(theta_v);
scale_factor = 0.5;
U_v_norm = scale_factor * U_v;
V_v_norm = scale_factor * V_v;
q_v = quiver(X_v, Y_v, U_v_norm, V_v_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
q_v.HandleVisibility = 'off';

% Plot vortex cluster trajectories (rotated 180 degrees)
for group_idx = 1:NUM_GROUPS
    if ~isempty(vortex_group_avg{group_idx})
        avg_traj = vortex_group_avg{group_idx};
        std_traj = vortex_group_std{group_idx};

        % Rotate trajectories by 180 degrees: (x,y) -> (-x,-y)
        avg_traj_rot = -avg_traj;

        % Plot trajectory line
        plot(avg_traj_rot(:,1), avg_traj_rot(:,2), 'Color', GROUP_COLORS(group_idx,:), ...
             'LineWidth', 2, 'HandleVisibility', 'off');

        % Plot confidence band (also rotated)
        x_band = [avg_traj_rot(:,1); flipud(avg_traj_rot(:,1))];
        y_band = [avg_traj_rot(:,2) + std_traj(:,2); flipud(avg_traj_rot(:,2) - std_traj(:,2))];
        fill(x_band, y_band, GROUP_COLORS(group_idx,:), 'FaceAlpha', 0.15, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Plot start point (circle) at rotated position
        plot(avg_traj_rot(1,1), avg_traj_rot(1,2), 'o', 'MarkerSize', 8, ...
             'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1, 'HandleVisibility', 'off');

        % Plot end point (square) at rotated position
        plot(avg_traj_rot(end,1), avg_traj_rot(end,2), 's', 'MarkerSize', 8, ...
             'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1, 'HandleVisibility', 'off');
    end
end

% Plot origin marker (red x)
plot(0, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'HandleVisibility', 'off');

% Create legend with dummy plots
h_start = plot(NaN, NaN, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
h_end = plot(NaN, NaN, 'sk', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
h_origin = plot(NaN, NaN, 'rx', 'MarkerSize', 12, 'LineWidth', 3);
legend([h_start, h_end, h_origin], {'Start', 'End', 'Origin'}, ...
       'Location', 'best', 'FontSize', LEGEND_FONT_SIZE);

% -------------------------------------------------------------------------
% BOTTOM SUBPLOT: SADDLE TRAJECTORIES ROTATED 180 DEGREES
% -------------------------------------------------------------------------

subplot(2, 1, 2);
hold on;
title('(b) Saddle Field Convergence', 'FontSize', 12, 'FontWeight', 'normal');
xlabel('x (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
ylabel('y (m)', 'FontSize', AXIS_LABEL_FONT_SIZE);
grid on;
axis equal;
xlim(SADDLE_XLIM);
ylim(SADDLE_YLIM);
set(gca, 'FontSize', TICK_FONT_SIZE);

% Add saddle field (NOT rotated)
[X_s, Y_s] = meshgrid(linspace(-1, 1, 15), linspace(-1, 1, 15));
r_s = sqrt((X_s - center_x).^2 + (Y_s - center_y).^2);
theta_s = atan2(Y_s - center_y, X_s - center_x);
U_s = -r_s .* sin(theta_s);  % Negated for correct saddle orientation
V_s = -r_s .* cos(theta_s);  % Negated for correct saddle orientation
U_s_norm = scale_factor * U_s;
V_s_norm = scale_factor * V_s;
q_s = quiver(X_s, Y_s, U_s_norm, V_s_norm, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
q_s.HandleVisibility = 'off';

% Plot saddle cluster trajectories (rotated 180 degrees)
for group_idx = 1:NUM_GROUPS
    if ~isempty(saddle_group_avg{group_idx})
        avg_traj = saddle_group_avg{group_idx};
        std_traj = saddle_group_std{group_idx};

        % Rotate trajectories by 180 degrees: (x,y) -> (-x,-y)
        avg_traj_rot = -avg_traj;

        % Plot trajectory line
        plot(avg_traj_rot(:,1), avg_traj_rot(:,2), 'Color', GROUP_COLORS(group_idx,:), ...
             'LineWidth', 2, 'HandleVisibility', 'off');

        % Plot confidence band (also rotated)
        x_band = [avg_traj_rot(:,1); flipud(avg_traj_rot(:,1))];
        y_band = [avg_traj_rot(:,2) + std_traj(:,2); flipud(avg_traj_rot(:,2) - std_traj(:,2))];
        fill(x_band, y_band, GROUP_COLORS(group_idx,:), 'FaceAlpha', 0.15, ...
             'EdgeColor', 'none', 'HandleVisibility', 'off');

        % Plot start point (circle) at rotated position
        plot(avg_traj_rot(1,1), avg_traj_rot(1,2), 'o', 'MarkerSize', 8, ...
             'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1, 'HandleVisibility', 'off');

        % Plot end point (square) at rotated position
        plot(avg_traj_rot(end,1), avg_traj_rot(end,2), 's', 'MarkerSize', 8, ...
             'MarkerFaceColor', GROUP_COLORS(group_idx,:), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 1, 'HandleVisibility', 'off');
    end
end

% Plot origin marker (red x)
plot(0, 0, 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'HandleVisibility', 'off');

% Create legend with dummy plots
h_start = plot(NaN, NaN, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
h_end = plot(NaN, NaN, 'sk', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'LineWidth', 1);
h_origin = plot(NaN, NaN, 'rx', 'MarkerSize', 12, 'LineWidth', 3);
legend([h_start, h_end, h_origin], {'Start', 'End', 'Origin'}, ...
       'Location', 'best', 'FontSize', LEGEND_FONT_SIZE);

% Save third combined figure
saveas(fig3, 'combined_trajectories_rotated180.png');
saveas(fig3, 'combined_trajectories_rotated180.fig');

fprintf('\nThird figure with rotated trajectories saved successfully!\n');
fprintf('  - combined_trajectories_rotated180.png\n');
fprintf('  - combined_trajectories_rotated180.fig\n');

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

% Function to plot trajectory with confidence band
function h = plotTrajectoryWithConfidence(avg_traj, std_traj, color, line_width, alpha_val, name)
    if isempty(avg_traj)
        h = [];
        return;
    end

    % Plot the average trajectory
    h = plot(avg_traj(:,1), avg_traj(:,2), 'Color', color, 'LineWidth', line_width, 'DisplayName', name);

    % Create confidence band (mean ± standard deviation)
    x_band = [avg_traj(:,1); flipud(avg_traj(:,1))];
    y_band = [avg_traj(:,2) + std_traj(:,2); flipud(avg_traj(:,2) - std_traj(:,2))];

    % Plot the confidence band with transparency
    fill(x_band, y_band, color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

% Define the trajectory averaging function
function [avg_traj, std_traj] = averageTrajectories(trajectories)
    % Find the minimum length among all trajectories
    traj_lengths = cellfun(@(x) size(x, 1), trajectories);
    min_length = min(traj_lengths);
    max_length = max(traj_lengths);

    fprintf('  Trajectory lengths vary from %d to %d points\n', min_length, max_length);

    % Number of points in the resampled trajectory (use min_length for safety)
    target_length = min_length;

    % Storage for resampled trajectories
    resampled_trajectories = cell(1, length(trajectories));

    % Resample all trajectories to the target length
    for i = 1:length(trajectories)
        original_traj = trajectories{i};
        original_length = size(original_traj, 1);

        if original_length == target_length
            % No resampling needed
            resampled_trajectories{i} = original_traj;
        else
            % Create a parametric representation based on cumulative distance
            % This preserves the spatial distribution better than uniform time sampling

            % Calculate cumulative distance along the trajectory
            dist = zeros(original_length, 1);
            for j = 2:original_length
                dist(j) = dist(j-1) + norm(original_traj(j,:) - original_traj(j-1,:));
            end

            % Normalize to [0, 1]
            if dist(end) > 0
                param = dist / dist(end);
            else
                param = linspace(0, 1, original_length)';
            end

            % Sample points at uniform parameter intervals
            new_param = linspace(0, 1, target_length)';
            resampled_x = interp1(param, original_traj(:,1), new_param, 'pchip');
            resampled_y = interp1(param, original_traj(:,2), new_param, 'pchip');

            resampled_trajectories{i} = [resampled_x, resampled_y];
        end
    end

    % Convert cell array to 3D array for easier computation
    traj_array = zeros(target_length, 2, length(trajectories));
    for i = 1:length(trajectories)
        traj_array(:,:,i) = resampled_trajectories{i};
    end

    % Calculate mean and standard deviation across runs
    avg_traj = mean(traj_array, 3);
    std_traj = std(traj_array, 0, 3);
end
