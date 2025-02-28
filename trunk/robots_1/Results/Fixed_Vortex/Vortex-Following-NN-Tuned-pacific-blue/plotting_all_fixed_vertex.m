clc; clear all; close all;
load('pacific_blue_vortex_follow_saturation_04.mat')

colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E'};
figure;
hold on;

data = {pose_04, pose_05, pose_06, pose_07};
labels = {'Sat 0.4', 'Sat 0.5', 'Sat 0.6', 'Sat 0.7'};

% Plot center of vortex
plot(0, 0, 'kx', 'MarkerSize', 10, 'DisplayName', 'Centre of Vortex');
all_velocities = [0,0,0,0];

for i = 1:length(data)
   
   %  Plot Main trajectory
   plot(data{i}(:,1), data{i}(:,2), 'Color', colors{i}, 'LineWidth', 2, 'DisplayName', labels{i});   % Start point
   plot(data{i}(1,1), data{i}(1,2), '^', 'Color', colors{i}, 'MarkerSize', 10, 'DisplayName', [labels{i} ' Start']);
   
   % Plot End points
   text(data{i}(1,1), data{i}(1,2), 'start', 'VerticalAlignment', 'bottom');
   plot(data{i}(end,1), data{i}(end,2), 's', 'Color', colors{i}, 'MarkerSize', 10, 'DisplayName', [labels{i} ' End']);
   text(data{i}(end,1), data{i}(end,2), 'end', 'VerticalAlignment', 'top');

   
    
    % Determine Velocities
    dx = diff(data{i}(:,1));
    dy = diff(data{i}(:,2));
    velocities = sqrt(dx.^2 + dy.^2);
    avg_velocity = mean(velocities);
    all_velocities(i)=avg_velocity;    
    

    % Calculate the goodness of circle
    current_pose = data{i};
    center_x = mean(current_pose(:,1));
    center_y = mean(current_pose(:,2));
    radii = sqrt((current_pose(:,1) - center_x).^2 + (current_pose(:,2) - center_y).^2);
    mean_radius = mean(radii);
    std_radius = std(radii);
    cv_radius = std_radius / mean_radius; % Coefficient of variation (relative consistency)
    
    % Print Statistics
    fprintf('\nResults for %s:\n', labels{i});
    fprintf('Average velocity for %s: %.4f m/timestep\n', labels{i}, avg_velocity);
    fprintf('Center: (%.4f, %.4f)\n', center_x, center_y);
    fprintf('Mean radius: %.4f m\n', mean_radius);
    fprintf('Std dev of radius: %.4f m\n', std_radius);
    fprintf('Coefficient of variation: %.4f (lower means more consistent)\n', cv_radius);
    

end

title('Vortex Following Trajectories');
xlabel('X Position (m)');
ylabel('Y Position (m)');
legend('Location', 'best');
grid on;
%hold off;


% Create grid
[x, y] = meshgrid(linspace(-1, 1, 255));
center_x = 0; 
center_y = 0;

% Calculate vector field
r = sqrt((x - center_x).^2 + (y - center_y).^2) + 1e-8;
theta = atan2(y - center_y, x - center_x);
u = -r .* sin(theta);
v = r .* cos(theta);

% Calculate magnitude and direction 
magnitude = sqrt(u.^2 + v.^2);
direction = (atan2(v, u) + pi) / (2*pi);  % Changed from atan2(v, u)
% Create HSV image
offset = 0.3;
magnitude_norm = (magnitude - min(magnitude(:))) / (max(magnitude(:)) - min(magnitude(:)));
magnitude_scaled = uint8(((magnitude_norm*(1-offset) + offset)*255));
direction_scaled = uint8(min(direction * 360, 360));

hsv = cat(3, direction_scaled, magnitude_scaled, 255*ones(size(magnitude_scaled), 'uint8'));
rgb = hsv2rgb(double(hsv)/255);



% Add quiver plot
skip = 12;
quiver(x(1:skip:end,1:skip:end), y(1:skip:end,1:skip:end), ...
    -u(1:skip:end,1:skip:end), -v(1:skip:end,1:skip:end), ...
    'Color', [0 0 0 0.1], 'ShowArrowHead', 'on', 'LineWidth', 0.1);
title('Vortex Flow Field');
xlabel('X Position');
ylabel('Y Position');
xlim([-1 1]);
ylim([-1 1]);
grid on;
hold off;

%% Predetermined Velocity

figure;
hold on;

data = {pose_diro};
labels = {'fixed velocity'};

% Plot center of vortex
plot(0, 0, 'kx', 'MarkerSize', 10, 'DisplayName', 'Centre of Vortex');

for i = 1:length(data)
   % Main trajectory
   plot(data{i}(:,1), data{i}(:,2), 'Color', colors{i}, 'LineWidth', 2, 'DisplayName', labels{i});   % Start point
   plot(data{i}(1,1), data{i}(1,2), '^', 'Color', colors{i}, 'MarkerSize', 10, 'DisplayName', [labels{i} ' Start']);
   text(data{i}(1,1), data{i}(1,2), 'start', 'VerticalAlignment', 'bottom');
   % End point
   plot(data{i}(end,1), data{i}(end,2), 's', 'Color', colors{i}, 'MarkerSize', 10, 'DisplayName', [labels{i} ' End']);
   text(data{i}(end,1), data{i}(end,2), 'end', 'VerticalAlignment', 'top');
end

title('Vortex Following Trajectories');
xlabel('X Position (m)');
ylabel('Y Position (m)');
legend('Location', 'best');
grid on;
%hold off;


% Create grid
[x, y] = meshgrid(linspace(-1, 1, 255));
center_x = 0; 
center_y = 0;

% Calculate vector field
r = sqrt((x - center_x).^2 + (y - center_y).^2) + 1e-8;
theta = atan2(y - center_y, x - center_x);
u = -r .* sin(theta);
v = r .* cos(theta);

% Calculate magnitude and direction 
magnitude = sqrt(u.^2 + v.^2);
direction = (atan2(v, u) + pi) / (2*pi);  % Changed from atan2(v, u)
% Create HSV image
offset = 0.3;
magnitude_norm = (magnitude - min(magnitude(:))) / (max(magnitude(:)) - min(magnitude(:)));
magnitude_scaled = uint8(((magnitude_norm*(1-offset) + offset)*255));
direction_scaled = uint8(min(direction * 360, 360));

hsv = cat(3, direction_scaled, magnitude_scaled, 255*ones(size(magnitude_scaled), 'uint8'));
rgb = hsv2rgb(double(hsv)/255);



% Add quiver plot
skip = 12;
quiver(x(1:skip:end,1:skip:end), y(1:skip:end,1:skip:end), ...
    -u(1:skip:end,1:skip:end), -v(1:skip:end,1:skip:end), ...
    'Color', [0 0 0 0.1], 'ShowArrowHead', 'on', 'LineWidth', 0.1);
title('Vortex Flow Field');
xlabel('X Position');
ylabel('Y Position');
xlim([-1 1]);
ylim([-1 1]);
grid on;