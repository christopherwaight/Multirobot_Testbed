% Assuming your data is in pose_04 with columns [x y]
x = pose_04(:,1);
y = pose_04(:,2);

% Compute FFT
Fs = 1; % Sampling frequency (adjust if you know the actual sampling rate)
N = length(x);
fx = fft(x);
fy = fft(y);

% Compute frequency axis
f = (0:N-1)*(Fs/N);

% Get magnitude spectrum
mag_x = abs(fx);
mag_y = abs(fy);

% Plot frequency spectrum
figure;
subplot(2,1,1);
plot(f(1:N/2), mag_x(1:N/2));
title('X Frequency Spectrum');
xlabel('Frequency');
ylabel('Magnitude');

subplot(2,1,2);
plot(f(1:N/2), mag_y(1:N/2));
title('Y Frequency Spectrum');
xlabel('Frequency');
ylabel('Magnitude');

% Find dominant frequency (excluding DC component)
[~, idx_x] = max(mag_x(2:N/2));
[~, idx_y] = max(mag_y(2:N/2));

freq_x = f(idx_x + 1);  % Add 1 because we excluded first element
freq_y = f(idx_y + 1);

period_x = 1/freq_x;
period_y = 1/freq_y;

fprintf('Estimated period from x: %.2f samples\n', period_x);
fprintf('Estimated period from y: %.2f samples\n', period_y);

% Plot original x-y trajectory
figure;
plot(x, y, 'b.-');
title('X-Y Trajectory');
xlabel('X Position');
ylabel('Y Position');
axis equal;
grid on;