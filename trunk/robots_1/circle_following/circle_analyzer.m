plot(circle45_3pi4agaom(:,1),circle45_3pi4agaom(:,2))
figure()
plot( linspace(0, length(circle45_3pi4agaom(:,1))/10, length(circle45_3pi4agaom(:,1))),circle45_3pi4agaom(:,1:2));ylim([-1,1])


x_array = circle30(1:150,1)*100;
y_array = circle30(1:150,2)*100;
a_min = -1; a_max = 1;  % x-range
b_min = -1; b_max = 1;  % y-range
r_min = 0.01; % Example minimum radius
r_max = max(sqrt((x_array - mean(x_array)).^2 + (y_array - mean(y_array)).^2)); % Estimate max radius

% Step size for a and b (adjust for desired resolution)
step_size = 0.01;

% Create accumulator array
a_range = a_min:step_size:a_max;
b_range = b_min:step_size:b_max;
r_range = r_min:step_size:r_max;
accumulator = zeros(length(a_range), length(b_range), length(r_range));

% Voting process
num_points = length(x_array);
for i = 1:num_points
    x = x_array(i);
    y = y_array(i);
    for a_idx = 1:length(a_range)
        a = a_range(a_idx);
        for b_idx = 1:length(b_range)
            b = b_range(b_idx);
            r = sqrt((x - a)^2 + (y - b)^2); % No need to round here
            r_idx = round((r - r_min) / step_size) + 1; % Find closest index in r_range
            if r_idx >= 1 && r_idx <= length(r_range)
                accumulator(a_idx, b_idx, r_idx) = accumulator(a_idx, b_idx, r_idx) + 1;
            end
        end
    end
end

% Find maxima (simplified)
[~, max_idx] = max(accumulator(:));
[a_idx, b_idx, r_idx] = ind2sub(size(accumulator), max_idx);
a = a_range(a_idx);
b = b_range(b_idx);
r = r_range(r_idx);

% Display results
%scatter(x_array, y_array); hold on;
%rectangle('position', [a-r, b-r, 2*r, 2*r], 'Curvature', [1,1], 'EdgeColor','r')
