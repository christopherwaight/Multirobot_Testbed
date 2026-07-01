%% Saturation Tuner %%

%% Loading and Pre-processing data
    % Uncomment if running without running hue_tuner first 

% data = readmatrix('pink-robot-calibration-20251301.csv');
% 
% % Extract columns 1 and 2 as Y and 3 to 6 as X
% Y = data(:, 1:2);
% X_0 = data(:, 3:6);
% 
% % Normalize RGBK values to [0, 1]
% X_norm = X_0;
% for i = 1:size(X_0, 2)
%     X_norm(:, i) = (X_0(:, i) - min(X_0(:, i))) ./ (max(X_0(:, i)) - min(X_0(:, i)));
% end
% X_norm=X_norm(1:rows_to_include,:);
% X_norm(X_norm<0)=0;
% X_norm(X_norm>1)=1;
% 
% X_hsv = rgb2hsv(X_norm(:,1:3));
% 
% X = [X_norm X_hsv(:,1:2)]; % X now has R G B K H S
% 
% %% Clipping off lowest 6 rows.
% rows_to_include = 576 -(24*6);
% X = X(1:rows_to_include,:);
% Ytrue = Y(1:rows_to_include,:);



%% Plotting the Data for analysis
% Define the number of bins
num_bins_sat = 20; 

% Calculate bin edges
bin_edges_sat = linspace(0, 1, num_bins_sat + 1);

% Initialize an array to store coefficients for each bin
all_coefficients_sat = zeros(num_bins_sat, 2);

% Loop through the bins
for i = 1:num_bins_sat
    % Logical indexing to filter data
    filter_idx = X(:,5) > bin_edges_sat(i) & X(:,5) < bin_edges_sat(i+1);
    Y_filtered = Y(filter_idx, 2);
    X_filtered = X(filter_idx, 6);

    figure();  % Create a new figure for each plot
    scatter(Y_filtered, X_filtered, 'b', 'filled'); 
    hold on;

    % Calculate line of best fit and store coefficients
    coefficients = polyfit(Y_filtered, X_filtered, 1);
    all_coefficients_sat(i, :) = coefficients;


    xFit = linspace(min(Y_filtered), max(Y_filtered), 100);  
    yFit = polyval(coefficients, xFit); 
    plot(xFit, yFit, 'r-', 'LineWidth', 2);

    xlabel('Sat (True)');
    ylabel('Sat (Estimated)');
    title(['Hue between ' num2str(bin_edges_sat(i)) ' and ' num2str(bin_edges_sat(i+1))]);
    legend('Data', 'Line of Best Fit');
    hold off;

    % Display the equation
    equation_str = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));
    text(0.5, 0.9, equation_str, 'Units', 'normalized', 'FontSize', 12); 
end




%% Modified saturation_adjust function

function s_adjusted = saturation_adjust(h, s, bin_edges, all_coefficients)
    
    bin_idx = find(h >= bin_edges(1:end-1) & h <= bin_edges(2:end)); 

    
    if ~isempty(bin_idx)
        coefficients = all_coefficients(bin_idx, :);  % Get coefficients for the bin
        s_adjusted = max(0, ((s - coefficients(2)) * 0.7/coefficients(1) + 0.3));
        s_adjusted = min(1, s_adjusted);  
    else %exception handling
        s_adjusted = s;  % Default to original value if h is outside the bins
    end
end


%% Adjusting the Saturation

for i=1:rows_to_include
    [X(i,6)] = saturation_adjust(X(i,5),X(i,6),bin_edges_sat, all_coefficients_sat);
end

%% Plotting Results %%

% Create a grid of points for interpolation
[X1grid, X2grid] = meshgrid(linspace(min(Ytrue(:,1)), max(Ytrue(:,1)), 50), ...
                           linspace(min(Ytrue(:,2)), max(Ytrue(:,2)), 50));

% Interpolate the hue values onto the grid
hueInterp = griddata(Ytrue(:,1), Ytrue(:,2), X(:,5), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, hueInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Hue (Estimated)');
title('Surface Plot of Estimated Hue');

% Interpolate the hue values onto the grid
satInterp = griddata(Ytrue(:,1), Ytrue(:,2), X(:,6), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, satInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Sat (Estimated)');
title('Surface Plot of Estimated Sat');

%% Capturing analysis information

X_max = max(X_0(:,:));
X_min = min(X_0(:,:));

hue_cal2_rmse = hue_cal1_rmse
sat_cal2_rmse = rmse(Ytrue(:,2), X(:,6),1)


