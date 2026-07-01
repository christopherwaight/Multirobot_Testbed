%% Saturation Tuner %%

%% Tuning between 0 and 0.2 %%
 
figure; 

% Logical indexing to filter data
X_augmented = X;
Y_augmented = Y;

filter_idx = X_augmented(:,5) >= 0 & X_augmented(:,5) <= 0.2; 
Y_filtered = Y_augmented(filter_idx, 2);
X_filtered = X_augmented(filter_idx, 6);

scatter(Y_filtered, X_filtered, 'b', 'filled');  % Scatter plot with filtered data
hold on;

% Calculate line of best fit using filtered data
coefficients = polyfit(Y_filtered, X_filtered, 1); 
xFit = linspace(min(Y_filtered), max(Y_filtered), 100);  
yFit = polyval(coefficients, xFit); 
plot(xFit, yFit, 'r-', 'LineWidth', 2); 

xlabel('Sat (True)');
ylabel('Sat (Estimated)');
title('Tune between 0 and 0.2');
legend('Data', 'Line of Best Fit');
hold off;

% Manually adjusting Saturation between 0 and 0.2

% Logical indexing to filter data (same as before)
filter_idx = X_augmented(:,5) >= 0 & X_augmented(:,5) <= 0.2; 

% Extract the 6th column of X_augmented
sat_estimated = X_augmented(:, 6);

% Apply transformation ONLY to the filtered values
sat_transformed = sat_estimated; % Initialize with original values
sat_transformed(filter_idx) = max(0,((sat_estimated(filter_idx) - 0.24)*1.25)) + 0.3; % Modify filtered values

% Clip the values to be between 0 and 1
sat_clipped = max(0, min(1, sat_transformed));

% Rewriting 
X_augmented(:, 6) = sat_clipped;

%% Plotting between 0.2 and 0.6 %%
 
figure; 

% Logical indexing to filter data
filter_idx = X_augmented(:,5) >= 0.2 & X_augmented(:,5) <= 0.6; 
Y_filtered = Y_augmented(filter_idx, 2);
X_filtered = X_augmented(filter_idx, 6);

scatter(Y_filtered, X_filtered, 'b', 'filled');  % Scatter plot with filtered data
hold on;

% Calculate line of best fit using filtered data
coefficients = polyfit(Y_filtered, X_filtered, 1); 
xFit = linspace(min(Y_filtered), max(Y_filtered), 100);  
yFit = polyval(coefficients, xFit); 
plot(xFit, yFit, 'r-', 'LineWidth', 2); 

xlabel('Sat (True)');
ylabel('Sat (Estimated)');
title('0.2 to 0.6');
legend('Data', 'Line of Best Fit');
hold off;

% Logical indexing to filter data (same as before)
filter_idx = X_augmented(:,5) >= 0.2 & X_augmented(:,5) <= 0.6; 

% Extract the 6th column of X_augmented
sat_estimated = X_augmented(:, 6);

% Apply transformation ONLY to the filtered values
sat_transformed = sat_estimated; % Initialize with original values
sat_transformed(filter_idx) = max(0,(sat_estimated(filter_idx) - 0.2) * 2) + 0.3;

% Clip the values to be between 0 and 1
sat_clipped = max(0, min(1, sat_transformed));

% Rewriting 
X_augmented(:, 6) = sat_clipped;

%% Tuning between 0.6 and 0.7 %%
 
figure; 

% Logical indexing to filter data
filter_idx = X_augmented(:,5) >= 0.6 & X_augmented(:,5) <= 0.7; 
Y_filtered = Y_augmented(filter_idx, 2);
X_filtered = X_augmented(filter_idx, 6);

scatter(Y_filtered, X_filtered, 'b', 'filled');  % Scatter plot with filtered data
hold on;

% Calculate line of best fit using filtered data
coefficients = polyfit(Y_filtered, X_filtered, 1); 
xFit = linspace(min(Y_filtered), max(Y_filtered), 100);  
yFit = polyval(coefficients, xFit); 
plot(xFit, yFit, 'r-', 'LineWidth', 2); 

xlabel('Sat (True)');
ylabel('Sat (Estimated)');
title('Tune between 0.6 and 0.7');
legend('Data', 'Line of Best Fit');
hold off;
%% Adjust
% Logical indexing to filter data (same as before)
filter_idx = X_augmented(:,5) >= 0.6 & X_augmented(:,5) <= 0.7; 

% Extract the 6th column of X_augmented
sat_estimated = X_augmented(:, 6);

% Apply transformation ONLY to the filtered values
sat_transformed = sat_estimated; % Initialize with original values
sat_transformed(filter_idx) = max(0,(sat_estimated(filter_idx) - 0.15) * 1.5) + 0.3;

% Clip the values to be between 0 and 1
sat_clipped = max(0, min(1, sat_transformed));

% Rewriting 
X_augmented(:, 6) = sat_clipped;

%% Plotting between 0.7 and 0.9

% Logical indexing to filter data
filter_idx = X_augmented(:,5) >= 0.7 & X_augmented(:,5) <= 0.9; 
Y_filtered = Y_augmented(filter_idx, 2);
X_filtered = X_augmented(filter_idx, 6);

figure();
scatter(Y_filtered, X_filtered, 'b', 'filled');  % Scatter plot with filtered data
hold on;

% Calculate line of best fit using filtered data
coefficients = polyfit(Y_filtered, X_filtered, 1); 
xFit = linspace(min(Y_filtered), max(Y_filtered), 100);  
yFit = polyval(coefficients, xFit); 
plot(xFit, yFit, 'r-', 'LineWidth', 2); 

xlabel('Sat (True)');
ylabel('Sat (Estimated)');
title('0.7 to 0.9');
legend('Data', 'Line of Best Fit');
hold off;

%% Adjust between 0.7 and 0.9

% Logical indexing to filter data (same as before)
filter_idx = X_augmented(:,5) >= 0.7 & X_augmented(:,5) <= 0.9; 

% Extract the 6th column of X_augmented
sat_estimated = X_augmented(:, 6);

% Apply transformation ONLY to the filtered values
sat_transformed = sat_estimated; % Initialize with original values
sat_transformed(filter_idx) = max(0,(sat_estimated(filter_idx)) - 0.11)*2.9 + 0.3;

% Clip the values to be between 0 and 1
sat_clipped = max(0, min(1, sat_transformed));

% Rewriting 
X_augmented(:, 6) = sat_clipped;

%% Plotting between 0.9 and 1

% Logical indexing to filter data
filter_idx = X_augmented(:,5) >= 0.9 & X_augmented(:,5) <= 1; 
Y_filtered = Y_augmented(filter_idx, 2);
X_filtered = X_augmented(filter_idx, 6);

figure();
scatter(Y_filtered, X_filtered, 'b', 'filled');  % Scatter plot with filtered data
hold on;

% Calculate line of best fit using filtered data
coefficients = polyfit(Y_filtered, X_filtered, 1); 
xFit = linspace(min(Y_filtered), max(Y_filtered), 100);  
yFit = polyval(coefficients, xFit); 
plot(xFit, yFit, 'r-', 'LineWidth', 2); 

xlabel('Sat (True)');
ylabel('Sat (Estimated)');
title('0.9 to 1');
legend('Data', 'Line of Best Fit');
hold off;
%% Adjust
% Logical indexing to filter data (same as before)
filter_idx = X_augmented(:,5) >= 0.9 & X_augmented(:,5) <= 1; 

% Extract the 6th column of X_augmented
sat_estimated = X_augmented(:, 6);

% Apply transformation ONLY to the filtered values
sat_transformed = sat_estimated; % Initialize with original values
sat_transformed(filter_idx) = max(0,(sat_estimated(filter_idx) - 0.2) * 2) + 0.3;

% Clip the values to be between 0 and 1
sat_clipped = max(0, min(1, sat_transformed));

% Rewriting 
X_augmented(:, 6) = sat_clipped;


%% Looking at Final Output


% Interpolate the saturation values onto the grid (using augmented data)
%satInterp = griddata(Y_augmented(:,1), Y_augmented(:,2), X_augmented(:,6), X1grid, X2grid);
satInterp = griddata(X_augmented(:,5), Y_augmented(:,2), Y_augmented(:,2), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, satInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Sat (Estimated)');
title('Surface Plot of Estimated Sat (Augmented Data)');
