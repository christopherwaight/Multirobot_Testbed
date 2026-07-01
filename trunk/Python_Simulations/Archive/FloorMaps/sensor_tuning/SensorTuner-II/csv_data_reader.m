%% Reading the CSV file

clc; clear all, close all;


%% Creating X and Ytrue variables
data = readmatrix('pink-robot-calibration-20251301.csv');

rows_to_include = 576 -(24*3);

% Extract columns 1 and 2 as Y 
Y = data(1:rows_to_include, 1:2);
Ytrue = Y;

% Extract columns 3 to 6 as X 
X_0 = data(1:576, 3:6);

% Normalize RGBK values to [0, 1]
X_norm = X_0;

for i = 1:size(X_0, 2)
    X_norm(:, i) = (X_0(:, i) - min(X_0(:, i))) ./ (max(X_0(:, i)) - min(X_0(:, i)));
end

X_norm=X_norm(1:rows_to_include,:);

X_norm(X_norm<0)=0;
X_norm(X_norm>1)=1;

X_hsv = rgb2hsv(X_norm(:,1:3));


% Concatenate everything
X = [X_norm X_hsv(:,1:2)]; % X now has R G B K H S



%% Plotting results %%

% Create a grid of points for interpolation
[X1grid, X2grid] = meshgrid(linspace(min(Y(:,1)), max(Y(:,1)), 50), ...
                           linspace(min(Y(:,2)), max(Y(:,2)), 50));

% Interpolate the hue values onto the grid
hueInterp = griddata(Y(:,1), Y(:,2), X(:,5), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, hueInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Hue (Estimated)');
title('Surface Plot of Estimated Hue');

% Interpolate the hue values onto the grid
satInterp = griddata(Y(:,1), Y(:,2), X(:,6), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, satInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Sat (Estimated)');
title('Surface Plot of Estimated Sat');

hue_raw_rmse = rmse(Y(Y(:,1) >= 0.021, 1), X(Y(:,1) >= 0.021, 5), 1)
sat_raw_rmse = rmse(Y(:,2), X(:,6),1)

%% Clear all unnecessary variables %%

% List of variables to keep
keepVars = {'X', 'Y', 'X_augmented', 'Y_augmented', 'Ytrue'};

% Get all variables in the workspace
allVars = who;

