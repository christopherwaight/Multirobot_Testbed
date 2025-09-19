% Read the CSV file, skipping the first row.

clc; clear all, close all;


data = readmatrix('hsv_training.csv', 'HeaderLines', 1);

% Extract columns 1 and 2 as Y (now with 548 rows)
Y = data(1:403, 2:3);
Ytrue = Y;

% Extract columns 3 to 6 as X (now with 548 rows)
X_0 = data(1:576, 4:7);

% Normalize RGB values to [0, 1]
X_norm = X_0;
for i = 1:size(X_0, 2)
    X_norm(:, i) = (X_0(:, i) - min(X_0(:, i))) ./ (max(X_0(:, i)) - min(X_0(:, i)));
end
X_norm=X_norm(1:403,:);

% 
X_norm(X_norm<0)=0;
X_norm(X_norm>1)=1;

% Convert RGB to HSV
X_augment1 = rgb2hsv(X_norm(:,1:3));


% Concatenate everything
X = [X_norm X_augment1(:,1:2)];

X_augmented = [];  % Initialize as empty
Y_augmented = [];  % Initialize as empty

% --- Noise generation and augmentation (repeated 10 times) ---
for j = 1:20
    % Generate normally distributed noise with std dev 0.01
    noise_X = 0.01 * randn(size(X)); 
    noise_Y = 0.01 * randn(size(Y)); 

    % Add noise to create augmented data
    X_plus = X + noise_X;
    Y_plus = Y + noise_Y;

    % Clip values to be between 0 and 1
    X_plus = max(0, min(1, X_plus)); 
    Y_plus = max(0, min(1, Y_plus)); 

    % Append augmented data 
    X_augmented = [X_augmented; X_plus];
    Y_augmented = [Y_augmented; Y_plus];

   
end

%% Manually adjusting Saturation.

% Extract the 6th column of X_augmented
sat_estimated = X_augmented(:, 6);

% Multiply the values by 2 and subtract 0.4
sat_transformed = sat_estimated * 1.2 - 0.1;

% Clip the values to be between 0 and 1
sat_clipped = max(0, min(1, sat_transformed)); 

% Rewriting 
%X_augmented(:, 6) = sat_clipped;



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

%% Clear all unnecessary variables %%

% List of variables to keep
keepVars = {'X', 'Y', 'X_augmented', 'Y_augmented', 'Ytrue'};

% Get all variables in the workspace
allVars = who;


%% Looking at Final Output
% Create a grid of points for interpolation (using augmented data)
[X1grid, X2grid] = meshgrid(linspace(min(Y_augmented(:,1)), max(Y_augmented(:,1)), 50), ...
                           linspace(min(Y_augmented(:,2)), max(Y_augmented(:,2)), 50));

% Interpolate the hue values onto the grid (using augmented data)
hueInterp = griddata(Y_augmented(:,1), Y_augmented(:,2), X_augmented(:,5), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, hueInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Hue (Estimated)');
title('Surface Plot of Estimated Hue (Augmented Data)');

% Interpolate the saturation values onto the grid (using augmented data)
satInterp = griddata(Y_augmented(:,1), Y_augmented(:,2), X_augmented(:,6), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, satInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Sat (Estimated)');
title('Surface Plot of Estimated Sat (Augmented Data)');


