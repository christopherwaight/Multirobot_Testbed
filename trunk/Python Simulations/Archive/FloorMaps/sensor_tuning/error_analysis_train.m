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
for j = 1:2
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

%% Saturation adjustment

function s_adjusted   = saturation_adjust(h,s)
    if 0 <= h && h <= 0.2
        s_adjusted = max(0, (s - 0.24) * 1.25) + 0.3;
    elseif 0.2 <= h && h <= 0.6
        s_adjusted = max(0, (s - 0.2) * 2) + 0.3;
    elseif 0.6 <= h && h <= 0.7
        s_adjusted = max(0, (s - 0.15) * 1.5) + 0.3;
    elseif 0.7 <= h && h <= 0.9
        s_adjusted = max(0, s - 0.11) * 2.9 + 0.3;
    elseif 0.9 <= h && h <= 1
        s_adjusted = max(0, (s - 0.2) * 2) + 0.3;
    else
        s_adjusted = s;  
    end
    
    s_adjusted = min(1, s_adjusted);
end


for i=1:403
    [X(i,6)] = saturation_adjust(X(i,5),X(i,6));
end






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



rmse_hue_train = sqrt(mean((Y(:,1) - X(:,5)).^2))
rmse_sat_train = sqrt(mean((Y(:,2) - X(:,6)).^2))


