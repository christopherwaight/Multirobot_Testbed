% Testing HSV

% Tuning copied from s-Tuner for pinK!

clc; clear all, close all;


data = readmatrix('hsv_testing.csv', 'HeaderLines', 1);

% Extract columns 1 and 2 as Y (now with 548 rows)
Ytest = data(1:70, 2:3);
Ytesttrue = Ytest;

% Extract columns 3 to 6 as X (now with 548 rows)
X_00 = data(1:70, 4:7);

% Normalize RGB values to [0, 1] (Values copied from S-Tuner Data for Pink

X_test_norm(:, 1) = (X_00(:, 1) - 250) ./ (1500-250);
X_test_norm(:, 2) = (X_00(:, 2) - 200) ./ (1700-200);
X_test_norm(:, 3) = (X_00(:, 3) - 200) ./ (1550-250);
X_test_norm(:, 4) = X_00(:, 4);

X_test_norm(X_test_norm<0)=0;
X_test_norm(X_test_norm>1)=1;

% Convert RGB to HSV
X_augment1 = rgb2hsv(X_test_norm(:,1:3));


% Concatenate everything
X = [X_test_norm X_augment1(:,1:2)];


%% Plotting

X_augmented = [];  % Initialize as empty
Y_augmented = [];  % Initialize as empty

% --- Noise generation and augmentation (repeated 10 times) ---
for j = 1:2
    % Generate normally distributed noise with std dev 0.01
    noise_X = 0.01 * randn(size(X)); 
    noise_Y = 0.01 * randn(size(Ytesttrue)); 

    % Add noise to create augmented data
    X_plus = X + noise_X;
    Y_plus = Ytesttrue + noise_Y;

    % Clip values to be between 0 and 1
    X_plus = max(0, min(1, X_plus)); 
    Y_plus = max(0, min(1, Y_plus)); 

    % Append augmented data 
    X_augmented = [X_augmented; X_plus];
    Y_augmented = [Y_augmented; Y_plus];

   
end
X_augmented;

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

for i=1:70
    [X_augmented(i,6)] = saturation_adjust(X_augmented(i,5),X_augmented(i,6));
end

for i=1:70
    [X(i,6)] = saturation_adjust(X(i,5),X(i,6));
end

%% Plotting results %%

% Create a grid of points for interpolation
[X1grid, X2grid] = meshgrid(linspace(min(Ytesttrue(:,1)), max(Ytesttrue(:,1)), 50), ...
                           linspace(min(Ytesttrue(:,2)), max(Ytesttrue(:,2)), 50));

% Interpolate the hue values onto the grid
hueInterp = griddata(Ytesttrue(:,1), Ytesttrue(:,2), X(:,5), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, hueInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Hue (Estimated)');
title('Surface Plot of Estimated Hue');

% Interpolate the hue values onto the grid
satInterp = griddata(Ytesttrue(:,1), Ytesttrue(:,2), X(:,6), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, satInterp);

% Add labels and title
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Sat (Estimated)');
title('Surface Plot of Estimated Sat');


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


