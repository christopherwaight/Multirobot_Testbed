%% Data File Reader %%
% Read the CSV file, skipping the first row
data = readmatrix('hsv_training.csv', 'HeaderLines', 1);

% Extract columns 1 and 2 as Y
Y = data(1:576, 2:3);

% Extract columns 3 to 6 as X
X_0 = data(1:576, 4:6);

X=X_0;
% Normalize RGB values to [0, 1]
X_norm = X_0;
for i = 1:size(X_0, 2)
    X_norm(:, i) = (X_0(:, i) - min(X(:, i))) ./ (max(X(:, i)) - min(X(:, i)));
end

% Convert RGB to HSV
X_augment1 = rgb2hsv(X_norm);

% Concatenate everything
X = [X_norm X_augment1];

% Create the 3D scatter plot
figure; % Create a new figure window
scatter3(Y(:,1), Y(:,2), X(:,4), 'filled'); % 'filled' fills the markers

% Add labels and title
xlabel('X1 (True Hue)'); 
ylabel('X2 (True Saturation)');
zlabel('X4 (Hue percieved)');
title('3D Scatter Plot of X1, X2 vs. X4 (Hue)');


% --- Add Mesh Grid ---

hold on; % Keep the scatter plot

% Create a meshgrid from X1 and X2
[X1grid, X2grid] = meshgrid(linspace(min(Y(:,1)), max(Y(:,1)), 50), ...
                           linspace(min(Y(:,2)), max(Y(:,2)), 50));

% Interpolate X4 (Hue) values to create the grid
X4grid = griddata(Y(:,1), Y(:,2), X(:,4), X1grid, X2grid);

% Create the mesh plot
mesh(X1grid, X2grid, X4grid); 

hold off; % Release the hold

%% Repeating for Saturation

% Create the 3D scatter plot
figure; % Create a new figure window
scatter3(Y(:,1), Y(:,2), X(:,5), 'filled'); % 'filled' fills the markers

% Add labels and title
xlabel('X1 (True Hue)'); 
ylabel('X2 (True Saturation)');
zlabel('X4 (Hue percieved)');
title('3D Scatter Plot of X1, X2 vs. X5 (Saturation)');


% --- Add Mesh Grid ---

hold on; % Keep the scatter plot

% Create a meshgrid from X1 and X2
[X1grid, X2grid] = meshgrid(linspace(min(Y(:,1)), max(Y(:,1)), 50), ...
                           linspace(min(Y(:,2)), max(Y(:,2)), 50));

% Interpolate X4 (Hue) values to create the grid
X4grid = griddata(Y(:,1), Y(:,2), X(:,5), X1grid, X2grid);

% Create the mesh plot
mesh(X1grid, X2grid, X4grid); 

hold off; % Release the hold