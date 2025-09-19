%% Verification plot for pink!


%% Extracting columns 1 and 2 as Y columns 3 to 6 as X

data = readmatrix('pink-robot-verification-20251301.csv');
Ytest = data(1:70, 1:2);
Ytesttrue = Ytest;
X_00 = data(1:70, 3:6);

%% Converting RGB Data to HSV - normalization, clip, convert.
% Be sure to run saturation_tuner first


X_test_norm(:, 1) = (X_00(:, 1) - X_min(1)) ./ (X_max(1)-X_min(1));
X_test_norm(:, 2) = (X_00(:, 2) - X_min(2)) ./ (X_max(2)-X_min(2));
X_test_norm(:, 3) = (X_00(:, 3) - X_min(3)) ./ (X_max(3)-X_min(3));
X_test_norm(:, 4) = X_00(:, 4);

X_test_norm(X_test_norm<0)=0;
X_test_norm(X_test_norm>1)=1;

X_hsv = rgb2hsv(X_test_norm(:,1:3));

X = [X_test_norm X_hsv(:,1:2)]; % X now has R G B K H S

%% Hue_adjust function

function h_adjusted = hue_adjust(h, bin_edges, all_coefficients)
    
    bin_idx = find(h >= bin_edges(1:end-1) & h <= bin_edges(2:end)); 

    
    if ~isempty(bin_idx)
        coefficients = all_coefficients(bin_idx, :);  % Get coefficients for the bin
        h_adjusted = max(0, ((h - coefficients(2)) * 1/coefficients(1)));
        h_adjusted = min(1, h_adjusted);  
    else %exception handling
        h_adjusted = h;  % Default to original value if h is outside the bins
    end
end


%% Adjusting the Hue

for i=1:70 % Magic literal. Only using 70 data points in verification
    [X(i,5)] = hue_adjust(X(i,5),bin_edges_hue, all_coefficients_hue);
end


%% Saturation adjustment function

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

for i=1:70
    [X(i,6)] = saturation_adjust(X(i,5),X(i,6),bin_edges_sat, all_coefficients_sat);
end






%% Plotting Results %%

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

hue_ver_rmse = rmse(Ytesttrue(:,1), X(:,5),1)
sat_ver_rmse = rmse(Ytesttrue(:,2), X(:,6),1)
