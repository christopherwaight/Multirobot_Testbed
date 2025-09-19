
% Assuming you have your X matrix, and trained weights (W1, b1, etc.)
y_pred = neural_network_inference_matrix(X, W1, b1, W2, b2, W3, b3,W4,b4, W_out, b_out);

Y = Ytrue;
% Create a grid of points for interpolation
[X1grid, X2grid] = meshgrid(linspace(min(Y(:,1)), max(Y(:,1)), 50), ...
                           linspace(min(Y(:,2)), max(Y(:,2)), 50));

% Interpolate the hue values onto the grid
hueInterp = griddata(Y(:,1), Y(:,2),  y_pred(:,1), X1grid, X2grid);

% Create the surface plot
figure;
surf(X1grid, X2grid, hueInterp);

% Add labels and titleX
xlabel('Hue (True)');
ylabel('Sat (True)');
zlabel('Hue (Estimated)');
title('Surface Plot of Estimated Hue');
