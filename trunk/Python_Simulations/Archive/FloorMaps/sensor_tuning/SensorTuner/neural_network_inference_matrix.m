function Y_pred = neural_network_inference_matrix(X, W1, b1, W2, b2, W3, b3, W4, b4, W_out, b_out) % Add W4, b4
    % X: Input matrix where each row is a data point (Nx6)
    % Y_pred: Output matrix where each row is the prediction for the corresponding data point in X

    % ReLU activation function
    function y = relu(x)
        y = max(0, x);
    end

    % Sigmoid activation function (for output)
    function y = sigmoid(x)
        y = 1 ./ (1 + exp(-x));
    end

    N = size(X, 1);
    Y_pred = zeros(N, size(W_out, 1));

    for i = 1:N
        u = X(i, :)'; 
        % Normalize the input (if necessary)
        u_norm = (u - min(u)) ./ (max(u) - min(u)); 

        % Forward propagation 
        z1 = W1 * u_norm + b1;
        a1 = relu(z1);  
        z2 = W2 * a1 + b2;
        a2 = relu(z2);
        z3 = W3 * a2 + b3;
        a3 = relu(z3);
        z4 = W4 * a3 + b4; % Forward pass for the new layer
        a4 = relu(z4);
        z_out = W_out * a4 + b_out; % Use a4 for output
        y = sigmoid(z_out);  
        Y_pred(i, :) = y';
    end
end