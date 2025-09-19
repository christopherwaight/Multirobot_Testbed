% Load X data
X_norm = X_augmented;
Y = Y_augmented;


% Layer sizes
input_size = 6;
hidden_size1 = 32;
hidden_size2 = 16;
hidden_size3 = 8;
output_size = 1;

% Xavier Glorot Initialization
W1 = rand(hidden_size1, input_size) * sqrt(2 / (input_size + hidden_size1)); 
b1 = zeros(hidden_size1, 1); 
W2 = rand(hidden_size2, hidden_size1) * sqrt(2 / (hidden_size1 + hidden_size2)); 
b2 = zeros(hidden_size2, 1);
W3 = rand(hidden_size3, hidden_size2) * sqrt(2 / (hidden_size2 + hidden_size3)); 
b3 = zeros(hidden_size3, 1);
W_out = rand(output_size, hidden_size3) * sqrt(2 / (hidden_size3 + output_size)); 
b_out = zeros(output_size, 1);

function [a1, a2, a3, output] = forward_propagation(X, W1, b1, W2, b2, W3, b3, W_out, b_out)
    z1 = W1 * X' + b1;  
    a1 = sigmoid(z1);     

    z2 = W2 * a1 + b2; 
    a2 = sigmoid(z2);     

    z3 = W3 * a2 + b3; 
    a3 = sigmoid(z3);     

    z_out = W_out * a3 + b_out; 
    output = sigmoid(z_out);   
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function [dW1, db1, dW2, db2, dW3, db3, dW_out, db_out] = backpropagation(X, Y, a1, a2, a3, output, W2, W3, W_out)
    m = size(X, 1); 

    d_out = output - Y'; 
    dW_out = (1/m) * d_out * a3';
    db_out = (1/m) * sum(d_out, 2); 

    d3 = (W_out' * d_out) .* (a3 .* (1 - a3));
    dW3 = (1/m) * d3 * a2';
    db3 = (1/m) * sum(d3, 2);

    d2 = (W3' * d3) .* (a2 .* (1 - a2));
    dW2 = (1/m) * d2 * a1';
    db2 = (1/m) * sum(d2, 2);

    d1 = (W2' * d2) .* (a1 .* (1 - a1));
    dW1 = (1/m) * d1 * X;
    db1 = (1/m) * sum(d1, 2);
end



% Split data into training and validation sets (e.g., 80/20 split)
train_ratio = 0.8;
data_size = size(X_norm, 1);
train_size = round(train_ratio * data_size);

indices = randperm(data_size);
train_indices = indices(1:train_size);
val_indices = indices(train_size+1:end);

X_train = X_norm(train_indices, :);
Y_train = Y(train_indices, 1);
X_val = X_norm(val_indices, :);
Y_val = Y(val_indices, 1);

% Training settings
epochs = 2*1e6; 
learning_rate = 0.2;
threshold = 0.004; % Loss threshold for early stopping

for i = 1:epochs
    % Forward propagation (training data)
    [a1_train, a2_train, a3_train, output_train] = forward_propagation(X_train, W1, b1, W2, b2, W3, b3, W_out, b_out);

    % Backpropagation
    [dW1, db1, dW2, db2, dW3, db3, dW_out, db_out] = backpropagation(X_train, Y_train, a1_train, a2_train, a3_train, output_train, W2, W3, W_out);

    % Update weights and biases
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    W3 = W3 - learning_rate * dW3;
    b3 = b3 - learning_rate * db3;
    W_out = W_out - learning_rate * dW_out;
    b_out = b_out - learning_rate * db_out;

    %% This part uses MSE
    % Calculate training and validation loss (MSE)
    loss_train = mean(mean((output_train - Y_train').^2));
    
    % Forward propagation (validation data)
    [~, ~, ~, output_val] = forward_propagation(X_val, W1, b1, W2, b2, W3, b3, W_out, b_out);
    loss_val = mean(mean((output_val - Y_val').^2));


    
    % Display losses at intervals
    if mod(i, 1000) == 0
        disp(['Epoch: ', num2str(i), ', Train Loss: ', num2str(loss_train), ', Val Loss: ', num2str(loss_val), ', lr: ', num2str(learning_rate)]);
        learning_rate = learning_rate *.97;
    end

    % Early stopping condition
    if loss_train < threshold && loss_val < threshold
        disp(['Stopping early at epoch ', num2str(i), ': Train Loss = ', num2str(loss_train), ', Val Loss = ', num2str(loss_val)]);
        break;
    end
end


