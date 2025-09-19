%% Training a NN to recognize Sat %%

%% HyperParameters
input_size = 6;
hidden_size1 = 5;
hidden_size2 = 4;
hidden_size3 = 3;
hidden_size4 = 2;
output_size = 1;
epochs = 2e4;
learning_rate = 0.0008; % Adam typically uses a smaller learning rate
batch_size = 64; % Example batch size 
beta1 = 0.9; % Adam parameters
beta2 = 0.999;
epsilon = 1e-8; 
threshold = 0.044; % Loss threshold for early stopping

%% Load Data
X_norm = X_augmented; 
Y = Y_augmented;

%Variable Xavier Glorot Initialization
% W1 = rand(hidden_size1, input_size) * sqrt(2 / (input_size + hidden_size1));
% b1 = zeros(hidden_size1, 1);
% W2 = rand(hidden_size2, hidden_size1) * sqrt(2 / (hidden_size1 + hidden_size2));
% b2 = zeros(hidden_size2, 1);
% W3 = rand(hidden_size3, hidden_size2) * sqrt(2 / (hidden_size2 + hidden_size3));
% b3 = zeros(hidden_size3, 1);
% W4 = rand(hidden_size4, hidden_size3) * sqrt(2 / (hidden_size3 + hidden_size4)); % New layer weights
% b4 = zeros(hidden_size4, 1); % New layer bias
% W_out = rand(output_size, hidden_size4) * sqrt(2 / (hidden_size4 + output_size)); 
% b_out = zeros(output_size, 1);

% Adam optimizer variables
mW1 = zeros(size(W1));
mb1 = zeros(size(b1));
vW1 = zeros(size(W1));
vb1 = zeros(size(b1));

mW2 = zeros(size(W2));
mb2 = zeros(size(b2));
vW2 = zeros(size(W2));
vb2 = zeros(size(b2));

mW3 = zeros(size(W3));
mb3 = zeros(size(b3));
vW3 = zeros(size(W3));
vb3 = zeros(size(b3));

mW4 = zeros(size(W4));  % For the new layer
mb4 = zeros(size(b4));  % For the new layer
vW4 = zeros(size(W4));  % For the new layer
vb4 = zeros(size(b4));  % For the new layer

mW_out = zeros(size(W_out));
mb_out = zeros(size(b_out));
vW_out = zeros(size(W_out));
vb_out = zeros(size(b_out)); 



%% Helper Functions
function y = relu(x)
  y = max(0, x);
end

% Forward propagation
function [a1, a2, a3, a4, output] = forward_propagation(X, W1, b1, W2, b2, W3, b3, W4, b4, W_out, b_out) 
    z1 = W1 * X' + b1;
    a1 = relu(z1);
    z2 = W2 * a1 + b2;
    a2 = relu(z2);
    z3 = W3 * a2 + b3;
    a3 = relu(z3);
    z4 = W4 * a3 + b4; % New layer forward pass
    a4 = relu(z4); 
    z_out = W_out * a4 + b_out;  % Output layer uses sigmoid
    output = sigmoid(z_out); 
end

% Sigmoid function (for output layer)
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Backpropagation 
function [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW_out, db_out] = backpropagation(X, Y, a1, a2, a3, a4, output, W2, W3, W4, W_out)
    m = size(X, 1);
    d_out = output - Y'; 
    dW_out = (1/m) * d_out * a4'; % Update for the new layer
    db_out = (1/m) * sum(d_out, 2);
    d4 = (W_out' * d_out) .* (a4 > 0); % Derivative of ReLU
    dW4 = (1/m) * d4 * a3';
    db4 = (1/m) * sum(d4, 2);
    d3 = (W4' * d4) .* (a3 > 0); 
    dW3 = (1/m) * d3 * a2';
    db3 = (1/m) * sum(d3, 2);
    d2 = (W3' * d3) .* (a2 > 0);
    dW2 = (1/m) * d2 * a1';
    db2 = (1/m) * sum(d2, 2);
    d1 = (W2' * d2) .* (a1 > 0);
    dW1 = (1/m) * d1 * X;
    db1 = (1/m) * sum(d1, 2);
end

%% Split data
train_ratio = 0.8;
data_size = size(X_norm, 1);
train_size = round(train_ratio * data_size);
indices = randperm(data_size);
train_indices = indices(1:train_size);
val_indices = indices(train_size+1:end);
X_train = X_norm(train_indices, :);
Y_train = Y(train_indices, 2);
X_val = X_norm(val_indices, :);
Y_val = Y(val_indices, 2);

%% Training loop
for i = 1:epochs

    train_indices = randperm(train_size);

    for j = 1:batch_size:size(X_train, 1) % Batching 
        X_batch = X_train(train_indices(j:min(j+batch_size-1, size(X_train, 1))), :);
        Y_batch = Y_train(train_indices(j:min(j+batch_size-1, size(X_train, 1))));

        % Forward propagation
        [a1_train, a2_train, a3_train, a4_train, output_train] = forward_propagation(X_batch, W1, b1, W2, b2, W3, b3, W4, b4, W_out, b_out);

        % Backpropagation
        [dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW_out, db_out] = backpropagation(X_batch, Y_batch, a1_train, a2_train, a3_train, a4_train, output_train, W2, W3, W4, W_out);

        % Adam optimizer updates
        % Update biased first moment estimate
        mW1 = beta1 * mW1 + (1 - beta1) * dW1;
        mb1 = beta1 * mb1 + (1 - beta1) * db1;
        mW2 = beta1 * mW2 + (1 - beta1) * dW2;
        mb2 = beta1 * mb2 + (1 - beta1) * db2;
        mW3 = beta1 * mW3 + (1 - beta1) * dW3;
        mb3 = beta1 * mb3 + (1 - beta1) * db3; 

        mW4 = beta1 * mW4 + (1 - beta1) * dW4;
        mb4 = beta1 * mb4 + (1 - beta1) * db4;
        mW_out = beta1 * mW_out + (1 - beta1) * dW_out;
        mb_out = beta1 * mb_out + (1 - beta1) * db_out;

        % Update biased second raw moment estimate
        vW1 = beta2 * vW1 + (1 - beta2) * dW1.^2;
        vb1 = beta2 * vb1 + (1 - beta2) * db1.^2;
        vW2 = beta2 * vW2 + (1 - beta2) * dW2.^2;
        vb2 = beta2 * vb2 + (1 - beta2) * db2.^2;
        vW3 = beta2 * vW3 + (1 - beta2) * dW3.^2;
        vb3 = beta2 * vb3 + (1 - beta2) * db3.^2;

        vW4 = beta2 * vW4 + (1 - beta2) * dW4.^2;
        vb4 = beta2 * vb4 + (1 - beta2) * db4.^2;
        vW_out = beta2 * vW_out + (1 - beta2) * dW_out.^2;
        vb_out = beta2 * vb_out + (1 - beta2) * db_out.^2;

        % Correct bias in first moment
        mW1_corrected = mW1 / (1 - beta1^i);
        mb1_corrected = mb1 / (1 - beta1^i);
        mW2_corrected = mW2 / (1 - beta1^i);
        mb2_corrected = mb2 / (1 - beta1^i);
        mW3_corrected = mW3 / (1 - beta1^i);
        mb3_corrected = mb3 / (1 - beta1^i);
        mW4_corrected = mW4 / (1 - beta1^i);
        mb4_corrected = mb4 / (1 - beta1^i);
        mW_out_corrected = mW_out / (1 - beta1^i);
        mb_out_corrected = mb_out / (1 - beta1^i);

        % Correct bias in second moment
        vW1_corrected = vW1 / (1 - beta2^i);
        vb1_corrected = vb1 / (1 - beta2^i);
        vW2_corrected = vW2 / (1 - beta2^i);
        vb2_corrected = vb2 / (1 - beta2^i);
        vW3_corrected = vW3 / (1 - beta2^i);
        vb3_corrected = vb3 / (1 - beta2^i);
        vW4_corrected = vW4 / (1 - beta2^i);
        vb4_corrected = vb4 / (1 - beta2^i);
        vW_out_corrected = vW_out / (1 - beta2^i);
        vb_out_corrected = vb_out / (1 - beta2^i);

        % Update weights and biases
        W1 = W1 - learning_rate * mW1_corrected ./ (sqrt(vW1_corrected) + epsilon);
        b1 = b1 - learning_rate * mb1_corrected ./ (sqrt(vb1_corrected) + epsilon);
        W2 = W2 - learning_rate * mW2_corrected ./ (sqrt(vW2_corrected) + epsilon);
        b2 = b2 - learning_rate * mb2_corrected ./ (sqrt(vb2_corrected) + epsilon);
        W3 = W3 - learning_rate * mW3_corrected ./ (sqrt(vW3_corrected) + epsilon);
        b3 = b3 - learning_rate * mb3_corrected ./ (sqrt(vb3_corrected) + epsilon);
        W4 = W4 - learning_rate * mW4_corrected ./ (sqrt(vW4_corrected) + epsilon);
        b4 = b4 - learning_rate * mb4_corrected ./ (sqrt(vb4_corrected) + epsilon);
        W_out = W_out - learning_rate * mW_out_corrected ./ (sqrt(vW_out_corrected) + epsilon);
        b_out = b_out - learning_rate * mb_out_corrected ./ (sqrt(vb_out_corrected) + epsilon);
    end

    % Calculate training and validation loss (e.g., MAE) 
    [~, ~, ~, ~, output_train] = forward_propagation(X_train, W1, b1, W2, b2, W3, b3, W4, b4, W_out, b_out);  
    loss_train = mean(abs(output_train - Y_train')); % MAE

    [~, ~, ~, ~, output_val] = forward_propagation(X_val, W1, b1, W2, b2, W3, b3, W4, b4, W_out, b_out);
    loss_val = mean(abs(output_val - Y_val')); % MAE

    % Display losses 
    if mod(i, 200) == 0
        disp(['Epoch: ', num2str(i), ', Train Loss: ', num2str(loss_train), ', Val Loss: ', num2str(loss_val)]);
    end

    % Early stopping condition
    if loss_train < threshold && loss_val < threshold
        disp(['Stopping early at epoch ', num2str(i), ': Train Loss = ', num2str(loss_train), ', Val Loss = ', num2str(loss_val)]);
        break; 
    end
end