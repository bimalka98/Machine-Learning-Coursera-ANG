function [J ,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

%% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Setup some useful variables

m = size(X, 1);

%% Part 1: Feedforward the neural network
%         and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%% Layer 2 calculations
X = [ones(m,1) X];
z_2 = X*Theta1';
a_2 = sigmoid(z_2);

%a_2 = sigmoid([ones(m,1) X]*Theta1');

%% Layer 3 calculations
a_2 = [ones(m,1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

%a_3 = sigmoid([ones(m,1) a_2] * Theta2');

%% Activation of last layer is the hypothesis
h_x = a_3;

% Our label vector consisits of numbers 1-10, we need to convert them into
% loigical arrays, it is done through a for loop. first colmn corresponds to
% the 1, second column --> 2.
Y = zeros(size(h_x));
for num = 1: num_labels
    Y(:,num) = (y == num);
end

%% Calculating cost unregularized

% J = -(1/m)*sum(sum((Y.*log(h_x) + (1-Y).*log(1-h_x))));

%% Calculating cost regularized
% First we need to neglect the parameters for bias term 1; we don't regularize
% theta correspond to bias terms.
reg_t1 = Theta1(:,(2:end));
reg_t2 = Theta2(:,(2:end));

J = (-(1/m)*sum(sum((Y.*log(h_x) + (1-Y).*log(1-h_x))))) ...
    + (lambda/(2*m))*sum([sum(reg_t1.^2)  sum(reg_t2.^2)]);


%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the first time.

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

%% Gradients with respect to theta2

delta3 = a_3 - Y;
Delta2 = Delta2 + delta3'*a_2;
Theta2_grad = (1/m)*Delta2;

%% Gradients with respect to theta1

delta2 = delta3*Theta2(:,(2:end)).*sigmoidGradient(z_2);
Delta1 = Delta1 + delta2'*X;
Theta1_grad = (1/m)*Delta1;

%%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
%% Regularizing the gradients
% First column of the network parameter Theta will not be regularized
% as they are corresponds to the bias term '1'. Therefore first column of
% theta is replaced by a column vector of zeros.

Theta2_grad = Theta2_grad +(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,(2:end))];
Theta1_grad = Theta1_grad + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,(2:end))];


%% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
