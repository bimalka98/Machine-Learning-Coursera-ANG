function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values

m = length(y); % number of training examples


%% Caculating hypothesis

h_x = X*theta;

%% Calculating  Regularized cost

% Since we do nor regularize the theta_0 term. ie theta(1) we have neglect
% that term. This  can be done by indexing the theta vector from its 2nd
% value to end.
J = (1/(2*m))* sum(( h_x - y).^2) + (lambda/(2*m))*sum(theta(2:end).^2);

%% Calculating gradients in order to use in gradient descent algorithms

% First take the gradient considering the regularization for all theta
% values.
% But  we do not regularize the theta_0.
% Therefore this expression is valid only to the theta(2:end)

grad = (1/m)*X'*(h_x - y) + (lambda/m)*theta;

% Therefore we need to undo what we did to theta_0.
grad(1) = grad(1) - (lambda/m)*theta(1);

% =========================================================================

grad = grad(:);

end
