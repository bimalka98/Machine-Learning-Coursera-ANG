function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
%Loop over lambda_vec with the following:
%% =========================================================================
      for i = 1:length(lambda_vec)
          lambda = lambda_vec(i);
          % Model is trained using regularization.
          theta = trainLinearReg(X, y, lambda);
          % We use the regularization parameter only when we train the
          % model
          % Once it is trained, in order to calculate training error or
          % cross validation error we need to neglect the regularization
          % and consider only the original cost function. This can be done
          % by simply sending lambda parameter zero into our regularized
          % cost function.
          [error_train(i),~] = linearRegCostFunction(X, y, theta, 0);
          [error_val(i),~] = linearRegCostFunction(Xval, yval, theta, 0); 

      end

end
