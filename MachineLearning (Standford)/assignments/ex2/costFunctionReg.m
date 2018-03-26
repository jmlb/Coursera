function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
restrictMatrix = ones(size(theta));
restrictMatrix(1) = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

co = -y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta));
thetaSqr = (restrictMatrix.*theta).^2;
J=(1/m)*sum(co)+lambda/(2*m)*sum(thetaSqr);
grad = (1/m)*X'*(sigmoid(X*theta)-y)+(lambda/m)*restrictMatrix.*theta;

% =============================================================

end
