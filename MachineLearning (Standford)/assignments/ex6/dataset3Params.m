function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = zeros(length(list)^2, 3);
trial = 1;
for jj = 1:length(list)
    C = list(jj)
    for kk = 1:length(list)
        sigma = list(kk)        
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        errorPredict = mean(double(predictions ~= yval))
        results(trial,:) = [C, sigma, errorPredict];
        trial = trial + 1;
    end
end
[minVal, indexMin] = min(results(:,3));
C = results(indexMin, 1)
sigma = results(indexMin, 2)


% =========================================================================

end
