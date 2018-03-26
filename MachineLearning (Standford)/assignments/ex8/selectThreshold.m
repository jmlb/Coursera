function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
combo = zeros(size(yval, 1), 2)
combo(:, 1) = yval
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    predictions = (pval < epsilon);
    combo(:, 2) = predictions;
    %Precision: select all rows where first column = 1 (true positive) or 0 (true negative).
    positive_samples = combo(combo(:,1)==1, : ); %samples true label is positive (anomalies)
    tp = sum(positive_samples(:, 2), 1) %sum all rows of predictions ==> gives total number of samples classified as psotive
    negative_samples = combo(combo(:,1)==0, :); %samples true label is negative (not anomalies)
    fp = sum(negative_samples(:, 2), 1) %sum all rows of predictions
    prec = tp/(tp + fp)
    
    %recall:
    fn = size(positive_samples,1) - tp %fn = ground truth says it's an anomaly(1) but our algorithm says it's not (0)
    rec = tp/(tp + fn)
    
    F1 = (2 * prec * rec)/(prec + rec )










    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
