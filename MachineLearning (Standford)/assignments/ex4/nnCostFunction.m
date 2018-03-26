function [J, grad] = nnCostFunction(nn_params, ...
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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); 
%layer 1 is the input
   
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1)); %pre allocating to save time
Theta2_grad = zeros(size(Theta2));
J = 0;
Theta1Delta = zeros(size(Theta1_grad)); %(25*401)
Theta2Delta = zeros(size(Theta2_grad)); %(10*26)
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

Xbiased = [ones(m,1),X]; %(5000*401)
vecY = repmat((1:num_labels),m,1); % (5000*10) matrix where each row (1:10)
vecY = bsxfun(@eq,vecY,y); % logical on row (%y is the actual digit value of example)
%logical arrays where for all indexes!=digit, vecY(i)=0, otherwise 1
%%%% Vectorized form %%
z2 = Theta1 * Xbiased'; %Theta1=(25*401), Xbias'=(401*5000) ==> (25*5000)
act2 = sigmoid(z2); %(25*5000)
act2biased = [ones(1,m);act2]; %(add row of 1: (26*5000)
z3 = Theta2 * act2biased; %Theta2(10*26), act2bias(26*5000) ==> (10*5000)
a3 = sigmoid(z3); %(10*5000)
cost = (- vecY.*log(a3')-(1-vecY).*log(1-a3'));
   %with regularization: we need to sum up all the Theta2^2 and all
   %Theta1^2 without the bias Theta associated with the unit (first column
   %of Theta1 and Theta2
J = (1/m) * sum(cost(:));
%Regularization term
T1Unbiased = Theta1(:,2:end); %Theta1(25*401) and T1Unbiased(25*400)
T2Unbiased = Theta2(:,2:end); %Theta2(10*26) and T2Unbiased(10*25)
Theta1sqr = T1Unbiased.^2; % element wise multiplication to get the square of Theta
RegTheta1 = sum(Theta1sqr(:)); %unroll the matrix and sum up  
Theta2sqr = T2Unbiased.^2; % element wise multiplication to get the square of Theta
RegTheta2 = sum(Theta2sqr(:)); %unroll the matrix and sum up
J = J + lambda/(2*m) *(RegTheta1 + RegTheta2);  

%Calculate the error on the output delta3
delta3 = a3 - vecY'; %a3(10*5000) - vecY(5000*10) ==> transpose vecY ==>delta3(10,5000)
delta2 = (T2Unbiased'*delta3).*sigmoidGradient(z2); %(10*25)'=(25*10)*(10*5000)=(25*5000).*(25*5000)=(25*5000)

Theta2Delta = Theta2Delta + delta3 * act2biased';    %Delta : delta3(10*1), act2biased(26*1)==> delta3*act2biased' is (10*1)*(1*26)=(10*26)
%Same dimension at Theta2
% X(5000*401) ==> needs to convert to (1*401), where all the elements in
% the same column are added ==> this is similar to a for loop thru m with
% and using accumulator

Theta1Delta = Theta1Delta + delta2 * Xbiased;    %delta2 is (25*5000) and X is 5000*401 ==> DeltaTheta1 is 25*401
%same dimension as Theta1

gradRegT1 = ones(size(Theta1));
gradRegT1(:,1) = 0;
gradRegT1 = (lambda/m) * gradRegT1 .* Theta1;
gradRegT2 = ones(size(Theta2));
gradRegT2(:,1) = 0;
gradRegT2 = (lambda/m) * gradRegT2 .* Theta2;

Theta1_grad = Theta1Delta/m + gradRegT1; %Theta1Delta = (25*401)  No regularization
Theta2_grad = Theta2Delta/m + gradRegT2;

%Theta1_grad = Theta1Delta/m; %Theta1Delta = (25*401)  No regularization
%Theta2_grad = Theta2Delta/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
