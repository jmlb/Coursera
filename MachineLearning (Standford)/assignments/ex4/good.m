for i = 1:m  
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
   %%%%%%%%%%%%%%%%  Perform forward  propagation %%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %layer 1 is the input
   thisX = [1, X(i,:)]; %add bias on input. this is a row 1*401
   vecY =(1:num_labels)'; %create a row vector 1-2-3-..10 - Transpose into col (10*1) for the classes
   digit = y(i); %actual digit value of example
   vecY = (vecY == digit); %logical arrays where for all indexes!=digit, vecY(i)=0, otherwise 1 

   
   %for activation value for layer2        
   z2 = Theta1 * thisX'; %Theta1 = 25*401 => thisX' = 401*1 ==> z2 = (25*401)*(401*1) = 25*1
   act2 = sigmoid(z2); %this is a vector 25*1
   act2biased = [1;act2]; %add the bias to activation vector a2biased=26*1 
   %activation value for layer3 and output
   z3 = Theta2 * act2biased;  %Theta2 = 10*26 ==> z3 = (10*26)*(26*1)=10*1 as expected
   a3 = sigmoid(z3); %activation value --> output (10*1)
   %Calculate JCost for this example
   %without regularization
   JIr = (1/m) * sum(- vecY.*log(a3)-(1-vecY).*log(1-a3));
   %with regularization: we need to sum up all the Theta2^2 and all
   %Theta1^2 without the bias Theta associated with the unit (first column
   %of Theta1 and Theta2
   J = J + JIr;
   
   % Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

%calculate the error on the output \delta^3
delta3 = a3 - vecY; %a3 is a (10*1) - vecY=(10*1) ==> delta3 is a (10*1)
%calculate the error on layer2 : hidden layer
%Theta2 is (10*26).
T1Unbiased = Theta1(:,2:end);  %Theta1 = 25*401 and T1Unbiased = (25*400)
T2Unbiased = Theta2(:,2:end);  % take out first column which has the Theta associated with a0^(2) T2unbiased is a (10*25)
delta2 = (T2Unbiased' * delta3) .* sigmoidGradient(z2);  %T2 unbiased (10*25) *delta3 is (10*1))=(25*1) and sigmoidGrad(z2)=25*1 ==> 25*1
Theta2Delta = Theta2Delta + delta3 * act2biased';    %Delta : delta3(10*1), act2biased(26*1)==> delta3*act2biased' is (10*1)*(1*26)=(10*26)
%Same dimension at Theta2
Theta1Delta = Theta1Delta + delta2 * thisX;    %delta2 is (25*1) and X is 1*401 ==> DeltaTheta1 is 25*401
%same dimension as Theta1
end