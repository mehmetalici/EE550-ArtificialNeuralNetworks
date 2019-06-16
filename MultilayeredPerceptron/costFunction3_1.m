function [J, Theta1_grad, Theta2_grad] = ...
                   costFunction3_1(X, y, Theta1, Theta2, lambda, treshold)                                                      
    %costFunction3_1 implements the neural network cost function for a two
    %layer neural network which performs classification.

    %% Initialization
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    m = size(y,1); 
    num_labels = size(Theta2,1); 
    num_of_layers = 3;
    
    %Converting y from 1 output to 3 outputs. Eg: Let y(1) = 3, 
    %New y(1,:) becomes [0 0 1].
    y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
    
    %% Feedforward Prop.
    A1 = [ones(size(X,1), 1), X];
    z = Theta1 * A1';
    A2 = sigmoid(z - treshold);
    A2 = [ones(1, size(A2,2)); A2];
    A3 = sigmoid(Theta2 * A2 - treshold);
    
    H = A3'; 
    
    %Calculating cost fn.
    J = -(1/m) * (sum(log(H).*y, 'all') +...
                              sum(log(1-H).*(1-y), 'all'));
    
    %Regularizing, i.e punishing high order terms, 
    %to prevent from overfitting.
    t1 = Theta1(:, 2:end);
    t2 = Theta2(:, 2:end);
    Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / ...
                                                    (2*m);
	J = J + Reg;
    
    
    
    %% Computing gradient with backprop algo
        
    Del3 = H - y;
    Del2 = (Del3 * Theta2)' .* A2 .* (1 - A2);    
    Delta1 = Del2(2:end, :) * A1;
    Delta2 = (A2 * Del3)';
    
    
    Theta1_grad = 1/m * ([Delta1(:,1), Delta1(:,2:end) + ...
                                        lambda*Theta1(:,2:end)]);
    Theta2_grad = 1/m * ([Delta2(:,1), Delta2(:,2:end) + ...
                                        lambda*Theta2(:,2:end)]);
    

    
end