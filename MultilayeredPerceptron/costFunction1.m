function [J, Theta1_grad, Theta2_grad] = costFunction(X, y, Theta1, Theta2)
                                                      
    

    %% Cost function
    
    J = 0;
    Theta1_grad = zeros(size(Theta1, 2) - 1, 1);
    Theta2_grad = zeros(size(Theta2, 2) - 1, 1);
    num_of_samples = length(y); 
    num_of_labels = size(Theta2,1); 
    num_of_layers = 3;
    
    
    
    A1 = [ones(size(X,1), 1), X];
    A2 = sigmoid(Theta1 * A1' );
    A2 = [ones(1, size(A2,2)); A2];
    A3 = sigmoid(Theta2 * A2 );
    
    
    H = A3'; 
    
    
    Y = y;
  
    
    J = -(1/num_of_samples) * (sum(log(H).*Y, 'all') +...
                              sum(log(1-H).*(1-Y), 'all'));


    %% Computing gradient with backprop algo
    
    
    Del3 = H - Y;
    Del2 = (Del3 * Theta2)' .* A2 .* (1 - A2);    
    Delta1 = Del2(2:end, :) * A1;
    Delta2 = (A2 * Del3)';
    
    
    Theta1_grad = 1/num_of_samples * Delta1;
    Theta2_grad = 1/num_of_samples * Delta2;
    
    
                 
      
    
end