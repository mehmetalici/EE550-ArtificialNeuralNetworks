function [J, Theta1_grad, Theta2_grad, Theta3_grad] = ...
                   costFunction_multi_3(X, y, Theta1, Theta2, Theta3,lambda)
                                                      
    

    %% Cost function
    
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    Theta3_grad = zeros(size(Theta3));
    m = length(y); 
    num_labels = size(Theta3,1); 

    
    A1 = [ones(size(X,1), 1), X];
    A2 = sigmoid(Theta1 * A1');
    
    A2 = [ones(1, size(A2,2)); A2];
    A3 = sigmoid(Theta2 * A2 );
    
    A3 = [ones(1, size(A3,2)); A3];
    A4 = sigmoid(Theta3 * A3 );
   
    
    H = A4'; 
    

    y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
    
    J = -(1/m) * (sum(log(H).*y, 'all') +...
                              sum(log(1-H).*(1-y), 'all'));

    t1 = Theta1(:, 2:end);
    t2 = Theta2(:, 2:end);
    Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / ...
                                                    (2*m);
	J = J + Reg;
    %% Computing gradient with backprop algo
    
    Del4 = H - y;
    Del3 = (Del4 * Theta3)' .* A3 .* (1 - A3); 
    %Del3 = Del3(2:end, :);
    
    Del2 = (Del3' * Theta2')' .* A2(2:end,:) .* (1 - A2(2:end,:));
    %Del2 = Del2(2:end, :);
    
    
    Delta1 = Del2 * A1;
    Delta2 = Del3 * A2';
    Delta3 = (A3 * Del4)';
    
    Theta1_grad = 1/m * Delta1;
    
    Theta2_grad = 1/m * Delta2(2:end, :);
    
    Theta3_grad = 1/m * Delta3;
    

    Theta1_grad(2:end,:) = Theta1_grad(2:end,:) + ...
                                lambda/m * Theta1(2:end,:);
    Theta2_grad(2:end,:) = Theta2_grad(2:end,:) +...
                                lambda/m * Theta2(2:end,:);             
    Theta3_grad(2:end,:) = Theta3_grad(2:end,:) + ...
                                lambda/m * Theta3(2:end,:);  
    
end