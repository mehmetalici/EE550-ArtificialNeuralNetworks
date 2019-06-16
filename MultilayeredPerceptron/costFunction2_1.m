function [J, Theta1_grad, Theta2_grad] = ...
                               costFunction2_1(X, y, Theta1, Theta2, lambda)
    % costFunction2_1 implements the neural network cost function for a two
    % layer neural network which performs classification.                                                  
    %
    % It takes input, output, parameters and lambda and outputs the cost
    % and the gradient matrices.
    
    %% Initializing
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    m = length(y); 
    num_of_labels = size(Theta2,1); 
    num_of_layers = 3;
    
    
    %% Feedforward Prop.
    A1 = [ones(size(X,1), 1), X];
    A2 = sigmoid(Theta1 * A1');
    A2 = [ones(1, size(A2,2)); A2];
    A3 = sigmoid(Theta2 * A2);
    
    H = A3'; 
    
    
    %For training the output was linearly spaced into the number of
    % output layer, which is 12. Then, the original sin(x) was classified
    % accordingly. Eg: Let sin(x) = 0.9432. The output will be 
    % Y(1,:) = [0 0 0 ..... 0 1], where size(Y, 1) = 12. 
    outputs = linspace(-1,1,13);
    
    Y = zeros(m, num_of_labels);
    for i = 1:length(y)
        for j = 1:length(outputs)
            if y(i) > outputs(j) && y(i) < outputs(j+1)
                Y(i,j) = 1;
                break;
            end
        end
    end
  
    %Calculating the cost function
    J = -(1/m) * (sum(log(H).*Y, 'all') +...
                              sum(log(1-H).*(1-Y), 'all'));

                          
    %Applying regularization to prevent from overfitting.                      
    t1 = Theta1(:, 2:end);
    t2 = Theta2(:, 2:end);
    Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / ...
                                                    (2*m);
	J = J + Reg;

    
    %% Computing gradient with backprop algo
    
    Del3 = H - Y;
    Del2 = (Del3 * Theta2)' .* A2 .* (1 - A2);    
    Delta1 = Del2 * A1;
    Delta1 = Delta1(2:end,:);
    
    Delta2 = (A2 * Del3)';
    
    Theta1_grad = 1/m * ([Delta1(:,1), Delta1(:,2:end) + ...
                                        lambda*Theta1(:,2:end)]);
    Theta2_grad = 1/m * ([Delta2(:,1), Delta2(:,2:end) + ...
                                        lambda*Theta2(:,2:end)]);
    
    
                 
      
    
end