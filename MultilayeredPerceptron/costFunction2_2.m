function [J, Theta1_grad, Theta2_grad, Theta3_grad] = ...
                   costFunction2_2(X, y, Theta1, Theta2, Theta3, lambda)
    % costFunction2_2 implements the neural network cost function for a
    % three layer neural network which performs classification.                                                  
    %
    % It takes input, output, parameters and lambda and outputs the cost
    % and the gradient matrices.                                                      
    

    %% Initialization
    
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    Theta3_grad = zeros(size(Theta3));
    m = length(y); 
    num_of_labels = size(Theta3,1); 
    num_of_layers = 4;
    
    
    %% Feedforward Prop.
    A1 = [ones(size(X,1), 1), X];
    A2 = sigmoid(Theta1 * A1');
    
    A2 = [ones(1, size(A2,2)); A2];
    A3 = sigmoid(Theta2 * A2 );
    
    A3 = [ones(1, size(A3,2)); A3];
    A4 = sigmoid(Theta3 * A3 );
   
    H = A4'; 
    
    %Classifying output
    outputs = linspace(-1,1,num_of_labels+1);
    
    Y = zeros(m, num_of_labels);
    for i = 1:length(y)
        for j = 1:length(outputs)
            if y(i) >= outputs(j) && y(i) <= outputs(j+1)
                Y(i,j) = 1;
                break;
            end
        end
    end
  
    %Calculating cost
    J = -(1/m) * (sum(log(H).*Y, 'all') +...
                              sum(log(1-H).*(1-Y), 'all'));


    %% Computing gradient with backprop algo
    
    Del4 = H - Y;
    Del3 = (Del4 * Theta3)' .* A3 .* (1 - A3); 
    Del3 = Del3(2:end, :);
    
    Del2 = (Del3' * Theta2)' .* A2 .* (1 - A2);
    Del2 = Del2(2:end, :);
    
    
    Delta1 = Del2 * A1;
    Delta2 = Del3 * A2';
    Delta3 = (A3 * Del4)';
    

    Theta1_grad = 1/m * ([Delta1(:,1), Delta1(:,2:end) + ...
                                        lambda*Theta1(:,2:end)]);
    Theta2_grad = 1/m * ([Delta2(:,1), Delta2(:,2:end) + ...
                                        lambda*Theta2(:,2:end)]);  
    Theta3_grad = 1/m * ([Delta3(:,1), Delta3(:,2:end) + ...
                                        lambda*Theta3(:,2:end)]); 
    
end