%% EE550 HW3 Q1: Simulating the MLP model with XOR


%% Initialization
clear ; close all; clc

%Creating dataset for XOR
X = [ 0 0;
      0 1;
      1 0;
      1 1
     ];

y = xor(X(:,1),X(:,2));

%Initializing the NN
input_layer_size = size(X,2);
hidden_layer_size = 9;
output_layer_size = 1;
Theta1 = rand(hidden_layer_size, input_layer_size+1);
Theta2 = rand(output_layer_size, hidden_layer_size+1);


%% Learning

alpha = 0.25; %Step Size
momentum = 0.5; %Averaging out the cost
num_iterations = 2000;  

%Keeping track of the past J.
past_J = zeros(1,num_iterations);
Theta1_old_grad = 0;
Theta2_old_grad = 0;

%Learning the parameters with gradient descent algo
for i = 1:num_iterations
    [J, Theta1_grad, Theta2_grad] = costFunction1(X, y, Theta1, Theta2);
    
    Theta1 = Theta1 - (alpha*Theta1_grad + momentum*Theta1_old_grad);
    Theta2 = Theta2 - (alpha*Theta2_grad + momentum*Theta2_old_grad);
    Theta1_old_grad = Theta1_grad;
    Theta2_old_grad = Theta2_grad;
    
    past_J(i) = J;
end

fprintf("Plotting the cost function vs epoches...\n");
%Plotting the cost function vs epoches.
figure;
plot(1:num_iterations, past_J);
xlabel("Nr. of epoches");
ylabel("Cost");
title("Cost function vs Nr. of Iterations");

fprintf("Program paused. Press any key to continue...\n");
pause;
%% Prediction

fprintf("\nDoing predictions...\n");
%Feedforward prop.
A1 = [ones(size(X,1), 1), X];
A2 = sigmoid(Theta1 * A1');
A2 = [ones(1, size(A2,2)); A2];
A3 = sigmoid(Theta2 * A2);
H = A3';

%If sigmoid outputs > 0.5, then it's 1 and o.w.
for i = 1:length(H)
    if H(i) > 0.5
        H(i) = 1;
    else
        H(i) = 0;
    end
    
    fprintf("For x1: %d, x2: %d => Predicted: %d, Actual: %d\n",...
    X(i,1), X(i,2), H(i), y(i));
end

fprintf("Training accuracy: %.2f", mean(H == y)*100);



