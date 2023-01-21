%% EE550 HW3 Q3: Iris Classification with four layers


%% Initialization
clear ; close all; clc

% Creating the dataset
dataset = load('iris.mat');
num_of_samples = 150;

%Mixing the input space
randIdxs = randperm(num_of_samples);
X_total = table2array(dataset.iris(randIdxs,1:4));
y_total = zeros(150,1);
species = string(table2array(dataset.iris(randIdxs,5)));

%Converting string to integer values.
y_total(species == 'Iris-setosa') = 1;
y_total(species == 'Iris-versicolor') = 2;
y_total(species == 'Iris-virginica') = 3;

%Seperating data points for training.
num_for_training = 125;
X = X_total(1:num_for_training,:);
y = y_total(1:num_for_training,:);

%NN Setup
input_layer_size = size(X,2);
first_hidden_layer_size = 10;
second_hidden_layer_size = 10;
output_layer_size = 3;
Theta1 = rand(first_hidden_layer_size, input_layer_size+1)*20 - 10;
Theta2 = rand(second_hidden_layer_size, first_hidden_layer_size+1)*20 - 10;
Theta3 = rand(output_layer_size, second_hidden_layer_size+1)*20 - 10;


%% Learning

%Configuration
alpha = 0.25;
momentum = 0.1;
num_iterations = 1000;
past_J = zeros(1,num_iterations);
Theta1_old_grad = 0;
Theta2_old_grad = 0;
Theta3_old_grad = 0;
lambda = 1.5;

%Learning the parameters via gradient descent algo.
for i = 1:num_iterations
    [J, Theta1_grad, Theta2_grad, Theta3_grad] = ...
                costFunction3_2(X, y, Theta1, Theta2, Theta3, lambda);
    
    %Updating parameters
    Theta1 = Theta1 - (alpha*Theta1_grad + momentum*Theta1_old_grad);
    Theta2 = Theta2 - (alpha*Theta2_grad + momentum*Theta2_old_grad);
    Theta3 = Theta3 - (alpha*Theta3_grad + momentum*Theta3_old_grad);
    
    Theta1_old_grad = Theta1_grad;
    Theta2_old_grad = Theta2_grad;
    Theta3_old_grad = Theta3_grad;
    past_J(i) = J;
end

fprintf("Plotting the cost function vs iteration index...\n");
figure;
plot(1:num_iterations, past_J);
title("Cost function vs Epoches");
xlabel("Epoch num.");
ylabel("Cost");

fprintf("Program paused. Press any key to continue...\n");
pause;

%% Prediction

%Getting datapoints for testing.
X_test = X_total(126:end, :);
Y_test = y_total(126:end, :);

%Feedforward Algo.
A1 = [ones(size(X_test,1), 1), X_test];
A2 = sigmoid(Theta1 * A1');
A2 = [ones(1, size(A2,2)); A2];
A3 = sigmoid(Theta2 * A2);
A3 = [ones(1, size(A3,2)); A3];
A4 = sigmoid(Theta3 * A3);
H = A4';

%Getting the max elt. in each row.
[~, p] = max(H, [], 2); 
predictions = ["Iris-Setosa" , "Iris-versicolor", "Iris-virginica"]; 
%Showing the output.
for i = 1:length(Y_test)
    fprintf("Predicted:%s - Actual:%s\n", predictions(p(i)),...
        predictions(Y_test(i)));
end

fprintf("Prediction accuracy: %.2f", mean(p == Y_test)*100);






