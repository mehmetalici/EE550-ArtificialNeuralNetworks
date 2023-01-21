%% EE550 HW3 Q3: Iris Classification


%% Initialization
clear ; close all; clc

% Creating Dataset
dataset = load('iris.mat');
num_of_samples = 150;

%Mixing the dataset. 
randIdxs = randperm(num_of_samples);
X_total = table2array(dataset.iris(randIdxs,1:4));
Y_total = zeros(150,1);
species = string(table2array(dataset.iris(randIdxs,5)));


Y_total(species == 'Iris-setosa',:) = 1;
Y_total(species == 'Iris-versicolor', :) = 2;
Y_total(species == 'Iris-virginica', :) = 3;


num_for_training = 125;
X = X_total(1:num_for_training,:);
Y = Y_total(1:num_for_training,:);

input_layer_size = size(X,2);
hidden_layer_size = 10;
output_layer_size = 3;
Theta1 = rand(hidden_layer_size, input_layer_size+1)*250 - 125;
Theta2 = rand(output_layer_size, hidden_layer_size+1)*250 - 125;


%% Learning

%Setup
alpha = 0.25;
momentum = 0.25;
num_iterations = 1000;
past_J = zeros(1,num_iterations);
Theta1_old_grad = 0;
Theta2_old_grad = 0;
lambda = 1.5;
treshold = 0;

%Learning the network parameters with gradient descent
for i = 1:num_iterations
    [J, Theta1_grad, Theta2_grad] = ...
                    costFunction3_1(X, Y, Theta1, Theta2, lambda, treshold);
                        
    Theta1 = Theta1 - (alpha*Theta1_grad + momentum*Theta1_old_grad);
    Theta2 = Theta2 - (alpha*Theta2_grad + momentum*Theta2_old_grad);
    
    Theta1_old_grad = Theta1_grad;
    Theta2_old_grad = Theta2_grad;
    past_J(i) = J;
end
fprintf("Plotting the cost function vs iteration index...\n");
figure;
plot(1:num_iterations, past_J);
xlabel("Epoches");
ylabel("Cost");
title("Cost Function vs Epoches");

fprintf("Program Paused. Press any key to continue...\n");
pause;

%% Prediction
X_test = X_total(126:end, :);
Y_test = Y_total(126:end, :);



A1 = [ones(size(X_test,1), 1), X_test];
A2 = sigmoid(Theta1 * A1' - treshold);
A2 = [ones(1, size(A2,2)); A2];
A3 = sigmoid(Theta2 * A2 - treshold);
H = A3';
[~, p] = max(H, [], 2); 
predictions = ["Iris-Setosa" , "Iris-versicolor", "Iris-virginica"]; 
fprintf("Test results...\n");
for i = 1:length(Y_test)
    fprintf("Predicted:%s - Actual:%s\n", predictions(p(i)),...
        predictions(Y_test(i)));
end

fprintf("Training accuracy: %.2f", mean(p == Y_test)*100);






