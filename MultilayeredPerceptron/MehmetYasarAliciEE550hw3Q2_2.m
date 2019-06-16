%% EE550 HW3 Q2_2: Function approximation - 4 Layer case


%% Initialization
clear ; close all; clc

% Creating Dataset
num_of_samples = 100;
samplePts = rand(num_of_samples,1)*2*pi;

%Precision of the digits after dot. 
precision = 5;
samples = num2str(samplePts, precision);

%Shaping the X Matrix for using in NN.
X = zeros(num_of_samples, precision);
for i = 1:size(samples,1)
    flag = 0;
    idx = 0;
    for j = 1:size(samples,2)
        if samples(i,j) == '.'
            X(i,1) = str2double(samples(i,j-1));
            idx = j;
            flag = 1;
            continue;
        end
        if flag == 1
            X(i,j-idx+1) = str2double(samples(i,j));
        end
    end
    
end
X = X(:,1:precision);
y = sin(samplePts);

%Initializing the NN parameters.
input_layer_size = size(X,2);
first_hidden_layer_size = 11;
second_hidden_layer_size = 11;
output_layer_size = 11;
Theta1 = rand(first_hidden_layer_size, input_layer_size+1)*20 - 10;
Theta2 = (rand(second_hidden_layer_size, first_hidden_layer_size+1)*20 - 10);
Theta3 = rand(output_layer_size, second_hidden_layer_size+1)*20 - 10;


%% Learning

%Optimization configuration and setup
alpha = 0.25;
momentum = 0.1;
num_iterations = 1000;
lambda = 1.5;
past_J = zeros(1,num_iterations);
Theta1_old_grad = 0;
Theta2_old_grad = 0;
Theta3_old_grad = 0;


%Finding the x* with gradient descent algorithm.
for i = 1:num_iterations
    [J, Theta1_grad, Theta2_grad, Theta3_grad] = ...
                    costFunction2_2(X, y, Theta1, Theta2, Theta3, lambda);
    
    Theta1 = Theta1 - (alpha*Theta1_grad + momentum*Theta1_old_grad);
    Theta2 = Theta2 - (alpha*(Theta2_grad) + momentum*Theta2_old_grad);
    Theta3 = Theta3 - (alpha*Theta3_grad + momentum*Theta3_old_grad);
    
    Theta1_old_grad = Theta1_grad;
    Theta2_old_grad = Theta2_grad;
    Theta3_old_grad = Theta3_grad;
    
    past_J(i) = J;
end

%Plotting
fprintf("Plotting the cost function vs nr of epoches...\n");
figure;
plot(1:num_iterations, past_J);
xlabel("Nr. of Iterations.");
ylabel("Cost");
title("Cost function vs. Number of Iterations");

fprintf("Program paused. Press any key to continue...\n");
pause;

%% Prediction

%Creating a dataset for testing.
samplePts = sort(rand(50,1)*2*pi);
y = sin(samplePts);
samples = num2str(samplePts, precision);

%Shaping the X matrix.
X = zeros(25, size(Theta1,1)+1);
for i = 1:size(samples,1)
    flag = 0;
    idx = 0;
    for j = 1:size(samples,2)
        if samples(i,j) == '.'
            X(i,1) = str2double(samples(i,j-1));
            idx = j;
            flag = 1;
            continue;
        end
        if flag == 1
            X(i,j-idx+1) = str2double(samples(i,j));
        end
    end
end
X = X(:,1:precision);


%Feedforward Prop.
A1 = [ones(size(X,1), 1), X];
A2 = sigmoid(Theta1 * A1');
A2 = [ones(1, size(A2,2)); A2];
A3 = sigmoid(Theta2 * A2);
A3 = [ones(1, size(A3,2)); A3];
A4 = sigmoid(Theta3 * A3);
H = A4'; 

%Getting the idxs of max elts.
[~, p] = max(H, [], 2); 

%Classifying output
outputs = linspace(-1,1,output_layer_size);

%Plotting the approximate and actual curves.
fprintf("Plotting the approximate and actual curves...\n");
figure;
plot(samplePts,y);
hold on;
plot(samplePts, outputs(p)); 
legend("Actual","Approx.");
title("Actual and approximated sine curves");
xlabel("Value of x in radians");
ylabel("Sin(x)");






