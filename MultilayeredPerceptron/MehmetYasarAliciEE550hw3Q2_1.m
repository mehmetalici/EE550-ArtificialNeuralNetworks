%% EE550 HW3 Q2: Function approximation - 3 Layer case

%% Initialization
clear ; close all; clc

% Creating dataset
num_of_samples = 100;
samplePts = rand(num_of_samples,1)*2*pi;
precision = 5;
samples = num2str(samplePts, precision);

%Creating the input matrix X. Each digit were counted as a feature. 
% Eg: let x = 3.1416, then x(1)=3, x(2)=1, x(3)=4... x(5)=6. 
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

% Initializing the NN
input_layer_size = size(X,2);
hidden_layer_size = 12;
output_layer_size = 12;
Theta1 = rand(hidden_layer_size, input_layer_size+1)*10 - 5;
Theta2 = rand(output_layer_size, hidden_layer_size+1)*10 - 5;


%% Learning

%Configuration and Initialiation
alpha = 0.5;
momentum = 0.1;
lambda = 2;
num_iterations = 1000;
past_J = zeros(1,num_iterations);
Theta1_old_grad = 0;
Theta2_old_grad = 0;

% Learning the parameters with gradient descent algo
for i = 1:num_iterations
    [J, Theta1_grad, Theta2_grad] = ...
                            costFunction2_1(X, y, Theta1, Theta2, lambda);
                            
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
xlabel("Epoches");
ylabel("Cost");
title("Cost over Nr. of Iterations");
fprintf("Program paused. Press any key to continue...\n");
pause;

%% Prediction


%Creating data for testing
samplePts = sort(rand(50,1)*2*pi);
y = sin(samplePts);
samples = num2str(samplePts, precision);

%Creating an input matrix X with the same logic 
%when creating for training. 
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




%Feedforwarding NN
A1 = [ones(size(X,1), 1), X];
A2 = sigmoid(Theta1 * A1');
A2 = [ones(1, size(A2,2)); A2];
A3 = sigmoid(Theta2 * A2);
H = A3';

%Getting the index of the max. elt. 
[~, p] = max(H, [], 2); 

%Classifying the approximations 
outputs = linspace(-1,1,output_layer_size);

%Plotting
fprintf("\nPlotting the approximated and actual sine...\n");
figure;
plot(samplePts,y); %Actual plot
hold on;
plot(samplePts, outputs(p)); %Approximated plot
legend("Actual", "Approx.");
xlabel("x in radians");
ylabel("Sin(x)");
title("Actual and Approximated Sine Curves");






