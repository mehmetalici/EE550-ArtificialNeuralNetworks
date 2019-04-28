%% Initialization
clear ; close all; clc

%% ==================== Creating the dataset ====================

%Creating positive and negative sample data points in 3D space.
pos_samples = [rand(50,3) * 100, ones(50,1)];
neg_samples = [rand(50,3) * -100, -ones(50,1)];
samples = [pos_samples; neg_samples];




%% ==================== Plotting Dataset  ====================
fprintf('Visualizing the dataset...\n');


color_positive = [0 0 1]; %Blue
color_negative = [1 0 0]; %Red

figure;
%Plotting the positive samples with blue.
scatter3(pos_samples(:,1),pos_samples(:,2),pos_samples(:,3),...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor', color_positive,...
    'DisplayName', 'Positive');
hold on;
%Plotting the negative samples with red.
scatter3(neg_samples(:,1),neg_samples(:,2),neg_samples(:,3),...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor', color_negative,...
    'DisplayName', 'Negative');
zlim([-100 100]);
title('Dataset');
lgd = legend;
lgd.FontSize = 11;
lgd.Title.String = 'Samples';
fprintf('Program paused. Press enter to continue...\n');
pause;


%% ==================== Training ====================

%Creating random 3x1 weight vector between (-100,100)
w = -150 + rand(3,1) * 300; 

eta = 0.5; %Learning coeff.
theta = 0; %Bias Point

%Selecting 40 samples from each specimen for training.
sampleAmount = 80;
sample_train = [pos_samples(1:sampleAmount/2,:);...
    neg_samples(1:sampleAmount/2,:)];

%Initializing the cost function
J = zeros(sampleAmount, 1);

for i = 1:sampleAmount
    %Selecting the sample and its output.
    x = sample_train(i,1:3)'; 
    d = sample_train(i,4); 
    
    %Prediction
    y = sign(w' * x - theta); 
    y(y == 0) = 1; 
    
    %Updating the weights
    w = w + eta * (d - y) * x; 

    %Calculating the cost function
    e = sample_train(:,4) - sign(sample_train(:,1:3)*w);
    J(i) = 1 / 2 * (e' * e);
end




%% ==================== Plotting the decision plane ====================
fprintf('\nPlotting the decision plane...\n');
hold on; 
%Creating linearly sampled points between [-100, 100]

[x1, x2] = meshgrid(-100:1:100); 
%Solving for x3
x3 = -w(1)/w(3)*x1 - w(2)/w(3)*x2;

%Plotting
p = surf(x1,x2,x3,...
    'DisplayName','Boundary');
lgd = legend;
lgd.Title.String = '';


title('Dataset with Decision Boundary');
fprintf('Program paused. Press enter to continue...\n');
pause;
%% ==================== Testing the Model ====================

%Selecting 10 samples from each specimen for testing.
sample_test = [pos_samples(41:50,:); neg_samples(41:50,:)];

fprintf('\nTesting the algorithm by 20 samples...\n')
X = sample_test(:,1:3);
y = sign(X * w); 
y(y == 0) = 1;

%You can manually compare them.
fprintf('\nTest completed. Training accuracy: %.2f\n', mean(double(y == sample_test(:,4))*100));
fprintf('Program paused. Press enter to continue...\n');
pause;


fprintf('\nVisualizing 5 predictions...\n');
figure;

%Number of samples to be visualized
sampleNum = 5;

%Selects the samples by their indexes.
X = sample_test(randperm(20,sampleNum),1:3);
y = sign(X * w);
y(y == 0) = 1;

X_pos = X(y > 0, :);
X_neg = X(y < 0, :);


scatter3(X_pos(:,1), X_pos(:,2), X_pos(:,3),...
            'MarkerEdgeColor', 'k',...
            'MarkerFaceColor', color_positive);
hold on;
scatter3(X_neg(:,1), X_neg(:,2), X_neg(:,3),...
            'MarkerEdgeColor', 'k',...
            'MarkerFaceColor', color_negative);
hold on;
plot3(x1,x2,x3);
lgd = legend('Positive','Negative');
lgd.FontSize = 10;
lgd.Title.String = 'Prediction';


title('Prediction of Five Data Points');
xlim([-100 100]);
ylim([-100 100]);
zlim([-100 100]);

fprintf('Program paused. Press enter to continue...\n');
pause;

%% ========== Plotting the cost function vs iteration index ==========

fprintf('\nPlotting the cost function...\n');
figure;
plot(1:sampleAmount, J);
title('Cost Function');
xlabel('Iteration Index');
ylabel('Cost');
fprintf('Program finished. Press enter to exit.');
pause;








