% EE550 HW4

%% Initialization
clear ; close all; clc


%% Generating 3-D Dataset

% Creating 3 Clusters from spherical coordinates
rng('shuffle');
rvals = 0.5*rand(30,1)-0.2;
elevation = asin(rvals);
azimuth_reg1 = pi/5*rand(30,1);
azimuth_reg2 = 2*pi/3 + pi/5*rand(30,1);
azimuth_reg3 = 4*pi/3 + pi/5*rand(30,1);

%Converting to cartesian
[x1,y1,z1] = sph2cart(azimuth_reg1,elevation,1);
[x2,y2,z2] = sph2cart(azimuth_reg2,elevation,1);
[x3,y3,z3] = sph2cart(azimuth_reg3,elevation,1);
[x4,y4,z4] = sphere ;




%% Plotting the sample vectors
fprintf("Plotting the sample vectors with random weights...\n");
figure
plot3(x1,y1,z1,'.', 'Color', 'red')
hold on;
plot3(x2,y2,z2,'.', 'Color', 'blue')
plot3(x3,y3,z3,'.', 'Color', 'green')

surf(x4,y4,z4,'FaceAlpha',0.05, 'LineStyle', 'none');
axis equal



%% Creating random normalized weight vectors

lineWidth = 5;
area = 200;
cross1_color = 'cyan';
cross2_color = 'yellow';
cross3_color = 'magenta';

w1 = .02*rand(3,1)-.01;
w1 = w1 ./ norm(w1);

w2 = .02*rand(3,1)-.01;
w2 = w2 ./ norm(w2);

w3 = .02*rand(3,1)-.01;
w3 = w3 ./ norm(w3);


%Plotting the initial random weights
scatter3(w1(1),w1(2),w1(3), area, 'x',...
    'LineWidth', lineWidth, 'MarkerEdgeColor', cross1_color);
scatter3(w2(1),w2(2),w2(3), area, 'x',...
    'LineWidth', lineWidth, 'MarkerEdgeColor', cross2_color);
scatter3(w3(1),w3(2),w3(3), area, 'x',...
    'LineWidth', lineWidth, 'MarkerEdgeColor', cross3_color);

title("Sample vectors with random weights");

fprintf("Program paused. Press any key to continue...\n\n");
pause;


%% Learning algorithm for winner take-all network

%Preparing the variables
w = [w1, w2, w3];
zeta_total = [x1(1:27), y1(1:27), z1(1:27); x2(1:27), y2(1:27), z2(1:27);...
    x3(1:27), y3(1:27), z3(1:27)];
zeta_total(randperm(length(zeta_total)));
w_past = zeros(81,3,3);
eta = 0.25;

%Learning
for i = 1:81
    %Keep track of past data
    w_past(i,:,:) = w;
    o = zeros(3,1);
    zeta = zeta_total(i, :);
    
    %Calculating distances
    dist1 = norm(zeta' - w(:,1));
    dist2 = norm(zeta' - w(:,2));
    dist3 = norm(zeta' - w(:,3));
   
    %Taking their minimum idx
    [~, winnerIdx]= min([dist1 dist2 dist3]);
    o(winnerIdx) = 1;
    
    %Updating only the winner
    w = w + eta * (o' .* (zeta' - w)); 
      
end



%% Plotting weights after learning
fprintf("Plotting trajectories on the same graph...\n");

%Plotting trajectories
plot3(w_past(:,1,1),w_past(:,2,1),w_past(:,3,1),...
    'Color', cross1_color, 'LineWidth', 1);
plot3(w_past(:,1,2),w_past(:,2,2),w_past(:,3,2),...
    'Color', cross2_color, 'LineWidth', 1);
plot3(w_past(:,1,3),w_past(:,2,3),w_past(:,3,3),...
    'Color', cross3_color,'LineWidth', 1);

%Plotting the converged weights
scatter3(w(1,1),w(2,1),w(3,1), area, 'x',...
'LineWidth', lineWidth, 'MarkerEdgeColor', cross1_color);
scatter3(w(1,2),w(2,2),w(3,2), area, 'x',...
'LineWidth', lineWidth, 'MarkerEdgeColor', cross2_color);
scatter3(w(1,3),w(2,3),w(3,3), area, 'x',...
'LineWidth', lineWidth, 'MarkerEdgeColor', cross3_color);

title("Weight vectors with convergence trajectories");

fprintf("Program paused. Press any key to continue...\n\n");
pause;


%% Testing the algorithm

fprintf("Testing the algorithm with 9 samples...\n");

%Getting the samples
zeta_t = [x1(28:end), y1(28:end), z1(28:end); x2(28:end), y2(28:end),...
    z2((28:end)); x3(28:end), y3(28:end), z3(28:end)];

%Plotting the weights
figure;
scatter3(w(1,1),w(2,1),w(3,1), area, 'x',...
'LineWidth', lineWidth, 'MarkerEdgeColor', cross1_color);
hold on;
scatter3(w(1,2),w(2,2),w(3,2), area, 'x',...
'LineWidth', lineWidth, 'MarkerEdgeColor', cross2_color);
scatter3(w(1,3),w(2,3),w(3,3), area, 'x',...
'LineWidth', lineWidth, 'MarkerEdgeColor', cross3_color);
surf(x4,y4,z4,'FaceAlpha',0.05, 'LineStyle', 'none');



for i = 1:9
    
    %Getting the net input
    h = zeta_t(i,:) * w; 
    
    %Getting the index of the most probable cluster 
    [~, prediction] = max(h);
    
    %Based on the prediction, assigning the colors of the weights.
    switch prediction
        case 1
            color = cross1_color;
        case 2
            color = cross2_color;
        case 3
            color = cross3_color;
    end
    
    %Plotting the new data points, with assigned colors.
    plot3(zeta_t(i,1), zeta_t(i,2), zeta_t(i,3), '.', 'Color', color,...
        'MarkerSize', 20);
    
end
title("Test results with 3 samples from each cluster");

fprintf("Program ended. Press any key to exit...\n\n");
pause;
close all;

















