%% EE550 Project 5: Continuous Type Hopfield Model
    % This project provides insights the energy contours, ...
    % equilibrium points and the effect of lambda by simulations. 

%% Initialization
clear ; close all; clc

T = [0 1; 1 0];

x1 = -1:0.001:1;
x2 = -1:0.001:1;


l = 1.4; %Initial Lambda


%% 1. Energy Contour Maps along with Equilibrium Points

%Governing D.Es in state space form 
x1_dot = -x1 + g(x2, l);
x2_dot = -x2 + g(x1, l);


%Finding equilibrium points by roots of f'(x) = 0.
idx1 = find(x1_dot < 1e-4 & x1_dot > -1e-4);
idx2 = find(x2_dot < 1e-4 & x2_dot > -1e-4);
eq_pts1 = x1(idx1);
eq_pts2 = x2(idx2);

fprintf("Equilibrum Points:\n");
fprintf("xphi_1: [%.3f %.3f]'\n", eq_pts1(1), eq_pts2(1));
fprintf("xphi_2: [%.3f %.3f]'\n", eq_pts1(2), eq_pts2(2));
fprintf("xphi_3: [%.3f %.3f]'\n", eq_pts1(3), eq_pts2(3));

[X1,X2] = meshgrid(x1,x2);

fprintf("Program paused. Press any key to continue...\n");
pause;

fprintf("\nCalculating the Energy Contours...");

E = zeros(size(X1));
%Finding the Energy Contours
for i = 1:length(X1)
    for j = 1:length(X2)
        x = [X1(i,j), X2(i,j)]';
        E(i,j) = -x(1)*x(2) + sum(g_inv_int(x, l));
    end    
end
fprintf("\nDone!\n");
fprintf("\nPlotting the Energy Contours with equilibrium points...\n");
visualize(X1, X2, E, eq_pts1, eq_pts2);
title("Energy Contours with Equilibrium Points");

fprintf("Program paused. Press any key to continue...\n");
pause;

%% 2. Convergence to the State Equilibrium Point
trajectory = zeros(100,2);
eq_pt = [eq_pts1(3), eq_pts2(3)]';

%Initializing a close state.
x = eq_pt + 0.25;

%First iteration
trajectory(1,:) = x;
v = g(T*x, l);
i = 2;
trajectory(2,:) = v;

%Calculating distance
d = dist(v, eq_pt);

%Iterate until convergence is achieved
while d > 1e-3
    i = i + 1;
    trajectory(i,:) = v;
    v = g(T*v, l);
    d = dist(v, eq_pt); 
end

fprintf("\nPlotting the convergence for the initial state ");
% Plotting the state trajectory
figure;
visualize(X1, X2, E, eq_pts1, eq_pts2);
title("Convergence of a State to the Stable Equilibrium Point");


fprintf("x0 = [%.2f %.2f]'...\n", x(1), x(2));

scatter(x(1), x(2), 'x', 'MarkerEdgeColor','r',...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1);
plot(trajectory(1:i,1), trajectory(1:i,2), '-.', 'Color','r');

fprintf("Program paused. Press any key to continue...\n");
pause;
%% 3. Movement of the stable equilibrium Points with increasing lambda

%Iterate for 100 times.
max_steps = 100;

%Initializing the trajectories.
trajectory_eq1 = zeros(max_steps,2);
trajectory_eq2 = zeros(max_steps,2);

%Initilalizing the equilibrium points.
eq_pt1 = eq_pt;
eq_pt2 = -eq_pt1; 

for i = 1:max_steps
    %Increasing Lambda
    l = l + 0.1;
    
    %Obtaining the dynamics of the whole space.
    x1_dot = -x1 + g(x2, l);
    x2_dot = -x2 + g(x1, l);
    
    %Getting the stable eq. pts' rate of changes.
    eq_pt1_1_dot = -eq_pt1(1) + g(eq_pt1(2), l);
    eq_pt1_2_dot = -eq_pt1(2) + g(eq_pt1(1), l);
    
    eq_pt2_1_dot = -eq_pt2(1) + g(eq_pt2(2), l);
    eq_pt2_2_dot = -eq_pt2(2) + g(eq_pt2(1), l);
    
    %Updating the eq. pts.
    eq_pt1 = [eq_pt1(1) + eq_pt1_1_dot; eq_pt1(2) + eq_pt1_2_dot];
    eq_pt2 = [eq_pt2(1) + eq_pt2_1_dot; eq_pt2(2) + eq_pt2_2_dot];
    
    %Keeping track of the past data.
    trajectory_eq1(i,:) = [eq_pt1(1),eq_pt1(2)];
    trajectory_eq2(i,:) = [eq_pt2(1),eq_pt2(2)];   
    
end
fprintf("\nCalcuating the energy contours for only the final lambda = %.2f...", l);
% Calculating the Energy Contours for the latest lambda.
E = zeros(size(X1));

for i = 1:length(X1)
    for j = 1:length(X2)
        x = [X1(i,j), X2(i,j)]';
        E(i,j) = -x(1)*x(2) + sum(g_inv_int(x, l));
    end    
end

fprintf("\nDone!\n");
fprintf("\nPlotting the trajectories of the stable equilibrium points...\n");
%Plotting the final Energy Contours with lambda = 1.4 + 100*0.1 = 11.4.
figure;
contour(X1,X2,E);
hold on;

%Plotting the initial stable eq. pts' locus.
sz = 90;
sz_final = 150;
LineWidth = 2.8;
LineWidthFinal = 3.4;
colorInit = 'r';
colorFinal = 'b';
scatter(eq_pts1(1), eq_pts2(1), sz ,'x', ...
    'MarkerEdgeColor',colorInit,...
              'MarkerFaceColor',colorInit,...
              'LineWidth',LineWidth);
scatter(eq_pts1(3), eq_pts2(3), sz, 'x', 'MarkerEdgeColor',colorInit,...
              'MarkerFaceColor',colorInit,...
              'LineWidth',LineWidth);
scatter(eq_pts1(2), eq_pts2(2), 'o', 'MarkerEdgeColor',[0 .5 .5],...
              'LineWidth',1.5);

%Plotting the trajectories of the eq. pts.
plot(trajectory_eq1(1:max_steps,1), trajectory_eq1(1:max_steps,2),...
    '-.', 'Color','r');
plot(trajectory_eq2(1:max_steps,1), trajectory_eq2(1:max_steps,2),...
    '-.', 'Color','r');

%Plotting the final eq. pts.
scatter(eq_pt1(1), eq_pt1(2), sz_final ,'x', 'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor','b',...
              'LineWidth',LineWidthFinal);
scatter(eq_pt2(1), eq_pt2(2), sz_final, 'x', 'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor','b',...
              'LineWidth',LineWidthFinal);


fprintf("Program ended. Press any key to exit...");
pause;




