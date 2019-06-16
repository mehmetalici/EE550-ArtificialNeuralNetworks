function visualize(X1, X2, E, eq_pts1, eq_pts2)

    %Initializing level sets.
    levels = [0.449, 0.156, 0.017, -0.003, -0.023, -0.041];
    % Plotting the contours with equilibrium points.
    contour(X1,X2,E, levels);
    hold on;
    sz = 90;
    LineWidth = 2.8;
    scatter(eq_pts1(1), eq_pts2(1), sz ,'x', 'MarkerEdgeColor',[0 .5 .5],...
                  'MarkerFaceColor',[0 .7 .7],...
                  'LineWidth',LineWidth);
    scatter(eq_pts1(3), eq_pts2(3), sz, 'x', 'MarkerEdgeColor',[0 .5 .5],...
                  'MarkerFaceColor',[0 .7 .7],...
                  'LineWidth',LineWidth);
    scatter(eq_pts1(2), eq_pts2(2), 'o', 'MarkerEdgeColor',[0 .5 .5],...
                  'LineWidth',1.5);

end