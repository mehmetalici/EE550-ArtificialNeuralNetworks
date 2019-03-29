%% Initialization
clear ; close all; clc

%% Create letters A, B, C, D, E
lettersGrid(:,:,1) = [
    
    0 0 1 1 1 1 0 0
    0 0 1 0 0 1 0 0
    0 0 1 1 1 1 0 0
    0 1 1 1 1 1 1 0
    0 1 0 0 0 0 1 0
    1 1 0 0 0 0 1 1
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    ];

lettersGrid(:,:,2) = [
    
    0 1 1 1 1 1 1 0
    0 1 0 0 0 0 1 0
    0 1 0 0 0 0 1 0
    0 1 0 0 0 0 1 0
    0 1 1 1 1 1 1 0
    0 1 0 0 0 0 1 0 
    0 1 0 0 0 0 1 0
    0 1 1 1 1 1 1 0
    
    ];

lettersGrid(:,:,3) = [
    
    0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 
    0 0 1 1 1 1 0 0 
    0 0 1 0 0 0 0 0
    0 0 1 0 0 0 0 0
    0 0 1 1 1 1 0 0
    0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0
    
    ];

lettersGrid(:,:,4) = [
    
    0 0 0 0 0 0 0 0
    0 1 1 1 1 1 0 0
    0 1 0 0 0 0 1 0
    0 1 0 0 0 0 1 0
    0 1 0 0 0 0 1 0
    0 1 0 0 0 0 1 0
    0 1 1 1 1 1 0 0
    0 0 0 0 0 0 0 0
    
    ];

lettersGrid(:,:,5) = [
   
    0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 0
    1 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 0
    1 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 0
    0 0 0 0 0 0 0 0
    
   ];



%% Visualize the letters

fprintf("Visualizing the sample patterns...\n");

letCount = size(lettersGrid,3);

figure;
sgtitle("Sample Patterns A through E");
s = char(65:70); %Titles A through E of the subplots. 

for i = 1:letCount
    subplot(2,3,i);
    visualize(lettersGrid(:,:,i));
    title(s(i));
end

fprintf("Program paused. Press enter to continue...\n");
pause;

%% Convert the letters to vectors.
gridXDim = size(lettersGrid,1);
gridYDim = size(lettersGrid,2);

letVecLen = gridXDim * gridYDim;

letters = zeros(letVecLen, letCount);

for i = 1:letCount
    letters(:,i) = reshape(lettersGrid(:,:,i), letVecLen, 1);
end

letters(letters == 0) = -1; %Converting all the zeros to -1s.

%% Learning


T = letters * letters' - letCount .* eye(letVecLen); 


%% Creating the distorted input vectors and testing the algorithm

sigmaVals = [0.4 0.6 0.8]'; % Three different sigma values.

num_iter = [2 2 2]'; % Number of iterations corresponding to sigma values.

for i = 1:size(sigmaVals)
    noise = normrnd(0,sigmaVals(i),letVecLen,5);
    inputs = letters + noise;


    %% visualize the input

    fprintf("\nDisplaying the distorted inputs with sigma %0.1f...\n" , sigmaVals(i));
    
    figure;
    tit = sprintf("Distorted Inputs with sigma %.1f\n", sigmaVals(i));
    sgtitle(tit);
    s = char(65:70);
    for j = 1:letCount
        subplot(2,3,j);
        visualize(reshape(inputs(:,j),size(lettersGrid,1),size(lettersGrid,2)));
        title(s(j));
    end

    fprintf("Program paused. Press enter to continue...\n");
    pause;
    
    %% Recognition
    fprintf("\nDisplaying the iteration results...\n");

    for k = 1:num_iter(i)
        inputs = sign(T * inputs);
        inputs(inputs == 0) = 1;

        figure;
        tit = sprintf("Output Iteration %d of %d (sigma %.1f)", k, num_iter(i), sigmaVals(i));
        sgtitle(tit);
        s = char(65:70);
        
        for l = 1:letCount 
            subplot(2,3,l);
            visualize(reshape(inputs(:,l),gridXDim, gridYDim));
            title(s(l));
        end
 
    end
    fprintf("End of the program. Press enter to exit...\n");
    pause;

end






