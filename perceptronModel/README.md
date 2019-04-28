# perceptronModel
Implementation of the Perceptron Model Neural Network. This project is a part of the EE550 Artificial Neural Networks Course.

## Introduction
Invented by Frank Rosenblatt in 1957, the Perceptron model is the simplest feedforward neural
network and laid the foundation of the deep learning which gained much attention by researchers
today. In this project this model was implemented using MATLAB. The program was built based
on a linear predictor function combining a set of weights with the feature vector. For the
implementation, 100 arbitrary points were picked in 3D space, 80 of which is used for learning
and 20 is used for testing the algorithm. It is shown that the algorithm converges after a few
iterations and has the %100 accuracy of prediction. 

## Approach and Methodology
100 sample data points, half of which are located in the first quadrant and the other in the eighth
sample data points in 3D space were created. To do this, a random linear disribution between --
100 and 100 were preferred for convenience. Also, a weight vector was initialized with arbitrary
numbers between -150 and 150, which will later be updated in the learning phase. The learning
coefficient 𝜂, which determines the tradeoff between aggressive and safe learning and should be
less than 1, was specified to be 0.5.

The dataset was visualized by colored circles at the specified data points in 3D space. The colors
blue and red were preferred to plot the positive and negative samples, respectively. For training
40 samples from each group were picked and used to update weights. Simultaneosly, a cost
function were calculated at each iteration and it converged to zero after a few iterations. In fact,
obviously the perceptron is guaranteed to converge as the training set is linearly seperable. When
a linearly inseperable training set D is given, the perceptron model will never get to the state.

The positive examples is seperated from the negative examples by a decision boundary and it is
visualized in the program with the same figure. To do this, the equation, 

<a href="https://www.codecogs.com/eqnedit.php?latex=w_1x_1&plus;w_2x_2&plus;w_3x_3=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_1x_1&plus;w_2x_2&plus;w_3x_3=0" title="w_1x_1+w_2x_2+w_3x_3=0" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=w_1,w_2,w_3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_1,w_2,w_3" title="w_1,w_2,w_3" /></a> are the elements of the adjusted weight vector and <a href="https://www.codecogs.com/eqnedit.php?latex=x_1,x_2,x_3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1,x_2,x_3" title="x_1,x_2,x_3" /></a>  are the points in the 3D space. Solving for <a href="https://www.codecogs.com/eqnedit.php?latex=x_3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_3" title="x_3" /></a>, we get,

<a href="https://www.codecogs.com/eqnedit.php?latex=x_3=-\frac{w_1}{w_3}x_1-\frac{w_2}{w_3}x_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_3=-\frac{w_1}{w_3}x_1-\frac{w_2}{w_3}x_2" title="x_3=-\frac{w_1}{w_3}x_1-\frac{w_2}{w_3}x_2" /></a>

Creating a meshgrid for <a href="https://www.codecogs.com/eqnedit.php?latex=x_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=x_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_2" title="x_2" /></a>  a nice decision hyperplane was obtained with surf function.


Selecting 10 samples of each classes the model is tested for prediction accuracy. To do this, the
specimens were multiplied with the trained weights and taking their sign functions the
predictions were obtained. Then, the prediction accuracy is obtained by, 
```
mean(double(y == sample_test(:,4))*100) 
```

where y and the sample_test’s forth column are predicted and the actual output, respectively.
This value turns out to be %100 as it validates the algorithm being implemented successfully
and it is printed to the command window for user. 


Five random points were taken from the test sample set and visualized with blue and red circles
based on having a positive or negative output, respectively. Here it is also validated that the
algorithm works without problem.
Finally, the calculated cost function were plotted against each iteration number to check its
convergence. It is shown that it converges to a steady state quickly after a maximum of a few
iterations. This is due to the fact that the training set is seperated linearly by a hyperplane. 

## Design Decisions
The program was organized in an interactive fashion. In certain waypoints, the program prints
out the current execution, information about the plots and also asks user’s confirmation to
continue. Furthermore, the source and plots were fully commented and vectorized
implementation is preferred wherever possible. Therefore, the user will be able to examine the
plots comfortably and stay in tune with the current execution and can have a better analysis on
the implementation. 

## Conclusion
In this project the perceptron model neural network was implemented in an interactive fashion.
The program trains with the data points in 3D space and tries to predict the output for each
sample data. It is shown that the cost function converges to zero after a number of iterations. 
