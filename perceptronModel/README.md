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

Creating a meshgrid for <a href="https://www.codecogs.com/eqnedit.php?latex=x_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /></a>
