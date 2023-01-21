## Winner Take All Network

**Note**: Please refer to `EE 550 Project 4 Description.pdf` for the problem description.

In this homework assignment the Winner Take-all Network which is a competitive learning 
model where the neurons compete each other to be activated, was implemented. Creating 
sample points and initializing random weights on a unit sphere, we demonstrated that the 
weights converge in the center after an all-or-nothing learning method. Having obtained 
weights, it is shown that we were able to categorize the newly arrived data based on the 
updated weights.

### Problem Definition
In this model, the objective is clustering or categorizing the input data. It is achieved by the 
model itself from the correlations in it. This is to say that the similar input was classified as 
being in the same categoty and fire the same output unit.  
Our main problem was how to find cluster categories in the input dataset and how to choose 
the weights accordingly. It is solved by finding the largest output value and then updating for 
only the winning unit only to make the winning weight closer to current vector.  

### Analysis
In this project, we are interested in the normalized weights thus the winner was the weight 
vector who is closest to the input vector ζ. Finding the winner, we only for it. This is achieved 
in an efficient manner via multiplying with the output vector whose only nonzero value was 
the winner neuron.  
After updating, it is shown that the weights converged to the centers of clusters following a 
trajectory. It was as expected since the update term is analygous to the gradient descent 
algorithm where the cost function is iteratively minimized.  
Finally, the algorithm’s correctness was tested via three inputs each from three clusters. After 
the test, it was shown that the algorithm is able to correctly cluster the new data. To do this, 
the active neuron’s index was obtained and assigned the color of the weights accordingly via a 
swich-case condition. When the predictions were plotted, it was shown that they perfectly 
matched with their clusters.

### Open Issues
The model does have some disadvangates. Firstly, it is not robust to failure. As one output 
neuron is used one category, if an output fails, the category is lost. Next, there is no hierarchy 
that no category can be defined within a category.  

