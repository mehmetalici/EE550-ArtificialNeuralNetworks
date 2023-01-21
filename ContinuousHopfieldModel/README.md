## Project 5: Continuous Hopfield Model

**Note**: Please refer to `EE 550 Project 5 Description.pdf` for the problem description.

In this project, the Continuous Hopfied Model which is an extension of the Binary Hopfield 
Model was implemented and its energy function was analyzed. This model was inspired by 
biological neurons which have continious I/O characheristics and some processing delays as in 
a capacitive network. This is to say that an input will lag behind the outputs Vj of other neurons 
due to the input capacitance “C” of the call membraine, transmembrance resistance “R” and 
finite impedance “𝑇𝑖𝑗ିଵ” between the output and cell body of cell i. 

### Analysis
We can use KCL to write the dynamics equations as 

$C_i \frac{u_i}{dt} + \frac{u_i}{R} = \sum T_{ij}V_j + I_j$,


assuming the cell as a node in an electrical circuit. This system can be written in the state space 
as 

$𝑥̇  =𝐶𝑇𝑦(𝑥) − 𝐺𝑥+𝐶𝐼$ 

With this form,  $𝑥̇ = 0$ yielded the points where the equilibrium point is achieved. Then, in 
order to find the convergence properties of the system, we have the energy function,  

$E=-\frac{1}{2} \sum \sum T_{ij}v_iv_j + \sum \frac{1}{R_i} \int_{0}^{v_i}g^{-1}(\zeta)d\zeta$,
 
where the second part of the RHS is a contribution of the continuous model over the discrete 
version. This is to say that the model operates within N-dim hypercube and the equilibrium 
points stays inside the cube. If the lambda increases too much, then the second term becomes 
negligible and the minimal states are found in the corners of the hypercube. 

In this project, the stability of differential equations were studied via the neurodynamic model. 
A two neuron example was handled and the minimal energy points and energy contour maps 
are shown. Furthermore, an initial point was shown to converge to the nearest equilibrium point 
after iterations.   

Finally, the effect of lambda was investigated. In the limiting case, when lambda is infinite, it is 
shown that the maxima and minima of the continuous Hopfield Network become identical to 
those of the corresponding discrete Hopfield Model.  