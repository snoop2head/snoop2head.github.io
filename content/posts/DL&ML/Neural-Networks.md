---
layout: single
title: "Neural Networks"


---

### Definition of Neural Networks

Neural Networks are function approximations that stack affine transformations followed by non-linear transformations. 

![Review: GoogLeNet (Inception v1)— Winner of ILSVRC 2014 (Image  Classification) | by Sik-Ho Tsang | Coinmonks | Medium](https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png)

### 1D Input Linear Neural Networks

![image-20210809212936712](../assets/images/2021-08-09-Neural-Networks/image-20210809212936712.png)

- input is 1d, output is 1d.
- Data is dots on 2d plane
- Model: `y_hat = wx + b`
- Loss: mean squared error(MSE) as loss function

Minimizing mean squared error loss function based on partial derivative. ![image-20210809213447949](../assets/images/2021-08-09-Neural-Networks/image-20210809213447949.png)

![image-20210809221240484](../assets/images/2021-08-09-Neural-Networks/image-20210809221240484.png)

- Backpropagation is (partial) differentiating loss function with all the parameters. 
- Gradient descent is the process of updating each individual weights based on partial differentiation value. 
- Eta(n) is stepsize. 

### Multi-Dimensional Input

- Model: `y = W_transpose * x + b`

![image-20210809221530430](../assets/images/2021-08-09-Neural-Networks/image-20210809221530430.png)

### Multi-layer perceptron

Stacking Layers of Matrices and adding non-linear transformation(activation function) in between stacks

![image-20210809221639897](../assets/images/2021-08-09-Neural-Networks/image-20210809221639897.png)

- Model: `W*p*W*x`
- Universal Approximation Theorem: There is single hidden layer feedforward netowrk that approximates any measurable function to any dessired degree of accuracy on some compact set K.

Loss function

![image-20210809223135091](../assets/images/2021-08-09-Neural-Networks/image-20210809223135091.png)

- Regression Task: Mean Squared Error Loss function
- Classification Task: Cross Entropy Loss Function
- Probabilistic Task: 

