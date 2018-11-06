# Ishhmael Rogers
# Robotics Engineering, Infinitely Deep Robotics Group
# www.idrg.io 
# 2018 


# Deep_Neural_Networks

# Introduction

speech recognition
computer vision 
Deep learning is a more sophisticated way to solve classification problems. They have been used previously to discover new medicines, understand natural language and even interperting the contents of a document. 
This repository contains code for implementing Deep Neural Networks in TensorFlow. This README provides an explanation of the code found in the repositiory

# Solving Problems 

Lots of data and complex problems

understanding what's in an image and translating a document to another language

A family of techniques that are adaptable to many problems. These techniques have a common infrastructure and langauage.


# Instructions for Installing TensorFlow 

---

# First TensforFlow program 

# Tensors 

In TensorFlow data is encapsulated in an object called a tensor. Tensor come in a variety of different sizes. 

A = tf.constant(1234) is a 0-dimensional int32 tensor
B = tf.constant([123, 456, 789]) is a 1-dimensional int32 tensor 
C = tf.constant(([123, 456, 789], [222, 333, 444]) is a 2-dimensional int32 tensor 

# Sessions 

TensorFlow's API is built around the concept of a computational graph. A "TensorFlow Session" is an environement for runnng a graph.  The session is responsible for allocating the operations to GPU(s) and/or CPU(s) 

Please run the helloTensor.py from this repository

# Classification Problems 

Is the task of taking an input and giving it a label. 

# Example : Trained brains  

Imagine exploring a jungle that only consisted of 4 different types of the following animals. 

1. Monkies
2. Spiders
3. Birds
4. Snakes

Based on previous observations (and the data obtained during the process), our trained brains would have no problems classifying each new animal they saw as one of the 4 animals found in this jungle. 

Computers do not yet posses the same capabilites as our brains. Therefore as a machine learning engineer, my job is to develop artificial neural networks so that my robot equiped with a RGB-D camera can perform the following tasks: 

1. Classify the animal 
2. Detect the animal ***  A topic that is explored in perception

Like all great brains, our network must also be trained using previously obtained data. To do this, it is best if we separate our data into different subsets. 

The training set has been sorted which means labels have been assigned to previously classified animals.

In this case, consider the previous data came from our robot's first ever exploration premeired yesterday. The robot captured 40 monkies, 20 spiders, 83 birds, and 25 snakes on camera. From there, I carefully analyzed each image and provided each animal with a correct lablel.


Within the first few seconds of the exploration our system detected a new example, now its job is to determine which class that animal belongs to. 



# Logistic Classifier 

A linear classifier takes in as inputs the pixels of an image, and applies a linear function to them to generate a prediction of what the animal could be. 

The linear function 

WX + b = y

where,

W = The Matrix of Weights
X = The Matrix of Inputs
b = The Bias 
y = The score

NOTE: Scores can also be called logits 

The process of training models involves finding the weights and bias necessary to allow the robot make a good prediction.  

Note: When classifying each animal of the jungle, is important to remember that each animal can have one and only one possible label. 

## Training 

# Question: How to use scores to perform classification?

We convert the scores into probabilities, where the probability of the correct classification is close to 1 and the every other class is close to 0. 

In order to covert scores into proper probablities, we use a Softmax function. The following are the characteristics of the proper probablities:   

1. Proper probablites always sum to 1 
2. They are directly proportional to the scores

Explore the softmax function in Tensor Flow  Deep_Neural_Networks/softmax.py

Results from the code if ran as is 

# [0.6590012  0.24243298 0.09856589]



## Cross-entropy in TensorFlow

The cross entropy function helps us to compare these new probabilities to the one-hot encoded labels. 

# One - hot encoded

Goes back to the idea that each animal can only recieve 1 label. 

To do this we come up with one variable for each of the classes.

If the input is a monkey, 
---
variable for moneky is 1
variable for bird is 0 
variable for spider is 0
variable for snake is 0  

---

....and the same for the other animals 


We can measure how well we are doing with 

Cross entropy equation measures the distance between two probablity vectors 

# Multinomial Logistic Classification 
## Insert equation

works like this:
1. Take in the input and plug it in to the linear model and then output the logits
2. Logits are not the inputs to the softmax function 
3. The softmax function converts the logits into probabilities
4. The distance from the softmax function's output from the 1 hot labels helps us measure how well we are doing


To do this we woll use two new functions provided by tensor flow 

tf.reduce_sum(
    input_tensor,
    axis=None,
    keepdims=None,
    name=None,
    reduction_indices=None,
    keep_dims=None
)

tf.math.log(
    x,
    name=None
)

See link for cross_entropy.py

Results of the "cross_entropy.py" file 

0.35667497

## Minimizing Cross Entropy

In order to find the best weights and bias we need to identify the weights and bias that result in the smallest cross entropy. We can use the concept of Loss 

# Loss =  Average Cross-Entropy

Mathematically we can see that the loss is a function of the weights and biases. Therefore we continually adjust the weights and bias until we obtain the set that results in the smallest possible loss. Now we must briefly shift gears from a classification problem to a numerical optimization problem. 

## Numerical Optimization

definition

To solve in the simpliest way

# Graident Descent 

To perform graident descent take the derivative of loss with respect to parameters, follow it by taking a step backwards and repeat until we get to the bottom. 

NOTE: The derivative can be a function of thousands of parameters. 

## Numerical stabilty

Adding a very large number to a very small number introduces a lot of error. We want the values to never get too big or too small

Variables need to zero mean and equal variance 

Badly condition vs well conditio 

Images 

Normalized 
take pixel value of image (0-255) subtract by 128 and divide by 128. Does nt change content 

Weights neef to be initialize at a good enough startign point

scheme 

draw weights randopmly from a gaussian distribution with mean zero and standard deviation sigma

sigma determines order of magnitude of the outputs at the initial point of the optimization. 

order also determines peakieness 

A large sigma means large peaks - very opionnated and certain
A small sigma means small peaks - very uncertain 

NOTE: Start off with a small peak and let the optimization gain confidence. 

Optimization package compute derivative of the loss with respect to the weights and biases and takes a step backwards in the direction opposite to that derivative. Do until we reach a minimum of the loss function. 





## How to feed image pixels to the classifier 


## Where to initialize the optimization





