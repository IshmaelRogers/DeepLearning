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

## How to feed image pixels to the classifier 
Images 

Normalized 
take pixel value of image (0-255) subtract by 128 and divide by 128. Does nt change content 

Weights neef to be initialize at a good enough startign point

scheme 

draw weights randomly from a gaussian distribution with mean zero and standard deviation sigma

sigma determines order of magnitude of the outputs at the initial point of the optimization. 

order also determines peakieness 

A large sigma means large peaks - very opionnated and certain
A small sigma means small peaks - very uncertain 

NOTE: Start off with a small peak and let the optimization gain confidence. 

## Where to initialize the optimization

Optimization package compute derivative of the loss with respect to the weights and biases and takes a step backwards in the direction opposite to that derivative. Do until we reach a minimum of the loss function. 

Udacity has provided me with a lab to test these concepts.

Please see link 


Training set 

Test set ***

Validation set 

Seperate the sets so that the data stays completely unbiased.

*** Depending on the size of the data set, take one or nmore subsets of data and do not perform any analysis on them. 

## Overfittng 

We use a validation set when training models to prevent the robot from learning from the test set. 

# Validation and Test Set size 

For most classification tasks I tend to hold back more than 30,000 examples for validation changes 0.1% in accuracy

If classes are not well balanced, for example, if some important classes are very rare the heiuristic is no good.

# Rule of 30


## Techniques for overfitting.  


The bigger the test set the less noisy the accuracy measure will be 

a change that effects 30 examples in your validation set one way or another is statistically significant, and can be trusted. 

RUle of 30 

Example 

Out of 3000 examples in validation set trust rule of 30 

which level of accuracy can you be confident is not noise? 

80% -> 81.0% 
80% -> 80.5%
80% -> 80.1%

Some math 
 ---
(1.0 x 3000)/100 = 30
(0.5 x 3000)/100 = 15
(0.1 x 3000)/100 = 3


In summary if at least 30 examples are going from incorrect to correct the methods are improving the models

Training models with gradient descent is good but comes with scalability issuses.

If computing the loss takes n floating point operations, computing the graident takes 3 times that to compute

So much data using an iterative process like Gradient Descent is slow and inefficient 


### Stochastic Gradient Descent 

The heart of deep learning 

Scales well with both big data and big model sizes

A bad optimizer but fast enough to get the job done.


Instead of calculating the loss, we compute a bad estimate of it and then spend time making it less terrible

The estimate comes from computing the average loss for a very small random fraction of the training data 

between 1 and 1000 training samples each time 

The way you pick your samples is important, it must be completely random or it wont work. 

0. Take a small piece of training data
1. compute the loss for that sample
2. compute the derivative for that sample
3. pretend that that derivate is the right direction to use to do gradient descent. 

It may increase the real loss and not reduce it 

We compensate by doing this many times taking very small steps each time so that each step becomes a lot cheaper to compute. In other words we have to take smaller steps instead of one large step.

SGD issues 



# Momentum and Learning Rate Decay

in order to improve the SGD 

Inputs need to have 
Mean = 0
equal variance (small)

Initial weight need to be random
Mean = 0
equal variance(small) 


## Momentum

at each step a small step in a random direction. Combided those steps take us towards the minimum of the loss 

knowledge accumulated from previous steps about where we should be heading 

running average of gradients 

use it instead of the current direction batch of the data.

## Learning Rate Decay


that step size should be smaller and smaller over 


## Parameter hyperspace 

Learning rate tuning 

never trust how quickly you learn it doesnt measure how well you train

Many hyper parameters

ininitial learning rate
learning decay rate 
momentum 
batch size
weight initialization 

Thing to remember: lower learning rate when things go wrong 

## ADAGRAD
ADAGRAD is a modification of SGD that implicity does momentum and learning rate decay. This makes learning less sensitive to hyper paramters 

## Mini Batching

Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.

randomly shuffle the data at the start of each epoch, then create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.


## Calculating Memory requirements 
 



### TensorFlow Mini-batching 

must divide data into branches 

Data can't always be divided equally. If we wanted to create 128 sample from a data set of 1000, we'd have 

7 batches of 128 samples 
1 batch of 104 samples 


Batch size will vary so we use TensorFlow's 

tf.placeholder()


function 

if each sample had n_input = 784 features and n_classes = 10 possible labels, the dimensions for: 

features would be [None, n_input] 
labels would be [None, n_classes]

NOTE: The None dimension is a placeholder for the batch size. 


This set up allows us to feed features and labels into the model with varying batch sizes

# Explore the mini batch functionn and example in the repository 


# Epochs 

a single forward and backward pass of the entire data set. 

used to increase the accuracy of the model without requiring more data.

## Epochs in TensorFlow

see Epochs example in repository 
 

Each epoch attempts to move to a lower cost, leading to better accuracy. 

# Parameter tuning 

# Epoch

Multiply epoch value by 10 and determine how well test accuracy improves.
Find a point where large changes in epoch result in small changes to accuracy. 

# Learning Rate

Lowering the learning rate requires more epochs, but achieves better accuracy. 

## Practice methods available in the lab from the repository 








