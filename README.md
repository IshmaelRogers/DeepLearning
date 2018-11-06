# Ishhmael Rogers
# Robotics Engineering, Infinitely Deep Robotics Group
# www.idrg.io 
# 2018 


# Deep Learning 

# Introduction

Deep learning architectures such as deep neural networks, deep belief networks and recurrent neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design and board game programs, where they have produced results comparable to and in some cases superior to human experts. In general, Deep Learning is family of techniques that are adaptable to many problems. These techniques have a common infrastructure and langauage.

This repository contains code for implementing the basics of Deep Learning in TensorFlow. This README provides an explanation of the code found in the repositiory

# Solving Problems 

Deep learning is necessary when there is a lot of data to work with and a complex problem to solve. Some common problems include understanding what's in an image and translating a document to another language.

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

Please run the "helloTensor.py" file contianed in this repository

# Classification Problems 

A common type of problem we'd like to solve is classification. It can be defined as the task of taking an input, feeding it into the network and giving it a label. 

# Example : Trained brains  

Imagine exploring a jungle that only consisted of 4 different types of the following animals. 

1. Monkies
2. Spiders
3. Birds
4. Snakes

Based on previous observations (and the data obtained during the process), our trained brains would have no problems classifying each new animal we saw as one of the 4 animals found in this jungle. 

Computers do not yet posses the same capabilites as our brains. Therefore as a machine learning engineer, my job is to develop artificial neural networks so that a robot equiped with a camera can perform the following tasks: 

1. Classify the animal 
2. Detect the animal ***  A topic that is explored in perception

Like all great brains, our artificial network must also be trained using previously obtained data. To do this, it is best if we separate our data into different subsets for optimal training. 

1. Training set
2. Validation set 
3. Testing set ***

*** Depending on the size of the data set, take one or more subsets of test data and do not perform any analysis on them. 
NOTE: It is best practice to always seperate data sets so that the data stays completely unbiased.
NOTE: The training set, the previous data has been sorted which means labels have been assigned to correctly classified animals.
NOTE: The validation set is used to prevent the classifier from learning directly from the test set.

## Overfitting 

An important concept to understand and prevent while training models.

In this case, consider the previous data that came from our robot's first ever exploration premeired yesterday. The robot captured 40,000 monkies, 20,000 spiders, 33,300 birds, and 25,000 snakes on camera. As mentioned previous, I spent some time carefully analyzing each image and provided each animal with a correct lablel.


Within the first few seconds of the exploration our system detected a new example, now its job is to determine which class that animal belongs to. 

Let's develop a Classifier to help us do this. 

# Logistic Classifier 

Ideal we'd want out linear classifier takes in pixels of an image as inputs, and apply a linear function to them to generate a prediction of what the animal could be. 

The linear function we use is as follow  

W x X + b = y  <----- Create equation 

where,

W = The Matrix of Weights
X = The Matrix of Inputs
b = The Bias 
y = The score

NOTE: Scores are also called logits 
NOTE: When classifying each animal of the jungle, it is important to remember that each animal can have one and only one possible label. 

## Training the classifier

# Question: How to use scores to perform classification?

In practice we wil convert the scores into proper probabilities such that, the probability of the correct classification is close to 1 and the every other classification is close to 0.

To do this, we use the Softmax function. 

## Softmax Function 

Explore the softmax function in Tensor Flow  Deep_Neural_Networks/softmax.py
Results from the code if ran as is: 

# [0.6590012  0.24243298 0.09856589]

# Proper Probabilities

The following are the characteristics of the proper probablities

1. Proper probablites always sum to 1 
2. They are directly proportional to the scores

## Cross-entropy in TensorFlow

The cross entropy function helps us to compare these new probabilities to the 1-Hot encoded labels by measuring the distance between the two probablity vectors. That distance can help determine how correct our prediction is.  We can measure how well we are doing with 

# 1-Hot encoding 

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

# Multinomial Logistic Classification 

## Insert equation

A multinomial Logistic Classifier works as follow:

1. Take in the image input and plug it in to the linear model and then output the logits (score)
2. Logits are now the inputs to the Softmax function 
3. The Softmax function converts the logits into probabilities
4. The distance from the softmax function's output from the 1 hot labels helps us measure how well we are classifying new objects. 


To do this we will use two functions provided by TensorFlow: 

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

# Cross_entropy.py

See "cross_entropy.py" file in this repository 

Results of the "cross_entropy.py" as seen in the repository  

0.35667497

## Minimizing Cross Entropy

In order to find the BEST weights and biases we need to identify the weights, W and biases, b that result in the smallest cross entropy. 

Let's introduce the concept of Loss 

# Loss =  Average Cross-Entropy

Mathematically we can see that the loss is a function of the weights and biases. Therefore we can continually adjust the weights and biases until we obtain the set that results in the smallest possible loss. 

Now we must briefly shift gears from a classification problem to a numerical optimization problem. 

## Numerical Optimization

Definition: the selection of a best element (with regard to some criterion) from some set of available alternatives. 

An optimization problem consists of maximizing or minimizing a real function by SYSTEMATICALLY choosing input values from within an allowed set and computing the value of the function. 

The best way to systematically choose the input values is with Gradient Descent.

# Graident Descent 

To perform graident descent:

1. Take the derivative of loss function with respect to parameters, 
2. Follow it by taking a step backwards
3. Repeat until we get reach a minimum of the function

NOTE: The derivative can be a function of thousands of parameters. 

## Numerical stabilty

Adding a very large number to a very small number introduces a lot of error. We want the values to never get too big or too small therefore we enforce the following requirements for variables.

Variables should be:
1. Mean = 0 
2. Equal variance 

Without these requirements we run the risk of having badly conditioned data.   

## Feeding image pixels to the classifier 
 
Images need to be normalized before being feed into our classifier.

To do this:

1. Take pixel value of image (usually 0-255) subtract by 128
2. Divide by 128. 

Performing this step does not change content of the data. Instead, it makes it easier for the classifier to perform repeated operations.  
NOTE: Weights and biases need to be initialize at a good enough starting point

# The Procedure 

Draw weights randomly from a gaussian distribution with mean zero and standard deviation sigma

NOTE: Sigma determines the order of magnitude of the outputs at the initial point of the optimization. 

NOTE: The order also determines peakieness of the graph.  

NOTE: A large sigma means large peaks - very opionnated and certain

NOTE: A small sigma means small peaks - very uncertain 

NOTE: Start off with a small peak and let the optimization gain confidence through training. 

## Where to initialize the optimization

The optimization package that is built into TensorFlow:

1. Computes the derivative of the loss with respect to the weights and biases
2. Takes a step backwards in the direction opposite to that derivative. 

These two steps are repeated until we reach a minimum of the loss function. 

## Neural Network Lab

Udacity has provided me with a lab to test these concepts.

Please see repository 


# Validation and Test Set size 

For most classification tasks it is necessary to hold back more than 30,000 examples. This can result in validation changes of 0.1% in accuracy

If classes are not well balanced, for example, if some important classes are very rare, this heiuristic is no good.

# Rule of 30

A change that effects 30 examples in your validation set one way or another is statistically significant, and can be trusted. 
NOTE: If at least 30 examples are going from incorrect to correct, the methods being implemented are improving the model.

# Example

Assume that we have 3000 examples in validation set and that we are confident about the rule of 30 

Which level of accuracy change can trusted to not be noise? 

80% -> 81.0% 
80% -> 80.5%
80% -> 80.1%

Some math 
 ---
(1.0 x 3000)/100 = 30 <--
(0.5 x 3000)/100 = 15
(0.1 x 3000)/100 = 3


## Techniques for preventing overfitting.  

NOTE: The bigger the test set the less noisy the accuracy measure will be 


Training models with gradient descent is good but comes with scalability issuses. With so much data using an iterative process like Gradient Descent the process can be slow and inefficient.

# If 
computing the loss takes N number of floating point operations, 

# Then 
computing the graident takes 3 times that to compute

### Stochastic Gradient Descent 

SGD is the heart of deep learning because it tends to scale well with both big data and big model sizes. In general, SGD is a bad optimizer but is fast enough to get the job done.

Instead of calculating the loss, we compute a bad estimate of it and then spend time making it less terrible

The estimate comes from computing the average loss for a very small random fraction of the training data

NOTE: Between 1 and 1000 training samples each time 
NOTE: The way you pick your samples is important, it must be completely random or it wont work. 

0. Take a small piece of training data
1. Compute the loss for that sample
2. Compute the derivative for that sample
3. Pretend that that derivate is the right direction to use to do gradient descent. 

You may notice that performing this method may increase the real loss and not reduce it. However, we can compensate by doing this many times taking very small steps each time so that each step becomes a lot cheaper to compute. In other words we have to take smaller steps instead of one large step.

SGD issues comes with issues. Below we talk about a common way to deal with those issues. 

# Momentum and Learning Rate Decay

In order to improve the performance of the S 

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

## Practice methods available in the lab in the NN-Lab folder







