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

Explore the softmax function in Tensor Flow 


x = tf.nn.softmax([2.0, 1.0, 0.2])





