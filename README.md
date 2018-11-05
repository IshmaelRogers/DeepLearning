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

The task of taking an input and giving it a label.
Once you can classify objects you can make efforts to detect them and rank them. 

The training set has been sorted 

Now we have a new example, now determine which class it belongs too. 

# Logistic Classifier 

a linear classifier takes in inputs, could be the pixels of an image, and applies a linear function to them to generate a prediction. 

WX + b = y

W = The Matrix of Weights
X = The Matrix of Inputs
b = The Bias 
y = The score

NOTE: Scores can also be called logits 

Training models involves finding the weights and bias necessary to make good predictions.  
Note: In the case of classifying letters of the alphabet, each image as input, can have one and only possible label. 

# Question: How to use scores to perform classification?

In order to covert scores into proper probablities we use a Softmax function. Thefollowing are the characteristics of the proper probablities:   

1. Proper probablites always sum to 1 
2. They will be large when scores are large
3. They will be small when score are small 







