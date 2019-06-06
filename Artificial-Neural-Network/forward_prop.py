#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:34:26 2018

@author: nitin
"""

# forward propagation example for deep learning in python class.
import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

# randomly initialize weights
D = 2 # dimensionality of input
M = 3 # hidden layer size        <-- It is also a hyperparameter
K = 3 # number of classes        <-- OR dimentionality of output
W1 = np.random.randn(D, M)       # the weights from input -> hidden
b1 = np.random.randn(M)          # bias units from input -> hidden
W2 = np.random.randn(M, K)       # the weights from hidden -> output
b2 = np.random.randn(K)          # bias unit from hidden -> output
                                 # All are irrespective of N (number of samples).
 

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1) # sigmoid     <-- 1500 X 3
    # Z = np.tanh(X.dot(W1) + b1) # tanh
    # Z = np.maximum(X.dot(W1) + b1, 0) # relu
    A = Z.dot(W2) + b2                  #  <-- 1500 X 3
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)  #  <-- 1500 X 3
    return Y

# determine the classification rate
# num correct / num total
def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

# verify we chose the correct axis
assert(len(P) == len(Y))

print("Classification rate for randomly chosen weights:", classification_rate(Y, P))
