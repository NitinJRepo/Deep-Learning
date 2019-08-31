#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:47:52 2018

@author: nitin
"""

import sys
sys.path.append('../common')
from process import get_data

from sklearn.neural_network import MLPClassifier

# get the data
Xtrain, Ytrain, Xtest, Ytest = get_data()

# create the neural network
model = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000)

# train the neural network
model.fit(Xtrain, Ytrain)

# print the train and test accuracy
train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)