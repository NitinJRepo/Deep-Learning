#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:27:44 2018

@author: nitin
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_transformed_data():
    print("Reading in and transforming data...")

    if not os.path.exists('../large_files/train.csv'):
        print('Looking for ../large_files/train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv('../large_files/train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    Y = data[:, 0].astype(np.int32)

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:]
    Ytest  = Y[-1000:]

    # center the data
    mu = Xtrain.mean(axis=0)
    Xtrain = Xtrain - mu
    Xtest  = Xtest - mu

    # transform the data
    pca = PCA()
    Ztrain = pca.fit_transform(Xtrain)
    Ztest  = pca.transform(Xtest)

    plot_cumulative_variance(pca)

    # take first 300 cols of Z
    Ztrain = Ztrain[:, :300]
    Ztest = Ztest[:, :300]

    # normalize Z
    mu = Ztrain.mean(axis=0)
    std = Ztrain.std(axis=0)
    Ztrain = (Ztrain - mu) / std
    Ztest = (Ztest - mu) / std

    return Ztrain, Ztest, Ytrain, Ytest


def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.show()
    return P


def forward(X, W, b):
    # softmax
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def predict(p_y):
    return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)


def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()


def gradW(t, y, X):
    return X.T.dot(t - y)


def gradb(t, y):
    return (t - y).sum(axis=0)


def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
