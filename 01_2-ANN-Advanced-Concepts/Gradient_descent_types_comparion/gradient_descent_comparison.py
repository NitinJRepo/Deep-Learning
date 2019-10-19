#!/usr/bin/env python3

# In this file we compare the progression of the cost function vs. iteration
# for 3 cases:
# 1) full gradient descent (OR batch GD)
# 2) Stochastic gradient descent (OR online GD)
# 3) mini-batch gradient descent
#
# We use the PCA-transformed data to keep the dimensionality down (D=300)
# I've tailored this example so that the training time for each is feasible.
# So what we are really comparing is how quickly each type of GD can converge,
# (but not actually waiting for convergence) and what the cost looks like at
# each iteration.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator

epoch = 50

def main():
    Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    # 1. Full GD
    W = np.random.randn(D, 10) / 28 # Square root of no. of dimentionality. i.e. 28 * 28 = 784
    b = np.zeros(10)
    loss_batch = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()

    for i in range(epoch):
        p_y = forward(Xtrain, W, b)

        W += lr * (gradW(Ytrain_ind, p_y, Xtrain) - reg * W)
        b += lr * (gradb(Ytrain_ind, p_y) - reg * b)

        p_y_test = forward(Xtest, W, b)
        temp_loss = cost(p_y_test, Ytest_ind)
        loss_batch.append(temp_loss)

        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, temp_loss))
                print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for full GD:", datetime.now() - t0)
    print("=======================================================")


    # 2. Stochastic GD
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    loss_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()

    for i in range(epoch): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)

        #for n in range(min(N, 500)): # shortcut so it won't take so long...
        for n in range(N):
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            p_y = forward(x, W, b)

            W += lr * (gradW(y, p_y, x) - reg * W)
            b += lr * (gradb(y, p_y) - reg  * b)

            p_y_test = forward(Xtest, W, b)
            loss = cost(p_y_test, Ytest_ind)
            loss_stochastic.append(loss)

        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, loss))
                print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for SGD:", datetime.now() - t0)
    print("=======================================================")


    # 3. Mini-batch GD
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    loss_mini_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N // batch_sz

    t0 = datetime.now()

    for i in range(epoch):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)

        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            temp_loss = cost(p_y_test, Ytest_ind)
            loss_mini_batch.append(temp_loss)

        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, temp_loss))
                print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for mini-batch GD:", datetime.now() - t0)


    # Plot graph
    x1 = np.linspace(0, 1, len(loss_batch))
    plt.plot(x1, loss_batch, label="full(batch) GD")
    x2 = np.linspace(0, 1, len(loss_stochastic))
    plt.plot(x2, loss_stochastic, label="stochastic GD")
    x3 = np.linspace(0, 1, len(loss_mini_batch))
    plt.plot(x3, loss_mini_batch, label="mini-batch GD")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()