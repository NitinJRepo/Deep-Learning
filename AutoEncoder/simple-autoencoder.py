from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
import pandas as pd

from plot_figure import plotFigure 

# Loads the training and test data sets (ignoring class labels)
(x_train, _), (x_test, _) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# The data set consists 3D arrays with 60K training and 10K test images. The images have a resolution of 28 x 28 (pixels).
print("x_train shape = ", x_train.shape)
print("x_test shape = ", x_test.shape)

# To work with the images as vectors, let's reshape the 3D arrays as matrices. In doing so, we'll reshape the 28 x 28 images into vectors of length 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("x_train shape = ", x_train.shape)
print("x_test shape = ", x_test.shape)

# Simple AutoEncoder

# Input dimension = 784
input_dim = x_train.shape[1]
encoding_dim = 32

compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

print("AutoEncoder Summary: ")
autoencoder.summary()

# We can extract the encoder model from the first layer of the autoencoder model.
# The reason we want to extract the encoder model is to examine what an encoded image looks like.
input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

print("Encoder Summary: ")
encoder.summary()

# Train the AutoEncoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Plot Encoded and Decoded images
plotFigure(x_test, encoder, autoencoder)
