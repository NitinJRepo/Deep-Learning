import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg') # For Fedora
import matplotlib.pyplot as plt

def plotFigure(dataset, encoder, autoencoder):
	num_images = 10
	np.random.seed(42)
	random_test_images = np.random.randint(dataset.shape[0], size=num_images)

	encoded_imgs = encoder.predict(dataset)
	decoded_imgs = autoencoder.predict(dataset)

	plt.figure(figsize=(18, 4))

	for i, image_idx in enumerate(random_test_images):
	    # plot original image
	    ax = plt.subplot(3, num_images, i + 1)
	    plt.imshow(dataset[image_idx].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	    
	    # plot encoded image
	    ax = plt.subplot(3, num_images, num_images + i + 1)
	    plt.imshow(encoded_imgs[image_idx].reshape(8, 4))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # plot reconstructed image
	    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
	    plt.imshow(decoded_imgs[image_idx].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()