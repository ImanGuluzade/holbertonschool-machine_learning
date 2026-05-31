#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid

if __name__ == '__main__':

    # Load the compressed MNIST digit dataset
    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    
    # Define a 3x3 Sobel kernel for edge extraction features
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    
    # Run the validation convolution pipeline function
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    # Plot the original sample input image array
    plt.imshow(images[0], cmap='gray')
    plt.show()
    
    # Plot the calculated convolved feature map output
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
