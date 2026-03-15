#!/usr/bin/env python3
"""Module to build a model with Keras Functional API"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.
    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout
    Returns:
        the keras model
    """
    # Define the input layer explicitly
    inputs = K.Input(shape=(nx,))

    # Define L2 regularization
    reg = K.regularizers.l2(lambtha)

    # Initialize the first hidden layer using the inputs
    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=reg
    )(inputs)

    # Loop through remaining layers
    for i in range(1, len(layers)):
        # Add dropout before the next Dense layer
        x = K.layers.Dropout(1 - keep_prob)(x)

        # Add the next Dense layer connected to the previous output (x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

    # Create the model by specifying inputs and outputs
    model = K.Model(inputs=inputs, outputs=x)

    return model
