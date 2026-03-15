#!/usr/bin/env python3
"""Module to build a model with Keras"""
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
    model = K.Sequential()
    # Define the L2 regularizer using the provided lambtha
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # First layer requires the input_dim parameter
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg,
                input_dim=nx
            ))
        else:
            # Subsequent layers infer input shape from previous layers
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            ))

        # Add Dropout layer after each Dense layer except the last one
        if i < len(layers) - 1:
            # Keras Dropout uses rate (1 - keep_prob)
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
