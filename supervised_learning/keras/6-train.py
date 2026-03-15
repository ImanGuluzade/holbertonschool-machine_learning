#!/usr/bin/env python3
"""Module to train a Keras model with early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) with the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate the model with, if not None
        early_stopping: boolean indicating whether to use early stopping
        patience: the patience used for early stopping
        verbose: boolean determining if output is printed during training
        shuffle: boolean determining whether to shuffle batches every epoch

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    # Early stopping should only be performed if validation_data exists
    if validation_data and early_stopping:
        # monitor='val_loss' is the default but we define it for clarity
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
