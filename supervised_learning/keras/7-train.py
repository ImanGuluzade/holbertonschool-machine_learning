#!/usr/bin/env python3
"""Module to train a Keras model with early stopping and LR decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping
    and learning rate decay.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) with the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate the model with, if not None
        early_stopping: boolean indicating whether to use early stopping
        patience: the patience used for early stopping
        learning_rate_decay: boolean indicating whether to use LR decay
        alpha: initial learning rate
        decay_rate: the decay rate
        verbose: boolean determining if output is printed
        shuffle: boolean determining whether to shuffle batches

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    # Early stopping logic (requires validation data)
    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    # Learning rate decay logic (requires validation data)
    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Calculates inverse time decay for the current epoch"""
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1  # Ensures Keras prints the LR update message
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
