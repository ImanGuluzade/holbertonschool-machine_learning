#!/usr/bin/env python3
"""Module to train a Keras model with early stopping, LR decay, and saving"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and saves the best model.

    Args:
        network: the model to train
        data: input data
        labels: one-hot labels
        batch_size: size of batch for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate model with
        early_stopping: boolean to use early stopping
        patience: patience for early stopping
        learning_rate_decay: boolean to use LR decay
        alpha: initial learning rate
        decay_rate: the decay rate
        save_best: boolean indicating whether to save the best model
        filepath: file path where the model should be saved
        verbose: boolean for output during training
        shuffle: boolean for shuffling batches

    Returns:
        The History object generated after training the model.
    """
    callbacks = []

    # Learning rate decay logic (requires validation data)
    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Calculates inverse time decay"""
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        ))

    # Early stopping logic (requires validation data)
    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    # Save best model logic (requires validation data and a filepath)
    if validation_data and save_best and filepath:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
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
