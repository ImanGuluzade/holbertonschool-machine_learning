#!/usr/bin/env python3
"""
Module containing the resnet50 function to build a ResNet-50 network.
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015).

    Returns:
    The compiled keras model
    """
    # 1. Input configuration layer
    inputs = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.HeNormal(seed=0)

    # 2. Stage 1: Initial Convolutional Processing block
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(inputs)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # 3. Stage 2: 3 blocks (1 projection, 2 identity)
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])
    X = identity_block(X, filters=[64, 64, 256])

    # 4. Stage 3: 4 blocks (1 projection with stride 2, 3 identity)
    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    # 5. Stage 4: 6 blocks (1 projection with stride 2, 5 identity)
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])

    # 6. Stage 5: 3 blocks (1 projection with stride 2, 2 identity)
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    # 7. Classification stage
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(X)

    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    return K.models.Model(inputs=inputs, outputs=outputs)
