#!/usr/bin/env python3
"""
Module for creating a vanilla autoencoder using TensorFlow Keras.
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an encoder, a decoder, and an autoencoder model.
    """
    encoder_input = K.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    latent = K.layers.Dense(latent_dims, activation='relu')(x)
    encoder = K.Model(encoder_input, latent, name='encoder')

    decoder_input = K.Input(shape=(latent_dims,))
    x = decoder_input
    reversed_layers = hidden_layers[::-1]
    for units in reversed_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    decoder_output = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(decoder_input, decoder_output, name='decoder')

    auto_output = decoder(encoder(encoder_input))
    auto = K.Model(encoder_input, auto_output, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
