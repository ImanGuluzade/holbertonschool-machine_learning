#!/usr/bin/env python3
"""
Module for creating a variational autoencoder (VAE) using TensorFlow Keras.
"""

import tensorflow as tf
import tensorflow.keras as K


def sampling(args):
    """
    Reparameterization trick by sampling from a standard normal distribution.
    """
    mu, log_sigma = args
    batch = K.backend.shape(mu)[0]
    dim = K.backend.int_shape(mu)[1]
    epsilon = K.backend.random_normal(shape=(batch, dim))
    return mu + K.backend.exp(0.5 * log_sigma) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an encoder, a decoder, and a variational autoencoder model.
    """
    # Encoder
    encoder_input = K.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    mu = K.layers.Dense(latent_dims, activation=None)(x)
    log_sigma = K.layers.Dense(latent_dims, activation=None)(x)

    # Sampling layer
    latent = K.layers.Lambda(
        sampling, output_shape=(latent_dims,)
    )([mu, log_sigma])

    encoder = K.Model(encoder_input, [latent, mu, log_sigma], name='encoder')

    # Decoder
    decoder_input = K.Input(shape=(latent_dims,))
    x = decoder_input
    reversed_layers = hidden_layers[::-1]
    for units in reversed_layers:
        x = K.layers.Dense(units, activation='relu')(x)
    decoder_output = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(decoder_input, decoder_output, name='decoder')

    # Full VAE model
    latent_tensor, mu_tensor, log_sigma_tensor = encoder(encoder_input)
    decoder_output_tensor = decoder(latent_tensor)
    auto = K.Model(encoder_input, decoder_output_tensor, name='autoencoder')

    # Custom VAE Loss (Reconstruction loss + KL divergence)
    reconstruction_loss = K.losses.binary_crossentropy(
        encoder_input, decoder_output_tensor
    )
    reconstruction_loss *= input_dims
    kl_loss = (
        1 + log_sigma_tensor
        - K.backend.square(mu_tensor)
        - K.backend.exp(log_sigma_tensor)
    )
    kl_loss = K.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
