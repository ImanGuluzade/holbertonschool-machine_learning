#!/usr/bin/env python3
"""Module that defines a Simple_GAN class"""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """Simple GAN trained with a generator and a discriminator"""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initializes the GAN, its losses and its optimizers"""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=generator.optimizer,
                               loss=generator.loss)

        # discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=discriminator.optimizer,
                                   loss=discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """Returns a fake sample of the given size"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size),
                              training=training)

    def get_real_sample(self, size=None):
        """Returns a random real sample of the given size"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Performs one training step: disc_iter updates of the
        discriminator, then one update of the generator

        Returns:
            dict with the discriminator loss and the generator loss
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real = self.get_real_sample()
                fake = self.get_fake_sample(training=False)
                real_out = self.discriminator(real, training=True)
                fake_out = self.discriminator(fake, training=True)
                discr_loss = self.discriminator.loss(real_out, fake_out)
            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake = self.get_fake_sample(training=True)
            fake_out = self.discriminator(fake, training=False)
            gen_loss = self.generator.loss(fake_out)
        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
