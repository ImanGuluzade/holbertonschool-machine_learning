#!/usr/bin/env python3
"""
Transfer Learning script to train a CNN on the CIFAR-10 dataset
using a pre-trained Keras application.
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the data for the MobileNetV2 model.

    Parameters:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 data
    Y: numpy.ndarray of shape (m,) containing CIFAR-10 labels

    Returns:
    X_p: Preprocessed X (normalized according to MobileNetV2 standards)
    Y_p: Preprocessed Y (converted to one-hot encoding representation)
    """
    X_p = K.applications.mobilenet_v2.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def train_model():
    """
    Loads pre-trained weights, structures a top dense classification head,
    trains the network on CIFAR-10, and saves the final output to disk.
    """
    # 1. Load data split partitions
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()

    # 2. Run preprocessing transformations
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)

    # 3. Define the pipeline input structure placeholder shape
    input_shape = (32, 32, 3)
    inputs = K.Input(shape=input_shape)

    # 4. Lambda layer to upscale images dynamically up to target resolutions
    # MobileNetV2 accepts 128x128 pixels as a standard input dimension bound
    upscale = K.layers.Lambda(
        lambda img: K.backend.resize_images(
            img, height_factor=4, width_factor=4,
            data_format='channels_last', interpolation='bilinear'
        )
    )(inputs)

    # 5. Initialize the pre-trained core base application network layout
    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_tensor=upscale
    )

    # Freeze the pre-trained weights entirely
    base_model.trainable = False

    # 6. Add a custom classification head on top
    x = base_model.output
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    # Assemble the final model architecture
    model = K.Model(inputs=inputs, outputs=outputs)

    # 7. Compile the network
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_with_crossentropy' if hasattr(
            K.losses, 'categorical_with_crossentropy'
        ) else 'categorical_crossentropy',
        metrics=['accuracy']
    )

    # 8. Train the model using early stopping to prevent overfitting
    early_stop = K.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=64,
        epochs=15,
        callbacks=[early_stop],
        verbose=1
    )

    # 9. Save the compiled model structure file
    model.save('cifar10.h5')


if __name__ == '__main__':
    train_model()
