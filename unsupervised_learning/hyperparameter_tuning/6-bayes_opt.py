#!/usr/bin/env python3
"""
Module for hyperparameter optimization using GPyOpt and TensorFlow/Keras.
Optimizes 5 different hyperparameters for a neural network on MNIST.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import GPyOpt

# Suppress TensorFlow oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def evaluate_model(parameters):
    """
    Objective function for GPyOpt.
    Parameters array format:
    [lr, num_units, dropout_rate, l2_reg, batch_size]
    """
    lr = float(parameters[0, 0])
    num_units = int(parameters[0, 1])
    dropout_rate = float(parameters[0, 2])
    l2_reg = float(parameters[0, 3])
    batch_size = int(parameters[0, 4])

    print(
        f"\n--- Training with: lr={lr:.5f}, units={num_units}, "
        f"dropout={dropout_rate:.2f}, l2={l2_reg:.5f}, batch={batch_size} ---"
    )

    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 784))
    x_val = x_val.reshape((-1, 784))

    # Build model with regularization and dropout
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            input_shape=(784,)
        ),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Checkpoint filename specifying the hyperparameter values
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = (
        f"model_lr_{lr:.5f}_u_{num_units}_drop_{dropout_rate:.2f}_"
        f"l2_{l2_reg:.5f}_bs_{batch_size}.keras"
    )
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    # Callbacks for Early Stopping and Model Checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path, monitor='val_accuracy',
            save_best_only=True, mode='max'
        )
    ]

    # Train model (using 10 epochs for efficient optimization)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Max validation accuracy achieved during training
    best_val_acc = np.max(history.history['val_accuracy'])
    print(f"Best Validation Accuracy for this run: {best_val_acc:.4f}")

    # GPyOpt minimizes the objective, so we return negative accuracy
    return -best_val_acc


if __name__ == '__main__':
    # Define bounds for the 5 hyperparameters:
    # 1. Learning rate (log scale represented via bounds)
    # 2. Number of hidden units
    # 3. Dropout rate
    # 4. L2 regularization weight
    # 5. Batch size
    domain = [
        {'name': 'lr', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (0.0, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128, 256)}
    ]

    print("Starting Bayesian Optimization using GPyOpt...")
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=evaluate_model,
        domain=domain,
        model_type='GP',
        acquisition_type='EI',
        exact_eval=True
    )

    # Run Bayesian optimization for a maximum of 30 iterations
    optimizer.run_optimization(max_iter=30)

    # Extract best hyperparameters and score
    best_params = optimizer.X[np.argmin(optimizer.Y)]
    best_score = -optimizer.fx_opt

    print("\n--- Optimization Complete ---")
    print(f"Best Accuracy: {best_score:.4f}")
    print(f"Best Hyperparameters:")
    print(f"  - Learning Rate: {best_params[0]:.5f}")
    print(f"  - Hidden Units:  {int(best_params[1])}")
    print(f"  - Dropout Rate:  {best_params[2]:.4f}")
    print(f"  - L2 Reg Weight: {best_params[3]:.5f}")
    print(f"  - Batch Size:    {int(best_params[4])}")

    # Save convergence plot
    optimizer.plot_convergence('convergence.png')
    plt.close()

    # Save optimization report to 'bayes_opt.txt'
    report_path = 'bayes_opt.txt'
    with open(report_path, 'w') as f:
        f.write("=== Bayesian Optimization Report ===\n\n")
        f.write(f"Optimal Validation Accuracy: {best_score:.4f}\n\n")
        f.write("Optimal Hyperparameters:\n")
        f.write(f"  - Learning Rate: {best_params[0]:.5f}\n")
        f.write(f"  - Hidden Units:  {int(best_params[1])}\n")
        f.write(f"  - Dropout Rate:  {best_params[2]:.4f}\n")
        f.write(f"  - L2 Reg Weight: {best_params[3]:.5f}\n")
        f.write(f"  - Batch Size:    {int(best_params[4])}\n\n")
        f.write("Optimization History (X, Y):\n")
        for i, (x, y) in enumerate(zip(optimizer.X, optimizer.Y)):
            f.write(f"Iteration {i+1}: Params={x}, Score={-y[0]:.4f}\n")

    print(f"Convergence plot saved to 'convergence.png'.")
    print(f"Report saved to '{report_path}'.")
