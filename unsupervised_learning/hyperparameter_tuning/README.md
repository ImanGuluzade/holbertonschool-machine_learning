# Supervised & Unsupervised Machine Learning - Autoencoders

This project contains implementations of various autoencoder architectures using Python and TensorFlow / Keras, developed as part of machine learning curriculum exercises.

## Technologies Used
* **Python 3**
* **TensorFlow / Keras**
* **NumPy & Matplotlib**
* **Pycodestyle** (for PEP 8 compliance)

## Implemented Architectures
1. **Vanilla Autoencoder (`0-vanilla.py`)**: Standard encoder-decoder architecture with dense layers and ReLU/sigmoid activations.
2. **Variational Autoencoder (`3-variational.py`)**: Generative model implementing the reparameterization trick, probabilistic latent spaces ($\mu$, $\log\sigma$), and a custom KL divergence + reconstruction loss function.

## Usage
Run the main evaluation scripts to train and visualize model reconstructions:
```bash
./0-main.py
./3-main.py
