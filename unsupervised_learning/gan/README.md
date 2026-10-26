# 0x00. Dimensionality Reduction

## Description
This project covers dimensionality reduction in unsupervised learning:
Principal Component Analysis (PCA) and t-SNE, implemented with NumPy.

## Learning Objectives
- What eigenvalues and eigenvectors are
- What the covariance matrix is
- What dimensionality reduction is and why it is used
- What PCA is and how it works
- What t-SNE is and how it works
- What the Shannon entropy and perplexity are
- What the P and Q affinities are
- How to compute the gradients and the cost of t-SNE

## Requirements
- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- Code follows the `pycodestyle` style (version 2.11.1)
- All files are executable
- All files end with a new line
- All modules, classes, and functions have documentation

## Files

| File | Description |
|------|-------------|
| `0-pca.py` | `def pca(X, var=0.95)`: performs PCA and returns the weights matrix `W` that maintains `var` fraction of the variance of `X` |
| `1-pca.py` | `def pca(X, ndim)`: performs PCA and returns the transformed data `T` of shape `(n, ndim)` |
| `2-P_init.py` | `def P_init(X, perplexity)`: initializes `D`, `P`, `betas` and `H` for the t-SNE P affinities |
| `3-entropy.py` | `def HP(Di, beta)`: calculates the Shannon entropy and P affinities relative to a data point |
| `4-P_affinities.py` | `def P_affinities(X, tol=1e-5, perplexity=30.0)`: calculates the symmetric P affinities of a dataset |
| `5-Q_affinities.py` | `def Q_affinities(Y)`: calculates the Q affinities and their numerator |
| `6-grads.py` | `def grads(Y, P)`: calculates the gradients of Y and the Q affinities |
| `7-cost.py` | `def cost(P, Q)`: calculates the cost of the t-SNE transformation |
| `8-tsne.py` | `def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500)`: performs a t-SNE transformation |

## Usage
Each task has a main file that imports its function, for example:

    ./0-main.py

## Author
Iman Guluzade
