#!/usr/bin/env python3
"""Module that computes the policy and its gradient."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix.

    Arguments:
        matrix: numpy.ndarray representing the state
        weight: numpy.ndarray representing the weights

    Returns:
        The policy (action probabilities) as a numpy.ndarray
    """
    z = np.dot(matrix, weight)
    e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e_z / np.sum(e_z, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte Carlo policy gradient based on a state and weight.

    Arguments:
        state: numpy.ndarray matrix of shape (1, s) representing the current state
        weight: numpy.ndarray of shape (s, a) representing the weights

    Returns:
        action: the selected action index
        grad: the gradient of the policy with respect to weights
    """
    # 1. Compute action probabilities using the policy function
    probs = policy(state, weight)

    # 2. Sample an action based on the probabilities
    action = np.random.choice(len(probs[0]), p=probs[0])

    # 3. Compute the gradient of the log-probability of the chosen action
    # For a softmax policy, d(log(pi_i)) / d(W) = x * (one_hot - pi)
    dsoftmax = -probs.T @ probs
    dsoftmax[action] += probs[0][action]

    # Alternatively expressed via the outer product of state and gradient vector:
    # grad = state.T @ (one_hot - probs)
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1.0
    d_z = one_hot - probs
    grad = state.T @ d_z

    return action, grad
