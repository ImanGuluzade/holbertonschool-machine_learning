#!/usr/bin/env python3
"""Module that uses epsilon-greedy to determine the next action."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Arguments:
        Q: numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation

    Returns:
        the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore: pick a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: pick the action with the highest Q-value for the state
        action = np.argmax(Q[state])

    return action
