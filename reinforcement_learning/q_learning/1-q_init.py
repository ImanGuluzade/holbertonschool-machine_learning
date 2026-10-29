#!/usr/bin/env python3
"""Module that initializes the Q-table."""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    Arguments:
        env: the FrozenLakeEnv instance

    Returns:
        Q: the Q-table as a numpy.ndarray of zeros
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
