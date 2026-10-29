#!/usr/bin/env python3
"""Module that has the trained agent play an episode."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    Arguments:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: the total rewards for the episode
        rendered_outputs: a list of rendered outputs representing the board state
    """
    state, _ = env.reset()
    rendered_outputs = []

    # Capture the initial state board rendering
    rendered_outputs.append(env.render())
    total_rewards = 0

    for _ in range(max_steps):
        # Always exploit the Q-table (choose action with highest Q-value)
        action = np.argmax(Q[state])

        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward

        # Capture the board state after the step
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
