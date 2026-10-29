#!/usr/bin/env python3
"""
Module containing a function that has a trained agent play an episode.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode.

    Parameters:
    - env: The FrozenLakeEnv instance.
    - Q (numpy.ndarray): The Q-table.
    - max_steps (int): The maximum number of steps in the episode.
      Defaults to 100.

    Returns:
    - total_rewards (float): The total rewards for the episode.
    - rendered_outputs (list): A list of strings representing the board
      state at each step.
    """
    state, _ = env.reset()

    rendered_outputs = []
    # Render initial state
    rendered_outputs.append(env.render())

    total_rewards = 0.0

    for _ in range(max_steps):
        # Always exploit: pick the action with the maximum Q-value
        # for the state
        action = np.argmax(Q[state, :])

        # Take the action
        state, reward, terminated, truncated, _ = env.step(action)

        # Accumulate reward
        total_rewards += reward

        # Render the board state and append it to the list
        rendered_outputs.append(env.render())

        # Check if the episode is finished
        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
