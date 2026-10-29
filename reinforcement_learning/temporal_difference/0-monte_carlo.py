#!/usr/bin/env python3
"""Module that performs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm

    Arguments:
        env: the environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns the next
            action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate

    Returns:
        V: the updated value estimate
    """
    for i in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append([state, action, reward, new_state])
            if terminated or truncated:
                break
            state = new_state

        episode = np.array(episode, dtype=int)
        G = 0
        for j, step in enumerate(episode[::-1]):
            s, _, r, _ = step
            G = gamma * G + r
            if s not in episode[:i - j]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
