#!/usr/bin/env python3
"""Module that performs the Monte Carlo algorithm for value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm (First-Visit MC)

    Arguments:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate

    Returns:
        V: the updated value estimate
    """
    for _ in range(episodes):
        # Compatible with both gym and gymnasium reset()
        reset_res = env.reset()
        state = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        episode_data = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            step_res = env.step(action)

            # Compatible with both gym (4 values) and gymnasium (5 values)
            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res

            episode_data.append((state, reward))
            state = next_state

            if done:
                break

        # First-Visit Monte Carlo update
        G = 0
        visited_states = set()

        for t in reversed(range(len(episode_data))):
            state, reward = episode_data[t]
            G = gamma * G + reward

            # First-visit check: update only if this is the first time
            # the state appeared in this episode
            if state not in visited_states:
                visited_states.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
