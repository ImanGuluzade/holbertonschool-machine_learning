#!/usr/bin/env python3
"""
Module containing the Q-learning training function.
"""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning on a FrozenLake environment.

    Parameters:
    - env: The FrozenLakeEnv instance from gymnasium.
    - Q (numpy.ndarray): The Q-table to update.
    - episodes (int): The total number of episodes to train over.
    - max_steps (int): The maximum number of steps per episode.
    - alpha (float): The learning rate.
    - gamma (float): The discount rate.
    - epsilon (float): The initial epsilon-greedy exploration threshold.
    - min_epsilon (float): The minimum epsilon value after decay.
    - epsilon_decay (float): The decay rate for epsilon between episodes.

    Returns:
    - tuple: The updated Q-table and a list of rewards per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        reward_sum = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            reward_sum += reward
            state = new_state

            if terminated or truncated:
                break

        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )
        total_rewards.append(reward_sum)

    return Q, total_rewards
