#!/usr/bin/env python3
"""Module that performs Q-learning."""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    Arguments:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q-table
        total_rewards: a list containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # When the agent falls in a hole, the reward should be -1
            if terminated and reward == 0:
                reward = -1

            # Q-learning update formula
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            Q[state, action] = old_value + alpha * (
                reward + gamma * next_max - old_value
            )

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        # Decay epsilon down to min_epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, total_rewards
