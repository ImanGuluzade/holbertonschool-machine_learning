#!/usr/bin/env python3
"""Module that performs SARSA(lambda)"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Chooses an action using the epsilon greedy policy

    Arguments:
        Q: numpy.ndarray of shape (s, a) containing the Q table
        state: the current state
        epsilon: the threshold for exploration

    Returns:
        the next action index
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(lambda)

    Arguments:
        env: the environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q table
        lambtha: the eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value epsilon should decay to
        epsilon_decay: the decay rate for epsilon between episodes

    Returns:
        Q: the updated Q table
    """
    init_epsilon = epsilon
    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        e = np.zeros_like(Q)
        for _ in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            e = e * lambtha * gamma
            e[state, action] += 1
            delta = (reward + gamma * Q[new_state, new_action]
                     - Q[state, action])
            Q += alpha * delta * e
            if terminated or truncated:
                break
            state = new_state
            action = new_action
        epsilon = (min_epsilon + (init_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))

    return Q
