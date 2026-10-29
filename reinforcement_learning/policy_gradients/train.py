#!/usr/bin/env python3
"""Module that implements a full training using policy gradients."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training loop using policy gradients (REINFORCE)

    Arguments:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: boolean, whether to render the environment every 1000 episodes

    Returns:
        scores: all values of the score (sum of all rewards during one episode)
    """
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    weights = np.random.rand(n_states, n_actions)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state[np.newaxis, :]  # Shape (1, n_states)
        episode_transitions = []

        while True:
            action, grad = policy_gradient(state, weights)

            # Take step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_transitions.append((state, action, reward, grad))

            state = next_state[np.newaxis, :]
            if terminated or truncated:
                break

        # Calculate scores and rewards-to-go for gradient update
        score = sum(t[2] for t in episode_transitions)
        scores.append(score)

        # Render environment every 1000 episodes if show_result is True
        if show_result and episode % 1000 == 0:
            env.render()

        # Monte Carlo policy gradient update (REINFORCE)
        G = 0
        for t in reversed(range(len(episode_transitions))):
            state, action, reward, grad = episode_transitions[t]
            G = gamma * G + reward
            weights += alpha * G * grad

        print(f"Episode: {episode} Score: {score}")

    return scores
