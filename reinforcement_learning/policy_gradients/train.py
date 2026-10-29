#!/usr/bin/env python3
"""Module that implements a full training using policy gradients."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


mo
def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a full training loop using policy gradients (REINFORCE)

    Arguments:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor

    Returns:
        scores: all values of the score (sum of all rewards during one episode)
    """
    # Initialize weights randomly with proper dimensions for CartPole-v1
    # CartPole observation space has 4 features, action space has 2 actions
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
        scores.append(sum(t[2] for t in episode_transitions))

        # Monte Carlo policy gradient update (REINFORCE)
        G = 0
        for t in reversed(range(len(episode_transitions))):
            state, action, reward, grad = episode_transitions[t]
            G = gamma * G + reward
            # Update weights using the policy gradient and accumulated discounted return
            weights += alpha * G * grad

        print(f"Episode: {episode} Score: {scores[-1]}")

    return scores
