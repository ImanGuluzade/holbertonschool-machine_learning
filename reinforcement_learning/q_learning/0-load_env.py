#!/usr/bin/env python3
"""Module that loads the FrozenLake environment."""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment

    Arguments:
        desc: list of lists outlining the custom map
        map_name: string of the map to load
        is_slippery: boolean to determine if ice is slippery

    Returns:
        The FrozenLakeEnv instance with render_mode='ansi'
    """
    return gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode='ansi'
    )
