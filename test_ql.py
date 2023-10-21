from qlearning import QLearningAgent

from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym
import os

print(os.getcwd())


def test_a():
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    n_actions = env.action_space.n  # type: ignore
    agent = QLearningAgent(
        learning_rate=0.5,
        epsilon=0.25,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
    )
