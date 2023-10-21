from qlearning import QLearningAgent
import pytest
from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym
import os

@pytest.fixture
def env():
    return gym.make("Taxi-v3", render_mode="rgb_array")

@pytest.fixture
def agent(env):
    n_actions = env.action_space.n
    return QLearningAgent(
        learning_rate=0.5,
        epsilon=0.25,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
    )

def test_a(env,agent):
    
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            best = agent.get_best_action(state)
            assert isinstance(best, int)
            assert best in range(env.action_space.n)
    
    
            
