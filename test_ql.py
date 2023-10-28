import os
import random
import typing as t
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pytest

from qlearning import QLearningAgent
from sarsa import SarsaAgent


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


@pytest.fixture
def softmax_sarsa_agent(env):
    n_actions = env.action_space.n
    return SarsaAgent(
        learning_rate=0.5,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
        policy="softmax",
    )


def test_a(env, agent):
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            best = agent.get_best_action(state)
            assert isinstance(best, int)
            assert best in range(env.action_space.n)


def test_b(env, agent):
    env.reset()
    done = False
    i = 0
    while not done:
        res = env.step(random.choice(range(env.action_space.n)))
        done = res[3]
        if i == 1:
            print(res)
        i += 1
    assert i <= 200


def test_softmax(softmax_sarsa_agent: SarsaAgent, env):
    env.reset()
    softmax_sarsa_agent.play_and_train(env, t_max=5)
