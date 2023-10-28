import gymnasium as gym
import numpy as np

from sarsa import SarsaAgent

# create random array of 5 integers between 0 and 10

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n
sarsa = SarsaAgent(
    learning_rate=0.5,
    gamma=0.99,
    legal_actions=list(range(n_actions)),
    policy="softmax",
)
state = env.reset()
sarsa.play_and_train(env, t_max=5)
