import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qlearning import QLearningAgent
from sarsa import SarsaAgent
from taxi import play_and_train

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=1000)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-i", "--img_output_file", type=str, default=None)
img_output_file = parser.parse_args().img_output_file
verbose = parser.parse_args().verbose
epochs = parser.parse_args().epochs
env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore
s, _ = env.reset()
env.step(0)
plt.imshow(env.render())
plt.savefig("taxi.png")
env.step(1)
plt.imshow(env.render())
plt.savefig("taxi1.png")
