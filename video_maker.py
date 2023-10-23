import argparse
import glob

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qlearning import QLearningAgent
from sarsa import SarsaAgent
from taxi import play_and_train
env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n
print(env.render())
fig, ax = plt.subplots()
