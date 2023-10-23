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


def q_learning(epochs=1000, verbose=True, img_output_file=None):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    n_actions = env.action_space.n
    agent = QLearningAgent(
        learning_rate=1,
        epsilon=0,
        gamma=1,
        legal_actions=list(range(n_actions)),
    )
    step = 100
    rewards = []
    mean_rewards = []
    for i in tqdm(range(epochs)):
        rewards.append(play_and_train(env, agent))
        if i % step == 0:
            if verbose:
                print("mean reward", np.mean(rewards[-step:]))
            mean_rewards.append(np.mean(rewards[-step:]))

    if img_output_file is not None:
        fig, ax = plt.subplots()
        smooth_curve = np.convolve(rewards, np.ones((step,)) / step, mode="valid")
        ax.plot(rewards, color="blue")
        ax.plot(smooth_curve, color="red")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Rewards")
        # ax.set_yscale("log")
        ax.set_title("Rewards per epoch")
        fig.savefig(img_output_file)
    assert np.mean(rewards[-step:]) > 0.0, print(np.mean(rewards[-step:]))
    print("mean reward", np.mean(rewards[-step:]))


# q_learning(epochs=epochs, verbose=verbose, img_output_file=img_output_file)
env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n
agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []


for i in tqdm(range(1000)):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
