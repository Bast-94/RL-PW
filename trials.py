import argparse

import gymnasium as gym
import matplotlib.pyplot as plt

from qlearning import QLearningAgent
from taxi import play_and_train, q_learning

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=1000)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-i", "--img_output_file", type=str, default=None)
img_output_file = parser.parse_args().img_output_file
verbose = parser.parse_args().verbose
epochs = parser.parse_args().epochs

q_learning(epochs=epochs, verbose=verbose, img_output_file=img_output_file)
