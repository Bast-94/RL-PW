import gymnasium as gym
from tqdm import tqdm

from qlearning import QLearningAgent
from sarsa import SarsaAgent


def train(
    agent: QLearningAgent | SarsaAgent,
    env: gym.Env,
    t_max: int = int(1e4),
    num_episodes: int = int(1e4),
):
    for _ in tqdm(range(num_episodes)):
        agent.play_and_train(env, t_max)
    return agent
