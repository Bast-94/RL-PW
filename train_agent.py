import gymnasium as gym
from tqdm import tqdm

from qlearning import QLearningAgent
from taxi import play_and_train


def train(
    agent: QLearningAgent,
    env: gym.Env,
    t_max: int = int(1e4),
    num_episodes: int = int(1e4),
):
    for _ in tqdm(range(num_episodes), desc="Training"):
        agent.play_and_train(env, t_max)
    return agent
