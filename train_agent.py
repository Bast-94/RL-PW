from taxi import play_and_train
from qlearning import QLearningAgent
from tqdm import tqdm
import gymnasium as gym

def train(agent:QLearningAgent, env:gym.Env, t_max:int = int(1e4), num_episodes:int = int(1e4), recording:bool = False):
    for i in tqdm(range(num_episodes)):
        play_and_train(env, agent, t_max)
    return agent

