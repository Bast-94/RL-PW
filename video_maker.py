import cv2
import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm

from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from taxi import play_and_train
from train_agent import train

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n
s, _ = env.reset()

height, width = env.render().shape[:2]
frameSize = (width, height)


t_max = int(1e3)
q_learning_agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)
eps_scheduling_agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)
sarsa_agent = SarsaAgent(
    learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions))
)

agents = [q_learning_agent, eps_scheduling_agent, sarsa_agent]
agent_names = ["q_learning", "eps_scheduling", "sarsa"]
fps = 16
nb_step = 10
artifact_dir = "artifacts"
ep_per_step = 100


def create_gif(agent, name,ep_per_step, nb_step,artifact_dir):
    
    gif_writer = imageio.get_writer(f"{artifact_dir}/{name}.gif", mode="I")
    for train_step in tqdm(range(1, nb_step + 1)):
        print(train_step)
        train(agent, env, t_max, num_episodes=ep_per_step)
        print("Done")
        print("Playing and recording agent: ", name)

        s, _ = env.reset()
        for i in tqdm(range(0, t_max)):
            action = agent.get_action(s)
            next_s, r, done, _, _ = env.step(action)
            s = next_s

            img = env.render()
            img = cv2.putText(
                img,
                f"Agent: {name} after {train_step*ep_per_step} episodes",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            gif_writer.append_data(img)
            if done:
                break

    gif_writer.close()
    
