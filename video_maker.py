import cv2
import gymnasium as gym
import numpy as np
from taxi import play_and_train
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from tqdm import tqdm
from train_agent import train
env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n
s, _ = env.reset() 

height,width = env.render().shape[:2]
frameSize = (width,height)





t_max = int(1e3)
q_learning_agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)
eps_scheduling_agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)
sarsa_agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions))) 

agents = [q_learning_agent, eps_scheduling_agent, sarsa_agent]
agent_names = ["q_learning", "eps_scheduling", "sarsa"]

outs = [ cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize) for name in agent_names]
for agent,name,out in zip(agents,agent_names,outs):
    print("Training agent: ", name)
    train(agent, env, t_max, num_episodes=1000, recording=False)
    print("Done")
    print("Playing and recording agent: ", name)
    s, _ = env.reset()            
    for i in tqdm(range(0,t_max)):
        action = agent.get_action(s)
        next_s, r, done, _, _ = env.step(action)
        s = next_s
        
        img = env.render()
        out.write(img)
        if done:
            break

    out.release()