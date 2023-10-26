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

agents = [q_learning_agent]#, eps_scheduling_agent, sarsa_agent]
agent_names = ["q_learning", "eps_scheduling", "sarsa"]
fps = 16
nb_step = 1
for agent,name in zip(agents,agent_names):
    print("Training agent: ", name)
    ep_per_step = 1000
    out = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
    for train_step in tqdm(range(1,nb_step+1)):
        print(train_step)
        train(agent, env, t_max, num_episodes=ep_per_step)
        print("Done")
        print("Playing and recording agent: ", name)
        
        s, _ = env.reset()         
        for i in tqdm(range(0,t_max)):
            action = agent.get_action(s)
            next_s, r, done, _, _ = env.step(action)
            s = next_s
              
            img = env.render()
            if(i==0):
                img = cv2.putText(img, f"Agent: {name} after {train_step*ep_per_step} episodes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(img)
            if done:
                break

    out.release()