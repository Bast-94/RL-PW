"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import logging
import os
import typing as t

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from video_maker import create_gif

# from video_maker import create_gif
env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore
IMG_DIR = "img"


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """

    return agent.play_and_train(env, t_max)


EP_STAGES = [250, 500, 1000]


def get_rewards_and_generate_gifs(agent, env, name):
    rewards = []
    nb_episode = max(EP_STAGES)
    for i in range(1, nb_episode + 1):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0:
            print("mean reward", np.mean(rewards[-100:]))
        if i in EP_STAGES:
            print("generate gif for stage", i)
            create_gif(
                agent=agent,
                name=f"{name}-{i}-ep",
                artifact_dir=IMG_DIR,
                t_max=int(1e4),
                env=env,
            )
    assert np.mean(rewards[-100:]) > 0.0
    return rewards


def smooth_curve(array: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Smooths the curve by averaging values with a given window size.
    """
    return np.convolve(array, np.ones(window_size) / window_size, mode="valid")


if __name__ == "__main__":
    
    trial = []
    #################################################
    # 1. Play with QLearningAgent
    #################################################
    print("QLEARNING")
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
    )
    ql_rewards = get_rewards_and_generate_gifs(agent, env, "qlearning")
    trial.append(dict(agent=agent,name="Q-Learning", rewards=ql_rewards,color_curve="blue"))
    #################################################
    # 2. Play with QLearningAgentEpsScheduling
    #################################################
    print("QLEARNING EPS")
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
    )

    ql_eps_rewards = get_rewards_and_generate_gifs(agent, env, "qlearning-eps")
    trial.append(dict(agent=agent,name="Q-Learning Epsilon Scheduling", rewards=ql_eps_rewards,color_curve="red"))
    

    print("SARSA")
    agent = SarsaAgent(
        learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)),epsilon=0.05
    )
    sarsa_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa")
    trial.append(dict(agent=agent,name="SARSA", rewards=sarsa_rewards,color_curve="green"))
    #####################################
    # 4. Play with SARSA with epsilon = 0
    #####################################
    print("SARSA EPSILON = 0")
    agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)),epsilon=0.0)
    sarsa_eps_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa-eps0")
    trial.append(dict(agent=agent,name="SARSA Epsilon = 0", rewards=sarsa_eps_rewards,color_curve="black"))
    #################################
    # 5. Play with SARSA with softmax
    #################################
    agent = SarsaAgent(
        learning_rate=0.5,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
        policy="softmax",
    )
    sarsa_softmax_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa-softmax")
    trial.append(dict(agent=agent,name="SARSA Softmax", rewards=sarsa_softmax_rewards,color_curve="orange"))
    #################################
    # 6. Play with Q-learning with softmax
    #################################
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5,
        epsilon=0.1,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
        policy="softmax",
    )
    ql_eps_softmax_rewards = get_rewards_and_generate_gifs(
        agent, env, "qlearning-eps-softmax"
    )
    trial.append(dict(agent=agent,name="Q-Learning Softmax", rewards=ql_eps_softmax_rewards,color_curve="purple"))
    # plot rewards
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # rolling window along the rewards to smooth the curve

    window = 100
    
    text_kwargs = dict(ha="center", va="center", fontsize=12, transform=ax.transAxes, color="black"
    )
    text_to_plot = f"Mean reward over last {window} episodes"
    for trial in trial:
        trial['final_reward'] = np.mean(trial['rewards'][-window:])
        trial['std_reward'] = np.std(trial['rewards'][-window:])
        trial['smooth_rewards'] = np.convolve(trial['rewards'], np.ones(window) / window, mode="valid")
        ax.plot(trial['smooth_rewards'], label=trial['name'],color=trial['color_curve'])
        text_to_plot += "\n" + f"{trial['name']}: {trial['final_reward']:.2f}, std: {trial['std_reward']:.2f}"
        
    
    ax.text(0.5, 0.5, text_to_plot, **text_kwargs)

    ax.set_xlabel("Episode")

    ax.set_ylabel("Reward")
    ax.set_title(
        f"Rewards for different algorithms, smoothed over window size {window}"
    )
    ax.legend()
    img_path = os.path.join(IMG_DIR, "rewards.png")
    plt.savefig(img_path)
