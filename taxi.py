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
                artifact_dir="artifacts",
                t_max=int(1e4),
                env=env,
            )
    assert np.mean(rewards[-100:]) > 0.0
    return rewards


if __name__ == "__main__":
    #################################################
    # 1. Play with QLearningAgent
    #################################################
    print("QLEARNING")
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
    )
    ql_rewards = get_rewards_and_generate_gifs(agent, env, "qlearning")

    #################################################
    # 2. Play with QLearningAgentEpsScheduling
    #################################################
    print("QLEARNING EPS")
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
    )

    ql_eps_rewards = get_rewards_and_generate_gifs(agent, env, "qlearning-eps")
    ####################
    # 3. Play with SARSA
    ####################

    print("SARSA")
    agent = SarsaAgent(
        learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions))
    )
    sarsa_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa")
    #################################
    # 3. Play with SARSA with softmax
    #################################
    agent = SarsaAgent(
        learning_rate=0.5,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
        policy="softmax",
    )
    sarsa_softmax_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa-softmax")
    # plot rewards
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # rolling window along the rewards to smooth the curve

    window = 50
    ql_rewards = np.array(ql_rewards)
    ql_eps_rewards = np.array(ql_eps_rewards)
    sarsa_rewards = np.array(sarsa_rewards)
    sarsa_softmax_rewards = np.array(sarsa_softmax_rewards)

    ql_rewards = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
    ql_eps_rewards = np.convolve(ql_eps_rewards, np.ones(window) / window, mode="valid")
    sarsa_rewards = np.convolve(sarsa_rewards, np.ones(window) / window, mode="valid")
    sarsa_softmax_rewards = np.convolve(
        sarsa_softmax_rewards, np.ones(window) / window, mode="valid"
    )

    ax.plot(ql_rewards, label="Q-Learning", color="red")
    ax.plot(ql_eps_rewards, label="Q-Learning Epsilon Scheduling", color="green")
    ax.plot(sarsa_rewards, label="SARSA", color="blue")
    ax.plot(sarsa_softmax_rewards, label="SARSA Softmax", color="orange")
    ax.set_xlabel("Episode")

    ax.set_ylabel("Reward")
    ax.set_title(
        f"Rewards for different algorithms, smoothed over window size {window}"
    )
    ax.legend()
    img_path = os.path.join("artifacts", "rewards.png")
    plt.savefig(img_path)
