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


#################################################
# 1. Play with QLearningAgent
#################################################
agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """

    return agent.play_and_train(env, t_max)


rewards = []
if __name__ == "__main__":
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0:
            print("mean reward", np.mean(rewards[-100:]))

    assert np.mean(rewards[-100:]) > 0.0
    create_gif(
        agent=agent,
        name="qlearning",
        ep_per_step=1000,
        nb_step=1,
        artifact_dir="artifacts",
        t_max=int(1e4),
        env=env,
    )

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)
if __name__ == "__main__":
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0:
            print("mean reward", np.mean(rewards[-100:]))

    assert np.mean(rewards[-100:]) > 0.0
    create_gif(
        agent=agent,
        name="qlearning_eps_scheduling",
        ep_per_step=1000,
        nb_step=1,
        artifact_dir="artifacts",
        t_max=int(1e4),
        env=env,
    )


####################
# 3. Play with SARSA
####################

if __name__ == "__main__":
    agent = SarsaAgent(
        learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions))
    )

    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0:
            print("mean reward", np.mean(rewards[-100:]))

    assert np.mean(rewards[-100:]) > 0.0
    create_gif(
        agent=agent,
        name="sarsa",
        ep_per_step=1000,
        nb_step=1,
        artifact_dir="artifacts",
        t_max=int(1e4),
        env=env,
    )
