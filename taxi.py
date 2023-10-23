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

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore
logging.basicConfig(level=logging.INFO)
output_file_log = open("output.log", "w")

#################################################
# 1. Play with QLearningAgent
#################################################


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for i in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION

        total_reward += r
        agent.update(state=s, action=a, next_state=next_s, reward=r)
        s = next_s
        if done:
            break
        # END SOLUTION

    return total_reward


def q_learning(epochs=1000, verbose=True, img_output_file=None):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    n_actions = env.action_space.n
    agent = QLearningAgent(
        learning_rate=0.5,
        epsilon=0.25,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
    )

    rewards = []
    for i in tqdm(range(epochs)):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0 and verbose:
            print("mean reward", np.mean(rewards[-100:]))
    if img_output_file is not None:
        fig, ax = plt.subplots()
        smooth_curve = np.convolve(rewards, np.ones((100,)) / 100, mode="valid")
        ax.plot(rewards, color="blue")
        ax.plot(smooth_curve, color="red")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Rewards")
        ax.set_title("Rewards per epoch")
        fig.savefig(img_output_file)
    assert np.mean(rewards[-100:]) > 0.0


if __name__ == "__main__":
    q_learning()
    # TODO: créer des vidéos de l'agent en action

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)
if __name__ == "__main__":
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0:
            print("mean reward", np.mean(rewards[-100:]))

    assert np.mean(rewards[-100:]) > 0.0

    # TODO: créer des vidéos de l'agent en action


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