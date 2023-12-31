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
        learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)), epsilon=0.05
    )
    sarsa_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa")
    #################################
    # 4. Play with SARSA with softmax
    #################################
    agent = SarsaAgent(
        learning_rate=0.5,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
        policy="softmax",
    )
    sarsa_softmax_rewards = get_rewards_and_generate_gifs(agent, env, "sarsa-softmax")
    #################################
    # 5. Play with Q-learning with softmax
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

    # plot rewards
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # rolling window along the rewards to smooth the curve

    window = 100
    final_ql_reward = np.mean(ql_rewards[-window:])
    final_ql_eps_reward = np.mean(ql_eps_rewards[-window:])
    final_sarsa_reward = np.mean(sarsa_rewards[-window:])
    final_sarsa_softmax_reward = np.mean(sarsa_softmax_rewards[-window:])
    final_ql_eps_softmax_reward = np.mean(ql_eps_softmax_rewards[-window:])

    # compute the final std for each algorithm

    std_ql_reward = np.std(ql_rewards[-window:])
    std_ql_eps_reward = np.std(ql_eps_rewards[-window:])
    std_sarsa_reward = np.std(sarsa_rewards[-window:])
    std_sarsa_softmax_reward = np.std(sarsa_softmax_rewards[-window:])
    std_ql_eps_softmax_reward = np.std(ql_eps_softmax_rewards[-window:])

    ql_rewards = np.array(ql_rewards)
    ql_eps_rewards = np.array(ql_eps_rewards)
    sarsa_rewards = np.array(sarsa_rewards)
    sarsa_softmax_rewards = np.array(sarsa_softmax_rewards)
    ql_eps_softmax_rewards = np.array(ql_eps_softmax_rewards)

    ql_rewards = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
    ql_eps_rewards = np.convolve(ql_eps_rewards, np.ones(window) / window, mode="valid")
    sarsa_rewards = np.convolve(sarsa_rewards, np.ones(window) / window, mode="valid")
    sarsa_softmax_rewards = np.convolve(
        sarsa_softmax_rewards, np.ones(window) / window, mode="valid"
    )
    ql_eps_softmax_rewards = np.convolve(
        ql_eps_softmax_rewards, np.ones(window) / window, mode="valid"
    )

    ax.plot(ql_rewards, label="Q-Learning", color="red")
    ax.plot(ql_eps_rewards, label="Q-Learning Epsilon Scheduling", color="green")
    ax.plot(sarsa_rewards, label="SARSA", color="blue")
    ax.plot(sarsa_softmax_rewards, label="SARSA Softmax", color="orange")
    ax.plot(
        ql_eps_softmax_rewards,
        label="Q-Learning Softmax",
        color="purple",
    )
    text_kwargs = dict(
        ha="center", va="center", fontsize=12, transform=ax.transAxes, color="black"
    )
    text_to_plot = f"Mean reward over last {window} episodes"
    text_to_plot += (
        "\n" + f"Q-Learning: {final_ql_reward:.2f}, std: {std_ql_reward:.2f}"
    )
    text_to_plot += (
        "\n"
        + f"Q-Learning Epsilon Scheduling: {final_ql_eps_reward:.2f}, std: {std_ql_eps_reward:.2f}"
    )
    text_to_plot += (
        "\n" + f"SARSA with $\varepsilon$ = 0.05 : {final_sarsa_reward:.2f}, std: {std_sarsa_reward:.2f}"
    )
    text_to_plot += (
        "\n"
        + f"SARSA Softmax: {final_sarsa_softmax_reward:.2f} ,std: {std_sarsa_softmax_reward:.2f}"
    )
    text_to_plot += (
        "\n"
        + f"Q-Learning with softmax: {final_ql_eps_softmax_reward:.2f}, std: {std_ql_eps_softmax_reward:.2f}"
    )
    ax.text(0.5, 0.5, text_to_plot, **text_kwargs)

    ax.set_xlabel("Episode")

    ax.set_ylabel("Reward")
    ax.set_title(
        f"Rewards for different algorithms, smoothed over window size {window}"
    )
    ax.legend()
    img_path = os.path.join(IMG_DIR, "rewards.png")
    plt.savefig(img_path)
