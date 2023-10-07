"""
Ce fichier contient des exercices à compléter sur la programmation dynamique.
Il est évalué automatiquement avec pytest, vous pouvez le lancer avec la
commande `pytest exercices.py`.
"""
import random
import typing as t

import gym
import numpy as np
import pytest
from gym import spaces

from dynamic_programming import MDP, GridWorldEnv


# Tests pour l'exercice 1
def test_mdp():
    mdp = MDP()
    assert mdp.P[0][0] == (1, -1, False)
    assert mdp.P[0][1] == (0, -1, False)
    assert mdp.P[1][0] == (0, -1, False)
    assert mdp.P[1][1] == (2, -1, False)
    assert mdp.P[2][0] == (2, 0, False)
    assert mdp.P[2][1] == (0, -1, False)

    mdp.reset()
    ret = mdp.step(0)
    assert ret[0] in [0, 1, 2]
    assert ret[1] in [0, -1]
    assert ret[2] in [True, False]
    assert isinstance(ret[3], dict)


# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for i in range(max_iter):
        prev_val = np.copy(values)

        for state in range(len(mdp.P)):
            mdp.reset_state(state)
            best_val = float("-inf")

            for action in range(len(mdp.P[0])):
                next_state, reward, _, _ = mdp.step(action, transition=False)
                current_val = reward + gamma * prev_val[next_state]

                if current_val > best_val:
                    prev_val[state] = current_val
                    best_val = current_val

        values = prev_val

    # END SOLUTION
    return values


def test_mdp_value_iteration(max_iter: int = 1000):
    mdp = MDP()
    values = mdp_value_iteration(mdp, max_iter=max_iter, gamma=1.0)
    assert np.allclose(values, [-2, -1, 0]), print(values)
    values = mdp_value_iteration(mdp, max_iter=max_iter, gamma=0.9)
    assert np.allclose(values, [-1.9, -1, 0])


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    diff = theta
    i = 0
    while i < max_iter and diff >= theta:
        prev_val = np.copy(values)
        for row in range(env.height):
            for col in range(env.width):
                best_val = float("-inf")
                state = (row, col)
                env.set_state(*state)
                for action in range(env.action_space.n):
                    next_state, reward, _, _ = env.step(action, make_move=False)

                    # env.current_position = state
                    current_val = (
                        reward + gamma * prev_val[next_state]
                    ) * env.moving_prob[row, col, action]
                    if current_val > best_val:
                        values[state] = current_val
                        best_val = current_val

        diff = np.max(np.abs(values - prev_val))
        i += 1

    return values
    # END SOLUTION


def test_grid_world_value_iteration(max_iter=1000):
    env = GridWorldEnv()

    values = grid_world_value_iteration(env, max_iter, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert np.allclose(values, solution), print(values)

    values = grid_world_value_iteration(env, max_iter, gamma=0.9)
    solution = np.array(
        [
            [0.81, 0.9, 1.0, 0.0],
            [0.729, 0.0, 0.9, 0.0],
            [0.6561, 0.729, 0.81, 0.729],
            [0.59049, 0.6561, 0.729, 0.6561],
        ]
    )
    assert np.allclose(values, solution)


# Exercice 4: GridWorld avec du bruit
# -----------------------------------
# Ecrire une fonction qui calcule la fonction de valeur pour le GridWorld
# avec du bruit.
# Le bruit est un mouvement aléatoire de l'agent vers sa gauche ou sa droite avec une probabilité de 0.1.


class StochasticGridWorldEnv(GridWorldEnv):
    def __init__(self):
        super().__init__()
        self.moving_prob = np.ones(shape=(self.height, self.width, self.action_space.n))
        zero_mask = (self.grid == "W") | (self.grid == "P") | (self.grid == "N")
        self.moving_prob[np.where(zero_mask)] = 0
        # self.moving_prob[np.where(~zero_mask)][:1] = 0.9
        # self.moving_prob[np.where(~zero_mask)][1:] = 0.05

    def _add_noise(self, action: int) -> int:
        prob = random.uniform(0, 1)
        if prob < 0.05:  # 5% chance to go left
            return (action - 1) % 4
        elif prob < 0.1:  # 5% chance to go right
            return (action + 1) % 4
        # 90% chance to go in the intended direction
        return action

    def get_next_states(
        self, action: int
    ) -> list[
        tuple[int, float, float, bool]
    ]:  # return list of (next_state, reward, probability, done)
        possible_actions = [(action - 1) % 4, (action + 1) % 4, action]
        probs = [0.05, 0.05, 0.9]
        res = []
        for action, prob in zip(possible_actions, probs):
            next_state, reward, is_done, _ = super().step(action, make_move=False)
            res.append((next_state, reward, prob, is_done, action))

        return res

    def step(self, action, make_move: bool = True):
        action = self._add_noise(action)
        return super().step(action, make_move)


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    diff = theta
    i = 0
    while i < max_iter and diff >= theta:
        prev_val = np.copy(values)
        for row in range(env.height):
            for col in range(env.width):
                best_val = float("-inf")
                state = (row, col)
                env.set_state(*state)
                for action in range(env.action_space.n):
                    next_states = env.get_next_states(action=action)
                    current_sum = 0
                    for next_state, reward, probability, _, _ in next_states:
                        # print(next_state, reward, probability, _, _)
                        current_sum += probability * (
                            reward + gamma * prev_val[next_state]
                        )
                    if current_sum > best_val:
                        best_val = current_sum
                        values[state] = best_val

        diff = np.max(np.abs(values - prev_val))
        i += 1

    return values
    # END SOLUTION
    return values


def test_stochastic_grid_world_value_iteration(max_iter=1000):
    env = StochasticGridWorldEnv()

    values = stochastic_grid_world_value_iteration(env, max_iter=max_iter, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(values, solution), print("  ", values)

    values = stochastic_grid_world_value_iteration(env, max_iter=max_iter, gamma=0.9)
    solution = np.array(
        [
            [0.77495822, 0.87063224, 0.98343293, 0.0],
            [0.68591168, 0.0, 0.77736888, 0.0],
            [0.60732544, 0.60891859, 0.68418232, 0.60570595],
            [0.54079452, 0.54500607, 0.60570595, 0.53914484],
        ]
    )
    assert np.allclose(values, solution), print("  ", values)


# Exercice 3: Evaluation de politique
# -----------------------------------
# Ecrire une fonction qui évalue la politique suivante:
#   - état 0, action 0
#   - état 1, action 0
#   - état 2, action 1


"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    # BEGIN SOLUTION
    if n in [0, 1]:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
    # END SOLUTION


# Tests pour l'exercice 1
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci(n, expected):
    assert fibonacci(n) == expected


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.


def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """

    # BEGIN SOLUTION
    def fibo_rec(n: int, a1: int, a2: int):
        if n == 0:
            return a1
        return fibo_rec(n - 1, a2, a1 + a2)

    return fibo_rec(n, 0, 1)
    # END SOLUTION


# Tests pour l'exercice 2
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci_memo(n, expected):
    assert fibonacci_memo(n) == expected


# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    # END SOLUTION


# Tests pour l'exercice 3
@pytest.mark.parametrize(
    "n,expected",
    [
        (1, 0),
        (2, 3),
        (3, 0),
        (4, 11),
        (5, 0),
        (6, 41),
        (7, 0),
        (8, 153),
        (9, 0),
        (10, 571),
    ],
)
def test_domino_paving(n, expected):
    assert domino_paving(n) == expected


def test_wall():
    env = GridWorldEnv()
    for i in range(2):
        env.step(0)
    old_position = env.current_position
    env.step(3)
    assert old_position == env.current_position
