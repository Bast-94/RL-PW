"""
Ce fichier contient des exercices à compléter sur la programmation dynamique.
Il est évalué automatiquement avec pytest, vous pouvez le lancer avec la
commande `pytest exercices.py`.
"""
import os
import random
import typing as t

import gym
import numpy as np
import pytest
from gym import spaces

from dynamic_programming import MDP, GridWorldEnv, StochasticGridWorldEnv
from dynamic_programming.values_iteration import (
    grid_world_value_iteration, mdp_value_iteration,
    stochastic_grid_world_value_iteration)


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


def test_mdp_value_iteration(max_iter: int = 1000):
    mdp = MDP()
    values = mdp_value_iteration(mdp, max_iter=max_iter, gamma=1.0)
    assert np.allclose(values, [-2, -1, 0]), print(values)
    values = mdp_value_iteration(mdp, max_iter=max_iter, gamma=0.9)
    assert np.allclose(values, [-1.9, -1, 0])


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
    if(n%2==1):
        return 0
    if(n<=0):
        return 1
    
    return 4*domino_paving(n-2) - domino_paving(n-4)
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
