import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

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
    for _ in range(max_iter):
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
