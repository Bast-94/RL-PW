import numpy as np

from dynamic_programming.mdp import MDP

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
