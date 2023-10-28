import numpy as np


def softmax_policy(agent, state):
    action_values = np.array(
        [agent.get_qvalue(state, action) for action in agent.legal_actions]
    )
    action_values -= np.max(action_values)
    probabilities = np.exp(action_values) / np.sum(np.exp(action_values))
    return np.random.choice(agent.legal_actions, p=probabilities)
