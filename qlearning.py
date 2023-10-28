import random
import typing as t
from collections import defaultdict

import gymnasium as gym
import numpy as np

Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        Q-Learning Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        value = 0.0
        # BEGIN SOLUTION
        action = self.get_best_action(state)
        value = self.get_qvalue(state, action)
        # END SOLUTION
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s') = r + gamma * max_a' Q(s', a')
           TD_error(s', a) = TD_target(s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s', a)
        """
        q_value = 0.0
        # BEGIN SOLUTION
        target = reward + self.gamma * self.get_value(next_state)
        td_error = target - self.get_qvalue(state, action)
        q_value = self.get_qvalue(state, action) + self.learning_rate * td_error
        # END SOLUTION

        self.set_qvalue(state, action, q_value)

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = [
            self.get_qvalue(state, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def epsilon_choice(self):
        return random.random() < self.epsilon

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration is done with epsilon-greedy. Namely, with probability self.epsilon, we should take a random action, and otherwise the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        action = self.legal_actions[0]

        # BEGIN SOLUTION
        if self.epsilon_choice():
            action = random.choice(self.legal_actions)
        else:
            action = self.get_best_action(state)
        # END SOLUTION

        return action

    def play_and_train(self, env: gym.Env, t_max: int = int(1e4)):
        total_reward = 0.0
        s, _ = env.reset()

        for i in range(t_max):
            # Get agent to pick action given state s
            a = self.get_action(s)

            next_s, r, done, _, _ = env.step(a)

            # Train agent for state s
            # BEGIN SOLUTION

            total_reward += r
            self.update(state=s, action=a, next_state=next_s, reward=r)
            s = next_s
            if done:
                break
            # END SOLUTION

        return total_reward
