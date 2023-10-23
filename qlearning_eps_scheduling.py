import random
import typing as t

import numpy as np

from qlearning import Action, QLearningAgent, State


class QLearningAgentEpsScheduling(QLearningAgent):
    def __init__(
        self,
        *args,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        **kwargs,
    ):
        """
        Q-Learning Agent with epsilon scheduling

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        super().__init__(*args, **kwargs)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.reset()

    def reset(self):
        """
        Reset epsilon to the start value.
        """
        self.epsilon = self.epsilon_start
        self.timestep = 0

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
        action = super().get_action(state)
        # END SOLUTION

        return action

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        super().update(state, action, reward, next_state)
        self.timestep += 1
        # update epsilon using self.timestep and self.epsilon_decay_steps

        # BEGIN SOLUTION
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.timestep / self.epsilon_decay_steps,
        )
