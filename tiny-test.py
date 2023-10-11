import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from exercices import (StochasticGridWorldEnv,
                       test_stochastic_grid_world_value_iteration)

env = StochasticGridWorldEnv()

env.render()

values = np.zeros((4, 4))
# BEGIN SOLUTION

theta = 1e-5
delta = theta
i = 0


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        print(f"For action {action} current sum is {current_sum}")
        values[row, col] = max(values[row, col], current_sum)
    print(values[row, col], prev_val[row, col])
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


import os

debug = False
max_iter = 1000
gamma = 1
if debug:
    while i < max_iter and delta >= theta:
        prev_val = np.copy(values)
        for row in range(env.height):
            for col in range(env.width):
                env.set_state(row, col)
                delta = value_iteration_per_state(env, values, gamma, prev_val, delta)
        i += 1

if os.path.exists("values.npy"):
    values = np.load("values.npy")
else:
    np.save("values.npy", values)

print(i)
values = np.array(
    [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)

env = GridWorldEnv()
for i in range(2):
    env.step(0)
env.render()
old_position = env.current_position
env.step(3)
env.render()
