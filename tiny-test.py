import numpy as np

from exercices import (StochasticGridWorldEnv,
                       test_stochastic_grid_world_value_iteration)

env = StochasticGridWorldEnv()
env.set_state(1, 3)
env.render()

# print(env.step(2, make_move=False))
env.direction_table = [
    env.up_position,
    env.right_postion,
    env.down_position,
    env.left_postion,
]
##print(env.get_next_states(3))
array = np.zeros((env.height, env.width, env.action_space.n, env.height, env.width, 2))
for row in range(env.height):
    for col in range(env.width):
        for action in range(env.action_space.n):
            env.set_state(row, col)
            next_states = env.get_next_states(action=action)
            for next_state, reward, probability, _, _ in next_states:
                # print(reward, probability, next_state)
                next_row, next_row = next_state
                array[row, col, action, next_row, next_row, 0] = probability
                array[row, col, action, next_row, next_row, 1] = reward

values = np.zeros((4, 4))
theta = 1e-5
# BEGIN SOLUTION
diff = theta
max_iter = 1000
i = 0
gamma = 1.0
"""env.direction_table = [
    env.up_position,
    env.right_postion,
    env.down_position,
    env.left_postion,
]"""
while i < max_iter and diff >= theta:
    prev_val = np.copy(values)

    for row in range(env.height):
        for col in range(env.width):
            best_val = float("-inf")
            for action in range(env.action_space.n):
                env.set_state(row, col)
                next_states = env.get_next_states(action=action)
                current_sum = 0
                for next_state, reward, probability, _, _ in next_states:
                    next_row, next_col = next_state
                    current_sum += (
                        probability
                        * (reward + gamma * prev_val[next_row, next_col])
                        * env.moving_prob[row, col, action]
                    )
                best_val = max(best_val, current_sum)

            values[row, col] = max(prev_val[row, col], best_val)
    # print(values)
    diff = np.max(np.abs(values - prev_val))

    i += 1
print(values)
# print(array[0, 0])
# test_stochastic_grid_world_value_iteration()
