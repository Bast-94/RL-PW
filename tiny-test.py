import numpy as np

from exercices import (StochasticGridWorldEnv,
                       test_stochastic_grid_world_value_iteration)

env = StochasticGridWorldEnv()
env.set_state(0, 2)
env.render()
print(env.get_next_states(2))
# est_grid_world_value_iteration(20)
test_stochastic_grid_world_value_iteration(50)
