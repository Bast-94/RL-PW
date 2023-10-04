import numpy as np

from exercices import (GridWorldEnv, StochasticGridWorldEnv,
                       test_grid_world_value_iteration,
                       test_stochastic_grid_world_value_iteration)

env = StochasticGridWorldEnv()

test_grid_world_value_iteration(20)
test_stochastic_grid_world_value_iteration()
