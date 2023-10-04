from exercices import (test_grid_world_value_iteration, test_mdp,
                       test_mdp_value_iteration)

test_mdp()
test_mdp_value_iteration(max_iter=3)
test_grid_world_value_iteration()
