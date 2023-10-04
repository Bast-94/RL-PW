from exercices import GridWorldEnv


def test_wall():
    env = GridWorldEnv()
    for i in range(2):
        env.step(0)
    old_position = env.current_position
    env.step(3)
    assert old_position == env.current_position
