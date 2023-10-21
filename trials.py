from qlearning import QLearningAgent
import gymnasium as gym
import matplotlib.pyplot as plt
from taxi import play_and_train

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore
agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)
s, _ = env.reset()
for i in range(250):
    a = agent.get_action(s)
    next_s, r, done, _, _ = env.step(a)
    s = next_s
    if done:
        break


print(i)

frame = env.render()
fig, ax = plt.subplots()
ax.axis("off")
ax.imshow(frame)
fig.savefig("taxi.png")
