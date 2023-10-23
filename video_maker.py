

import cv2
import gymnasium as gym
import numpy as np
env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n
env.reset()
fourcc = cv2.VideoWriter_fourcc(*"XVID")
height, width = 600, 400
video = cv2.VideoWriter("ouptut/video.avi", fourcc, 20.0, (height, width))
for i in range(1000):
    random_img = np.random.randint(255, size=(height, width, 3), dtype=np.uint8)
    video.write(random_img)
video.release()