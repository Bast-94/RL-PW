import os

import cv2
import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm


def create_gif(agent, env, name, artifact_dir=".", t_max=int(1e4), text=""):
    frames = []

    s, _ = env.reset()
    total_reward = 0.0
    for i in tqdm(range(0, t_max)):
        action = agent.get_action(s)
        next_s, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        s = next_s
        done = terminated or truncated
        img = env.render()
        img = cv2.putText(
            img,
            text=f"{name} agent, {i} steps, total reward: {total_reward}" + text,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        frames.append(img)
        assert i <= 200, "Agent didn't finish the game in 200 steps"
        if done:
            break
    output_file_path = os.path.join(artifact_dir, f"{name}.gif")
    imageio.mimsave(output_file_path, frames, fps=8, loop=0)
