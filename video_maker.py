import cv2
import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm


def create_gif(
    agent, name, ep_per_step, nb_step, artifact_dir, t_max, env, policy=None
):
    from train_agent import train

    gif_writer = imageio.get_writer(f"{artifact_dir}/{name}.gif", mode="I")
    for train_step in tqdm(range(1, nb_step + 1)):
        print(train_step)
        train(agent, env, t_max, num_episodes=ep_per_step)
        print("Done")
        print("Playing and recording agent: ", name)

        s, _ = env.reset()
        for i in tqdm(range(0, t_max)):
            action = agent.get_action(s)
            next_s, r, done, _, _ = env.step(action)
            s = next_s

            img = env.render()
            img = cv2.putText(
                img,
                text=f"{name} agent after {train_step*ep_per_step} episodes \n {i} steps \n reward: {r}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            gif_writer.append_data(img)
            if done:
                break

    gif_writer.close()
