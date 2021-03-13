"""Runner for CustomLavaEnv.

Owner: kyeongmin woo
Email: wgm0601@gmail.com

Reference:
    - https://github.com/maximecb/gym-minigrid
"""

from typing import Any

import gym
from gym.wrappers import Monitor

from agent.random_policy import RandomPolicy
from env.custom_minigrid import CustomLavaEnv


def runner(
    policy: Any,
    environment: gym.Env,
    max_length: int = 100,
    save_video: bool = True,
    save_dir: str = "./video",
) -> None:
    """Define runner function for grid world."""
    if save_video:
        environment = Monitor(environment, save_dir, force=True)

    # Initialize environment
    obs = environment.reset()

    done = False
    episode_reward = 0
    episode_length = 0

    # Run until done == True
    for _ in range(max_length):
        # Take a step
        action = policy.get_action(obs)
        obs, reward, done, _ = environment.step(action)

        episode_reward += reward
        episode_length += 1

        if done:
            break

    print("Total reward:", episode_reward)
    print("Total length:", episode_length)

    env.close()


# Test that the logging function is working
grid_size = (5, 5)
env = CustomLavaEnv(
    width=grid_size[0], height=grid_size[1], obstacle_pos=[(2, 2), (3, 3)]
)
rand_policy = RandomPolicy(env.action_space)
runner(rand_policy, env, save_video=True, save_dir="./result")
