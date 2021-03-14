"""Runner for CustomLavaEnv.

Owner: kyeongmin woo
Email: wgm0601@gmail.com

Reference:
    - https://github.com/maximecb/gym-minigrid
"""

import argparse
from typing import Any

import gym
from gym.wrappers import Monitor

from agent.dp import DPStateAgent
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


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--policy", default="random", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    grid_height, grid_width = 3, 4
    env = CustomLavaEnv(
        width=grid_width, height=grid_height, obstacle_pos=[(1, 1), (1, 2)]
    )
    if args.policy == "random":
        run_policy = RandomPolicy(env)
    elif args.policy == "dpstate":
        run_policy = DPStateAgent(env, lamb=0.1)
    else:
        raise NotImplementedError
    runner(run_policy, env, save_video=True, save_dir="./result")
