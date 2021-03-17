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

AGENT_CONFIG = {"lambda": 0.1, "threshold": 0.00001, "max_evaluation": 1000}

ENV_CONFIG = {"width": 5, "height": 5, "obstacle_pos": ((2, 0), (2, 2), (2, 3), (2, 4))}


def runner(
    policy: Any,
    environment: gym.Env,
    update_iter: int = 10,
    max_length: int = 100,
    save_video: bool = True,
    save_dir: str = "./video",
) -> None:
    """Define runner function for grid world."""
    if save_video:
        environment = Monitor(environment, save_dir, force=True)

    for update_step in range(update_iter):
        # Initialize environment
        obs = environment.reset()

        done = False
        episode_reward = 0
        episode_length = 0
        for _ in range(max_length):
            # Take a step
            action = policy.get_action(obs["pos"])
            obs, reward, done, _ = environment.step(action)

            update_info = dict(
                agent_pos=obs["pos"],
                reward_grid=obs["reward_grid"],
            )

            episode_reward += reward
            episode_length += 1

            if done:
                break

        print("Update Step: {} | Total reward: {}".format(update_step, episode_reward))
        print("Update Step: {} | Total length: {}".format(update_step, episode_length))
        print()
        max_diff = policy.update_policy(update_info)
        print("max_diff: ", round(max_diff, 6))
        policy.print_policy()
        policy.print_value()
        print(obs["reward_grid"])
        print()

    env.close()


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--policy", default="random", type=str)
args = parser.parse_args()


if __name__ == "__main__":
    grid_height, grid_width = 3, 4
    env = CustomLavaEnv(**ENV_CONFIG)
    if args.policy == "random":
        run_policy = RandomPolicy(env)
    elif args.policy == "dpstate":
        run_policy = DPStateAgent(env, AGENT_CONFIG)
    else:
        raise NotImplementedError
    runner(run_policy, env, save_video=True, save_dir="./result", update_iter=100)
