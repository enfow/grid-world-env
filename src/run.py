"""Runner for CustomLavaEnv.

Owner: kyeongmin woo
Email: wgm0601@gmail.com

Reference:
    - https://github.com/maximecb/gym-minigrid
"""

import argparse
from typing import Any, Dict, List

import gym
from gym.wrappers import Monitor

from agent.abstract_agent import AbstractAgent
from agent.dynamic_programming import PolicyIteration, ValueIteration
from agent.monte_carlo import MCAgent
from agent.random_policy import RandomPolicy
from agent.temporal_difference import QLearningAgent, SARSAAgent
from env.custom_minigrid import CustomLavaEnv

AGENT_CONFIG = {
    "policy": "qlearning",
    "lambda": 0.01,
    "threshold": 0.00001,
    "max_evaluation": 100,
    # SARSA
    "lr": 0.01,
    "epsilon": 0.1,
}

ENV_CONFIG = {"height": 3, "width": 4, "obstacle_pos": ((1, 1), (0, 3))}


class Runner:
    """Define runner for reinforcement learning update on Grid World."""

    def __init__(
        self,
        policy: str,
        env_config: Dict[str, Any],
        agent_config: Dict[str, Any],
        n_episode: int = 10,
        max_length: int = 100,
        save_video: bool = True,
        save_dir: str = "./video",
    ) -> None:
        """Initialize."""
        self.env = CustomLavaEnv(**env_config)
        self.policy = policy
        self.agent = self.get_agent(policy, self.env, agent_config)

        self.env_config = env_config
        self.agent_config = agent_config

        self.n_episode = n_episode
        self.max_length = max_length
        self.save_video = save_video
        self.save_dir = save_dir

        self.episode_lengths: List[int] = []
        self.episode_rewards: List[float] = []

    def get_agent(
        self, policy: str, env: gym.Env, agent_config: Dict[str, Any]
    ) -> AbstractAgent:
        """Get agent with policy."""
        if policy == "random":
            agent = RandomPolicy(env)
        elif policy == "pi":
            agent = PolicyIteration(env, agent_config)
        elif policy == "vi":
            agent = ValueIteration(env, agent_config)
        elif policy == "mc":
            agent = MCAgent(env, agent_config)
        elif policy == "sarsa":
            agent = SARSAAgent(env, agent_config)
        elif policy == "qlearning":
            agent = QLearningAgent(env, agent_config)
        else:
            raise NotImplementedError
        return agent

    def run(self) -> None:
        """Start Agent-Environment Interaction and update policy."""
        if self.save_video:
            self.env = Monitor(self.env, self.save_dir, force=True)

        for episode in range(self.n_episode):

            if self.policy in ["pi", "vi"]:
                self.run_dynamic_programming()
            elif self.policy in ["mc"]:
                self.run_monte_carlo()
            elif self.policy in ["sarsa", "qlearning"]:
                self.run_temporal_difference()

            print(
                "Episode: {} | Episode Length: {} | Episode reward: {}".format(
                    episode, self.episode_lengths[-1], self.episode_rewards[-1]
                )
            )
            print()
            self.agent.print_results()

        self.env.close()

    def run_dynamic_programming(self) -> None:
        """Run single episode and update DP methods."""
        done = False
        episode_reward = 0

        obs = self.env.reset()

        for _step in range(self.max_length):
            cur_state = obs["pos"]
            action = self.agent.get_action(cur_state)
            obs, reward, done, _ = self.env.step(action)

            episode_reward += reward

            if done:
                break

        update_info = dict(agent_pos=obs["pos"], reward_grid=obs["reward_grid"])

        self.agent.update_policy(update_info)

        self.episode_lengths.append(_step + 1)
        self.episode_rewards.append(episode_reward)

    def run_monte_carlo(self) -> None:
        """Run single episode and update MC methods."""
        done = False
        episode_reward = 0

        obs = self.env.reset()
        transactions = []

        for _step in range(self.max_length):
            cur_state = obs["pos"]
            action = self.agent.get_action(cur_state)
            obs, reward, done, _ = self.env.step(action)
            next_state = obs["pos"]

            transactions.append((cur_state, action, next_state, reward))

            episode_reward += reward

            if done:
                break

        update_info = dict(transactions=transactions)
        self.agent.update_policy(update_info)

        self.episode_lengths.append(_step + 1)
        self.episode_rewards.append(episode_reward)

    def run_temporal_difference(self) -> None:
        """Run single episode and update TD methods."""
        done = False
        episode_reward = 0

        obs = self.env.reset()

        for _step in range(self.max_length):
            cur_state = obs["pos"]
            action = self.agent.get_action(cur_state)
            obs, reward, done, _ = self.env.step(action)
            next_state = obs["pos"]

            update_info = dict(
                state=cur_state,
                action=action,
                reward=reward,
                next_state=next_state,
            )

            self.agent.update_policy(update_info)

            episode_reward += reward

            if done:
                break

        self.episode_lengths.append(_step + 1)
        self.episode_rewards.append(episode_reward)


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--policy", default="random", type=str)
parser.add_argument("--n_episode", default=10, type=int)
parser.add_argument("--max_length", default=100, type=int)
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--save_dir", default="video", type=str)
args = parser.parse_args()

if __name__ == "__main__":

    runner = Runner(
        policy=args.policy,
        env_config=ENV_CONFIG,
        agent_config=AGENT_CONFIG,
        n_episode=args.n_episode,
        max_length=args.max_length,
        save_video=args.save_video,
        save_dir=args.save_dir,
    )
    runner.run()
