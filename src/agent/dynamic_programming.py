"""Define Agents with Dynamic Programming Algorithm."""

from typing import Any, Dict, Tuple

import gym
import numpy as np

from agent.abstract_agent import AbstractAgent


class PolicyIteration(AbstractAgent):
    """Define Policy Iteration."""

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
    ) -> None:
        """Initialize.

        References:
            - Reinforcement Learning The introduction(Sutton) p80
        """
        super().__init__(env)
        self.value_v: Dict[Tuple[int, int], float] = self._get_initial_value_v()
        self.lamb: float = config["lambda"]
        self.threshold: float = config["threshold"]
        self.max_eval: int = config["max_evaluation"]

    def policy_evaluation(
        self,
        reward_grid: np.ndarray,
    ) -> float:
        """Update value with full backup.

        Params:
            - reward_grid: reward value information for all next states.
        """
        for _ in range(self.max_eval):
            new_value: Dict[Tuple[int, int], float] = self._get_initial_value_v()
            max_diff: float = 0.0
            # loop for each state
            for state in self.all_states:
                new_value[state] = self.__backup(state, reward_grid)
                max_diff = max(max_diff, abs(new_value[state] - self.value_v[state]))
            self.value_v = new_value

            if max_diff < self.threshold:
                break

        return max_diff

    def __backup(self, state: Tuple[int, int], reward_grid: np.ndarray) -> float:
        """Update all of the states."""
        value: float = 0.0
        for action in self.policy[state]:
            next_state = self._get_next_state(state, action)
            reward = reward_grid[next_state]
            value += self.policy[state][action] * (
                reward + self.lamb * self.value_v[next_state]
            )
        return value

    def policy_improvement(self) -> None:
        """Update greedy policy with self.value_v."""
        self.update_policy_with_value_v()

    def update_policy(self, update_info: Dict[str, Any]) -> float:
        """Updata policy with policy evaluation and policy improvement.

        Notes:
            - policy evaluation: Update value
            - policy improvement: Update policy with new value
        """
        reward_grid = update_info["reward_grid"]

        max_diff = self.policy_evaluation(reward_grid)
        self.policy_improvement()

        return max_diff

    def print_results(self) -> None:
        """Print results for dynamic programming update."""
        self.print_policy()
        self.print_state_value()


class ValueIteration(PolicyIteration):
    """Define Value Iteration.

    References:
        - Reinforcement Learning The introduction(Sutton) p83
    """

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
    ) -> None:
        """Initialize."""
        super().__init__(env, config)
        self.max_eval = 1
