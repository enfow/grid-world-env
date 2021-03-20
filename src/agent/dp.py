"""Define Agents with Dynamic Programming Algorithm."""

from typing import Any, Dict, Tuple

import gym
import numpy as np

from agent.abstract_agent import AbstractAgent


class DPAgent(AbstractAgent):
    """Define Dynamic Programming Abstract Class."""

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
    ) -> None:
        """Initialize.

        Args:
            - row, col: the size of grid
            - actions: mapping action_name to int value for environmnet
                - left : 0, right : 1, up : 2, down : 3
            - policy: mappint state(grid cell) to action and probability pair
            - value: value of the each state
            - all_states: the list of all states(grid cell)
            - lamb: lambda value for return
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
        """Define Policy Evaluation."""
        raise NotImplementedError

    def policy_improvement(self) -> None:
        """Define Policy Improvement."""
        raise NotImplementedError

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


class PolicyIteration(DPAgent):
    """Define Dynamic Programming Algorithm."""

    def policy_evaluation(
        self,
        reward_grid: np.ndarray,
    ) -> float:
        """Update value with full backup.

        Params:
            - reward_grid: reward value information for all next states.

        Returns:
            - max_diff: maximum of the diffs

        Notes:
            - max_diff should be calculated with the absolute.
        """
        for _ in range(self.max_eval):
            new_value: Dict[Tuple[int, int], float] = self._get_initial_value_v()
            max_diff: float = 0.0
            for state in self.all_states:
                new_value[state] = self.__backup(state, reward_grid)
                max_diff = max(max_diff, abs(new_value[state] - self.value_v[state]))
            self.value_v = new_value

            if max_diff < self.threshold:
                break

        return max_diff

    def policy_improvement(self) -> None:
        """Update greedy policy with self.value_v."""
        for state in self.all_states:
            greedy_actions = list()
            for idx, action in enumerate(self.policy[state]):
                next_state = self._get_next_state(state, action)
                next_value = self.value_v[next_state]
                if idx == 0:
                    max_value = next_value
                if next_value > max_value:
                    greedy_actions = [action]
                    max_value = next_value
                elif next_value == max_value:
                    greedy_actions.append(action)

            for action in self.policy[state]:
                if action in greedy_actions:
                    self.policy[state][action] = 1 / len(greedy_actions)
                else:
                    self.policy[state][action] = 0.0

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
