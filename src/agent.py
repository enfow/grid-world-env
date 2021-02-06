"""Define Agents."""

import itertools
from typing import Dict, List, Set, Tuple

import numpy as np

from env import VALID_ACTIONS


class DPStateAgent:
    """Define Dynamic Programming Algorithm."""

    def __init__(
        self,
        grid: Tuple[int, int],
        lamb: float,
    ) -> None:
        """Initialize."""
        self.row, self.col = grid
        self.policy: Dict[
            Tuple[int, int], Dict[str, float]
        ] = self.__get_initial_policy()
        self.value: np.ndarray = self.__get_initial_value()
        self.all_states: List[Tuple[int, int]] = self.__get_all_states()
        self.lamb: float = lamb

    def policy_evaluation(
        self,
        reward_grid: np.ndarray,
    ) -> None:
        """Update model with single backup or full backup."""
        new_value = self.__get_initial_value()
        for state in self.all_states:
            new_value[state] = self.__single_backup(state, reward_grid)
        self.value = new_value

    def policy_improvement(self) -> None:
        """Update policy with self.value."""
        raise NotImplementedError

    def __single_backup(self, state: Tuple[int, int], reward_grid: np.ndarray) -> float:
        """Update all of the states."""
        value: float = 0.0
        for action in self.policy[state]:
            row, col = state
            row_move, col_move = VALID_ACTIONS[action]
            reward = reward_grid[row + row_move, col + col_move]
            value += self.policy[state][action] * (
                reward + self.lamb * self.value[state]
            )
        return value

    def __get_all_states(self) -> List[Tuple[int, int]]:
        """Get all of the availavle states."""
        return list(itertools.product(range(self.row), range(self.col)))

    def __get_initial_value(self) -> np.ndarray:
        """Get initial value of the each state."""
        return np.zeros((self.row, self.col))

    def __get_initial_policy(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Get initial policy for each state."""
        policy = dict()

        states = itertools.product(range(self.row), range(self.col))
        for state in states:
            actions = self.__available_actions_on_state(state)
            policy[state] = {action: 1 / len(actions) for action in actions}

        return policy

    def __available_actions_on_state(self, state: Tuple[int, int]) -> Set[str]:
        """Get available actions list on certain state."""
        row, col = state
        availiable_actions = set(VALID_ACTIONS)

        for action, move in VALID_ACTIONS.items():
            row_move, col_move = move
            if (row + row_move < 0) or (row + row_move >= self.row):
                availiable_actions.remove(action)
            elif (col + col_move < 0) or (col + col_move >= self.col):
                availiable_actions.remove(action)

        return availiable_actions
