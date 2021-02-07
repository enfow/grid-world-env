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
    ) -> float:
        """Update value with full backup.

        Params:
            - reward_grid: reward value information for all next states.

        Returns:
            - max_diff: maximum of the diffs

        Notes:
            - max_diff should be calculated with the absolute.
        """
        new_value: np.ndarray = self.__get_initial_value()
        max_diff: float = 0.0
        for state in self.all_states:
            new_value[state] = self.__backup(state, reward_grid)
            max_diff = max(max_diff, abs(new_value[state] - self.value[state]))
        self.value = new_value
        return max_diff

    def policy_improvement(self) -> None:
        """Update policy with self.value."""
        for state in self.all_states:
            actions = self.policy[state].keys()
            for action in actions:
                row_move, col_move = VALID_ACTIONS[action]
                next_state = self.__get_next_state(state, action)

    def __backup(self, state: Tuple[int, int], reward_grid: np.ndarray) -> float:
        """Update all of the states."""
        value: float = 0.0
        for action in self.policy[state]:
            next_state = self.__get_next_state(state, action)
            reward = reward_grid[next_state]
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

    def __get_next_state(self, cur_state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Get next state with current state and current action.
        
        Notes:
            - If the action can not be choosen at the current state, raise error.

        Return:
            - next_state
        """
        move_row, move_col = VALID_ACTIONS[action]
        next_state = (cur_state[0] + move_row, cur_state[1] + move_col)
        return next_state 

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
