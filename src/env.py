"""Define environments."""

from typing import Tuple

import numpy as np

VALID_ACTIONS = dict(
    left=(0, -1),
    right=(0, 1),
    up=(-1, 0),
    down=(1, 0),
)


class GridWorld:
    """Define GridWorld Environemnt.

    Arguments:
        - reward_grid: how much reward the agent get when they arrived at each
            grid cell.
    """

    def __init__(
        self, grid: Tuple[int, int], start: Tuple[int, int], finish: Tuple[int, int]
    ) -> None:
        """Initialize."""
        self.row: int = grid[0]
        self.col: int = grid[1]
        self.start: Tuple[int, int] = start
        self.finish: Tuple[int, int] = finish
        self.reward_grid: np.ndarray = np.zeros(grid)

        assert self.__is_existing_state(self.start), "state param is out of bound"
        assert self.__is_existing_state(self.finish), "finish param is out of bound"

    def update_reward(self, state: Tuple[int, int], reward: int) -> None:
        """Update reward_grid."""
        self.reward_grid[state] = reward

    def get_reward(self, state: Tuple[int, int]) -> int:
        """Get reward value of certain state."""
        return self.reward_grid[state]

    def __is_existing_state(self, state: Tuple[int, int]) -> bool:
        """Check certain state is valid."""
        row_validity: bool = (state[0] >= 0) & (state[0] < self.row)
        col_validity: bool = (state[1] >= 0) & (state[1] < self.col)
        return row_validity & col_validity
