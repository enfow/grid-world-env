"""Define AbstractAgent."""
import itertools
import random
from typing import Any, Dict, List, Tuple

import gym
import numpy as np


class AbstractAgent:
    """Define Abstract Agent."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize."""
        self.actions: Dict[str, int] = env.actions
        self.action_space = env.action_space
        self.row, self.col = env.grid_size
        self.cur_pos = None
        self.all_states = self._get_all_states()
        self.value = self._get_initial_value()
        self.policy: Dict[
            Tuple[int, int], Dict[str, float]
        ] = self._get_initial_policy()

    def get_action(self, state: Tuple[int, int]) -> int:
        """Get action according to current policy."""
        raise NotImplementedError

    def _get_action_with_v(self, state: Tuple[int, int]) -> int:
        """Get action given state.

        Params:
            - state: the position of agent on the grid

        Returns:
            - action: the proper actions for input state.
        """
        action_to_prob: Dict[str, float] = self.policy[state]
        max_prob, max_actions = None, []
        for action, prob in action_to_prob.items():
            if max_prob is None:
                max_prob = prob
                max_actions.append(action)
            else:
                if prob > max_prob:
                    max_prob = prob
                    max_actions = [action]
                elif prob == max_prob:
                    max_actions.append(action)
        return self.actions[random.choice(max_actions)]

    def _get_all_states(self) -> List[Tuple[int, int]]:
        """Get all of the availavle states."""
        return list(itertools.product(range(self.row), range(self.col)))

    def _get_initial_policy(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Get initial policy for each state."""
        policy = dict()

        states = itertools.product(range(self.row), range(self.col))
        for state in states:
            actions = self.available_actions_on_state(state).keys()
            policy[state] = {action: 1 / len(actions) for action in actions}

        return policy

    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences."""
        raise NotImplementedError

    def _get_next_state(
        self, cur_state: Tuple[int, int], action: str
    ) -> Tuple[int, int]:
        """Get next state with current state and current action.

        Notes:
            - If the action can not be choosen at the current state, raise error.

        Return:
            - next_state
        """
        move_row, move_col = self.action_to_move[action]
        next_state = (cur_state[0] + move_row, cur_state[1] + move_col)
        return next_state

    def _get_initial_value(self) -> np.ndarray:
        """Get initial value of the each state."""
        return np.zeros((self.row, self.col))

    def available_actions_on_state(
        self, state: Tuple[int, int]
    ) -> Dict[str, Tuple[int, int]]:
        """Get available actions list on certain state."""
        row, col = state
        available_actions = dict()

        for action in self.actions.keys():
            row_move, col_move = self.action_to_move[action]
            if (
                not (
                    (row + row_move < 0)
                    or (row + row_move >= self.row)
                    or (col + col_move < 0)
                    or (col + col_move >= self.col)
                )
                or action == "done"
            ):
                available_actions[action] = (row + row_move, col + col_move)

        return available_actions

    def print_policy(self) -> None:
        """Print state value."""
        for state, action_prob in self.policy.items():
            max_prob = 0.0
            actions = list()
            for action, prob in action_prob.items():
                if max_prob < prob:
                    max_prob = prob
                    actions = [action]
                elif max_prob == prob:
                    actions.append(action)
            print("{} | {}".format(state, actions))

    def print_value(self) -> None:
        """Print state value."""
        print(self.value)

    @property
    def action_to_move(self) -> Dict[str, Tuple[int, int]]:
        """Get direction of each actions.

        Notes:
            - the return value is dictionary
                - key of it is action name like 'left'
                - value of it is direction to move with each key action
                - direction is tuple like (Row direction, Col direction)
            - the upper-right corner is (0,0)
        """
        return {
            "left": (0, -1),
            "right": (0, 1),
            "up": (-1, 0),
            "down": (1, 0),
        }
