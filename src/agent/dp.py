"""Define Agents."""

import itertools
import random
from typing import Any, Dict, List, Tuple

import gym
import numpy as np

from agent.abstract_agent import AbstractAgent


class DPStateAgent(AbstractAgent):
    """Define Dynamic Programming Algorithm."""

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
        self.policy: Dict[
            Tuple[int, int], Dict[str, float]
        ] = self.__get_initial_policy()
        self.value: np.ndarray = self.__get_initial_value()
        self.all_states: List[Tuple[int, int]] = self.__get_all_states()
        self.lamb: float = config["lambda"]
        self.threshold: float = config["threshold"]
        self.max_eval: int = config["max_evaluation"]

    def get_action(self, state: Tuple[int, int]) -> int:
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

    def update_policy(self, update_info: Dict[str, Any]) -> float:
        """Updata policy with policy evaluation and policy improvement.

        Notes:
            - policy evaluation: Update value
            - policy improvement: Update policy with new value
        """
        reward_grid = update_info["reward_grid"]

        for _ in range(self.max_eval):
            max_diff = self.policy_evaluation(reward_grid)
            if max_diff < self.threshold:
                break

        self.policy_improvement()
        return max_diff

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
        """Update greedy policy with self.value."""
        for state in self.all_states:
            greedy_actions = list()
            for idx, action in enumerate(self.policy[state]):
                next_state = self.__get_next_state(state, action)
                next_value = self.value[next_state]
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
            next_state = self.__get_next_state(state, action)
            reward = reward_grid[next_state]
            value += self.policy[state][action] * (
                reward + self.lamb * self.value[next_state]
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
            actions = self.available_actions_on_state(state).keys()
            policy[state] = {action: 1 / len(actions) for action in actions}

        return policy

    def __get_next_state(
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
