"""Define AbstractAgent."""
import itertools
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import gym

STATE = Tuple[int, int]
ACTION = int


class AbstractAgent(ABC):
    """Define Abstract Agent."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize.

        Attributes:
            - actions
            - action_space
            - row, col
            - cur_pos
            - all_states
            - value: value of each state. It has same size with grid world.
            - policy: action probability on each state. pi(a | s)
                e.g. self.policy[(0,0)]["left"] <- probability of action left
                    on state (0, 0)
        """
        self.actions: Dict[ACTION, str] = {
            action_int: action_str for action_str, action_int in env.actions.items()
        }
        self.action_space = env.action_space
        self.row, self.col = env.grid_size
        self.cur_pos: STATE = env.agent_pos
        self.all_states = self._get_all_states()
        self.value_v: Dict[STATE, float] = self._get_initial_value_v()
        self.value_q: Dict[STATE, Dict[ACTION, float]] = self._get_initial_value_q()
        self.policy: Dict[STATE, Dict[ACTION, float]] = self._get_initial_policy()

    @abstractmethod
    def print_results(self) -> None:
        """Print results like policy, values.

        Notes:
            - Use methods below
                - print_policy
                - print_state_value
                - print_action_value
        """
        pass

    @abstractmethod
    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences.

        Notes:
            - Use methods below
                - update_policy_with_value_v
                - update_policy_with_value_q
        """
        pass

    def update_policy_with_value_v(self) -> None:
        """Update greedy policy with self.value_v.

        References:
            - Reinforcement Learning The introduction(Sutton) p80
        """
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

    def update_policy_with_value_q(self) -> None:
        """Update Policy with q value."""
        for state in self.all_states:
            max_q = None
            max_actions = list()
            for action, action_q in self.value_q[state].items():
                if max_q is None:
                    max_q = action_q
                    max_actions = [action]
                if max_q < action_q:
                    max_q = action_q
                    max_actions = [action]
                elif max_q == action_q:
                    max_actions.append(action)
            for action in self.policy[state]:
                if action in max_actions:
                    self.policy[state][action] = 1 / len(max_actions)
                else:
                    self.policy[state][action] = 0.0

    def get_action(self, state: STATE, epsilon: float = 0.0) -> int:
        """Get action with episilon greedy.

        Notes:
            - random.random(): random floating point number in the range [0.0, 1.0)
        """
        if epsilon < random.random():
            action = self.get_action_with_policy(state)
        else:
            action = self.get_random_action(state)
        return action

    def get_random_action(self, state: STATE) -> ACTION:
        """Get random action on certain state."""
        available_actions = list(self.get_possible_actions(state))
        action = random.choice(available_actions)
        return action

    def get_action_with_policy(self, state: STATE) -> ACTION:
        """Get action given state with current policy."""
        action_to_prob: Dict[ACTION, float] = self.policy[state]
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
        return random.choice(max_actions)

    def _get_all_states(self) -> List[STATE]:
        """Get all of the availavle states."""
        return list(itertools.product(range(self.row), range(self.col)))

    def _get_initial_value_v(self) -> Dict[STATE, float]:
        """Get initial value of the each state."""
        return {state: 0.0 for state in self.all_states}

    def _get_initial_value_q(self) -> Dict[STATE, Dict[ACTION, float]]:
        """Get initial value of the each state."""
        value_q = dict()

        states = itertools.product(range(self.row), range(self.col))
        for state in states:
            actions = self.get_possible_actions(state).keys()
            value_q[state] = {action: 0.0 / len(actions) for action in actions}

        return value_q

    def _get_initial_policy(self) -> Dict[STATE, Dict[ACTION, float]]:
        """Get initial policy for each state."""
        policy = dict()

        states = itertools.product(range(self.row), range(self.col))
        for state in states:
            actions = self.get_possible_actions(state).keys()
            policy[state] = {action: 1 / len(actions) for action in actions}

        return policy

    def _get_next_state(self, cur_state: STATE, action: ACTION) -> STATE:
        """Get next state with current state and current action.

        Notes:
            - If the action can not be choosen at the current state, raise error.

        Return:
            - next_state
        """
        move_row, move_col = self.action_to_move[action]
        next_state = (cur_state[0] + move_row, cur_state[1] + move_col)
        return next_state

    def get_possible_actions(self, state: STATE) -> Dict[ACTION, STATE]:
        """Get available actions list on certain state."""
        row, col = state
        available_actions = dict()

        for action in self.actions:
            row_move, col_move = self.action_to_move[action]
            if not (
                (row + row_move < 0)
                or (row + row_move >= self.row)
                or (col + col_move < 0)
                or (col + col_move >= self.col)
            ):
                available_actions[action] = (row + row_move, col + col_move)
        return available_actions

    def print_policy(self) -> None:
        """Print state value."""
        logging.info("POLICY")
        for state, action_prob in self.policy.items():
            max_prob = 0.0
            actions = list()
            for action, prob in action_prob.items():
                if max_prob < prob:
                    max_prob = prob
                    actions = [action]
                elif max_prob == prob:
                    actions.append(action)
            logging.info(
                "%s | %s", state, [self.action_to_string(action) for action in actions]
            )
        logging.info("")

    def print_state_value(self) -> None:
        """Print state value."""
        logging.info("STATE VALUE")
        for state, state_value in self.value_v.items():
            logging.info("%s | %s", state, round(state_value, 10))
        logging.info("")

    def print_action_value(self) -> None:
        """Print state value."""
        logging.info("ACTION VALUE")
        for state, action_to_value in self.value_q.items():
            string_for_print = str(
                [
                    (self.action_to_string(action), round(action_value, 4))
                    for action, action_value in action_to_value.items()
                ]
            )
            logging.info("%s | %s", state, string_for_print)
        logging.info("")

    def action_to_string(self, action: ACTION) -> str:
        """Get action string value matching with action int value."""
        return self.actions[action]

    def action_to_int(self, inp_action_str: str) -> int:
        """Get action int value matching with action string value."""
        for action_int, action_str in self.actions.items():
            if action_str == inp_action_str:
                return action_int
        raise RuntimeError

    @property
    def action_to_move(self) -> Dict[int, STATE]:
        """Get direction of each actions.

        Notes:
            - the return value is dictionary
                - key of it is action name like 'left'
                - value of it is direction to move with each key action
                - direction is tuple like (Row direction, Col direction)
            - the upper-right corner is (0,0)
        """
        return {
            self.action_to_int("left"): (0, -1),
            self.action_to_int("right"): (0, 1),
            self.action_to_int("up"): (-1, 0),
            self.action_to_int("down"): (1, 0),
        }
