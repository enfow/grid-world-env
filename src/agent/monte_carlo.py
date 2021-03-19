"""Define Agents with Monte Carlo Method."""

from typing import Any, Dict, List, Tuple

import gym
import numpy as np

from agent.abstract_agent import AbstractAgent

STATE = Tuple[int, int]
ACTION = int
REWARD = int
DONE = bool
SAR = Tuple[STATE, ACTION, REWARD]


class MCAgent(AbstractAgent):
    """Define Monte Calro Method Agent."""

    def __init__(self, env: gym.Env, config: Dict[str, Any]) -> None:
        """Initialize."""
        super().__init__(env)
        self.transactions: List[SAR] = list()
        self.visited_state: Dict[STATE, int] = dict()
        self.value = self._get_initial_value()
        self.lamb = config["lambda"]

    def reset(self) -> None:
        """Reset the agent's state."""
        self.transactions = list()
        self.visited_state = dict()

    def get_action(self, state: Tuple[int, int]) -> int:
        """Get action according to current policy."""
        return super()._get_action_with_v(state)

    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences."""
        self.__analyze_episode(update_info["episode"])

        self.update_value_v()
        self.policy_improvement()

        self.reset()

    def update_value_v(self) -> None:
        """Update value v with monte carlo method."""
        return_g = 0

        state_to_returns: Dict[STATE, List[float]] = {
            state: [] for state in self.visited_state
        }

        for step in reversed(range(len(self.transactions))):
            state, _, reward = self.transactions[step]

            return_g = (self.lamb * return_g) + reward

            if state not in self.visited_state:
                raise RuntimeError

            if self.visited_state[state] == 1:
                state_to_returns[state].append(return_g)
                self.value[state] = np.mean(state_to_returns[state])
            elif self.visited_state[state] > 1:
                self.visited_state[state] -= 1
                state_to_returns[state].append(return_g)
            else:
                raise RuntimeError

    def policy_improvement(self) -> None:
        """Update greedy policy with self.value."""
        for state in self.all_states:
            greedy_actions = list()
            for idx, action in enumerate(self.policy[state]):
                next_state = self._get_next_state(state, action)
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

    def __analyze_episode(
        self, episode: List[Tuple[STATE, ACTION, STATE, REWARD]]
    ) -> None:
        """Add (state, action, reward) pair to episode."""
        for cur_state, action, _, reward in episode:
            self.transactions.append((cur_state, action, reward))
            if cur_state in self.visited_state:
                self.visited_state[cur_state] += 1
            else:
                self.visited_state[cur_state] = 1
