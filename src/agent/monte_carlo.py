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
    """Define Monte Calro Method Agent.

    References:
        - Reinforcement Learning The introduction(Sutton) p92
    """

    def __init__(self, env: gym.Env, config: Dict[str, Any]) -> None:
        """Initialize.

        Configurations:
            - lambda: Coefficient for return value
        """
        super().__init__(env)
        self.transactions: List[SAR] = list()
        self.visited_state: Dict[STATE, int] = dict()
        self.lamb = config["lambda"]

    def reset(self) -> None:
        """Reset the agent's state."""
        self.transactions = list()
        self.visited_state = dict()

    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences."""
        self.__analyze_episode(update_info["transactions"])

        self.monte_carlo_update()
        self.update_policy_with_value_v()

        self.reset()

    def monte_carlo_update(self) -> None:
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
                self.value_v[state] = np.mean(state_to_returns[state])
            elif self.visited_state[state] > 1:
                self.visited_state[state] -= 1
                state_to_returns[state].append(return_g)
            else:
                raise RuntimeError

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

    def print_results(self) -> None:
        """Print results for monte carlo update."""
        self.print_policy()
        self.print_state_value()
