"""Define Agents with Temporal Difference Method."""

from typing import Any, Dict, Tuple

import gym

from agent.abstract_agent import AbstractAgent

STATE = Tuple[int, int]
ACTION = int
REWARD = int
DONE = bool
SARSA = Tuple[STATE, ACTION, REWARD, STATE, ACTION]


class SARSAAgent(AbstractAgent):
    """Define SARSA Agent."""

    def __init__(self, env: gym.Env, config: Dict[str, Any]) -> None:
        """Initialize.

        Configurations:
            - lambda: Coefficient for Next Q value
            - lr: Coefficient for Temporal Difference
            - epsilon: Random Action Probability
        """
        super().__init__(env)
        self.lamb = config["lambda"]
        self.learning_rate = config["lr"]
        self.epsilon = config["epsilon"]

    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences."""
        cur_s = update_info["state"]
        cur_a = update_info["action"]
        reward = update_info["reward"]
        next_s = update_info["next_state"]
        self.update_value_q(cur_s, cur_a, reward, next_s)
        self.update_policy_with_value_q()

    def update_value_q(
        self, cur_s: STATE, cur_a: ACTION, reward: REWARD, next_s: STATE
    ) -> None:
        """Update q value.

        Notes:
            - sarsa update: q(s,a) = q(s,a) + lr * (r + (lambda * q(s',a')) - q(s,a))
        """
        next_a = self.get_action(next_s, epsilon=self.epsilon)
        cur_q = self.value_q[cur_s][cur_a]
        next_q = self.value_q[next_s][next_a]
        self.value_q[cur_s][cur_a] = cur_q + self.learning_rate * (
            reward + (self.lamb * next_q) - cur_q
        )

    def print_results(self) -> None:
        """Print results for temporal difference update."""
        self.print_policy()
        self.print_action_value()


class QLearningAgent(SARSAAgent):
    """Define Q-learning Agnet."""

    def update_value_q(
        self, cur_s: STATE, cur_a: ACTION, reward: REWARD, next_s: STATE
    ) -> None:
        """Update q value.

        Notes:
            - q-learning update: q(s,a) = q(s,a) + lr * (
                r + (gamma * argmax q(s')) - q(s,a)
            )
        """
        cur_q = self.value_q[cur_s][cur_a]
        next_a = super().get_action(next_s, epsilon=self.epsilon)
        next_q = self.value_q[next_s][next_a]
        self.value_q[cur_s][cur_a] = cur_q + self.learning_rate * (
            reward + (self.lamb * next_q) - cur_q
        )
