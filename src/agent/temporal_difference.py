"""Define Agents with Temporal Difference Method."""

import random
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
        """Initialize."""
        super().__init__(env)
        self.lamb = config["lambda"]
        self.learning_rate = config["lr"]
        self.epsilon = config["epsilon"]
        self.action_int_to_str = {
            self.actions[state_s]: state_s for state_s in self.actions
        }

    def get_action(self, state: STATE) -> int:
        """Get action with episilon greedy.

        Notes:
            - random.random(): random floating point number in the range [0.0, 1.0)
        """
        if self.epsilon < random.random():
            # greedy
            action = super().get_action(state)
        else:
            # random action
            available_actions = list(self.available_actions_on_state(state))
            action = self.actions[random.choice(available_actions)]
        return action

    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences."""
        cur_s = update_info["state"]
        cur_a = update_info["action"]
        reward = update_info["reward"]
        next_s = update_info["next_state"]
        self.update_q_value(cur_s, cur_a, reward, next_s)
        self.update_policy_with_q_value()

    def update_policy_with_q_value(self) -> None:
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

    def update_q_value(
        self, cur_s: STATE, cur_a: ACTION, reward: REWARD, next_s: STATE
    ) -> None:
        """Update q value.

        Notes:
            - sarsa update: q(s,a) = q(s,a) + lr * (r + (gamma * q(s',a')) - q(s,a))
        """
        next_a = self.get_action(next_s)
        cur_a = self.action_int_to_str[cur_a]
        next_a = self.action_int_to_str[next_a]
        cur_q = self.value_q[cur_s][cur_a]
        next_q = self.value_q[next_s][next_a]
        self.value_q[cur_s][cur_a] = cur_q + self.learning_rate * (
            reward + (self.lamb * next_q) - cur_q
        )


class QLearningAgent(SARSAAgent):
    """Define Q-learning Agnet."""

    def update_q_value(
        self, cur_s: STATE, cur_a: ACTION, reward: REWARD, next_s: STATE
    ) -> None:
        """Update q value.

        Notes:
            - q-learning update: q(s,a) = q(s,a) + lr * (
                r + (gamma * argmax q(s')) - q(s,a)
            )
        """
        cur_a = self.action_int_to_str[cur_a]
        cur_q = self.value_q[cur_s][cur_a]
        next_a = super().get_action(next_s)
        next_a = self.action_int_to_str[next_a]
        next_q = self.value_q[next_s][next_a]
        self.value_q[cur_s][cur_a] = cur_q + self.learning_rate * (
            reward + (self.lamb * next_q) - cur_q
        )
