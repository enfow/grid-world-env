"""Test cases for environments and agents."""
import io
import itertools
import math
import sys
from typing import Dict, List, Set, Tuple

import numpy as np

from agent.dp import PolicyIteration
from env.custom_minigrid import CustomLavaEnv

ROW = 3
COL = 4


class TestCustomLavaEnv:
    """Test cases for CustomLavaEnv."""

    def setup_class(self):
        """Initialize common attributes."""
        self.grid = (ROW, COL)
        self.all_states = list(itertools.product(range(ROW), range(COL)))
        self.obstacle_pos = [(1, 1), (0, 3)]

    def setup_method(self):
        """Initialize CustomLavaEnv for each test cases."""
        self.env = CustomLavaEnv(width=COL, height=ROW, obstacle_pos=self.obstacle_pos)

    def test_initial_observation(self):
        """Check initial observation which is return of the reset method."""
        valid_reward_grid = self.get_reward_grid
        valid_agent_pos = (0, 0)

        initial_obs = self.env.reset()

        assert initial_obs["pos"] == valid_agent_pos
        assert (initial_obs["reward_grid"] == valid_reward_grid).all()

    def test_observation_after_step(self):
        """Check change of observation with step method."""

        for name, action_list in self.get_action_lists.items():
            obs = self.env.reset()
            for idx, action in enumerate(action_list):
                obs, _, _, _ = self.env.step(action)
                valid_pos = self.get_position_with_action_lists[name][idx]
                assert obs["pos"] == valid_pos, "set {}, action {}, state {}".format(
                    name, action, obs["pos"]
                )

    @property
    def get_reward_grid(self) -> np.array:
        return np.array(
            [
                [0, 0, 0, -10],
                [0, -10, 0, 0],
                [0, 0, 0, 100],
            ]
        )

    @property
    def get_action_lists(self) -> Dict[str, List[int]]:
        return dict(
            action1=[1, 1, 1, 3, 3],
            action2=[3, 3, 1, 1, 1],
            action3=[3, 3, 1, 0, 1, 1, 1],
            action4=[3, 1, 1, 3, 1],
        )

    @property
    def get_position_with_action_lists(self) -> Dict[str, List[Tuple[int, int]]]:
        return dict(
            action1=[(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)],
            action2=[(1, 0), (2, 0), (2, 1), (2, 2), (2, 3)],
            action3=[(1, 0), (2, 0), (2, 1), (2, 0), (2, 1), (2, 2), (2, 3)],
            action4=[(1, 0), (1, 1), (1, 2), (2, 2), (2, 3)],
        )


class TestPolicyIteration:
    """Test cases for PolicyIteration."""

    def setup_class(self):
        """Initialize common attributes."""
        self.grid = (ROW, COL)
        self.all_states = list(itertools.product(range(ROW), range(COL)))
        self.obstacle_pos = [(1, 1), (0, 3)]

    def setup_method(self):
        """Initialize PolicyIteration for each test case"""
        self.env = CustomLavaEnv(width=COL, height=ROW, obstacle_pos=self.obstacle_pos)
        self.agent = PolicyIteration(
            env=self.env,
            config={
                "lambda": 0.1,
                "threshold": 0.1,
                "max_evaluation": 1,
            },
        )

    def test_initial_policy(self):
        """Check initial policy of agent is correct.

        CheckList:
            - the agent.policy is dictionary
            - the keys of env.policy should include all possible states.
        """
        valid_value = ROW * COL

        assert isinstance(self.agent.policy, dict)
        assert len(self.agent.policy) == valid_value

        valid_value = self.get_available_actions
        for state in self.all_states:
            actions = self.agent.policy[state].keys()
            probs = list(self.agent.policy[state].values())
            assert actions == valid_value[state], "problem on state {}".format(state)
            assert probs == [1 / len(actions) for _ in actions]

    def test_initial_value(self):
        """Check size of the value dictionary

        CheckList:
            - agent.value is dictionary
            - the keys of env.value should include all possible states.
        """
        valid_value = self.get_valid_state_values["initial"]

        for grid_cell in valid_value:
            assert self.agent.value_v[grid_cell] == valid_value[grid_cell]

    def test_single_policy_evaluation(self):
        """Check policy_evaluation method with single mode is correct."""
        mock_reward_grid = self.get_reward_grid
        valid_value = self.get_valid_state_values["single_iter"]
        valid_max_diff = max(abs(min(valid_value.values())), max(valid_value.values()))
        lamb = 0.1

        max_diff = self.agent.policy_evaluation(mock_reward_grid)

        for state in self.all_states:
            assert math.isclose(valid_value[state], self.agent.value_v[state])

        assert math.isclose(max_diff, valid_max_diff)

    def test_single_policy_improvement(self):
        """Check policy_evaluation method with single mode is correct."""
        mock_reward_grid = self.get_reward_grid
        valid_value = self.get_valid_policies["single_iter"]
        lamb = 0.1

        self.agent.policy_evaluation(mock_reward_grid)
        self.agent.policy_improvement()

        for state in self.all_states:
            assert self.agent.policy[state] == valid_value[state], f"{state}"

    def test_get_action_with_initial_policy(self):
        """Check correct action according to the initial value."""
        for state, action in self.get_valid_policies["initial"].items():
            # actions which can be the output of the initial policy
            valid_actions = set()
            for action_str in action:
                valid_actions.add(self.agent.actions[action_str])
            # actions which are returns of the agent get_action method
            output_actions = set()
            for _ in range(100):
                action = self.agent.get_action(state)
                output_actions.add(action)
                if len(output_actions) == len(valid_actions):
                    break
            assert output_actions == valid_actions

    def test_get_action_with_single_iter_policy(self):
        """Check correct action according to the initial value."""

        mock_reward_grid = self.get_reward_grid
        lamb = 0.1

        self.agent.policy_evaluation(mock_reward_grid)
        self.agent.policy_improvement()

        for state, action in self.get_valid_policies["single_iter"].items():
            # actions which can be the output of the initial policy
            valid_actions_str = self.get_max_prob_actions(
                self.get_valid_policies["single_iter"][state]
            )

            valid_actions = set()
            for action_str in valid_actions_str:
                valid_actions.add(self.agent.actions[action_str])
            # actions which are returns of the agent get_action method
            output_actions = set()
            for _ in range(100):
                action = self.agent.get_action(state)
                output_actions.add(action)
                if len(output_actions) == len(valid_actions):
                    break
            assert output_actions == valid_actions, "with state {}".format(state)

    def get_max_prob_actions(self, action_prob: Dict[str, float]) -> Set[str]:
        """Get actions with max probabilities."""
        max_actions = set()
        max_prob = 0.0
        for action, prob in action_prob.items():
            if prob > max_prob:
                max_actions = {action}
                max_prob = prob
            elif prob == max_prob:
                max_actions.add(action)
        return max_actions

    @property
    def get_reward_grid(self) -> np.array:
        return np.array(
            [
                [0, 0, 0, -10],
                [0, -10, 0, 0],
                [0, 0, 0, 100],
            ]
        )

    @property
    def get_valid_state_values(self) -> Dict[str, np.ndarray]:

        initial = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        single_iter = np.array(
            [
                [0, -10 / 3, -10 / 3, 0],
                [-10 / 3, 0, -10 / 4, 30],
                [0, -10 / 3, 100 / 3, 0],
            ]
        )

        valid_state_values = dict(initial=dict(), single_iter=dict())
        for grid_cell in itertools.product(range(ROW), range(COL)):
            valid_state_values["initial"][grid_cell] = initial[grid_cell]
            valid_state_values["single_iter"][grid_cell] = single_iter[grid_cell]

        return valid_state_values

    @property
    def get_valid_policies(self) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
        """Get valid policies."""
        valid_policies = dict(
            # policy after single policy iteration
            initial={
                (0, 0): {"down": 1 / 2, "right": 1 / 2},
                (0, 1): {"left": 1 / 3, "down": 1 / 3, "right": 1 / 3},
                (0, 2): {"left": 1 / 3, "down": 1 / 3, "right": 1 / 3},
                (0, 3): {"left": 1 / 2, "down": 1 / 2},
                (1, 0): {
                    "up": 1 / 3,
                    "down": 1 / 3,
                    "right": 1 / 3,
                },
                (1, 1): {"up": 1 / 4, "left": 1 / 4, "down": 1 / 4, "right": 1 / 4},
                (1, 2): {"up": 1 / 4, "left": 1 / 4, "down": 1 / 4, "right": 1 / 4},
                (1, 3): {"up": 1 / 3, "left": 1 / 3, "down": 1 / 3},
                (2, 0): {"up": 1 / 2, "right": 1 / 2},
                (2, 1): {"left": 1 / 3, "up": 1 / 3, "right": 1 / 3},
                (2, 2): {"left": 1 / 3, "up": 1 / 3, "right": 1 / 3},
                (2, 3): {"left": 1 / 2, "up": 1 / 2},
            },
            # policy after single policy iteration
            single_iter={
                (0, 0): {"down": 1 / 2, "right": 1 / 2},
                (0, 1): {"left": 1 / 2, "down": 1 / 2, "right": 0},
                (0, 2): {"left": 0, "down": 0, "right": 1},
                (0, 3): {"left": 0, "down": 1},
                (1, 0): {
                    "up": 1 / 3,
                    "down": 1 / 3,
                    "right": 1 / 3,
                },
                (1, 1): {"up": 0, "left": 0, "down": 0, "right": 1},
                (1, 2): {"up": 0, "left": 0, "down": 1, "right": 0},
                (1, 3): {"up": 1 / 2, "left": 0, "down": 1 / 2},
                (2, 0): {"up": 1 / 2, "right": 1 / 2},
                (2, 1): {"left": 0, "up": 0, "right": 1},
                (2, 2): {"left": 0, "up": 0, "right": 1},
                (2, 3): {"left": 1, "up": 0},
            },
            # policy only up and down
            up_and_down={
                (0, 0): {"down": 1, "right": 0},
                (0, 1): {"left": 0, "down": 1, "right": 0},
                (0, 2): {"left": 0, "down": 1, "right": 0},
                (0, 3): {"left": 0, "down": 1},
                (1, 0): {
                    "up": 0,
                    "down": 1,
                    "right": 0,
                },
                (1, 1): {"up": 0, "left": 0, "down": 1, "right": 0},
                (1, 2): {"up": 0, "left": 0, "down": 1, "right": 0},
                (1, 3): {"up": 0, "left": 0, "down": 1},
                (2, 0): {"up": 1, "right": 0},
                (2, 1): {"left": 0, "up": 1, "right": 0},
                (2, 2): {"left": 0, "up": 1, "right": 0},
                (2, 3): {"left": 0, "up": 1},
            },
        )
        return valid_policies

    def get_invalid_policies(
        self,
    ) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
        """Get invalid policies."""
        invalid_policies = dict(
            # invalid policy
            # - (0,0)'s sum is over 1
            # - (1,1)'s sum is under 1
            # - (1,3)'s action probability is all 0
            # - (2,2)'s action probability is all 1
            sum_is_not_one={
                (0, 0): {"down": 1, "right": 2},
                (0, 1): {"left": 1 / 2, "down": 1 / 2, "right": 0},
                (0, 2): {"left": 0, "down": 0, "right": 1},
                (0, 3): {"left": 0, "down": 1},
                (1, 0): {
                    "up": 1 / 3,
                    "down": 1 / 3,
                    "right": 1 / 3,
                },
                (1, 1): {"up": 0, "left": 0, "down": 1 / 2, "right": 1 / 3},
                (1, 2): {"up": 0, "left": 0, "down": 1, "right": 0},
                (1, 3): {"up": 0, "left": 0, "down": 0},
                (2, 0): {"up": 1 / 2, "right": 1 / 2},
                (2, 1): {"left": 0, "up": 0, "right": 1},
                (2, 2): {"left": 1, "up": 1, "right": 1},
                (2, 3): {"left": 1, "up": 0},
            }
        )
        return invalid_policies

    @property
    def get_available_actions(self):
        """Get available actions for each state of ROW=3, COL=4"""
        return {
            (0, 0): {
                "down",
                "right",
            },
            (0, 1): {
                "left",
                "down",
                "right",
            },
            (0, 2): {
                "left",
                "down",
                "right",
            },
            (0, 3): {
                "left",
                "down",
            },
            (1, 0): {
                "up",
                "down",
                "right",
            },
            (1, 1): {
                "up",
                "left",
                "down",
                "right",
            },
            (1, 2): {
                "up",
                "left",
                "down",
                "right",
            },
            (1, 3): {
                "up",
                "left",
                "down",
            },
            (2, 0): {
                "up",
                "right",
            },
            (2, 1): {
                "up",
                "left",
                "right",
            },
            (2, 2): {
                "up",
                "left",
                "right",
            },
            (2, 3): {
                "up",
                "left",
            },
        }
