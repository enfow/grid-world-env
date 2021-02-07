"""Test cases for environments and agents."""
import itertools
import math

import numpy as np

from agent import DPStateAgent
from env import VALID_ACTIONS, GridWorld

ROW = 3
COL = 4


class TestGridWorld:
    """Test cases for GridWorld."""

    def setup_class(self):
        """Initialize common attributes.."""
        self.grid = (ROW, COL)
        self.all_states = list(itertools.product(range(ROW), range(COL)))

    def setup_method(self):
        """Initialize the GridWorld environment for each test case."""
        self.env = GridWorld(grid=self.grid, start=(0, 0), finish=(ROW - 1, COL - 1))

    def test_reward_grid_of_env(self):
        """Check the shape of reward_grid(row/column) is correct.

        CheckList:
            - row of the env.reward_gird
            - column of the env.reward_grid
        """
        row, col = self.env.reward_grid.shape
        assert row == ROW
        assert col == COL

    def test_initiale_reward_grid(self):
        """Check changing reward correctly with env.update_reward method.

        CheckList:
            - env.update_reward method update single reward_grid cell at once.
            - env.update_reward method update only the input state's reward.
        """
        for state in self.all_states:
            assert self.env.reward_grid[state] == 0

    def test_update_reward_grid(self):
        """Check changing reward correctly with env.update_reward method.

        CheckList:
            - env.update_reward method update single reward_grid cell at once.
            - env.update_reward method update only the input state's reward.
        """
        valid_values = {
            (1, 1): 10,
            (1, 2): -10,
            (2, 3): 5,
        }

        for update_state in valid_values:
            self.env.reward_grid[update_state] = valid_values[update_state]

        for state in self.all_states:
            if state in valid_values:
                assert self.env.reward_grid[state] == valid_values[state]
            else:
                assert self.env.reward_grid[state] == 0


class TestDPStateAgent:
    """Test cases for DPStateAgent."""

    def setup_class(self):
        """Initialize commne attributes."""
        self.grid = (ROW, COL)
        self.all_states = list(itertools.product(range(ROW), range(COL)))

    def setup_method(self):
        """Initialize DPStateAgent for each test case"""
        self.agent = DPStateAgent(grid=self.grid, lamb=0.1)

    def test_initial_policy(self):
        """Check initial policy of agent is correct.

        CheckList:
            - the agent.policy is dictionary
            - the keys of env.policy should include all possible states.
        """
        valid_value = ROW * COL

        assert isinstance(self.agent.policy, dict)
        assert len(self.agent.policy) == valid_value

        valid_vaue = self.get_available_actions
        for state in self.all_states:
            actions = self.agent.policy[state].keys()
            probs = list(self.agent.policy[state].values())
            assert actions == valid_vaue[state], "problem on state {}".format(state)
            assert probs == [1 / len(actions) for _ in actions]

    def test_initial_value(self):
        """Check size of the value dictionary

        CheckList:
            - agent.value is dictionary
            - the keys of env.value should include all possible states.
        """
        valid_value = ROW * COL

        assert isinstance(self.agent.value, np.ndarray)
        assert self.agent.value.size == valid_value

    def test_single_policy_improvement(self):
        """Check policy_evaluation method with single mode is correct."""
        mock_reward_grid = np.array(
            [
                [0, 0, 0, -10],
                [0, -10, 0, 0],
                [0, 0, 0, 100],
            ]
        )
        lamb = 0.1
        valid_value = np.array(
            [
                [0, -10 / 3, -10 / 3, 0],
                [-10 / 3, 0, -10 / 4, 30],
                [0, -10 / 3, 100 / 3, 0],
            ]
        )
        valid_max_diff = (abs(valid_value)).max()

        max_diff = self.agent.policy_evaluation(mock_reward_grid)

        for state in self.all_states:
            assert math.isclose(valid_value[state], self.agent.value[state])

        assert math.isclose(max_diff, valid_max_diff)

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
