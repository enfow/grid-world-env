"""Define Custom Environment.

Owner: kyeongmin woo
Email: wgm0601@gmail.com

Reference:
    - https://github.com/maximecb/gym-minigrid
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gym_minigrid.minigrid import (Goal, Grid, Lava, MiniGridEnv, WorldObj,
                                   spaces)

VALID_ACTIONS: Dict[str, int] = dict(left=0, right=1, up=2, down=3)


class CustomLavaEnv(MiniGridEnv):
    """Define custom lava environment.

    Notes:
        - grid world
        - there are 4 actions: left, right, up, down
        - when agent arrives at lava, get -10 point
        - when agent arrives at goal, get 100 point
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        max_steps: int = 100,
        see_through_walls: bool = False,
        seed: int = 1,
        agent_view_size: int = 7,
        obstacle_type: WorldObj = Lava,
        obstacle_pos: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """Initialize."""
        self.obstacle_type: WorldObj = obstacle_type
        self.obstacle_pos = obstacle_pos

        # Action enumeration for this environment
        self.actions = VALID_ACTIONS

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict({"image": self.observation_space})

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.grid_size: Tuple[int, int] = (width, height)
        self.width: int = width + 2  # add 2 for surrounding wall
        self.height: int = height + 2  # add 2 for surrounding wall
        self.max_steps: int = max_steps
        self.see_through_walls: bool = see_through_walls

        # Current position and direction of the agent
        self.agent_pos: Tuple[int, int] = (1, 1)
        self.agent_dir: int = 0

        # Initialize the RNG
        self.seed(seed=seed)

        self.mission = None

        # Initialize the state
        self.reset()

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate grid space.

        Jobs:
            - create grid world
            - create wall
            - set starting point
            - set goal
            - setl lava
        """
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Create wall
        self.grid.wall_rect(0, 0, width, height)

        # Starting point
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Create Goal
        self.goal_pos: np.ndarray = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate Lava with obstacle_pos
        if self.obstacle_pos:
            for lava_pos in self.obstacle_pos:
                self.grid.set(*lava_pos, self.obstacle_type())

    def step(self, action: int) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """Take action."""
        self.step_count += 1

        reward, done = self.step_forward(action)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def step_forward(self, action: int) -> Tuple[int, bool]:
        """Move agent with action."""
        reward = 0
        done = False

        # change direction with action value
        if action == self.actions["left"]:
            self.agent_dir = 0
        elif action == self.actions["up"]:
            self.agent_dir = 3
        elif action == self.actions["right"]:
            self.agent_dir = 2
        elif action == self.actions["down"]:
            self.agent_dir = 1

        # get information about the forward cell
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # forward cell is empty
        if fwd_cell is None:
            self.agent_pos = fwd_pos
        # forward cell is goal
        elif fwd_cell.type == "goal":
            self.agent_pos = fwd_pos
            reward = self.get_goal_reward()
            done = True
        # forward cell is lava
        elif fwd_cell is not None and fwd_cell.type == "lava":
            self.agent_pos = fwd_pos
            reward = self.get_lava_reward()
        # forward cell is Wall
        elif fwd_cell is not None and fwd_cell.type == "wall":
            pass
        # unknown type
        else:
            AssertionError("unknown action")

        return reward, done

    def get_goal_reward(self) -> int:
        """Get the reward that agent receive when it arrive at the goal."""
        return 100

    def get_lava_reward(self) -> int:
        """Get the reward that agent receive when it arrive at the lava."""
        return -10
