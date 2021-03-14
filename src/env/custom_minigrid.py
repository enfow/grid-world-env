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
        - Inherit MiniGridEnv
        - there are 4 actions: left, right, up, down
        - when agent arrives at lava, get -10 point
        - when agent arrives at goal, get 100 point
        - Valid Area:
            - The boundary cells are always wall.
            - The argument width and height define the size of the valid area.
                which does not include the wall.
            - It is the reason why 2 is added to width and height.
            - The start, obastacle and goal positions should consider the walls
                too. It can be checked and adjusted using
                `self.__adjust_pos_consider_walls` method.
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
        self.valid_width = width
        self.valid_height = height
        self.width: int = width + 2  # add 2 for surrounding wall
        self.height: int = height + 2  # add 2 for surrounding wall

        # Current position and direction of the agent
        agent_initial_pos = [(0, 0)]
        self.agent_pos: Tuple[int, int] = self.__adjust_pos_consider_walls(
            agent_initial_pos
        )[0]
        self.agent_dir: int = 0

        # Setting for obstacles
        self.obstacle_type: WorldObj = obstacle_type
        self.obstacle_pos = obstacle_pos

        self.goal_pos = [(self.valid_height - 1, self.valid_width - 1)]

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
        self.reward_range = (-10, 100)
        self.reward_grid: np.ndarray = np.zeros((self.valid_height, self.valid_width))

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.grid_size: Tuple[int, int] = (height, width)
        self.max_steps: int = max_steps
        self.see_through_walls: bool = see_through_walls

        # Initialize the RNG
        self.seed(seed=seed)

        self.mission = None

        # Initialize the state
        self.reset()

    def __set_grid_type(self, height: int, width: int, grid_type: WorldObj) -> None:
        """Set grid type.

        Notes:
            - Grid.set() method's argument order is little bit confusing.
                (width, hieght, type) not (height, width, type).
        """
        self.grid.set(width, height, grid_type)

    def __get_grid_type(self, height: int, width: int) -> WorldObj:
        """Set grid type.

        Notes:
            - Grid.set() method's argument order is little bit confusing.
                (width, hieght, type) not (height, width, type).
        """
        return self.grid.get(width, height)

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate grid space.

        Jobs:
            - create grid world
            - create wall
            - set starting point
            - set goal
            - set lava
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
        for goal_pos in self.__adjust_pos_consider_walls(self.goal_pos):
            self.__set_grid_type(*goal_pos, Goal())

        # Create Lava
        if self.obstacle_pos:
            for lava_pos in self.__adjust_pos_consider_walls(self.obstacle_pos):
                self.__set_grid_type(*lava_pos, self.obstacle_type())

        # Settings for reward_grid
        for goal_pos in self.goal_pos:
            self.reward_grid[goal_pos] = self.get_goal_reward()
        if self.obstacle_pos:
            for lava_pos in self.obstacle_pos:
                self.reward_grid[lava_pos] = self.get_lava_reward()

    def __adjust_pos_consider_walls(
        self, positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Check validity of the input positions and adjust it with walls."""
        if not positions:
            return list()
        position_with_walls = list()
        for row, col in positions:
            assert row >= 0
            assert row <= self.height - 2
            assert col >= 0
            assert col <= self.width - 2
            position_with_walls.append((row + 1, col + 1))

        return position_with_walls

    def __get_pos_on_valid_area(self) -> Tuple[int, int]:
        """Get agent position."""
        row, col = self.agent_pos
        return (row - 1, col - 1)

    def gen_obs(self) -> Dict[str, Any]:
        """Wrap the parent's gen_obs method for additional observation.

        Notes:
            - original obs: image(np.array)
            - Added obs: pos(Tuple[int, int]), reward_grid(np.array)
        """
        obs = super().gen_obs()
        obs.update(
            pos=self.__get_pos_on_valid_area(),
            reward_grid=self.reward_grid,
        )

        return obs

    def step(self, action: int) -> Tuple[Dict[str, Any], int, bool, Dict[str, Any]]:
        """Take action."""
        self.step_count += 1

        reward, done = self.step_forward(action)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def __get_forward_pos_and_agent_dir(
        self, action: int
    ) -> Tuple[Tuple[int, int], int]:
        """Get forward position with action.

        Notes:
            - actions:
                - left: 0
                - right: 1
                - up: 2
                - down: 3
            - agent_dir(MiniGridEnv):
                - left: 2
                - right: 0
                - up: 3
                - down: 1
            - agent_pos in MiniGridEnv has form (column, row) not (row, column).
                So the return value switch the order of the agent position for
                forward position.
        """
        cur_c, cur_r = self.agent_pos
        # change direction with action value
        if action == self.actions["right"]:
            agent_dir = 0
            cur_c += 1
        elif action == self.actions["down"]:
            agent_dir = 1
            cur_r += 1
        elif action == self.actions["left"]:
            agent_dir = 2
            cur_c -= 1
        elif action == self.actions["up"]:
            agent_dir = 3
            cur_r -= 1
        else:
            raise NotImplementedError("Unknown action {}".format(action))

        return (cur_r, cur_c), agent_dir

    def step_forward(self, action: int) -> Tuple[int, bool]:
        """Move agent with action."""
        reward = 0
        done = False

        # get information about the forward cell
        fwd_pos, self.agent_dir = self.__get_forward_pos_and_agent_dir(action)
        fwd_cell = self.__get_grid_type(*fwd_pos)
        fwd_r, fwd_c = fwd_pos

        # forward cell is empty
        if fwd_cell is None:
            self.agent_pos = (fwd_c, fwd_r)
        # forward cell is goal
        elif fwd_cell.type == "goal":
            self.agent_pos = (fwd_c, fwd_r)
            reward = self.get_goal_reward()
            done = True
        # forward cell is lava
        elif fwd_cell is not None and fwd_cell.type == "lava":
            self.agent_pos = (fwd_c, fwd_r)
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
