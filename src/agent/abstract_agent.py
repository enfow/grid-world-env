"""Define AbstractAgent."""
from typing import Any, Dict, Tuple

import gym


class AbstractAgent:
    """Define Abstract Agent."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize."""
        self.actions: Dict[str, int] = env.actions
        self.action_space = env.action_space
        self.row, self.col = env.grid_size
        self.cur_pos = None

    def get_action(self, state: Tuple[int, int]) -> int:
        """Get action according to current policy."""
        raise NotImplementedError

    def update_policy(self, update_info: Dict[str, Any]) -> None:
        """Update policy with experiences."""
        raise NotImplementedError
