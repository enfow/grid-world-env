"""Define random policy."""

from typing import Any, Dict

import gym


class RandomPolicy:
    """Define random policy for grid-world."""

    def __init__(self, action_space: gym.spaces.discrete.Discrete) -> None:
        """Initialize."""
        self.action_space = action_space

    def get_action(self, obs: Dict[str, Any]) -> int:  # pylint: disable=unused-argument
        """Get random action.

        Notes:
            - unuse parameter obs - To share the same interface with other policy
        """
        return self.action_space.sample()
