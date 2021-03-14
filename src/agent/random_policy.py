"""Define random policy."""

from typing import Any, Dict

from agent.abstract_agent import AbstractAgent


class RandomPolicy(AbstractAgent):
    """Define random policy for grid-world."""

    def get_action(
        self, state: Dict[str, Any]
    ) -> int:  # pylint: disable=unused-argument
        """Get random action."""
        return self.action_space.sample()

    def update_policy(
        self, update_info: Dict[str, Any]
    ) -> None:  # pylint: disable=unused-argument
        """Pass because it is random policy."""
        pass
